"""
=============================================================================
文件: completion_service.py
描述: 补全模型调用编排层

架构职责：
1. 统一通过 chat/completions 发起补全请求，避免不同后端接口能力不一致。
2. 主补全后端失败或空返回时，按配置回退到本地兼容补全。
3. 补全约束、上下文分类和后验校验统一下沉到 completion_policy.py。
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

import httpx

from app.core.config import settings
from app.services.completion.completion_policy import (
    build_chat_messages,
    get_profile,
    infer_completion_context,
    normalize_completion_candidate,
    normalize_trigger_mode,
    post_process_completion_with_reason,
    truncate_context,
)

logger = logging.getLogger(__name__)

HTTP_TIMEOUT_SECONDS = 10.0
LOG_TEXT_PREVIEW_CHARS = 120
MAX_COMPLETION_ATTEMPTS = 2
RETRYABLE_REJECT_REASONS_BY_MODE = {
    'auto': frozenset(),
    'manual': frozenset({
        'fragmented_sentence_start',
        'unfinished_bridge_tail',
        'unterminated_sentence',
    }),
}
RETRYABLE_REJECT_REASONS_FALLBACK = frozenset({
    'fragmented_sentence_start',
    'unfinished_bridge_tail',
    'unterminated_sentence',
})
_http_client: httpx.AsyncClient | None = None


@dataclass(frozen=True)
class CompletionBackendRequest:
    backend_name: str
    model: str
    api_base: str
    url: str
    payload: dict[str, Any]
    headers: dict[str, str]
    timeout: float


class CompletionBackendError(Exception):
    def __init__(self, request: CompletionBackendRequest, exc: Exception):
        """把底层请求异常包装成补全链路可识别的错误类型。

        包装后的异常会同时保留“这次发给了哪个后端、哪个模型、什么地址”的上下文，
        便于日志层区分是主补全失败、回退补全失败，还是某个特定模型返回异常。
        """
        super().__init__(str(exc))
        self.request = request
        self.exc = exc


def get_completion_http_client() -> httpx.AsyncClient:
    """获取补全链路复用的 HTTP 客户端。

    文本补全会被高频触发，如果每次请求都新建连接，延迟和资源消耗都会偏高。
    因此这里把 HTTP 客户端做成进程级复用对象，统一复用连接池与超时配置。
    """
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS,
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=10),
        )
    return _http_client


async def close_completion_http_client() -> None:
    """关闭补全服务使用的 HTTP 客户端。"""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def _auth_headers(api_key: str | None) -> dict[str, str]:
    """生成补全请求所需的鉴权请求头。"""
    if api_key and api_key != 'ollama':
        return {'Authorization': f'Bearer {api_key}'}
    return {}


def _has_local_fallback() -> bool:
    """判断当前配置是否启用了本地补全兜底。"""
    return bool(
        settings.COMPLETION_FALLBACK_ENABLED
        and settings.COMPLETION_FALLBACK_MODEL
        and settings.COMPLETION_FALLBACK_BASE_URL
    )


def _build_chat_completion_request(
    prefix: str,
    suffix: str,
    language: str | None,
    trigger_mode: str | None,
    attempt: int,
    *,
    previous_reject_reason: str | None,
    model: str,
    api_base: str,
    api_key: str | None,
    backend_name: str,
    disable_thinking: bool = False,
) -> CompletionBackendRequest:
    """构建一次 chat/completions 请求描述。

    该函数只负责“把业务语义翻译成请求体”，不真正发网络请求。
    这样主后端、回退后端和重试请求都能复用同一套消息构造逻辑，
    并且可以在日志里完整记录本次补全调用的配置快照。
    """
    context = infer_completion_context(prefix, language)
    profile = get_profile(trigger_mode)
    payload: dict[str, Any] = {
        'model': model,
        'messages': build_chat_messages(
            prefix,
            suffix,
            context,
            trigger_mode,
            attempt,
            previous_reject_reason=previous_reject_reason,
        ),
        'max_tokens': profile.max_tokens,
        'temperature': profile.temperature,
        'stream': False,
    }
    if disable_thinking:
        payload['thinking'] = {'type': 'disabled'}
    return CompletionBackendRequest(
        backend_name=backend_name,
        model=model,
        api_base=api_base,
        url=f"{api_base.rstrip('/')}/chat/completions",
        payload=payload,
        headers=_auth_headers(api_key),
        timeout=profile.timeout,
    )


async def _execute_completion_request(request: CompletionBackendRequest) -> str:
    """执行一次底层补全请求。

    这里把 HTTP 层错误统一转换成 `CompletionBackendError`，让上层只处理
    “这次补全是否成功”和“是否需要回退/重试”，而不用关心 httpx 的具体异常种类。
    """
    client = get_completion_http_client()
    try:
        response = await client.post(
            request.url,
            json=request.payload,
            headers=request.headers,
            timeout=request.timeout,
        )
        response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise CompletionBackendError(request, exc) from exc

    result = response.json()
    if 'choices' in result and result['choices']:
        choice = result['choices'][0]
        if choice.get('message'):
            return choice['message'].get('content', '')
        return choice.get('text', '')
    return result.get('content', '')


async def _request_completion(
    prefix: str,
    suffix: str,
    language: str | None,
    trigger_mode: str | None,
    attempt: int,
    previous_reject_reason: str | None = None,
) -> tuple[str, CompletionBackendRequest]:
    """请求补全结果，并在需要时回退到兼容后端。

    当前策略是优先走主补全后端；若主后端报错，或成功返回但内容为空，
    才会按配置切到本地兼容后端。返回值除了文本本身，还会带回实际生效的请求信息，
    供后面的日志和调试信息使用。
    """
    primary_succeeded = False
    primary_request = _build_chat_completion_request(
        prefix,
        suffix,
        language,
        trigger_mode,
        attempt,
        previous_reject_reason=previous_reject_reason,
        model=settings.COMPLETION_MODEL,
        api_base=settings.completion_api_base,
        api_key=settings.completion_api_key,
        backend_name='primary_chat_completion',
        disable_thinking=settings.completion_provider == 'deepseek',
    )
    try:
        text = await _execute_completion_request(primary_request)
        primary_succeeded = True
        if text.strip() or not _has_local_fallback():
            return text, primary_request
        logger.info(
            '[GhostBackend] 主补全后端返回空结果，回退本地补全',
            extra={
                'extra_data': {
                    'event': 'completion_provider_fallback',
                    'from_backend': primary_request.backend_name,
                    'from_model': primary_request.model,
                    'reason': 'empty_model_response',
                }
            },
        )
    except CompletionBackendError as exc:
        if not _has_local_fallback():
            raise
        logger.warning(
            '[GhostBackend] 主补全后端请求失败，回退本地补全: %s',
            exc,
            extra={
                'extra_data': {
                    'event': 'completion_provider_fallback',
                    'from_backend': exc.request.backend_name,
                    'from_model': exc.request.model,
                    'from_api_base': exc.request.api_base,
                    'reason': 'primary_request_failed',
                }
            },
        )

    fallback_request = _build_chat_completion_request(
        prefix,
        suffix,
        language,
        trigger_mode,
        attempt,
        previous_reject_reason=previous_reject_reason,
        model=settings.COMPLETION_FALLBACK_MODEL,
        api_base=settings.COMPLETION_FALLBACK_BASE_URL,
        api_key=settings.COMPLETION_FALLBACK_API_KEY,
        backend_name='local_fallback_completion',
    )
    try:
        return await _execute_completion_request(fallback_request), fallback_request
    except CompletionBackendError as exc:
        if primary_succeeded:
            logger.warning(
                '[GhostBackend] 主补全后端已正常返回空结果，但本地回退失败；按无建议处理: %s',
                exc,
                extra={
                    'extra_data': {
                        'event': 'completion_fallback_failed_after_empty_primary',
                        'from_backend': primary_request.backend_name,
                        'from_model': primary_request.model,
                        'fallback_backend': exc.request.backend_name,
                        'fallback_model': exc.request.model,
                    }
                },
            )
            return '', primary_request
        raise


def _preview_text(text: str, limit: int = LOG_TEXT_PREVIEW_CHARS) -> str:
    """把长文本压缩成适合日志展示的预览片段。

    补全链路会在日志里记录前后文片段，但不能把整段正文直接打满日志。
    这里统一做换行转义和长度截断，既方便排查，又避免日志过大。
    """
    normalized = (text or '').replace('\n', '\\n').strip()
    if len(normalized) <= limit:
        return normalized
    return f'{normalized[:limit - 3]}...'


def _should_retry_after_rejection(reason: str | None, attempt: int, mode: str) -> bool:
    """判断拒绝结果是否允许继续重试。"""
    retryable_reasons = RETRYABLE_REJECT_REASONS_BY_MODE.get(mode, RETRYABLE_REJECT_REASONS_FALLBACK)
    return attempt < MAX_COMPLETION_ATTEMPTS - 1 and reason in retryable_reasons


def _should_retry_after_error(attempt: int) -> bool:
    """判断异常场景下是否还要继续重试。"""
    return attempt < MAX_COMPLETION_ATTEMPTS - 1


async def complete_text(
    prefix: str,
    suffix: str,
    language: str | None = None,
    trigger_mode: str | None = None,
) -> str:
    """执行一次补全并返回可直接插入正文的文本。

    这是面向普通调用方的轻量入口。如果上层并不关心命中的是哪个后端、
    是否触发重试、最终拒绝原因是什么，就直接走这个简化接口。
    """
    result = await complete_text_detailed(prefix, suffix, language=language, trigger_mode=trigger_mode)
    return result['completion']


async def complete_text_detailed(
    prefix: str,
    suffix: str,
    language: str | None = None,
    trigger_mode: str | None = None,
) -> dict[str, str | None]:
    """执行一次补全并返回详细元数据。

    这是补全模块的核心编排入口，整体会完成：
    1. 前后文裁剪与上下文类型识别；
    2. 构造补全请求；
    3. 处理后端失败、空返回和可重试拒绝；
    4. 对候选结果做后验审核与规范化；
    5. 输出补全文本、后端信息、拒绝原因和耗时等元数据。

    之所以保留详细返回结构，是因为前端状态提示、后端日志和实验排查
    都需要知道“这次为什么没给补全”而不只是空字符串。
    """
    started_at = time.perf_counter()
    mode = normalize_trigger_mode(trigger_mode)
    safe_prefix, safe_suffix = truncate_context(prefix, suffix, mode)
    context = infer_completion_context(safe_prefix, language)

    logger.info(
        '[GhostBackend] 收到补全请求',
        extra={
            'extra_data': {
                'event': 'completion_request_received',
                'mode': mode,
                'language': language or '',
                'block_kind': context.block_kind,
                'prefer_sentence_continuation': context.prefer_sentence_continuation,
                'prefix_len': len(prefix),
                'suffix_len': len(suffix),
                'prefix_tail': _preview_text(safe_prefix[-80:]),
                'suffix_head': _preview_text(safe_suffix[:80]),
            }
        },
    )

    last_reason: str | None = None
    for attempt in range(MAX_COMPLETION_ATTEMPTS):
        try:
            generated_text, backend_request = await _request_completion(
                safe_prefix,
                safe_suffix,
                language,
                mode,
                attempt,
                previous_reject_reason=last_reason,
            )
            normalized_candidate = normalize_completion_candidate(generated_text, safe_prefix, safe_suffix, mode)
            logger.info(
                '[GhostBackend] 模型已返回补全文本',
                extra={
                    'extra_data': {
                        'event': 'completion_model_response',
                        'mode': mode,
                        'backend': backend_request.backend_name,
                        'model': backend_request.model,
                        'attempt': attempt + 1,
                        'raw_len': len(generated_text or ''),
                        'raw_preview': _preview_text(generated_text),
                        'normalized_preview': _preview_text(normalized_candidate),
                        'elapsed_ms': int((time.perf_counter() - started_at) * 1000),
                    }
                },
            )
            candidate, reject_reason = post_process_completion_with_reason(
                generated_text,
                safe_prefix,
                safe_suffix,
                context,
                mode,
            )
            if candidate:
                logger.info(
                    '[GhostBackend] 接受补全',
                    extra={
                        'extra_data': {
                            'event': 'completion_candidate_accepted',
                            'mode': mode,
                            'backend': backend_request.backend_name,
                            'model': backend_request.model,
                            'attempt': attempt + 1,
                            'chars': len(candidate),
                            'candidate_preview': _preview_text(candidate),
                            'elapsed_ms': int((time.perf_counter() - started_at) * 1000),
                        }
                    },
                )
                return {'completion': candidate, 'reason': None}
            last_reason = reject_reason
            logger.info(
                '[GhostBackend] 丢弃补全',
                extra={
                    'extra_data': {
                        'event': 'completion_candidate_rejected',
                        'mode': mode,
                        'backend': backend_request.backend_name,
                        'model': backend_request.model,
                        'attempt': attempt + 1,
                        'reason': reject_reason,
                        'raw_len': len(generated_text or ''),
                        'raw_preview': _preview_text(generated_text),
                        'normalized_preview': _preview_text(normalized_candidate),
                        'elapsed_ms': int((time.perf_counter() - started_at) * 1000),
                    }
                },
            )
            if not _should_retry_after_rejection(reject_reason, attempt, mode):
                return {'completion': '', 'reason': last_reason}
        except CompletionBackendError as exc:
            logger.error(
                'LLM 请求失败: %s',
                exc,
                extra={
                    'extra_data': {
                        'event': 'completion_request_failed',
                        'mode': mode,
                        'attempt': attempt + 1,
                        'backend': exc.request.backend_name,
                        'model': exc.request.model,
                        'api_base': exc.request.api_base,
                        'elapsed_ms': int((time.perf_counter() - started_at) * 1000),
                    }
                },
            )
            last_reason = 'request_failed'
            if _should_retry_after_error(attempt):
                continue
            return {'completion': '', 'reason': last_reason}
        except Exception as exc:
            logger.exception(
                '[GhostBackend] 代码补全服务异常: %s',
                exc,
                extra={
                    'extra_data': {
                        'event': 'completion_internal_error',
                        'mode': mode,
                        'attempt': attempt + 1,
                        'elapsed_ms': int((time.perf_counter() - started_at) * 1000),
                    }
                },
            )
            last_reason = 'internal_error'
            if _should_retry_after_error(attempt):
                continue
            return {'completion': '', 'reason': last_reason}

    return {'completion': '', 'reason': last_reason or 'no_suggestion'}
