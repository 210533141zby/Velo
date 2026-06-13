from __future__ import annotations

import re
from typing import Any

from app.core.config import settings
from app.logger import logger
from app.services.model_factory import get_rag_judge_model
from app.services.rag.llm_json_utils import extract_json_dict
from app.services.rag.prompt_templates import build_document_judge_prompt
from app.services.rag.text_utils import compact_text

JUDGE_EVIDENCE_LIMIT = 160
JUDGE_ANSWER_LIMIT = 120


def _normalize_yes_no_flag(value: Any) -> bool | None:
    """把模型返回的各种“是/否”表达折叠成布尔值。

    判别模型可能返回 yes/no、true/false、中文是/否，甚至带少量额外词语；
    这里统一收口，减少后续判断分支。
    """
    if isinstance(value, bool):
        return value

    normalized = re.sub(r'[^0-9a-z\u4e00-\u9fff]+', '', str(value or '').strip().lower())
    if normalized in {'yes', 'true', '1', '是', '能', '可以', '可回答', '通过', 'accept'}:
        return True
    if normalized in {'no', 'false', '0', '否', '不能', '不可以', '不可回答', '拒绝', 'reject'}:
        return False
    return None


def _normalize_document_judge_result(payload: dict[str, Any]) -> dict[str, Any]:
    """清洗单文档判别结果，补齐默认值并裁剪可展示字段。

    这个步骤把模型输出从“原始 JSON”转换成后续评分器可直接使用的稳定结构，
    同时约束证据摘录、简答摘要和拒绝原因的长度。
    """
    core_topic_match = _normalize_yes_no_flag(payload.get('core_topic_match') or payload.get('topic_match'))
    contains_direct_evidence = _normalize_yes_no_flag(
        payload.get('contains_direct_evidence') or payload.get('direct_evidence') or payload.get('supported')
    )
    answerable = _normalize_yes_no_flag(payload.get('answerable') or payload.get('can_answer'))
    if answerable is None:
        answerable = bool(core_topic_match) and bool(contains_direct_evidence)
    elif core_topic_match is False or contains_direct_evidence is False:
        answerable = False

    evidence_quote = compact_text(str(payload.get('evidence_quote') or payload.get('evidence') or ''), JUDGE_EVIDENCE_LIMIT)
    answer_brief = compact_text(str(payload.get('answer_brief') or payload.get('answer') or ''), JUDGE_ANSWER_LIMIT)
    reason = compact_text(str(payload.get('reason') or payload.get('diagnosis') or ''), 24)

    if not answerable:
        evidence_quote = ''
        answer_brief = ''

    return {
        'core_topic_match': bool(core_topic_match) if core_topic_match is not None else False,
        'contains_direct_evidence': bool(contains_direct_evidence) if contains_direct_evidence is not None else False,
        'answerable': bool(answerable),
        'evidence_quote': evidence_quote,
        'answer_brief': answer_brief,
        'reason': reason,
    }


async def judge_rag_document(query: str, title: str, content: str) -> dict[str, Any]:
    """调用判别模型，判断一篇候选文档能否直接支撑当前问题。

    当系统启用判别步骤时，这个函数会让模型分别判断主题是否对齐、
    是否存在直接证据，以及能否安全作答；失败时则回退到更保守的规则兜底。
    """
    if not settings.RAG_JUDGE_ENABLED:
        return {
            'core_topic_match': True,
            'contains_direct_evidence': True,
            'answerable': True,
            'evidence_quote': '',
            'answer_brief': '',
            'reason': '',
        }

    payload: dict[str, Any] = {}
    compact_content = compact_text(content, settings.RAG_JUDGE_CONTEXT_CHARS)
    try:
        response = await get_rag_judge_model().ainvoke(build_document_judge_prompt(query, title, compact_content))
        payload = extract_json_dict(getattr(response, 'content', ''))
    except Exception:
        logger.exception(
            'RAG 文档判别失败，回退到规则防御',
            extra={'extra_data': {'event': 'rag_document_judge_failed', 'query': query, 'title': title}},
        )
        return {
            'core_topic_match': True,
            'contains_direct_evidence': True,
            'answerable': True,
            'evidence_quote': '',
            'answer_brief': '',
            'reason': 'judge_fallback',
        }

    return _normalize_document_judge_result(payload)
