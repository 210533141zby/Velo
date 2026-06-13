from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.config import settings

_chat_model_instance = None
_rag_judge_model_instance = None
_rag_contextual_model_instance = None


def _build_chat_model(temperature: float | None = None, max_tokens: int | None = None) -> ChatOpenAI:
    """按统一配置构建一个 `ChatOpenAI` 实例。

    所有聊天类模型都从这里出厂，保证 API 地址、模型名和超时设置口径一致；
    不同用途之间只通过温度和输出长度做少量差异化。
    """
    resolved_temperature = settings.CHAT_TEMPERATURE if temperature is None else temperature
    return ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.chat_api_base,
        model=settings.CHAT_MODEL,
        temperature=resolved_temperature,
        timeout=120,
        max_retries=1,
        max_tokens=max_tokens,
    )


def get_chat_model() -> ChatOpenAI:
    """返回系统默认对话模型的单例实例。

    主回答链路会频繁调用该模型，做成单例可以避免重复初始化底层客户端。
    """
    global _chat_model_instance
    if _chat_model_instance is None:
        _chat_model_instance = _build_chat_model()
    return _chat_model_instance


def get_rag_judge_model() -> ChatOpenAI:
    """返回 RAG 判别阶段使用的低温模型实例。

    证据判断更强调输出稳定性，因此这里固定使用更保守的温度和更短的输出预算。
    """
    global _rag_judge_model_instance
    if _rag_judge_model_instance is None:
        _rag_judge_model_instance = _build_chat_model(
            temperature=0.0,
            max_tokens=settings.RAG_JUDGE_MAX_TOKENS,
        )
    return _rag_judge_model_instance


def get_rag_contextual_model() -> ChatOpenAI:
    """返回用于 contextual retrieval 前缀生成的模型实例。

    该模型不直接回答用户问题，而是给切分片段补充最小必要的文档级背景。
    """
    global _rag_contextual_model_instance
    if _rag_contextual_model_instance is None:
        _rag_contextual_model_instance = _build_chat_model(
            temperature=0.0,
            max_tokens=settings.RAG_CONTEXTUAL_MAX_TOKENS,
        )
    return _rag_contextual_model_instance
