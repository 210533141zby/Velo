from __future__ import annotations

import asyncio
from typing import Any, Sequence

from app.core.config import settings
from app.logger import logger
from app.services.model_factory import get_chat_model
from app.services.rag.evidence_packager import EvidencePackager
from app.services.rag.llm_json_utils import extract_json_dict
from app.services.rag.pipeline_models import EvidenceAssessment, RetrievedCandidate
from app.services.rag.prompt_templates import build_deep_review_prompt
from app.services.rag.text_utils import compact_text

_packager = EvidencePackager()


def _review_rank(candidate: RetrievedCandidate) -> tuple[float, float, float]:
    """返回复核结果排序时使用的分数字段。"""
    return (
        float(candidate.rerank_score),
        float(candidate.adaptive_score),
        float(candidate.coverage_score),
    )


def _review_score(candidate: RetrievedCandidate) -> float:
    """把候选文档的多路信号压成深度复核使用的单一分数。

    深度复核阶段不再重新做一整轮证据评分，而是基于已有的 rerank、自适应分
    和覆盖分给候选文档一个轻量综合分，用来决定谁值得进入复核提示词。
    """
    return max(
        0.0,
        min(
            1.0,
            float(candidate.rerank_score) * 0.56
            + float(candidate.adaptive_score) * 0.32
            + float(candidate.coverage_score) * 0.12,
        ),
    )


def _candidate_assessments(candidates: Sequence[RetrievedCandidate]) -> list[EvidenceAssessment]:
    """把候选文档临时包装成 EvidenceAssessment 列表。

    EvidencePackager 的输入是 assessment，因此这里把深度复核候选文档
    先临时映射成“默认可用”的评估对象，方便直接复用已有打包逻辑。
    """
    ordered_candidates = sorted(candidates, key=_review_rank, reverse=True)
    assessments: list[EvidenceAssessment] = []
    for candidate in ordered_candidates[: settings.RAG_DEEP_REVIEW_TOP_K]:
        assessments.append(
            EvidenceAssessment(
                candidate=candidate,
                final_score=_review_score(candidate),
                usable=True,
            )
        )
    return assessments


def _build_context(query: str, candidates: Sequence[RetrievedCandidate]) -> str:
    """为深度复核提示词构建证据上下文。

    这里会复用证据打包器，把前几名候选整理成适合复核模型阅读的证据包，
    避免 deep review 路径再维护一套独立的上下文拼装逻辑。
    """
    packet = _packager.pack(
        query,
        _candidate_assessments(candidates),
        max_units=min(max(settings.RAG_DEEP_REVIEW_TOP_K, 1), 8),
        char_budget=settings.RAG_DEEP_REVIEW_CONTEXT_CHARS,
        prefer_full_content=True,
    )
    return _packager.render_for_review(packet)


def _normalize_used_ranks(value: Any, limit: int) -> tuple[int, ...]:
    """规范化复核阶段记录的使用排名。"""
    if not isinstance(value, list):
        return ()
    ranks: list[int] = []
    for item in value:
        try:
            rank = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= rank <= limit and rank not in ranks:
            ranks.append(rank)
    return tuple(ranks)


def _empty_review() -> dict[str, Any]:
    """生成空复核结果。"""
    return {
        'verdict': 'no_context',
        'answer': '',
        'used_ranks': (),
    }


def _fallback_review_answer(candidates: Sequence[RetrievedCandidate]) -> dict[str, Any]:
    """在深度复核超时或失败时，直接用高排位候选拼出保底答案。

    该兜底仍然保持“引用自已检索证据”的约束，只是跳过额外模型判断，
    以避免演示场景里因为复核模型波动而整条回答变空。
    """
    snippets: list[str] = []
    used_ranks: list[int] = []
    for index, candidate in enumerate(candidates[:3], start=1):
        text = compact_text(candidate.chunk_text or candidate.full_content or getattr(candidate.doc, 'page_content', ''), 220).strip()
        if not text:
            continue
        snippets.append(text)
        used_ranks.append(index)
    if not snippets:
        return _empty_review()
    return {
        'verdict': 'answer',
        'answer': compact_text('；'.join(snippets), 1600),
        'used_ranks': tuple(used_ranks),
    }


async def deep_review_and_answer(query: str, candidates: Sequence[RetrievedCandidate]) -> dict[str, Any]:
    """执行深度复核并生成答案。"""
    if not candidates:
        return _empty_review()

    review_candidates = sorted(candidates, key=_review_rank, reverse=True)[: settings.RAG_DEEP_REVIEW_TOP_K]
    try:
        response = await asyncio.wait_for(
            get_chat_model().ainvoke(build_deep_review_prompt(query, _build_context(query, review_candidates))),
            timeout=20.0,
        )
        payload = extract_json_dict(getattr(response, 'content', ''))
        verdict = str(payload.get('verdict') or '').strip().lower()
        answer = compact_text(str(payload.get('answer') or '').strip(), 1600)
        used_ranks = _normalize_used_ranks(payload.get('used_ranks'), len(review_candidates))
        result = {
            'verdict': 'answer' if verdict == 'answer' and answer else 'no_context',
            'answer': answer,
            'used_ranks': used_ranks,
        }
        if result['verdict'] == 'answer' and not result['used_ranks']:
            result['used_ranks'] = tuple(range(1, min(3, len(review_candidates)) + 1))
        return result
    except asyncio.TimeoutError:
        logger.warning(
            'Deep-Evidence Reviewer 超时，回退到证据式答案',
            extra={'extra_data': {'event': 'rag_deep_review_timeout', 'query': query}},
        )
        return _fallback_review_answer(review_candidates)
    except Exception:
        logger.exception(
            'Deep-Evidence Reviewer 执行失败',
            extra={'extra_data': {'event': 'rag_deep_review_failed', 'query': query}},
        )
        return _fallback_review_answer(review_candidates)
