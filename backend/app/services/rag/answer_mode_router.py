from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.services.rag.pipeline_models import (
    AnswerMode,
    AnswerPlan,
    EvidenceAssessment,
    EvidenceRequirement,
    QuestionFocusType,
    QueryIntent,
)
from app.services.rag.score_distribution import summarize_scores


@dataclass(frozen=True)
class RouteSignals:
    unique_doc_count: int
    usable_count: int
    primary_score: float
    secondary_score: float
    primary_gap: float
    clear_margin: float
    support_margin: float
    dominance_ratio: float
    primary_is_clear: bool
    direct_evidence_ratio: float
    extractive_support_ratio: float
    multi_source_support: bool
    support_span: int


def _source_doc_ids(assessments: Sequence[EvidenceAssessment]) -> tuple[int, ...]:
    """按出现顺序提取参与作答的来源文档编号集合。

    路由结果会把这组文档 id 回填到 `AnswerPlan`，供后续生成器和前端引用展示
    共用，因此这里需要去重但保持原始优先级顺序。
    """
    doc_ids: list[int] = []
    seen: set[int] = set()
    for assessment in assessments:
        doc_id = assessment.candidate.doc_id
        if doc_id is None or doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
    return tuple(doc_ids)


def _rank_assessments(assessments: Sequence[EvidenceAssessment]) -> list[EvidenceAssessment]:
    """只保留可用证据，并按最终分数与直接证据属性重新排序。"""
    return sorted(
        (assessment for assessment in assessments if assessment.usable),
        key=lambda assessment: (
            float(assessment.final_score),
            1.0 if assessment.direct_evidence else 0.0,
            1.0 if assessment.supports_extractive else 0.0,
        ),
        reverse=True,
    )


def _supports_atomic_short_answer(
    primary: EvidenceAssessment | None,
    intent: QueryIntent,
    signals: RouteSignals,
) -> bool:
    """判断当前问题是否足够稳定，值得走短答案抽取路径。

    这一步的目标不是“能不能勉强抽一句”，而是判断抽取式回答能否在当前证据
    条件下比结构化综合更稳、更不容易漏掉限定条件。
    """
    if primary is None:
        return False
    if not intent.wants_short_answer or intent.evidence_requirement is not EvidenceRequirement.ATOMIC_SPAN:
        return False
    if intent.question_focus.category is QuestionFocusType.REASON:
        return False
    if not signals.primary_is_clear:
        return False
    if primary.answer_brief.strip() or primary.evidence_quote.strip():
        return True
    if primary.direct_evidence and primary.supports_extractive:
        return True
    if not intent.question_focus.has_explicit_cue:
        return False
    if not primary.direct_evidence:
        return False
    if signals.primary_score < max(0.72, signals.secondary_score + signals.support_margin):
        return False
    return True


def _unique_doc_count(assessments: Sequence[EvidenceAssessment]) -> int:
    """统计当前证据集合实际覆盖了多少不同来源文档。"""
    doc_keys: set[object] = set()
    for assessment in assessments:
        doc_id = assessment.candidate.doc_id
        if doc_id is not None:
            doc_keys.add(doc_id)
            continue
        title = assessment.candidate.title.strip()
        if title:
            doc_keys.add(title)
    return len(doc_keys)


def _build_route_signals(ranked_assessments: Sequence[EvidenceAssessment]) -> RouteSignals:
    """把证据分布压缩成路由决策所需的一组信号特征。

    这些信号会被后续回答模式路由共同使用，用来判断当前更像单证据直答、
    单文档综合，还是多文档整合问题。
    """
    unique_doc_count = _unique_doc_count(ranked_assessments)
    usable_count = len(ranked_assessments)
    score_distribution = summarize_scores(
        [float(assessment.final_score) for assessment in ranked_assessments],
        fallback_clear_margin=0.12,
        min_clear_margin=0.06,
        max_clear_margin=0.16,
        min_support_margin=0.04,
        max_support_margin=0.12,
    )
    primary_score = score_distribution.leader
    secondary_score = score_distribution.runner_up if usable_count > 1 else 0.0
    primary_gap = score_distribution.local_gap if usable_count > 1 else primary_score
    primary_is_clear = usable_count <= 1 or score_distribution.dominance_ratio >= 1.0
    direct_evidence_ratio = (
        sum(1 for assessment in ranked_assessments if assessment.direct_evidence) / usable_count if usable_count else 0.0
    )
    extractive_support_ratio = (
        sum(1 for assessment in ranked_assessments if assessment.supports_extractive) / usable_count if usable_count else 0.0
    )
    support_span = score_distribution.support_cluster_size
    multi_source_support = unique_doc_count > 1 and support_span > 1
    return RouteSignals(
        unique_doc_count=unique_doc_count,
        usable_count=usable_count,
        primary_score=primary_score,
        secondary_score=secondary_score,
        primary_gap=primary_gap,
        clear_margin=score_distribution.adaptive_clear_margin,
        support_margin=score_distribution.adaptive_support_margin,
        dominance_ratio=score_distribution.dominance_ratio,
        primary_is_clear=primary_is_clear,
        direct_evidence_ratio=direct_evidence_ratio,
        extractive_support_ratio=extractive_support_ratio,
        multi_source_support=multi_source_support,
        support_span=support_span,
    )


class AnswerModeRouter:
    @classmethod
    def route(
        cls,
        intent: QueryIntent,
        usable_assessments: Sequence[EvidenceAssessment],
    ) -> AnswerPlan:
        """根据问题意图和证据分布选择最终回答模式。

        这是后端回答链路的总路由器，负责在 `NO_CONTEXT`、`EXTRACTIVE`、
        `STRUCTURED` 和 `GENERATIVE` 之间做最终决策。
        """
        ranked_assessments = _rank_assessments(usable_assessments)
        source_doc_ids = _source_doc_ids(ranked_assessments)
        primary = ranked_assessments[0] if ranked_assessments else None
        primary_doc_id = primary.candidate.doc_id if primary is not None else None

        if not ranked_assessments:
            return AnswerPlan(
                mode=AnswerMode.NO_CONTEXT,
                reason='no_usable_evidence',
                primary_doc_id=None,
                source_doc_ids=(),
                generator_name='no_context_generator',
                trace_data={'usable_doc_count': 0},
            )

        signals = _build_route_signals(ranked_assessments)
        trace_data = {
            'usable_doc_count': signals.usable_count,
            'unique_doc_count': signals.unique_doc_count,
            'primary_score': signals.primary_score,
            'secondary_score': signals.secondary_score,
            'primary_gap': round(signals.primary_gap, 4),
            'clear_margin': round(signals.clear_margin, 4),
            'support_margin': round(signals.support_margin, 4),
            'dominance_ratio': round(signals.dominance_ratio, 4),
            'primary_is_clear': signals.primary_is_clear,
            'direct_evidence_ratio': round(signals.direct_evidence_ratio, 4),
            'extractive_support_ratio': round(signals.extractive_support_ratio, 4),
            'multi_source_support': signals.multi_source_support,
            'support_span': signals.support_span,
            'question_focus': intent.question_focus.category.value,
            'question_focus_explicit': intent.question_focus.has_explicit_cue,
        }

        if intent.evidence_requirement is EvidenceRequirement.FULL_DOCUMENT:
            return AnswerPlan(
                mode=AnswerMode.GENERATIVE,
                reason='full_document_requirement',
                primary_doc_id=primary_doc_id,
                source_doc_ids=source_doc_ids,
                generator_name='generative_generator',
                trace_data=trace_data,
            )

        if _supports_atomic_short_answer(primary, intent, signals):
            return AnswerPlan(
                mode=AnswerMode.EXTRACTIVE,
                reason='atomic_evidence_available',
                primary_doc_id=primary_doc_id,
                source_doc_ids=source_doc_ids,
                generator_name='extractive_generator',
                trace_data=trace_data,
            )

        if intent.evidence_requirement is EvidenceRequirement.ATOMIC_SPAN:
            if signals.multi_source_support:
                return AnswerPlan(
                    mode=AnswerMode.GENERATIVE,
                    reason='multi_source_atomic_resolution',
                    primary_doc_id=primary_doc_id,
                    source_doc_ids=source_doc_ids,
                    generator_name='generative_generator',
                    trace_data=trace_data,
                )
            return AnswerPlan(
                mode=AnswerMode.STRUCTURED,
                reason='atomic_synthesis_required',
                primary_doc_id=primary_doc_id,
                source_doc_ids=source_doc_ids,
                generator_name='structured_generator',
                trace_data=trace_data,
            )

        if signals.unique_doc_count > 1:
            return AnswerPlan(
                mode=AnswerMode.GENERATIVE,
                reason='ambiguous_multi_source_synthesis' if not signals.primary_is_clear else 'multi_document_synthesis',
                primary_doc_id=primary_doc_id,
                source_doc_ids=source_doc_ids,
                generator_name='generative_generator',
                trace_data=trace_data,
            )

        return AnswerPlan(
            mode=AnswerMode.STRUCTURED,
            reason='evidence_grounded_synthesis',
            primary_doc_id=primary_doc_id,
            source_doc_ids=source_doc_ids,
            generator_name='structured_generator',
            trace_data=trace_data,
        )
