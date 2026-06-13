from __future__ import annotations

from typing import Sequence

from app.services.rag.pipeline_models import EvidenceAssessment, RetrievedCandidate


def _clamp_unit(value: float | int | None) -> float:
    """把证据分裁剪到合法范围。"""
    if value is None:
        return 0.0
    return max(0.0, min(float(value), 1.0))


def _clamp_percent(value: float, *, floor: int = 0, ceiling: int = 99) -> int:
    """把百分比分值裁剪到合法范围。"""
    return max(floor, min(ceiling, int(round(value))))


def _assessment_rank_key(assessment: EvidenceAssessment) -> tuple[float, float, float]:
    """生成评估项排序键。"""
    return (
        float(assessment.final_score),
        1.0 if assessment.direct_evidence else 0.0,
        1.0 if assessment.supports_extractive else 0.0,
    )


def _candidate_rank_key(candidate: RetrievedCandidate) -> tuple[float, float, float]:
    """生成候选项排序键。"""
    return (
        max(float(candidate.rerank_score), float(candidate.adaptive_score), 0.0),
        float(candidate.coverage_score),
        float(candidate.identifier_overlap),
    )


def _leader_gap_bonus(top_score: float, next_score: float | None) -> float:
    """计算领先项的分差奖励。"""
    if next_score is None:
        return 3.0
    gap = top_score - next_score
    if gap >= 0.15:
        return 4.0
    if gap >= 0.08:
        return 2.0
    return 0.0


def calibrated_assessment_confidence(
    assessment: EvidenceAssessment,
    *,
    rank_index: int = 0,
    ranked_assessments: Sequence[EvidenceAssessment] = (),
) -> int:
    """计算校准后的评估项置信度。"""
    candidate = assessment.candidate
    final_score = _clamp_unit(assessment.final_score)
    confidence = 20.0 + final_score * 70.0

    if assessment.usable:
        confidence += 2.0
    if assessment.direct_evidence:
        confidence += 6.0
    elif assessment.supports_extractive:
        confidence += 3.0
    if assessment.judge and assessment.judge.invoked and assessment.judge.passed:
        confidence += 2.0

    rerank_score = _clamp_unit(candidate.rerank_score)
    if rerank_score >= 0.90:
        confidence += 4.0
    elif rerank_score >= 0.75:
        confidence += 2.0

    anchor_strength = max(_clamp_unit(candidate.identifier_overlap), _clamp_unit(candidate.coverage_score))
    if anchor_strength >= 0.75:
        confidence += 4.0
    elif anchor_strength >= 0.45:
        confidence += 2.0
    elif anchor_strength > 0.0:
        confidence += 1.0

    if float(candidate.dense_score) > 0.0 and float(candidate.bm25_score) > 0.0:
        confidence += 1.0

    if rank_index == 0:
        next_score = None
        if len(ranked_assessments) > 1:
            next_score = float(ranked_assessments[1].final_score)
        confidence += _leader_gap_bonus(float(assessment.final_score), next_score)

    return _clamp_percent(confidence, floor=28, ceiling=96)


def calibrated_candidate_confidence(
    candidate: RetrievedCandidate,
    *,
    rank_index: int = 0,
    ranked_candidates: Sequence[RetrievedCandidate] = (),
) -> int:
    """计算校准后的候选项置信度。"""
    primary_score = max(_clamp_unit(candidate.rerank_score), _clamp_unit(candidate.adaptive_score))
    confidence = 18.0 + primary_score * 72.0

    anchor_strength = max(_clamp_unit(candidate.identifier_overlap), _clamp_unit(candidate.coverage_score))
    if anchor_strength >= 0.75:
        confidence += 5.0
    elif anchor_strength >= 0.45:
        confidence += 3.0
    elif anchor_strength > 0.0:
        confidence += 1.0

    if float(candidate.dense_score) > 0.0 and float(candidate.bm25_score) > 0.0:
        confidence += 1.0

    if rank_index == 0:
        next_score = None
        if len(ranked_candidates) > 1:
            next_score = max(
                float(ranked_candidates[1].rerank_score),
                float(ranked_candidates[1].adaptive_score),
            )
        confidence += _leader_gap_bonus(primary_score, next_score)

    return _clamp_percent(confidence, floor=24, ceiling=94)


def rank_assessments_for_confidence(assessments: Sequence[EvidenceAssessment]) -> list[EvidenceAssessment]:
    """计算并排序面向置信度的评估项列表。"""
    return sorted(assessments, key=_assessment_rank_key, reverse=True)


def rank_candidates_for_confidence(candidates: Sequence[RetrievedCandidate]) -> list[RetrievedCandidate]:
    """计算并排序面向置信度的候选项列表。"""
    return sorted(candidates, key=_candidate_rank_key, reverse=True)
