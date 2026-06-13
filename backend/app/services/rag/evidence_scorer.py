from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from app.logger import logger
from app.services.rag.evidence_judge import judge_rag_document
from app.services.rag.hybrid_search import has_identifier, tokenize_for_bm25
from app.services.rag.pipeline_models import (
    DefenseProfile,
    EvidenceAssessment,
    EvidenceRequirement,
    JudgeDecision,
    QueryIntent,
    RetrievedCandidate,
    ScoreContribution,
)
from app.services.rag.rerank_service import normalize_lookup_text
from app.services.rag.score_distribution import percentile, summarize_scores

JudgeCallable = Callable[[str, str, str], Awaitable[dict]]

BASE_RELEVANCE_WEIGHTS = {
    'adaptive_signal': 0.50,
    'rerank_signal': 0.50,
}
BASE_SIGNAL_WEIGHT = 0.50
TOPIC_SIGNAL_WEIGHTS = {
    DefenseProfile.STRICT: 0.18,
    DefenseProfile.MODERATE: 0.14,
    DefenseProfile.LOOSE: 0.10,
}
JUDGE_SIGNAL_WEIGHTS = {
    DefenseProfile.STRICT: 0.24,
    DefenseProfile.MODERATE: 0.18,
    DefenseProfile.LOOSE: 0.0,
}
PROFILE_RELEVANCE_FLOORS = {
    DefenseProfile.STRICT: 0.40,
    DefenseProfile.MODERATE: 0.34,
    DefenseProfile.LOOSE: 0.28,
}
PROFILE_TOPIC_FLOORS = {
    DefenseProfile.STRICT: 0.16,
    DefenseProfile.MODERATE: 0.06,
    DefenseProfile.LOOSE: 0.03,
}
PROFILE_FINAL_FLOORS = {
    DefenseProfile.STRICT: 0.56,
    DefenseProfile.MODERATE: 0.43,
    DefenseProfile.LOOSE: 0.40,
}
MAX_JUDGE_CANDIDATES = 5
HIGH_CONFIDENCE_RERANK_BASE = 0.56


@dataclass(frozen=True)
class CandidateSnapshot:
    candidate: RetrievedCandidate
    title: str
    content: str
    adaptive_score: float
    query_tokens: set[str]
    title_tokens: set[str]
    content_tokens: set[str]
    topic_alignment: float
    title_alignment: float
    base_relevance: float


@dataclass(frozen=True)
class BatchCalibration:
    relevance_floor: float
    topic_floor: float
    weak_evidence_floor: float
    direct_evidence_floor: float
    title_alignment_floor: float
    extractive_relevance_floor: float
    extractive_topic_floor: float
    final_floor: float
    judge_candidate_limit: int
    high_confidence_base_floor: float
    high_confidence_topic_floor: float
    high_confidence_gap_floor: float
    high_confidence_rerank_floor: float
    relevance_clear_margin: float
    relevance_support_margin: float
    support_cluster_size: int


def _clamp_score(value: float | int | None) -> float:
    """把分数裁剪到合法范围内。"""
    if value is None:
        return 0.0
    return max(0.0, min(float(value), 1.0))


def _compact_text(value: str, limit: int = 2200) -> str:
    """压缩文本中的空白并保留核心内容。"""
    normalized = ' '.join(str(value or '').split()).strip()
    return normalized[:limit]


def _token_set(value: str) -> set[str]:
    """把文本整理成词集合。"""
    return {
        normalize_lookup_text(token)
        for token in tokenize_for_bm25(value)
        if normalize_lookup_text(token)
    }


def _coverage_ratio(left: set[str], right: set[str]) -> float:
    """计算候选内容对查询词的覆盖率。"""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _candidate_title(candidate: RetrievedCandidate) -> str:
    """提取候选文档标题。"""
    if candidate.title:
        return str(candidate.title)
    if candidate.doc is not None:
        metadata = getattr(candidate.doc, 'metadata', {}) or {}
        return str(metadata.get('source') or '')
    return ''


def _candidate_chunk_text(candidate: RetrievedCandidate) -> str:
    """提取候选证据最适合评分的片段正文。

    统一证据评分主要看局部证据强度，因此优先使用候选片段；
    若片段字段缺失，再回退到元数据里可用的摘要或正文预览。
    """
    if candidate.doc is not None:
        page_content = str(getattr(candidate.doc, 'page_content', '') or '')
        if page_content.strip():
            return _compact_text(page_content, limit=2200)
    if candidate.chunk_text.strip():
        return _compact_text(candidate.chunk_text, limit=2200)
    if candidate.doc is not None:
        metadata = getattr(candidate.doc, 'metadata', {}) or {}
        for key in ('content', 'content_preview', 'summary'):
            text = str(metadata.get(key) or '')
            if text.strip():
                return _compact_text(text, limit=2200)
    for key in ('content', 'content_preview', 'summary'):
        text = str(candidate.metadata.get(key) or '')
        if text.strip():
            return _compact_text(text, limit=2200)
    return ''


def _build_contribution(name: str, raw_value: float, weight: float, reason: str) -> ScoreContribution:
    """构建单条证据分数贡献明细。"""
    normalized_value = _clamp_score(raw_value)
    return ScoreContribution(
        name=name,
        raw_value=normalized_value,
        weight=weight,
        weighted_value=normalized_value * weight,
        reason=reason,
    )


def _average_contributions(contributions: Sequence[ScoreContribution]) -> float:
    """计算贡献明细的平均值。"""
    total_weight = sum(contribution.weight for contribution in contributions)
    if total_weight <= 0:
        return 0.0
    return sum(contribution.weighted_value for contribution in contributions) / total_weight


def _normalize_judge_payload(payload: dict | JudgeDecision) -> JudgeDecision:
    """规范化判断模型的请求载荷。"""
    if isinstance(payload, JudgeDecision):
        return payload

    topic_match = bool(payload.get('core_topic_match') or payload.get('topic_match'))
    direct_evidence = bool(payload.get('contains_direct_evidence') or payload.get('direct_evidence'))
    answerable = bool(payload.get('answerable'))
    passed = answerable and topic_match and direct_evidence
    return JudgeDecision(
        invoked=True,
        passed=passed,
        topic_match=topic_match,
        direct_evidence=direct_evidence,
        answerable=answerable,
        evidence_quote=str(payload.get('evidence_quote') or ''),
        answer_brief=str(payload.get('answer_brief') or ''),
        reason=str(payload.get('reason') or ''),
    )


class UnifiedEvidenceScorer:
    def __init__(
        self,
        *,
        judge_timeout_seconds: float = 6.0,
        judge_callable: JudgeCallable | None = None,
    ) -> None:
        """初始化当前对象需要的状态和依赖。"""
        self.judge_timeout_seconds = judge_timeout_seconds
        self.judge_callable = judge_callable or judge_rag_document

    async def assess_concurrently(
        self,
        candidates: Sequence[RetrievedCandidate],
        intent: QueryIntent,
    ) -> list[EvidenceAssessment]:
        """并行评估一批候选证据。"""
        snapshots = [self._build_snapshot(candidate, intent) for candidate in candidates]
        calibration = self._build_calibration(snapshots, intent)
        judge_decisions = await self._run_judge_batch(snapshots, intent, calibration)

        assessments: list[EvidenceAssessment] = []
        for snapshot, judge_decision in zip(snapshots, judge_decisions):
            assessment = self._build_assessment(snapshot, intent, judge_decision, calibration)
            self._emit_trace_log(assessment)
            assessments.append(assessment)
        return assessments

    def _build_snapshot(self, candidate: RetrievedCandidate, intent: QueryIntent) -> CandidateSnapshot:
        """构建候选证据的快照信息。"""
        title = _candidate_title(candidate)
        content = _candidate_chunk_text(candidate)
        query_tokens = _token_set(intent.keyword_query or intent.normalized_query)
        title_tokens = _token_set(title)
        content_tokens = _token_set(content)
        topic_alignment = max(
            _coverage_ratio(query_tokens, title_tokens),
            _coverage_ratio(query_tokens, content_tokens),
        )
        title_alignment = _coverage_ratio(query_tokens, title_tokens)
        adaptive_score = _clamp_score(candidate.adaptive_score)
        base_relevance = self._compute_base_relevance(candidate)
        return CandidateSnapshot(
            candidate=candidate,
            title=title,
            content=content,
            adaptive_score=adaptive_score,
            query_tokens=query_tokens,
            title_tokens=title_tokens,
            content_tokens=content_tokens,
            topic_alignment=topic_alignment,
            title_alignment=title_alignment,
            base_relevance=base_relevance,
        )

    def _compute_base_relevance(self, candidate: RetrievedCandidate) -> float:
        """计算候选证据的基础相关性。"""
        contributions = (
            _build_contribution(
                'adaptive_signal',
                candidate.adaptive_score,
                BASE_RELEVANCE_WEIGHTS['adaptive_signal'],
                'hybrid_adaptive_score',
            ),
            _build_contribution(
                'rerank_signal',
                candidate.rerank_score,
                BASE_RELEVANCE_WEIGHTS['rerank_signal'],
                'cross_encoder_rerank',
            ),
        )
        return _average_contributions(contributions)

    async def _run_judge_batch(
        self,
        snapshots: Sequence[CandidateSnapshot],
        intent: QueryIntent,
        calibration: BatchCalibration,
    ) -> list[JudgeDecision]:
        """批量执行证据判断。"""
        decisions = [self._skipped_judge_decision() for _ in snapshots]
        if not self._should_invoke_judge(intent):
            return decisions

        skipped_indexes = self._select_high_confidence_skip_indexes(snapshots, calibration)
        for index in skipped_indexes:
            decisions[index] = self._skipped_judge_decision(reason='high_confidence_gap')

        judge_indexes = tuple(
            index for index in self._select_judge_indexes(snapshots, calibration)
            if index not in skipped_indexes
        )
        tasks = [self._run_single_judge(snapshots[index], intent, calibration) for index in judge_indexes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for index, result in zip(judge_indexes, results):
            if isinstance(result, JudgeDecision):
                decisions[index] = result
                continue
            if isinstance(result, Exception):
                decisions[index] = JudgeDecision(
                    invoked=True,
                    passed=False,
                    topic_match=False,
                    direct_evidence=False,
                    answerable=False,
                    reason='judge_exception',
                    latency_ms=0,
                )
                continue
            decisions[index] = _normalize_judge_payload(result)
        return decisions

    def _skipped_judge_decision(self, reason: str = 'judge_skipped') -> JudgeDecision:
        """生成跳过判断时的默认结果。"""
        return JudgeDecision(
            invoked=False,
            passed=True,
            topic_match=True,
            direct_evidence=False,
            answerable=False,
            reason=reason,
            latency_ms=0,
        )

    def _select_judge_indexes(
        self,
        snapshots: Sequence[CandidateSnapshot],
        calibration: BatchCalibration,
    ) -> tuple[int, ...]:
        """选择需要送入判断模型的候选下标。"""
        ranked_indexes = sorted(
            range(len(snapshots)),
            key=lambda index: snapshots[index].base_relevance,
            reverse=True,
        )
        selected: list[int] = []
        for index in ranked_indexes:
            if len(selected) >= calibration.judge_candidate_limit:
                break
            if selected and snapshots[index].base_relevance < max(0.0, calibration.relevance_floor - 0.08):
                continue
            selected.append(index)

        if not selected and ranked_indexes:
            selected.append(ranked_indexes[0])
        return tuple(sorted(selected))

    def _select_high_confidence_skip_indexes(
        self,
        snapshots: Sequence[CandidateSnapshot],
        calibration: BatchCalibration,
    ) -> tuple[int, ...]:
        """选择可直接跳过判断的高置信候选。"""
        if not snapshots:
            return ()
        ranked_indexes = sorted(
            range(len(snapshots)),
            key=lambda index: snapshots[index].base_relevance,
            reverse=True,
        )
        best_index = ranked_indexes[0]
        best_snapshot = snapshots[best_index]
        second_score = snapshots[ranked_indexes[1]].base_relevance if len(ranked_indexes) > 1 else 0.0
        if best_snapshot.base_relevance < calibration.high_confidence_base_floor:
            return ()
        if best_snapshot.topic_alignment < calibration.high_confidence_topic_floor:
            return ()
        if not self._has_retrieval_consensus(best_snapshot):
            return ()
        if best_snapshot.candidate.rerank_score < calibration.high_confidence_rerank_floor:
            return ()
        if best_snapshot.base_relevance - second_score < calibration.high_confidence_gap_floor:
            return ()
        return (best_index,)

    def _has_retrieval_consensus(self, snapshot: CandidateSnapshot) -> bool:
        """判断多路检索结果是否形成一致性。"""
        candidate = snapshot.candidate
        has_dual_support = candidate.dense_score > 0.0 and candidate.bm25_score > 0.0
        has_anchor_support = candidate.identifier_overlap > 0.0 or snapshot.title_alignment >= 0.18
        return has_dual_support or has_anchor_support

    def _has_anchor_lexical_support(
        self,
        snapshot: CandidateSnapshot,
        topic_alignment: float,
        calibration: BatchCalibration,
    ) -> bool:
        """把精确锚点的词法一致性视作一种可校准支撑。

        当答案片段嵌在较长正文里时，交叉编码器有时会低估这类短锚点查询。
        因此这里保留一条通用恢复路径：只要查询锚点确实出现、BM25 也给出支持，
        且候选内容对问题主题的覆盖达到阈值，就允许把它视为有效支撑。
        """
        candidate = snapshot.candidate
        if candidate.identifier_overlap <= 0.0 or candidate.bm25_score <= 0.0:
            return False
        if topic_alignment < max(calibration.topic_floor, 0.24):
            return False
        lexical_strength = max(
            float(candidate.coverage_score),
            float(candidate.identifier_overlap),
            float(snapshot.title_alignment),
        )
        return lexical_strength >= 0.45

    def _should_skip_judge_for_snapshot(
        self,
        snapshot: CandidateSnapshot,
        intent: QueryIntent,
        calibration: BatchCalibration,
    ) -> bool:
        """判断当前快照是否可以跳过证据判断。"""
        if intent.evidence_requirement is not EvidenceRequirement.ATOMIC_SPAN:
            return False
        if not has_identifier(intent.normalized_query):
            return False
        if snapshot.base_relevance < calibration.high_confidence_base_floor:
            return False
        if snapshot.topic_alignment < calibration.high_confidence_topic_floor:
            return False
        if snapshot.candidate.rerank_score < calibration.high_confidence_rerank_floor:
            return False
        return self._has_retrieval_consensus(snapshot)

    async def _run_single_judge(
        self,
        snapshot: CandidateSnapshot,
        intent: QueryIntent,
        calibration: BatchCalibration,
    ) -> JudgeDecision:
        """执行单条候选证据判断。"""
        if not self._should_invoke_judge(intent):
            return self._skipped_judge_decision()
        if self._should_skip_judge_for_snapshot(snapshot, intent, calibration):
            return self._skipped_judge_decision(reason='high_confidence_anchor')

        started_at = time.perf_counter()
        judge_content = snapshot.content
        if intent.evidence_requirement is not EvidenceRequirement.ATOMIC_SPAN:
            full_content = ''
            if snapshot.candidate.doc is not None:
                full_content = str(getattr(snapshot.candidate.doc, 'full_content', '') or '')
            if not full_content.strip():
                full_content = snapshot.candidate.full_content
            if full_content.strip():
                judge_content = _compact_text(full_content, limit=6000)
        try:
            payload = await asyncio.wait_for(
                self.judge_callable(intent.normalized_query, snapshot.title, judge_content),
                timeout=self.judge_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return JudgeDecision(
                invoked=True,
                passed=False,
                topic_match=False,
                direct_evidence=False,
                answerable=False,
                reason='judge_timeout',
                timed_out=True,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )

        decision = _normalize_judge_payload(payload)
        return JudgeDecision(
            invoked=decision.invoked,
            passed=decision.passed,
            topic_match=decision.topic_match,
            direct_evidence=decision.direct_evidence,
            answerable=decision.answerable,
            evidence_quote=decision.evidence_quote,
            answer_brief=decision.answer_brief,
            reason=decision.reason,
            timed_out=decision.timed_out,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )

    def _should_invoke_judge(self, intent: QueryIntent) -> bool:
        """判断当前候选是否需要进入判断模型。"""
        if not intent.needs_judge:
            return False
        return JUDGE_SIGNAL_WEIGHTS[intent.defense_profile] > 0.0

    def _build_assessment(
        self,
        snapshot: CandidateSnapshot,
        intent: QueryIntent,
        judge_decision: JudgeDecision,
        calibration: BatchCalibration,
    ) -> EvidenceAssessment:
        """构建评估项。"""
        contributions = [
            _build_contribution(
                'base_signal',
                snapshot.base_relevance,
                BASE_SIGNAL_WEIGHT,
                'adaptive_plus_rerank',
            )
        ]
        contributions.append(
            _build_contribution(
                'topic_signal',
                snapshot.topic_alignment,
                TOPIC_SIGNAL_WEIGHTS[intent.defense_profile],
                'query_topic_alignment',
            )
        )
        if judge_decision.invoked:
            contributions.append(
                _build_contribution(
                    'judge_signal',
                    1.0 if judge_decision.passed else 0.0,
                    JUDGE_SIGNAL_WEIGHTS[intent.defense_profile],
                    'semantic_judge',
                )
            )

        base_relevance = snapshot.base_relevance
        final_score = _average_contributions(contributions)
        flags = list(self._build_flags(snapshot, intent, judge_decision, base_relevance, final_score, calibration))
        direct_evidence = self._determine_direct_evidence(snapshot, intent, judge_decision, base_relevance, calibration)
        supports_extractive = self._determine_extractive_support(intent, judge_decision, base_relevance, snapshot, calibration)
        usable, reject_reason = self._determine_usability(
            snapshot=snapshot,
            intent=intent,
            judge_decision=judge_decision,
            base_relevance=base_relevance,
            topic_alignment=snapshot.topic_alignment,
            final_score=final_score,
            direct_evidence=direct_evidence,
            calibration=calibration,
        )

        if usable:
            flags.append('USABLE')
        elif reject_reason:
            flags.append(reject_reason)

        trace_summary = {
            'doc_id': snapshot.candidate.doc_id,
            'title': snapshot.title[:80],
            'adaptive_score': round(snapshot.adaptive_score, 4),
            'base_relevance': round(base_relevance, 4),
            'topic_match': 'PASS' if snapshot.topic_alignment >= calibration.topic_floor else 'FAIL',
            'judge_latency_ms': judge_decision.latency_ms,
            'flags': flags,
            'usable': usable,
            'final_score': round(final_score, 4),
            'judge_status': self._judge_status(judge_decision),
            'relevance_floor': round(calibration.relevance_floor, 4),
            'topic_floor': round(calibration.topic_floor, 4),
            'final_floor': round(calibration.final_floor, 4),
            'judge_candidate_limit': calibration.judge_candidate_limit,
            'relevance_clear_margin': round(calibration.relevance_clear_margin, 4),
            'relevance_support_margin': round(calibration.relevance_support_margin, 4),
            'support_cluster_size': calibration.support_cluster_size,
        }

        return EvidenceAssessment(
            candidate=snapshot.candidate,
            contributions=tuple(contributions),
            judge=judge_decision,
            final_score=final_score,
            usable=usable,
            reject_reason=reject_reason,
            direct_evidence=direct_evidence,
            supports_extractive=supports_extractive,
            evidence_quote=judge_decision.evidence_quote,
            answer_brief=judge_decision.answer_brief,
            flags=tuple(flags),
            trace_summary=trace_summary,
        )

    def _build_flags(
        self,
        snapshot: CandidateSnapshot,
        intent: QueryIntent,
        judge_decision: JudgeDecision,
        base_relevance: float,
        final_score: float,
        calibration: BatchCalibration,
    ) -> tuple[str, ...]:
        """为当前 assessment 生成可解释的诊断标记。

        这些 flag 不直接参与前端展示，但会进入 trace 日志和误差分析，
        用来说明某条证据是因为基础分低、主题对齐弱、编号不匹配还是 judge 失败
        而被压分或拒绝。
        """
        flags: list[str] = []
        if base_relevance < calibration.relevance_floor:
            flags.append('LOW_BASE_RELEVANCE')
        if snapshot.topic_alignment < calibration.topic_floor:
            flags.append('WEAK_TOPIC_ALIGNMENT')
        if has_identifier(intent.normalized_query) and snapshot.candidate.identifier_overlap <= 0.0:
            flags.append('IDENTIFIER_MISMATCH')
        if intent.evidence_requirement is EvidenceRequirement.ATOMIC_SPAN and snapshot.topic_alignment < calibration.weak_evidence_floor:
            flags.append('WEAK_EVIDENCE')
        if judge_decision.invoked and judge_decision.timed_out:
            flags.append('LLM_JUDGE_TIMEOUT')
        elif judge_decision.invoked and judge_decision.passed:
            flags.append('PASSED_LLM_JUDGE')
        elif judge_decision.invoked:
            flags.append('FAILED_LLM_JUDGE')
        if final_score < calibration.final_floor:
            flags.append('LOW_FINAL_SCORE')
        return tuple(flags)

    def _determine_direct_evidence(
        self,
        snapshot: CandidateSnapshot,
        intent: QueryIntent,
        judge_decision: JudgeDecision,
        base_relevance: float,
        calibration: BatchCalibration,
    ) -> bool:
        """判断当前候选是否可视为直接证据。

        对原子片段型问题，系统更希望拿到能直接回答的证据，而不是泛相关材料。
        这里会综合 judge 结果、主题对齐和标题对齐，决定这条候选能否被当作 direct evidence。
        """
        if judge_decision.invoked:
            return judge_decision.direct_evidence
        if intent.evidence_requirement is EvidenceRequirement.FULL_DOCUMENT:
            return False
        return snapshot.topic_alignment >= calibration.direct_evidence_floor and (
            snapshot.title_alignment >= calibration.title_alignment_floor
            or base_relevance >= calibration.extractive_relevance_floor
        )

    def _determine_extractive_support(
        self,
        intent: QueryIntent,
        judge_decision: JudgeDecision,
        base_relevance: float,
        snapshot: CandidateSnapshot,
        calibration: BatchCalibration,
    ) -> bool:
        """判断当前候选是否适合进入抽取式回答路径。

        这一步只对原子片段问题开放。它的目标是把“可用于生成式综合的证据”
        与“已经足以支撑直接抽句的证据”区分开来，供后续回答路由使用。
        """
        if intent.evidence_requirement is not EvidenceRequirement.ATOMIC_SPAN:
            return False
        if judge_decision.invoked and judge_decision.answerable:
            return snapshot.topic_alignment >= calibration.extractive_topic_floor
        return (
            base_relevance >= calibration.extractive_relevance_floor
            and snapshot.topic_alignment >= calibration.extractive_topic_floor
        )

    def _determine_usability(
        self,
        *,
        snapshot: CandidateSnapshot,
        intent: QueryIntent,
        judge_decision: JudgeDecision,
        base_relevance: float,
        topic_alignment: float,
        final_score: float,
        direct_evidence: bool,
        calibration: BatchCalibration,
    ) -> tuple[bool, str]:
        """综合多路信号判断这条候选是否可用。

        可用性判断是统一证据评分的最终门槛，会同时考虑主题对齐、judge 结论、
        最终分数、防御等级和是否具备直接证据属性，并给出对应的拒绝原因。
        """
        if topic_alignment < calibration.topic_floor:
            return False, 'REJECT_TOPIC_MISMATCH'
        if judge_decision.invoked and not judge_decision.passed:
            if self._has_calibrated_retrieval_support(snapshot, base_relevance, topic_alignment, calibration):
                return True, ''
            if judge_decision.timed_out:
                return False, 'REJECT_JUDGE_TIMEOUT'
            return False, 'REJECT_FAILED_LLM_JUDGE'
        if final_score < calibration.final_floor and not direct_evidence:
            return False, 'REJECT_LOW_FINAL_SCORE'
        if intent.defense_profile is DefenseProfile.STRICT and intent.evidence_requirement is EvidenceRequirement.ATOMIC_SPAN:
            if not direct_evidence:
                return False, 'REJECT_WEAK_DIRECT_EVIDENCE'
        if not judge_decision.invoked and base_relevance < calibration.relevance_floor and topic_alignment < calibration.weak_evidence_floor:
            return False, 'REJECT_LOW_FINAL_SCORE'
        return True, ''

    def _has_calibrated_retrieval_support(
        self,
        snapshot: CandidateSnapshot,
        base_relevance: float,
        topic_alignment: float,
        calibration: BatchCalibration,
    ) -> bool:
        """当多路检索信号强一致时，允许恢复候选证据。

        LLM 判断器适合充当噪声过滤器，但在检索层已经给出稳定交叉支持时，
        它不应成为一票否决。这里的恢复规则保持通用，只依赖分数分布、
        主题覆盖和检索一致性，不依赖领域词或特定基准实体。
        """
        if self._has_anchor_lexical_support(snapshot, topic_alignment, calibration):
            return True
        if base_relevance < calibration.high_confidence_base_floor:
            return False
        if topic_alignment < calibration.high_confidence_topic_floor:
            return False
        if snapshot.candidate.rerank_score < calibration.high_confidence_rerank_floor:
            return False
        return self._has_retrieval_consensus(snapshot)

    def _build_calibration(
        self,
        snapshots: Sequence[CandidateSnapshot],
        intent: QueryIntent,
    ) -> BatchCalibration:
        """根据本批候选分布自适应生成评分阈值。

        这里不是使用一组完全固定的硬阈值，而是结合当前批次的基础相关度、
        主题对齐和重排分布，动态推导 relevance/topic/final floor，
        让严格问题和宽松问题都能保持相对稳定的筛选行为。
        """
        profile_relevance = PROFILE_RELEVANCE_FLOORS[intent.defense_profile]
        profile_topic = PROFILE_TOPIC_FLOORS[intent.defense_profile]
        profile_final = PROFILE_FINAL_FLOORS[intent.defense_profile]
        atomic_requirement = intent.evidence_requirement is EvidenceRequirement.ATOMIC_SPAN
        close_margin = 0.10 if atomic_requirement else 0.14

        if not snapshots:
            return BatchCalibration(
                relevance_floor=profile_relevance,
                topic_floor=profile_topic,
                weak_evidence_floor=max(profile_topic + 0.10, 0.20),
                direct_evidence_floor=max(profile_topic + 0.08, 0.20),
                title_alignment_floor=max(profile_topic * 0.75, 0.10),
                extractive_relevance_floor=profile_relevance + 0.06,
                extractive_topic_floor=profile_topic + 0.08,
                final_floor=profile_final,
                judge_candidate_limit=1,
                high_confidence_base_floor=profile_relevance + 0.18,
                high_confidence_topic_floor=profile_topic + 0.10,
                high_confidence_gap_floor=0.10,
                high_confidence_rerank_floor=HIGH_CONFIDENCE_RERANK_BASE,
                relevance_clear_margin=0.10,
                relevance_support_margin=0.08,
                support_cluster_size=0,
            )

        base_values = sorted((snapshot.base_relevance for snapshot in snapshots), reverse=True)
        topic_values = sorted((snapshot.topic_alignment for snapshot in snapshots), reverse=True)
        rerank_values = sorted((_clamp_score(snapshot.candidate.rerank_score) for snapshot in snapshots), reverse=True)
        base_distribution = summarize_scores(
            base_values,
            fallback_clear_margin=close_margin,
            min_clear_margin=0.06,
            max_clear_margin=0.16,
            min_support_margin=0.04,
            max_support_margin=0.12,
        )
        topic_distribution = summarize_scores(
            topic_values,
            fallback_clear_margin=0.08,
            min_clear_margin=0.04,
            max_clear_margin=0.14,
            min_support_margin=0.03,
            max_support_margin=0.10,
        )

        judge_candidate_limit = min(
            max(base_distribution.support_cluster_size, 1),
            min(len(base_values), MAX_JUDGE_CANDIDATES),
        )
        if not atomic_requirement:
            judge_candidate_limit = max(judge_candidate_limit, min(len(base_values), 3))

        relevance_floor = max(
            profile_relevance,
            min(
                base_distribution.leader - base_distribution.adaptive_clear_margin * 1.4,
                base_distribution.upper_quartile - base_distribution.adaptive_support_margin * 0.5
                if len(base_values) > 1
                else base_distribution.leader - base_distribution.adaptive_clear_margin,
            ),
        )
        topic_floor = max(
            profile_topic,
            min(
                topic_distribution.leader - topic_distribution.adaptive_clear_margin * 1.2,
                topic_distribution.upper_quartile - topic_distribution.adaptive_support_margin * 0.4
                if len(topic_values) > 1
                else topic_distribution.leader - topic_distribution.adaptive_clear_margin,
            ),
        )
        weak_evidence_floor = max(
            topic_floor + max(topic_distribution.adaptive_support_margin, 0.06),
            0.22 if atomic_requirement else 0.18,
        )
        direct_evidence_floor = max(
            topic_floor + max(topic_distribution.adaptive_support_margin * 0.8, 0.05),
            0.24 if atomic_requirement else 0.18,
        )
        title_alignment_floor = max(topic_floor * 0.75, 0.10)
        extractive_relevance_floor = max(
            relevance_floor + max(base_distribution.adaptive_support_margin, 0.05),
            base_distribution.leader - max(base_distribution.adaptive_support_margin, 0.08),
        )
        extractive_topic_floor = max(
            direct_evidence_floor,
            topic_floor + max(topic_distribution.adaptive_support_margin, 0.04),
            topic_distribution.leader - max(topic_distribution.adaptive_support_margin, 0.06),
        )
        final_floor = max(
            profile_final,
            min(
                0.76,
                extractive_relevance_floor * 0.78 + topic_floor * 0.14,
            ),
        )
        high_confidence_base_floor = max(
            profile_relevance + 0.18,
            base_distribution.leader - base_distribution.adaptive_support_margin * 0.5,
        )
        high_confidence_topic_floor = max(
            topic_floor + max(topic_distribution.adaptive_support_margin, 0.05),
            topic_distribution.leader - topic_distribution.adaptive_support_margin,
            0.20,
        )
        high_confidence_rerank_floor = max(HIGH_CONFIDENCE_RERANK_BASE, percentile(rerank_values, 0.75))

        return BatchCalibration(
            relevance_floor=max(relevance_floor, 0.0),
            topic_floor=max(topic_floor, 0.0),
            weak_evidence_floor=min(max(weak_evidence_floor, 0.0), 1.0),
            direct_evidence_floor=min(max(direct_evidence_floor, 0.0), 1.0),
            title_alignment_floor=min(max(title_alignment_floor, 0.0), 1.0),
            extractive_relevance_floor=min(max(extractive_relevance_floor, 0.0), 1.0),
            extractive_topic_floor=min(max(extractive_topic_floor, 0.0), 1.0),
            final_floor=min(max(final_floor, 0.0), 1.0),
            judge_candidate_limit=judge_candidate_limit,
            high_confidence_base_floor=min(max(high_confidence_base_floor, 0.0), 1.0),
            high_confidence_topic_floor=min(max(high_confidence_topic_floor, 0.0), 1.0),
            high_confidence_gap_floor=base_distribution.adaptive_clear_margin,
            high_confidence_rerank_floor=min(max(high_confidence_rerank_floor, 0.0), 1.0),
            relevance_clear_margin=base_distribution.adaptive_clear_margin,
            relevance_support_margin=base_distribution.adaptive_support_margin,
            support_cluster_size=base_distribution.support_cluster_size,
        )

    def _judge_status(self, judge_decision: JudgeDecision) -> str:
        """把 judge 决策对象压缩成便于日志记录的状态码。"""
        if not judge_decision.invoked:
            return 'SKIP'
        if judge_decision.timed_out:
            return 'TIMEOUT'
        if judge_decision.passed:
            return 'PASS'
        return 'FAIL'

    def _emit_trace_log(self, assessment: EvidenceAssessment) -> None:
        """输出单条证据评分的追踪日志。

        统一证据评分是整条 RAG 主链路里最需要解释性的环节之一，
        因此这里会把摘要化的 trace 信息打到日志里，便于答辩演示和线上排查。
        """
        logger.info(
            'RAG 统一证据评分完成',
            extra={
                'extra_data': {
                    'event': 'rag_evidence_assessed',
                    **assessment.trace_summary,
                }
            },
        )
