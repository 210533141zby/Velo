from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryIntentType(str, Enum):
    LOOKUP = 'lookup'
    FACTOID = 'factoid'
    RELATION = 'relation'
    SUMMARY = 'summary'
    OVERVIEW = 'overview'
    REASON = 'reason'
    LOCATION = 'location'


class DefenseProfile(str, Enum):
    STRICT = 'strict'
    MODERATE = 'moderate'
    LOOSE = 'loose'


class EvidenceRequirement(str, Enum):
    ATOMIC_SPAN = 'atomic_span'
    MULTI_SPAN = 'multi_span'
    FULL_DOCUMENT = 'full_document'


class AnswerMode(str, Enum):
    NO_CONTEXT = 'no_context'
    EXTRACTIVE = 'extractive'
    STRUCTURED = 'structured'
    GENERATIVE = 'generative'


class QuestionFocusType(str, Enum):
    DESCRIPTION = 'description'
    PERSON = 'person'
    LOCATION = 'location'
    TIME = 'time'
    QUANTITY = 'quantity'
    REASON = 'reason'
    METHOD = 'method'
    CHOICE = 'choice'
    DEFINITION = 'definition'
    ATTRIBUTE = 'attribute'


@dataclass(frozen=True)
class QuestionFocus:
    category: QuestionFocusType = QuestionFocusType.DESCRIPTION
    cue_text: str = ''
    slot_terms: tuple[str, ...] = ()
    prompt_hint: str = '直接概括用户索要的核心事实'
    expects_multiple_points: bool = False

    @property
    def has_explicit_cue(self) -> bool:
        """判断问题里是否出现了明确提示回答焦点的线索词。

        例如“时间”“原因”“地点”等提示，一旦显式出现，就说明下游生成器
        可以更积极地按槽位填充方式组织答案。
        """
        return bool(self.cue_text.strip())

    @property
    def prefers_slot_filling(self) -> bool:
        """判断当前问题是否更适合按“主体 + 槽位值”方式作答。

        时间、地点、数量、原因等类型通常不需要长篇综合，而是更适合把答案
        压成一条直接命中的事实表述。
        """
        return self.category in {
            QuestionFocusType.PERSON,
            QuestionFocusType.LOCATION,
            QuestionFocusType.TIME,
            QuestionFocusType.QUANTITY,
            QuestionFocusType.REASON,
            QuestionFocusType.METHOD,
            QuestionFocusType.CHOICE,
            QuestionFocusType.DEFINITION,
            QuestionFocusType.ATTRIBUTE,
        }


@dataclass(frozen=True)
class QueryIntent:
    original_query: str
    normalized_query: str
    keyword_query: str
    intent_type: QueryIntentType
    retrieval_depth: int
    defense_profile: DefenseProfile
    evidence_requirement: EvidenceRequirement
    wants_short_answer: bool
    needs_judge: bool
    trace_tags: tuple[str, ...] = ()
    question_focus: QuestionFocus = field(default_factory=QuestionFocus)

    @property
    def prefers_extractive(self) -> bool:
        """判断当前问题是否更适合直接抽取短答案而非生成式综合。

        当问题类型偏查找、证据需求只需原子片段时，抽取式路径通常更稳，
        也更容易控制幻觉。
        """
        return self.intent_type in {
            QueryIntentType.LOOKUP,
            QueryIntentType.FACTOID,
            QueryIntentType.LOCATION,
        } and self.evidence_requirement is EvidenceRequirement.ATOMIC_SPAN


@dataclass(frozen=True)
class RetrievedCandidate:
    doc: Any = None
    doc_id: int | None = None
    title: str = ''
    adaptive_score: float = 0.0
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    coverage_score: float = 0.0
    identifier_overlap: float = 0.0
    chunk_text: str = ''
    full_content: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreContribution:
    name: str
    raw_value: float
    weight: float
    weighted_value: float
    reason: str


@dataclass(frozen=True)
class JudgeDecision:
    invoked: bool
    passed: bool
    topic_match: bool
    direct_evidence: bool
    answerable: bool
    evidence_quote: str = ''
    answer_brief: str = ''
    reason: str = ''
    timed_out: bool = False
    latency_ms: int = 0


@dataclass(frozen=True)
class EvidenceAssessment:
    candidate: RetrievedCandidate
    contributions: tuple[ScoreContribution, ...] = ()
    judge: JudgeDecision | None = None
    final_score: float = 0.0
    usable: bool = False
    reject_reason: str = ''
    direct_evidence: bool = False
    supports_extractive: bool = False
    evidence_quote: str = ''
    answer_brief: str = ''
    flags: tuple[str, ...] = ()
    trace_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerPlan:
    mode: AnswerMode
    reason: str
    primary_doc_id: int | None
    source_doc_ids: tuple[int, ...]
    generator_name: str
    trace_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    doc_id: int | None
    title: str
    text: str
    source_rank: int
    source_score: float
    relevance_score: float
    query_coverage: float
    identifier_coverage: float
    focus_score: float = 0.0


@dataclass(frozen=True)
class EvidencePacket:
    units: tuple[EvidenceUnit, ...]
    char_budget: int
