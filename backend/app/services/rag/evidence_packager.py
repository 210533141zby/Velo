from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from app.services.rag.hybrid_search import extract_identifiers
from app.services.rag.pipeline_models import (
    EvidenceAssessment,
    EvidencePacket,
    EvidenceUnit,
    QueryIntent,
    QuestionFocus,
    QuestionFocusType,
)
from app.services.rag.rerank_service import normalize_lookup_text, tokenize_text
from app.services.rag.score_distribution import summarize_scores
from app.services.rag.text_utils import compact_structured_text, compact_text, split_paragraphs, split_text_segments

DEFAULT_SOURCE_CHAR_LIMIT = 12000
TIME_PATTERN = re.compile(r'(?:\d{2,4}年|\d{1,2}月|\d{1,2}日|\d{1,2}[点时分]|上午|下午|凌晨|当日|当天)')
NUMBER_PATTERN = re.compile(r'(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万两]+)(?:[%％]|个|人|名|项|次|家|位|届|天|年|月|小时|分钟|公里|米|亿元?|万元?|倍|级)?')
REASON_MARKERS = ('因为', '由于', '原因', '缘于', '所以', '因此', '致使')
METHOD_MARKERS = ('通过', '采用', '利用', '借助', '依靠', '方式', '方法', '步骤')
LIST_MARKERS = ('包括', '包含', '如下', '分别', '以及', '及', '、')
LOCATION_MARKERS = ('位于', '坐落', '地处', '地址', '地点', '在')
DEFINITION_MARKERS = ('是', '为', '指', '叫做', '称为', '又称')


def _coverage_ratio(left: set[str], right: set[str]) -> float:
    """计算候选内容对查询词的覆盖率。"""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _token_set(value: str) -> set[str]:
    """把文本整理成词集合。"""
    return tokenize_text(value)


def _unit_texts(content: str) -> list[str]:
    """把一段正文拆成适合下游综合使用的证据单元。

    优先保留天然段落结构；如果正文本身只有一个长段，再退回到按句切分并做滑动窗口。
    目标不是做严格切块，而是给后续综合器提供几段长度受控、语义相对完整的候选单元。
    """
    paragraphs = split_paragraphs(content)
    if len(paragraphs) >= 2:
        return [compact_text(paragraph, 720) for paragraph in paragraphs if compact_text(paragraph, 720)]

    sentences = split_text_segments(content)
    if not sentences:
        compacted = compact_text(content, 720)
        return [compacted] if compacted else []

    if len(sentences) <= 2:
        compacted = compact_structured_text('\n'.join(sentences), 720)
        return [compacted] if compacted else []

    units: list[str] = []
    for index in range(len(sentences)):
        compacted = compact_structured_text('\n'.join(sentences[index : index + 2]), 720)
        if compacted:
            units.append(compacted)
    return units


def _unit_similarity(left: set[str], right: set[str]) -> float:
    """计算两个证据单元之间的词项相似度。

    这个值主要用于打包阶段的去冗余控制，避免最终上下文包里出现大量内容近似、
    只是表述略有差异的证据单元。
    """
    if not left or not right:
        return 0.0
    shared = len(left & right)
    total = len(left | right)
    return shared / total if total else 0.0


@dataclass(frozen=True)
class _CandidateUnit:
    unit_id: str
    doc_id: int | None
    title: str
    text: str
    source_rank: int
    source_score: float
    relevance_score: float
    query_coverage: float
    identifier_coverage: float
    focus_score: float
    tokens: set[str]


@dataclass(frozen=True)
class _PackingStrategy:
    target_source_count: int
    support_cluster_size: int
    dominance_ratio: float


@dataclass(frozen=True)
class _RenderOptions:
    include_doc_id: bool = False
    include_source_rank: bool = False
    include_relevance_score: bool = False
    include_answer_signal: bool = False


class EvidencePackager:
    """为下游模型调用构建紧凑且有证据约束的上下文包。"""

    def __init__(self, *, source_char_limit: int = DEFAULT_SOURCE_CHAR_LIMIT) -> None:
        """初始化证据打包器的字符预算配置。"""
        self.source_char_limit = source_char_limit

    def pack(
        self,
        query: str,
        assessments: Sequence[EvidenceAssessment],
        *,
        max_units: int,
        char_budget: int,
        prefer_full_content: bool,
        intent: QueryIntent | None = None,
    ) -> EvidencePacket:
        """打包一组用于作答的证据单元。"""
        query_tokens = _token_set(query)
        query_identifiers = extract_identifiers(query)
        question_focus = intent.question_focus if intent is not None else QuestionFocus()
        candidates = self._build_candidates(
            query_tokens,
            query_identifiers,
            question_focus,
            assessments,
            prefer_full_content=prefer_full_content,
        )
        selected = self._select_units(candidates, max_units=max_units, char_budget=char_budget)
        return EvidencePacket(
            units=tuple(
                EvidenceUnit(
                    unit_id=item.unit_id,
                    doc_id=item.doc_id,
                    title=item.title,
                    text=item.text,
                    source_rank=item.source_rank,
                    source_score=item.source_score,
                    relevance_score=item.relevance_score,
                    query_coverage=item.query_coverage,
                    identifier_coverage=item.identifier_coverage,
                    focus_score=item.focus_score,
                )
                for item in selected
            ),
            char_budget=char_budget,
        )

    def render_for_synthesis(self, packet: EvidencePacket) -> str:
        """渲染综合作答阶段使用的证据文本。"""
        return self._render(packet, _RenderOptions(include_answer_signal=True))

    def render_for_review(self, packet: EvidencePacket) -> str:
        """渲染深度复核阶段使用的证据文本。"""
        return self._render(packet, _RenderOptions(include_source_rank=True))

    def render(self, packet: EvidencePacket) -> str:
        """兼容旧接口的统一渲染入口。"""
        return self.render_for_review(packet)

    def _render(self, packet: EvidencePacket, options: _RenderOptions) -> str:
        """把证据包渲染成可直接送给模型的文本。

        渲染时会按不同调用场景决定是否展示 doc_id、来源排序或回答相关度，
        最终再统一做一次结构压缩，保证输出仍落在预算范围内。
        """
        blocks: list[str] = []
        for index, unit in enumerate(packet.units, start=1):
            lines = [
                f'证据单元 {index}',
                f'unit_id: {unit.unit_id}',
                f'标题：{unit.title}',
            ]
            if options.include_doc_id:
                lines.append(f'doc_id: {unit.doc_id}')
            if options.include_source_rank:
                lines.append(f'来源排序：{unit.source_rank}')
            if options.include_relevance_score:
                lines.append(f'相关度：{round(float(unit.relevance_score), 4)}')
            if options.include_answer_signal:
                answer_signal = max(float(unit.relevance_score), float(unit.focus_score))
                lines.append(f'回答相关度：{round(answer_signal, 4)}')
            lines.append(f'内容：\n{unit.text}')
            blocks.append('\n'.join(lines))
        return compact_structured_text('\n\n'.join(blocks), packet.char_budget)

    def _build_candidates(
        self,
        query_tokens: set[str],
        query_identifiers: set[str],
        question_focus: QuestionFocus,
        assessments: Sequence[EvidenceAssessment],
        *,
        prefer_full_content: bool,
    ) -> list[_CandidateUnit]:
        """把 assessment 列表展开成可筛选的证据单元候选。

        每条 assessment 可能对应多个局部单元。这里会把来源分、查询覆盖、
        标识符覆盖和焦点分一并计算出来，形成后续打包阶段统一消费的候选对象。
        """
        candidates: list[_CandidateUnit] = []
        for source_rank, assessment in enumerate(assessments, start=1):
            content = self._source_content(assessment, prefer_full_content=prefer_full_content)
            if not content:
                continue
            title_tokens = _token_set(assessment.candidate.title)
            title_overlap = _coverage_ratio(query_tokens, title_tokens)
            unit_texts = _unit_texts(content)
            for unit_index, text in enumerate(unit_texts, start=1):
                unit_tokens = _token_set(text)
                query_coverage = _coverage_ratio(query_tokens, unit_tokens)
                identifier_coverage = _coverage_ratio(query_identifiers, extract_identifiers(text))
                focus_score = self._focus_score(question_focus, text, unit_tokens)
                if (
                    query_tokens
                    and query_coverage <= 0.0
                    and identifier_coverage <= 0.0
                    and focus_score <= 0.0
                    and unit_index > 1
                ):
                    continue
                relevance = self._relevance_score(
                    source_score=float(assessment.final_score),
                    query_coverage=query_coverage,
                    identifier_coverage=identifier_coverage,
                    focus_score=focus_score,
                    title_overlap=title_overlap,
                    position=unit_index,
                )
                candidates.append(
                    _CandidateUnit(
                        unit_id=f'{assessment.candidate.doc_id or source_rank}:{unit_index}',
                        doc_id=assessment.candidate.doc_id,
                        title=assessment.candidate.title,
                        text=text,
                        source_rank=source_rank,
                        source_score=float(assessment.final_score),
                        relevance_score=relevance,
                        query_coverage=query_coverage,
                        identifier_coverage=identifier_coverage,
                        focus_score=focus_score,
                        tokens=unit_tokens,
                    )
                )
        return sorted(
            candidates,
            key=lambda item: (
                item.relevance_score,
                item.focus_score,
                item.query_coverage,
                item.identifier_coverage,
                -item.source_rank,
            ),
            reverse=True,
        )

    def _source_content(self, assessment: EvidenceAssessment, *, prefer_full_content: bool) -> str:
        """提取来源文档的正文内容。"""
        if prefer_full_content and assessment.candidate.full_content.strip():
            return compact_structured_text(assessment.candidate.full_content, self.source_char_limit)
        if assessment.candidate.chunk_text.strip():
            return compact_structured_text(assessment.candidate.chunk_text, self.source_char_limit)
        if assessment.candidate.full_content.strip():
            return compact_structured_text(assessment.candidate.full_content, self.source_char_limit)
        page_content = str(getattr(assessment.candidate.doc, 'page_content', '') or '')
        return compact_structured_text(page_content, self.source_char_limit)

    def _relevance_score(
        self,
        *,
        source_score: float,
        query_coverage: float,
        identifier_coverage: float,
        focus_score: float,
        title_overlap: float,
        position: int,
    ) -> float:
        """计算证据的相关性分数。"""
        score = source_score * 0.54
        score += query_coverage * 0.22
        score += identifier_coverage * 0.08
        score += focus_score * 0.12
        score += title_overlap * 0.04
        score -= max(position - 1, 0) * 0.015
        return max(score, 0.0)

    def _focus_score(
        self,
        question_focus: QuestionFocus,
        text: str,
        unit_tokens: set[str],
    ) -> float:
        """估算证据单元对问题焦点的对齐程度。

        这一步并不直接判断“能不能回答”，而是估计当前单元更像在提供时间、
        数量、原因、方法还是定义类信息，供打包阶段优先保留更贴题的证据。
        """
        slot_alignment = _coverage_ratio(set(question_focus.slot_terms), unit_tokens)
        if question_focus.category is QuestionFocusType.DESCRIPTION and not question_focus.slot_terms:
            return slot_alignment

        marker_score = 0.0
        if question_focus.category is QuestionFocusType.REASON:
            marker_score = 1.0 if any(marker in text for marker in REASON_MARKERS) else 0.0
            if not question_focus.slot_terms:
                return marker_score
            if marker_score <= 0.0:
                return slot_alignment * 0.25
            return min(1.0, slot_alignment * 0.4 + marker_score * 0.6)
        elif question_focus.category is QuestionFocusType.METHOD:
            marker_score = 1.0 if any(marker in text for marker in METHOD_MARKERS) else 0.0
            if not question_focus.slot_terms:
                return marker_score
            if marker_score <= 0.0:
                return slot_alignment * 0.25
            return min(1.0, slot_alignment * 0.4 + marker_score * 0.6)
        elif question_focus.category is QuestionFocusType.TIME:
            marker_score = 1.0 if TIME_PATTERN.search(text) else 0.0
        elif question_focus.category is QuestionFocusType.QUANTITY:
            marker_score = 1.0 if NUMBER_PATTERN.search(text) else 0.0
        elif question_focus.category in {QuestionFocusType.CHOICE, QuestionFocusType.ATTRIBUTE}:
            marker_score = max(
                self._marker_ratio(text, LIST_MARKERS),
                1.0 if NUMBER_PATTERN.search(text) else 0.0,
                1.0 if TIME_PATTERN.search(text) else 0.0,
            )
        elif question_focus.category is QuestionFocusType.LOCATION:
            marker_score = self._marker_ratio(text, LOCATION_MARKERS)
        elif question_focus.category is QuestionFocusType.DEFINITION:
            marker_score = self._marker_ratio(text, DEFINITION_MARKERS)
        elif question_focus.category is QuestionFocusType.PERSON:
            marker_score = self._marker_ratio(text, DEFINITION_MARKERS)

        if not question_focus.slot_terms:
            return marker_score
        if marker_score <= 0.0:
            return slot_alignment
        return min(1.0, slot_alignment * 0.55 + marker_score * 0.45)

    def _marker_ratio(self, text: str, markers: Sequence[str]) -> float:
        """计算标记词在文本中的占比。"""
        if not markers:
            return 0.0
        matched = sum(1 for marker in markers if marker in text)
        return matched / len(markers) if matched else 0.0

    def _source_key(self, candidate: _CandidateUnit) -> int | str | None:
        """生成证据单元所属来源的稳定键。

        打包阶段需要按来源做去重和首轮选拔，因此这里统一把 doc_id 或标题
        归并成一个可比较的来源标识。
        """
        if candidate.doc_id is not None:
            return candidate.doc_id
        return candidate.title.strip() or None

    def _build_strategy(
        self,
        candidates: Sequence[_CandidateUnit],
        *,
        max_units: int,
    ) -> _PackingStrategy:
        """生成证据打包策略。"""
        source_best_scores: dict[int | str | None, float] = {}
        for candidate in candidates:
            source_key = self._source_key(candidate)
            if source_key is None:
                continue
            existing = source_best_scores.get(source_key)
            if existing is None or candidate.relevance_score > existing:
                source_best_scores[source_key] = candidate.relevance_score

        source_count = len(source_best_scores)
        if source_count <= 1:
            return _PackingStrategy(target_source_count=min(max_units, 1), support_cluster_size=source_count, dominance_ratio=0.0)

        distribution = summarize_scores(
            list(source_best_scores.values()),
            fallback_clear_margin=0.10,
            min_clear_margin=0.05,
            max_clear_margin=0.16,
            min_support_margin=0.04,
            max_support_margin=0.12,
        )

        if distribution.dominance_ratio >= 1.0 and max_units <= 4:
            target_source_count = 1
        else:
            target_source_count = min(
                source_count,
                max_units,
                max(2, distribution.support_cluster_size),
            )

        return _PackingStrategy(
            target_source_count=max(1, target_source_count),
            support_cluster_size=distribution.support_cluster_size,
            dominance_ratio=distribution.dominance_ratio,
        )

    def _select_units(
        self,
        candidates: Sequence[_CandidateUnit],
        *,
        max_units: int,
        char_budget: int,
    ) -> list[_CandidateUnit]:
        """在数量与字符预算约束下选择最终证据单元。

        选择过程会先保证来源覆盖，再做去重和预算控制，尽量让最终证据包
        同时具备多来源支撑、内容互补和长度可控三个特征。
        """
        if not candidates or max_units <= 0 or char_budget <= 0:
            return []

        selected: list[_CandidateUnit] = []
        seen_texts: set[str] = set()
        total_chars = 0
        strategy = self._build_strategy(candidates, max_units=max_units)

        top_by_doc: dict[int | None, _CandidateUnit] = {}
        for candidate in candidates:
            top_by_doc.setdefault(self._source_key(candidate), candidate)
        initial_pool = sorted(top_by_doc.values(), key=lambda item: item.relevance_score, reverse=True)

        for candidate in initial_pool:
            if len(selected) >= max_units:
                break
            normalized_text = normalize_lookup_text(candidate.text)
            if not normalized_text or normalized_text in seen_texts:
                continue
            if selected and total_chars + len(candidate.text) > char_budget:
                continue
            selected.append(candidate)
            seen_texts.add(normalized_text)
            total_chars += len(candidate.text)
            if len(selected) >= strategy.target_source_count:
                break

        while len(selected) < max_units:
            best_candidate: _CandidateUnit | None = None
            best_score = float('-inf')
            for candidate in candidates:
                normalized_text = normalize_lookup_text(candidate.text)
                if not normalized_text or normalized_text in seen_texts:
                    continue
                if selected and total_chars + len(candidate.text) > char_budget:
                    continue
                redundancy = max(
                    (_unit_similarity(candidate.tokens, item.tokens) for item in selected),
                    default=0.0,
                )
                score = candidate.relevance_score - redundancy * 0.18
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            if best_candidate is None:
                break
            selected.append(best_candidate)
            seen_texts.add(normalize_lookup_text(best_candidate.text))
            total_chars += len(best_candidate.text)

        if not selected and candidates:
            selected.append(candidates[0])
        return selected
