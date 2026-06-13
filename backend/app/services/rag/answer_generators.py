from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, Sequence

from app.services.rag.answer_synthesizer import AnswerSynthesizer, align_short_answer_to_focus
from app.services.rag.hybrid_search import has_identifier, tokenize_for_bm25
from app.services.rag.pipeline_models import (
    AnswerMode,
    AnswerPlan,
    EvidenceAssessment,
    EvidenceRequirement,
    QueryIntent,
    QuestionFocusType,
)
from app.services.rag.prompt_templates import build_no_context_answer
from app.services.rag.rerank_service import normalize_lookup_text
from app.services.rag.source_confidence import calibrated_assessment_confidence, rank_assessments_for_confidence
from app.services.rag.text_utils import compact_text, split_text_segments

try:
    import jieba.posseg as pseg
except Exception:  # pragma: no cover - 依赖缺失时回退到分词结构特征。
    pseg = None

MAX_DIRECT_EVIDENCE_LENGTH = 120
MAX_EXTRACTIVE_SPAN_LENGTH = 240
MULTI_ANCHOR_COVERAGE_FLOOR = 0.45
FUNCTIONAL_POS_PREFIXES = ('c', 'p', 'r', 'u')
FUNCTIONAL_POS_TAGS = {'e', 'o', 'w', 'x', 'y'}
CONTENT_TOKEN_PATTERN = re.compile(r'[0-9a-z\u4e00-\u9fff]+', re.IGNORECASE)
TIME_SPAN_PATTERN = re.compile(r'(?:\d{2,4}年|\d{1,2}月|\d{1,2}日|\d{1,2}[点时分]|上午|下午|凌晨|当日|当天)')
NUMBER_SPAN_PATTERN = re.compile(r'(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万两]+)(?:[%％]|个|人|名|项|次|家|位|届|天|年|月|小时|分钟|公里|米|亿元?|万元?|倍|级)?')
REASON_SPAN_MARKERS = ('因为', '由于', '原因', '缘于', '因此', '所以', '致使')
METHOD_SPAN_MARKERS = ('通过', '采用', '利用', '借助', '依靠', '方式', '方法', '步骤')


class FallbackRequiredError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueryAnchor:
    token: str
    weight: float


@dataclass(frozen=True)
class QueryAnchorProfile:
    anchors: tuple[QueryAnchor, ...]
    window_frequency: dict[str, int]
    total_weight: float


@dataclass(frozen=True)
class AnchorMatch:
    coverage: float
    matched_count: int
    matched_weight: float
    token_density: float


def _assessment_rank(assessment: EvidenceAssessment) -> tuple[float, float, float]:
    """生成评估项在抽取式作答阶段使用的排序键。

    这里优先看统一证据评分的最终分数，再把“是否含直接证据”“是否支持抽取”
    两个布尔信号作为次级排序依据。这样后面的抽取式生成器会先尝试最有希望
    直接给出答案的证据。
    """
    return (
        float(assessment.final_score),
        1.0 if assessment.direct_evidence else 0.0,
        1.0 if assessment.supports_extractive else 0.0,
    )


def _select_assessments(
    plan: AnswerPlan,
    assessments: Sequence[EvidenceAssessment],
) -> list[EvidenceAssessment]:
    """选择评估项列表。"""
    ranked = sorted(assessments, key=_assessment_rank, reverse=True)
    if not ranked:
        return []
    if not plan.source_doc_ids:
        return ranked[:3]

    selected_ids = set(plan.source_doc_ids)
    selected = [assessment for assessment in ranked if assessment.candidate.doc_id in selected_ids]
    return selected or ranked[:3]


def build_sources(assessments: Sequence[EvidenceAssessment]) -> list[dict]:
    """把评估结果转换成前端可展示的来源列表。

    生成器返回答案时，前端还需要同步看到标题、文档编号、排序和证据分。
    这里统一复用校准后的 assessment 置信度，避免不同回答模式各自拼一套来源结构。
    """
    ranked = rank_assessments_for_confidence(assessments)
    return [
        {
            'title': assessment.candidate.title,
            'doc_id': assessment.candidate.doc_id,
            'rank': index,
            'confidence': calibrated_assessment_confidence(
                assessment,
                rank_index=index - 1,
                ranked_assessments=ranked,
            ),
        }
        for index, assessment in enumerate(assessments, start=1)
    ]


def _candidate_chunk_text(assessment: EvidenceAssessment) -> str:
    """提取候选证据的片段级正文。

    抽取式回答优先依赖与问题最接近的局部片段，因此这里先尝试使用 chunk_text，
    若候选对象里没有显式缓存，再回退到底层文档对象的 `page_content`。
    """
    candidate = assessment.candidate
    if candidate.chunk_text.strip():
        return candidate.chunk_text
    page_content = str(getattr(candidate.doc, 'page_content', '') or '')
    return page_content


def _candidate_full_content(assessment: EvidenceAssessment) -> str:
    """提取候选证据可用的全文级正文。

    当局部片段无法稳定抽出答案时，生成器会回头在更长的正文范围里再试一次。
    这里统一封装全文回退逻辑，保证 chunk 和 full content 的优先级始终一致。
    """
    candidate = assessment.candidate
    if candidate.full_content.strip():
        return candidate.full_content
    full_content = str(getattr(candidate.doc, 'full_content', '') or '')
    if full_content.strip():
        return full_content
    return _candidate_chunk_text(assessment)


def _window_tokens(value: str) -> set[str]:
    """把候选窗口文本整理成可匹配的归一化词集合。

    这个集合只服务于锚点覆盖计算，因此会走和检索侧一致的归一化规则，
    方便后面判断某个片段是否真正覆盖了问题里的关键实体、编号或属性词。
    """
    return {
        normalize_lookup_text(token)
        for token in tokenize_for_bm25(value)
        if normalize_lookup_text(token)
    }


def _query_pos_flags(query: str) -> dict[str, set[str]]:
    """为查询词构建词性索引表。

    抽取式回答不会对所有查询词一视同仁。这里先收集每个归一化词元对应的词性集合，
    供后面的“锚点筛选”和“功能词过滤”判断使用。
    """
    if pseg is None:
        return {}

    flags_by_token: dict[str, set[str]] = {}
    for item in pseg.cut(query):
        normalized = normalize_lookup_text(item.word)
        if not normalized:
            continue
        flags_by_token.setdefault(normalized, set()).add(str(item.flag or ''))
    return flags_by_token


def _is_functional_pos(flag: str) -> bool:
    """判断词性是否更像功能词而非内容词。

    连词、介词、语气词一类词项通常不适合作为问题锚点，
    因此这里把它们单独识别出来，避免后面的覆盖计算被噪声词拖偏。
    """
    if not flag or flag == 'eng':
        return False
    if flag in FUNCTIONAL_POS_TAGS:
        return True
    return flag.startswith(FUNCTIONAL_POS_PREFIXES)


def _is_informative_query_token(token: str, flags: set[str]) -> bool:
    """判断查询词元是否足够有信息量，值得作为锚点。

    规则会保留编号、较长的实体词和非功能词，同时剔除过短、过虚的词项。
    目的是让抽取式回答围绕真正表达问题约束的词来做窗口匹配。
    """
    if not token or not CONTENT_TOKEN_PATTERN.search(token):
        return False
    if has_identifier(token):
        return True
    if len(token) <= 1:
        return False
    if flags and all(_is_functional_pos(flag) for flag in flags):
        return False
    return True


def _is_semantic_backbone(flags: set[str]) -> bool:
    """判断词性集合中是否包含可视为语义骨架的成分。

    名词、时间词、方位词以及英文实体词通常更能代表问题主题。
    这个判断主要用于多锚点场景下，决定哪些锚点即便局部未命中也值得保留。
    """
    for flag in flags:
        if flag == 'eng':
            return True
        if flag.startswith(('n', 't', 's')):
            return True
    return False


def _build_anchor_profile(query: str, title: str, windows: Sequence[str]) -> QueryAnchorProfile:
    """为当前问题构建锚点画像。

    锚点画像是抽取式回答的核心中间表示，里面会记录：
    1. 哪些查询词值得当锚点；
    2. 每个锚点的权重；
    3. 每个锚点在候选窗口中出现了多少次。

    后面的窗口筛选、最佳句抽取和覆盖充分性判断都会依赖这份画像。
    """
    title_tokens = _window_tokens(title)
    pos_flags = _query_pos_flags(query)
    seen: set[str] = set()
    raw_anchors: list[QueryAnchor] = []

    for token in tokenize_for_bm25(query):
        normalized = normalize_lookup_text(token)
        if normalized in seen or not _is_informative_query_token(normalized, pos_flags.get(normalized, set())):
            continue
        seen.add(normalized)
        weight = 1.0
        if has_identifier(token) or has_identifier(normalized):
            weight += 1.0
        if len(normalized) >= 4:
            weight += 0.25
        raw_anchors.append(QueryAnchor(token=normalized, weight=weight))

    if not raw_anchors:
        return QueryAnchorProfile(anchors=(), window_frequency={}, total_weight=0.0)

    window_token_sets = [_window_tokens(window) for window in windows]
    window_frequency: dict[str, int] = {}
    for anchor in raw_anchors:
        window_frequency[anchor.token] = sum(1 for token_set in window_token_sets if anchor.token in token_set)

    supported_anchor_count = sum(1 for anchor in raw_anchors if window_frequency.get(anchor.token, 0) > 0)
    has_non_title_anchor = any(anchor.token not in title_tokens for anchor in raw_anchors)
    anchors: list[QueryAnchor] = []
    for anchor in raw_anchors:
        flags = pos_flags.get(anchor.token, set())
        if (
            supported_anchor_count >= 2
            and window_frequency.get(anchor.token, 0) == 0
            and not has_identifier(anchor.token)
            and not _is_semantic_backbone(flags)
        ):
            continue
        weight = anchor.weight
        if has_non_title_anchor and anchor.token in title_tokens:
            weight = max(0.7, weight - 0.15)
        anchors.append(QueryAnchor(token=anchor.token, weight=weight))

    if not anchors:
        return QueryAnchorProfile(anchors=(), window_frequency=window_frequency, total_weight=0.0)

    total_weight = sum(_effective_anchor_weight(anchor, window_frequency.get(anchor.token, 0)) for anchor in anchors)
    return QueryAnchorProfile(
        anchors=tuple(anchors),
        window_frequency=window_frequency,
        total_weight=total_weight,
    )


def _effective_anchor_weight(anchor: QueryAnchor, frequency: int) -> float:
    """根据窗口出现频次调整锚点的有效权重。

    如果某个词在大量窗口里都出现，它区分候选片段的能力会下降。
    这里用一个轻量的频次惩罚，让“到处都有的词”在匹配时贡献变小。
    """
    if frequency <= 1:
        return anchor.weight
    return anchor.weight / (frequency ** 0.5)


def _match_text(text: str, profile: QueryAnchorProfile) -> AnchorMatch:
    """计算一段文本对锚点画像的匹配情况。

    返回值不仅包含匹配覆盖率，还记录命中的锚点数、累计权重和词密度，
    供后面的窗口排序与“证据是否足够”判断共同使用。
    """
    text_tokens = _window_tokens(text)
    if not text_tokens or not profile.anchors or profile.total_weight <= 0.0:
        return AnchorMatch(coverage=0.0, matched_count=0, matched_weight=0.0, token_density=0.0)

    matched_weight = 0.0
    matched_count = 0
    for anchor in profile.anchors:
        if anchor.token not in text_tokens:
            continue
        matched_count += 1
        matched_weight += _effective_anchor_weight(anchor, profile.window_frequency.get(anchor.token, 0))

    coverage = matched_weight / profile.total_weight if profile.total_weight else 0.0
    token_density = matched_weight / max(len(text_tokens), 1)
    return AnchorMatch(
        coverage=coverage,
        matched_count=matched_count,
        matched_weight=matched_weight,
        token_density=token_density,
    )


def _match_sort_key(match: AnchorMatch, text: str) -> tuple[float, int, float, int]:
    """生成窗口或句子匹配结果的排序键。

    排序优先级依次是覆盖率、命中锚点个数、单位长度的信息密度，
    最后才用更短的文本做轻微偏好，避免在信息量接近时总挑过长片段。
    """
    return (
        match.coverage,
        match.matched_count,
        match.token_density,
        -len(text),
    )


def _candidate_windows(content: str) -> list[str]:
    """把正文切成适合抽取答案的局部窗口。

    对短文本直接整体使用；对较长文本则按句滑动切出多组窗口，
    让抽取式回答可以先定位最相关的小范围，再决定是否抽一句还是两句。
    """
    sentences = split_text_segments(content)
    if not sentences:
        compacted = compact_text(content, 480)
        return [compacted] if compacted else []
    if len(sentences) <= 3:
        return ['\n'.join(sentences)]

    windows: list[str] = []
    for index in range(len(sentences)):
        window = compact_text('\n'.join(sentences[index : index + 3]), 480)
        if window:
            windows.append(window)
    return windows


def _clean_extractive_text(value: str) -> str:
    """清洗抽取得到的候选答案文本。

    抽取式回答经常会带出列表符号、Markdown 标记或多余分隔符。
    这里先做一次结构清洗，保证后续长度判断和焦点对齐判断基于更干净的文本执行。
    """
    cleaned = str(value or '').strip()
    cleaned = re.sub(r'^\s*[-*•]+\s*', '', cleaned)
    cleaned = cleaned.replace('**', '')
    cleaned = cleaned.replace('\\', '')
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip('：:，,；; ')


def _ensure_sentence(value: str) -> str:
    """确保句子已准备就绪。"""
    cleaned = compact_text(value, 640).strip()
    if not cleaned:
        return ''
    if cleaned[-1] not in '。！？!?':
        cleaned += '。'
    return cleaned


def _final_answer_limit(intent: QueryIntent) -> int:
    """根据证据需求类型决定最终答案的长度上限。

    原子片段回答应更短，多片段或全文型回答可以适当放宽。
    这个上限既影响抽取结果裁剪，也影响后续统一答案规范化。
    """
    if intent.evidence_requirement is EvidenceRequirement.FULL_DOCUMENT:
        return 720
    if intent.evidence_requirement is EvidenceRequirement.MULTI_SPAN:
        return 480
    return 240


def _normalize_atomic_answer(value: str, *, intent: QueryIntent) -> str:
    """把抽取式短答案整理成最终可返回的自然语言句子。

    这里会依次做文本清洗、长度压缩、短焦点扩写和句尾闭合，
    让抽取到的片段从“局部证据”变成“前端可以直接展示的答案”。
    """
    cleaned = _clean_extractive_text(compact_text(value, _final_answer_limit(intent)))
    if not cleaned:
        return ''
    cleaned = re.sub(r'\s*[-*•]\s*', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = _expand_short_focus_answer(intent.normalized_query, intent, cleaned)
    return _ensure_sentence(cleaned)


def _usable_direct_evidence(value: str) -> str:
    """判断一段直接证据是否适合作为最终短答案直接复用。

    某些 judge 返回的 `answer_brief` 或 `evidence_quote` 已经足够短且结构完整，
    这时无需再次抽句。这里专门过滤掉过长、过碎或明显带列表结构的候选。
    """
    cleaned = compact_text(value, 240).strip()
    if not cleaned:
        return ''
    if len(cleaned) > MAX_DIRECT_EVIDENCE_LENGTH:
        return ''
    if cleaned.count('：') > 1:
        return ''
    if ' - ' in cleaned or '•' in cleaned:
        return ''
    return cleaned


def _best_atomic_candidate(window: str, profile: QueryAnchorProfile) -> tuple[str, AnchorMatch]:
    """在单个窗口内挑选最适合直接回答的句子或句对。

    逻辑会先尝试单句，再在必要时尝试相邻双句组合，最后选择匹配度最高、
    同时长度仍受控的候选，服务于原子片段问题的抽取式回答。
    """
    sentences = split_text_segments(window) or [window]
    sentence_candidates = [
        (_clean_extractive_text(sentence), _match_text(sentence, profile))
        for sentence in sentences
        if _clean_extractive_text(sentence)
    ]
    if not sentence_candidates:
        return '', AnchorMatch(coverage=0.0, matched_count=0, matched_weight=0.0, token_density=0.0)

    best_sentence, best_match = max(sentence_candidates, key=lambda item: _match_sort_key(item[1], item[0]))
    if len(sentence_candidates) <= 1 or best_match.matched_count >= 2:
        return best_sentence, best_match

    pair_candidates: list[tuple[str, AnchorMatch]] = []
    for index in range(len(sentences) - 1):
        combined = compact_text(f'{sentences[index]} {sentences[index + 1]}', MAX_EXTRACTIVE_SPAN_LENGTH)
        cleaned = _clean_extractive_text(combined)
        if cleaned:
            pair_candidates.append((cleaned, _match_text(cleaned, profile)))

    if not pair_candidates:
        return best_sentence, best_match

    best_pair, best_pair_match = max(pair_candidates, key=lambda item: _match_sort_key(item[1], item[0]))
    if _match_sort_key(best_pair_match, best_pair) > _match_sort_key(best_match, best_sentence):
        return best_pair, best_pair_match
    return best_sentence, best_match


def _match_is_sufficient(match: AnchorMatch, profile: QueryAnchorProfile) -> bool:
    """判断当前锚点匹配结果是否足以支撑抽取回答。

    单锚点问题只要命中即可，多锚点问题则必须满足一定覆盖比例，
    以避免只抓住问题中的一个局部词，就误把不完整片段当成最终答案。
    """
    anchor_count = len(profile.anchors)
    if anchor_count == 0 or match.matched_count == 0:
        return False
    if anchor_count == 1:
        return True
    if match.matched_count < 2:
        return False
    return match.coverage >= MULTI_ANCHOR_COVERAGE_FLOOR


def _answer_tagged_tokens(answer: str) -> list[tuple[str, str]]:
    """对候选答案分词并保留词性标记。

    这一步主要服务于人名、地名等焦点对齐判断，
    用来检查答案里是否出现了查询中没有的新实体词。
    """
    if pseg is None:
        return []
    tagged: list[tuple[str, str]] = []
    for item in pseg.cut(answer):
        normalized = normalize_lookup_text(item.word)
        if not normalized:
            continue
        tagged.append((normalized, str(item.flag or '')))
    return tagged


def _contains_novel_pos_token(query: str, answer: str, *, prefixes: tuple[str, ...]) -> bool:
    """判断答案是否包含查询中未出现、但词性符合要求的新词元。

    对人员、地点类问题，答案往往需要给出一个新的实体词。
    这里通过词性前缀检测这类“新信息”是否真正出现。
    """
    query_tokens = _window_tokens(query)
    for normalized, flag in _answer_tagged_tokens(answer):
        if normalized in query_tokens:
            continue
        if flag.startswith(prefixes):
            return True
    return False


def _contains_marker(answer: str, markers: tuple[str, ...]) -> bool:
    """判断答案里是否出现指定类型的提示标记词。"""
    return any(marker in answer for marker in markers)


def _focus_alignment_is_clear(query: str, intent: QueryIntent, answer: str) -> bool:
    """判断候选答案是否真正对齐了问题焦点。

    即使片段命中了查询锚点，也未必回答了“时间 / 数量 / 原因 / 方法”等核心槽位。
    这里按焦点类型做一层轻量校验，减少答非所问的抽取结果。
    """
    focus = intent.question_focus
    if not focus.has_explicit_cue:
        return True

    if focus.category is QuestionFocusType.TIME:
        return bool(TIME_SPAN_PATTERN.search(answer) or NUMBER_SPAN_PATTERN.search(answer))
    if focus.category is QuestionFocusType.QUANTITY:
        return bool(NUMBER_SPAN_PATTERN.search(answer))
    if focus.category is QuestionFocusType.REASON:
        return _contains_marker(answer, REASON_SPAN_MARKERS)
    if focus.category is QuestionFocusType.METHOD:
        return _contains_marker(answer, METHOD_SPAN_MARKERS)
    if focus.category is QuestionFocusType.PERSON:
        return _contains_novel_pos_token(query, answer, prefixes=('nr',))
    if focus.category is QuestionFocusType.LOCATION:
        return _contains_novel_pos_token(query, answer, prefixes=('ns', 's'))
    return True


def _expand_short_focus_answer(query: str, intent: QueryIntent, answer: str) -> str:
    """在安全前提下，把过短答案补成更自然的焦点表述。

    这一步并不新增事实，只是把非常短的槽位值重新嵌回问题框架，
    让最终回答更像自然中文，而不是孤立的一个时间点或地点名。
    """
    return align_short_answer_to_focus(query, intent, answer)


def _extractive_answer_from_content(
    query: str,
    intent: QueryIntent,
    primary: EvidenceAssessment,
    content: str,
) -> str:
    """从候选正文中抽取一个与问题最对齐的答案片段。

    这是抽取式生成器的主干流程：先切窗口、再做锚点匹配、再选句子，
    最后再叠加焦点对齐与覆盖充分性约束，尽量把“能抽出来”和“值得返回”
    区分开来。
    """
    if not content.strip():
        return ''

    windows = _candidate_windows(content)
    profile = _build_anchor_profile(query, primary.candidate.title, windows)
    if not profile.anchors or not windows:
        return ''

    scored_windows = [
        (window, _match_text(window, profile))
        for window in windows
    ]
    best_window, best_window_match = max(scored_windows, key=lambda item: _match_sort_key(item[1], item[0]))
    if not _match_is_sufficient(best_window_match, profile):
        return ''

    if intent.evidence_requirement is EvidenceRequirement.ATOMIC_SPAN:
        candidate_answer, candidate_match = _best_atomic_candidate(best_window, profile)
        if not _match_is_sufficient(candidate_match, profile):
            return ''
    else:
        candidate_answer = _clean_extractive_text(best_window)
        candidate_match = best_window_match
    if len(candidate_answer) < 6:
        return ''
    if not _match_is_sufficient(candidate_match, profile):
        return ''
    if not _focus_alignment_is_clear(query, intent, candidate_answer):
        return ''
    return candidate_answer


class ExtractiveGenerator:
    """优先直接复用高置信证据，必要时再抽取最合适的支撑片段。"""

    async def generate(
        self,
        plan: AnswerPlan,
        query: str,
        intent: QueryIntent,
        assessments: Sequence[EvidenceAssessment],
    ) -> str | None:
        """基于高置信证据生成抽取式答案。

        先尝试直接复用 judge 给出的短答案或证据引文；若不满足条件，
        再退回到正文窗口中做抽取。返回 `None` 表示当前模式无法稳定生成答案，
        由上层触发降级。
        """
        del plan
        if not assessments:
            return None

        ranked_assessments = sorted(assessments, key=_assessment_rank, reverse=True)
        for assessment in ranked_assessments:
            brief_answer = _usable_direct_evidence(assessment.answer_brief)
            if brief_answer and _focus_alignment_is_clear(query, intent, brief_answer):
                return _normalize_atomic_answer(brief_answer, intent=intent)

            evidence_quote = _usable_direct_evidence(assessment.evidence_quote)
            if evidence_quote and _focus_alignment_is_clear(query, intent, evidence_quote):
                return _normalize_atomic_answer(evidence_quote, intent=intent)

        for assessment in ranked_assessments:
            candidate_answer = _extractive_answer_from_content(query, intent, assessment, _candidate_chunk_text(assessment))
            if not candidate_answer:
                full_content = _candidate_full_content(assessment)
                chunk_text = _candidate_chunk_text(assessment)
                if full_content.strip() and full_content.strip() != chunk_text.strip():
                    candidate_answer = _extractive_answer_from_content(query, intent, assessment, full_content)
            if candidate_answer:
                return _normalize_atomic_answer(candidate_answer, intent=intent)
        return None


class SynthesisGenerator:
    """基于统一证据综合的非抽取式生成器。"""

    def __init__(self, model_getter: Callable):
        """初始化生成式综合器依赖。"""
        self.synthesizer = AnswerSynthesizer(model_getter)

    async def generate(
        self,
        plan: AnswerPlan,
        query: str,
        intent: QueryIntent,
        assessments: Sequence[EvidenceAssessment],
    ) -> str:
        """基于统一证据包生成非抽取式答案。"""
        content = await self.synthesizer.synthesize(plan, query, intent, assessments)
        if not content:
            raise FallbackRequiredError(f'{plan.mode.value}_empty')
        return content


class GeneratorFactory:
    def __init__(self, model_getter: Callable):
        """初始化不同回答模式共用的生成器集合。"""
        self.extractive = ExtractiveGenerator()
        self.synthesis = SynthesisGenerator(model_getter)

    async def execute(
        self,
        plan: AnswerPlan,
        query: str,
        intent: QueryIntent,
        assessments: Sequence[EvidenceAssessment],
    ) -> tuple[str, list[dict]]:
        """按当前回答计划执行对应的生成器。

        这里是回答生成阶段的统一分发入口：根据 plan 选择抽取式、
        结构化综合式或泛生成式路径，并同时回填该答案实际使用的来源列表。
        """
        selected = _select_assessments(plan, assessments)
        if plan.mode is AnswerMode.NO_CONTEXT or not selected:
            return build_no_context_answer(), []

        if plan.mode is AnswerMode.EXTRACTIVE:
            extracted = await self.extractive.generate(plan, query, intent, selected)
            if extracted is None:
                raise FallbackRequiredError('extractive_failed')
            return extracted, build_sources(selected[:1])

        if plan.mode in {AnswerMode.STRUCTURED, AnswerMode.GENERATIVE}:
            return await self.synthesis.generate(plan, query, intent, selected), build_sources(selected[:3])

        return build_no_context_answer(), []

    @staticmethod
    def downgrade(plan: AnswerPlan) -> AnswerPlan:
        """在当前生成模式失败时生成下一档降级计划。

        降级顺序遵循“抽取式 -> 结构化 -> 泛生成 -> 无上下文”，
        这样既尽量保留证据约束，也避免单一生成模式失败后整条链路直接中断。
        """
        if plan.mode is AnswerMode.EXTRACTIVE:
            return replace(
                plan,
                mode=AnswerMode.STRUCTURED,
                reason='extractive_fallback',
                generator_name='structured_generator',
            )
        if plan.mode is AnswerMode.STRUCTURED:
            return replace(
                plan,
                mode=AnswerMode.GENERATIVE,
                reason='structured_fallback',
                generator_name='generative_generator',
            )
        return replace(
            plan,
            mode=AnswerMode.NO_CONTEXT,
            reason='generation_exhausted',
            generator_name='no_context_generator',
            source_doc_ids=(),
            primary_doc_id=None,
        )
