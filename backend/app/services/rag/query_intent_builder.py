from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.rag.hybrid_search import extract_identifiers, has_identifier, tokenize_for_bm25
from app.services.rag.pipeline_models import (
    DefenseProfile,
    EvidenceRequirement,
    QuestionFocus,
    QuestionFocusType,
    QueryIntent,
    QueryIntentType,
)
from app.services.rag.rerank_service import normalize_lookup_text

try:
    import jieba.posseg as pseg
except Exception:  # pragma: no cover - 依赖缺失时退回分词结构特征。
    pseg = None

TRAILING_NOISE = '，,。.!！？?：:；; '
CLAUSE_SEPARATOR_PATTERN = re.compile(r'[，,；;。！？?]+')
QUOTED_SPAN_PATTERN = re.compile(r'(?:《[^》]{1,80}》|“[^”]{1,80}”|"[^"\n]{1,80}")')
LABELLED_BLOCK_PATTERN = re.compile(r'(?:^|\n)\s*[\w\u4e00-\u9fff（）()《》-]{1,12}\s*[:：]')
FUNCTIONAL_POS_PREFIXES = ('c', 'p', 'r', 'u')
FUNCTIONAL_POS_TAGS = {'e', 'o', 'w', 'x', 'y'}
CONTENT_TOKEN_PATTERN = re.compile(r'[0-9a-z\u4e00-\u9fff]+', re.IGNORECASE)
PERSON_CUE_PATTERN = re.compile(r'(?:谁|哪位|何人|是谁)')
LOCATION_CUE_PATTERN = re.compile(r'(?:在哪里|在哪儿|在何处|位于哪里|位于何处|何处|何地)')
TIME_CUE_PATTERN = re.compile(
    r'(?:什么时候|何时|哪一年|哪年|哪天|几时|几月几日|几月几号|几月|几号|几日|多久)'
)
QUANTITY_CUE_PATTERN = re.compile(
    r'(?:多少[^\s，。！？?]{0,4}|第几[^\s，。！？?]{0,2}|几次|几人|几名|几位|几年|几个月|几天|几家|几项|几种|多大|多长|多远|多高|多宽|多深|多重)'
)
REASON_CUE_PATTERN = re.compile(r'(?:为什么|为何|何故|缘何|原因是什么|原因在哪(?:里|儿)?|缘由是什么)')
METHOD_CUE_PATTERN = re.compile(r'(?:如何|怎么|怎样|怎么办|方法是什么|方式是什么|途径是什么)')
LIST_CUE_PATTERN = re.compile(
    r'(?:哪些[^\s，。！？?]{0,4}|哪几[^\s，。！？?]{0,4}|什么内容|包括什么|包含什么|分别是什么)'
)
CHOICE_CUE_PATTERN = re.compile(
    r'(?:哪个[^\s，。！？?]{0,4}|哪一[^\s，。！？?]{0,4}|什么[^\s，。！？?]{0,4})'
)
DEFINITION_CUE_PATTERN = re.compile(r'(?:是什么|指什么|叫什么|名称是什么|名字是什么|含义是什么|意思是什么)')
ATTRIBUTE_TIME_SUFFIX_PATTERN = re.compile(r'(?:时间|日期|年份|年限|时长|时点)\s*$')
ATTRIBUTE_LOCATION_SUFFIX_PATTERN = re.compile(r'(?:地点|位置|所在地)\s*$')
ATTRIBUTE_REASON_SUFFIX_PATTERN = re.compile(r'(?:原因|缘由|依据)\s*$')
ATTRIBUTE_METHOD_SUFFIX_PATTERN = re.compile(r'(?:方式|方法|途径)\s*$')
ATTRIBUTE_QUANTITY_SUFFIX_PATTERN = re.compile(r'(?:数量|规模|次数|金额|价格|比例|数据)\s*$')
DEFAULT_RETRIEVAL_DEPTH = {
    QueryIntentType.LOOKUP: 6,
    QueryIntentType.FACTOID: 8,
    QueryIntentType.RELATION: 10,
    QueryIntentType.SUMMARY: 10,
    QueryIntentType.OVERVIEW: 10,
    QueryIntentType.REASON: 10,
    QueryIntentType.LOCATION: 8,
}


@dataclass(frozen=True)
class QueryStructureProfile:
    token_count: int
    semantic_token_count: int
    clause_count: int
    connector_count: int
    possessive_split_count: int
    quoted_anchor_count: int
    identifier_count: int
    has_identifier: bool
    has_question_mark: bool
    has_structured_blocks: bool
    is_fragment_like: bool


@dataclass(frozen=True)
class TaggedToken:
    raw: str
    normalized: str
    flag: str
    start: int
    end: int


def _collapse_spaces(value: str) -> str:
    """压缩连续空白并返回单行文本。"""
    return ' '.join(str(value or '').split()).strip()


def _is_functional_pos(flag: str) -> bool:
    """判断词性是否属于功能词。"""
    if not flag or flag == 'eng':
        return False
    if flag in FUNCTIONAL_POS_TAGS:
        return True
    return flag.startswith(FUNCTIONAL_POS_PREFIXES)


def _is_semantic_backbone(flag: str) -> bool:
    """判断词性是否属于语义骨干成分。"""
    return flag == 'eng' or flag.startswith(('n', 't', 's'))


def _tagged_tokens(query: str) -> list[TaggedToken]:
    """提取带词性标记的查询词元。

    这一层不是为了做通用中文分词，而是为后面的查询规范化、关键词压缩和结构画像
    提供“词元 + 词性 + 原始位置”三类信息。位置索引尤其重要，因为裁剪问题首尾噪声、
    抽取焦点槽位时都依赖字符级边界。
    """
    if pseg is None:
        return []

    tagged: list[TaggedToken] = []
    cursor = 0
    for item in pseg.cut(query):
        raw = str(item.word or '')
        if not raw:
            continue
        start = query.find(raw, cursor)
        if start < 0:
            start = query.find(raw)
            if start < 0:
                continue
        end = start + len(raw)
        cursor = end
        tagged.append(
            TaggedToken(
                raw=raw,
                normalized=normalize_lookup_text(raw),
                flag=str(item.flag or ''),
                start=start,
                end=end,
            )
        )
    return tagged


def _is_semantic_token(token: TaggedToken) -> bool:
    """判断词元是否属于语义骨干。"""
    if not token.normalized or not CONTENT_TOKEN_PATTERN.search(token.normalized):
        return False
    if has_identifier(token.raw) or has_identifier(token.normalized):
        return True
    if len(token.normalized) <= 1:
        return _is_semantic_backbone(token.flag)
    if _is_functional_pos(token.flag):
        return False
    return True


def _is_boundary_anchor(token: TaggedToken) -> bool:
    """判断词元是否属于边界锚点。"""
    if not _is_semantic_token(token):
        return False
    if has_identifier(token.raw) or has_identifier(token.normalized):
        return True
    if _is_semantic_backbone(token.flag):
        return True
    return len(token.normalized) >= 3


def _trim_query_edges(query: str) -> str:
    """裁剪问题首尾的噪声成分。"""
    tagged_tokens = _tagged_tokens(query)
    semantic_tokens = [token for token in tagged_tokens if _is_semantic_token(token)]
    if not semantic_tokens:
        return query.strip(TRAILING_NOISE)
    leading_anchor = next((token for token in semantic_tokens if _is_boundary_anchor(token)), semantic_tokens[0])
    start = leading_anchor.start
    end = semantic_tokens[-1].end
    return query[start:end].strip(TRAILING_NOISE)


def normalize_query(query: str) -> tuple[str, tuple[str, ...]]:
    """规范化查询。

    这里主要做两件事：先压缩空白，再尽量去掉问题首尾对检索价值不高的噪声成分。
    返回值除了规范化后的问题文本，还会带上 trace tag，方便后端日志回看
    “这一轮问题是否经历过清洗”。
    """
    original = _collapse_spaces(query)
    normalized = _trim_query_edges(original)

    trace_tags: list[str] = []
    if normalized != original:
        trace_tags.append('query_normalized')
    return normalized or original, tuple(trace_tags)


def _ranked_query_terms(text: str, *, limit: int) -> list[str]:
    """按重要性返回查询词项。

    该函数服务于混合检索里的 BM25 查询构造。整体思路不是简单保留全部分词，
    而是优先保留编号、实体词和语义骨干词，尽量丢掉长度过短或功能性太强的词项，
    让稀疏检索看到的查询更集中、更像人工提炼后的关键词。
    """
    tagged_lookup = {token.normalized: token.flag for token in _tagged_tokens(text) if token.normalized}
    tokens: list[str] = []
    seen: set[str] = set()
    ranked_tokens: list[tuple[int, int, str]] = []
    for index, token in enumerate(tokenize_for_bm25(text)):
        normalized = normalize_lookup_text(token)
        if (
            not normalized
            or normalized in seen
        ):
            continue
        flag = tagged_lookup.get(normalized, '')
        if not has_identifier(token):
            if len(normalized) <= 1:
                continue
            if _is_functional_pos(flag):
                continue
        seen.add(normalized)
        priority = 2
        if has_identifier(token):
            priority = 0
        elif _is_semantic_backbone(flag):
            priority = 1
        ranked_tokens.append((priority, index, str(token)))

    ranked_tokens.sort(key=lambda item: (item[0], item[1]))
    for _priority, _index, token in ranked_tokens[:limit]:
        tokens.append(token)
    return tokens


def build_keyword_query(query: str) -> str:
    """构建检索用关键词查询。

    返回结果会作为词项检索侧的输入，与原始自然语言问题形成区分：
    稠密检索保留语义表达，稀疏检索则更强调实体、编号和关键属性。
    """
    terms = _ranked_query_terms(query, limit=8)
    return ' '.join(terms).strip() or query


def _count_clauses(query: str) -> int:
    """统计问题中显式分句的数量。"""
    clauses = [segment for segment in CLAUSE_SEPARATOR_PATTERN.split(query) if segment.strip()]
    return len(clauses) or 1


def _focus_prompt_hint(
    category: QuestionFocusType,
    *,
    expects_multiple_points: bool,
) -> str:
    """生成问题焦点提示。"""
    base_hint_map = {
        QuestionFocusType.PERSON: '明确回答被询问的人员或主体身份',
        QuestionFocusType.LOCATION: '明确回答被询问的位置或地点',
        QuestionFocusType.TIME: '明确回答被询问的时间、时段或日期',
        QuestionFocusType.QUANTITY: '明确回答被询问的数量、规模或数值',
        QuestionFocusType.REASON: '明确回答被询问的原因、依据或动因',
        QuestionFocusType.METHOD: '明确回答被询问的方法、做法或途径',
        QuestionFocusType.CHOICE: '明确回答被询问的对象、选项或并列项',
        QuestionFocusType.DEFINITION: '明确回答被询问的定义、名称或属性值',
        QuestionFocusType.ATTRIBUTE: '明确给出问题所求的属性值',
        QuestionFocusType.DESCRIPTION: '直接概括用户索要的核心事实',
    }
    hint = base_hint_map[category]
    if expects_multiple_points:
        return f'{hint}，并按问题维度逐条组织'
    return hint


def _focus_slot_terms(
    query: str,
    *,
    cue_start: int,
    cue_end: int,
) -> tuple[str, ...]:
    """提取焦点槽位词项。"""
    tagged_tokens = _tagged_tokens(query)
    if tagged_tokens:
        candidates: list[tuple[int, str]] = []
        seen: set[str] = set()
        for token in tagged_tokens:
            if not _is_semantic_token(token):
                continue
            if token.start >= cue_start and token.end <= cue_end:
                continue
            distance = min(abs(token.start - cue_end), abs(cue_start - token.end))
            normalized = token.normalized
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append((distance, token.raw))
        candidates.sort(key=lambda item: item[0])
        return tuple(token for _distance, token in candidates[:4])

    terms: list[str] = []
    seen_fallback: set[str] = set()
    for token in tokenize_for_bm25(query):
        normalized = normalize_lookup_text(token)
        if not normalized or normalized in seen_fallback:
            continue
        seen_fallback.add(normalized)
        terms.append(str(token))
    return tuple(terms[:4])


def _question_focus_expects_multiple_points(
    category: QuestionFocusType,
    query: str,
    profile: QueryStructureProfile,
) -> bool:
    """判断问题焦点是否要求多点回答。"""
    strong_multi_signal = (
        profile.clause_count >= 2
        or '分别' in query
        or bool(LIST_CUE_PATTERN.search(query))
        or profile.quoted_anchor_count >= 2
    )
    if category in {QuestionFocusType.REASON, QuestionFocusType.METHOD}:
        return strong_multi_signal
    if category is QuestionFocusType.CHOICE:
        return True
    return strong_multi_signal or profile.connector_count >= 1 or _prefers_multi_span(profile)


def infer_question_focus(query: str, profile: QueryStructureProfile) -> QuestionFocus:
    """推断问题的主要关注点。"""
    expects_multiple_points = _question_focus_expects_multiple_points(QuestionFocusType.DESCRIPTION, query, profile)
    explicit_detectors = [
        (
            QuestionFocusType.REASON,
            REASON_CUE_PATTERN,
            _question_focus_expects_multiple_points(QuestionFocusType.REASON, query, profile),
        ),
        (
            QuestionFocusType.METHOD,
            METHOD_CUE_PATTERN,
            _question_focus_expects_multiple_points(QuestionFocusType.METHOD, query, profile),
        ),
        (
            QuestionFocusType.TIME,
            TIME_CUE_PATTERN,
            _question_focus_expects_multiple_points(QuestionFocusType.TIME, query, profile),
        ),
        (QuestionFocusType.LOCATION, LOCATION_CUE_PATTERN, False),
        (QuestionFocusType.QUANTITY, QUANTITY_CUE_PATTERN, False),
        (QuestionFocusType.PERSON, PERSON_CUE_PATTERN, False),
        (QuestionFocusType.CHOICE, LIST_CUE_PATTERN, True),
        (QuestionFocusType.DEFINITION, DEFINITION_CUE_PATTERN, False),
        (QuestionFocusType.CHOICE, CHOICE_CUE_PATTERN, expects_multiple_points),
    ]
    for category, pattern, focus_multi in explicit_detectors:
        matched = pattern.search(query)
        if not matched:
            continue
        return QuestionFocus(
            category=category,
            cue_text=matched.group(0),
            slot_terms=_focus_slot_terms(query, cue_start=matched.start(), cue_end=matched.end()),
            prompt_hint=_focus_prompt_hint(category, expects_multiple_points=focus_multi),
            expects_multiple_points=focus_multi,
        )

    suffix_detectors = (
        (QuestionFocusType.TIME, ATTRIBUTE_TIME_SUFFIX_PATTERN),
        (QuestionFocusType.LOCATION, ATTRIBUTE_LOCATION_SUFFIX_PATTERN),
        (QuestionFocusType.REASON, ATTRIBUTE_REASON_SUFFIX_PATTERN),
        (QuestionFocusType.METHOD, ATTRIBUTE_METHOD_SUFFIX_PATTERN),
        (QuestionFocusType.QUANTITY, ATTRIBUTE_QUANTITY_SUFFIX_PATTERN),
    )
    for category, pattern in suffix_detectors:
        matched = pattern.search(query)
        if not matched:
            continue
        return QuestionFocus(
            category=category,
            cue_text='',
            slot_terms=_focus_slot_terms(query, cue_start=matched.start(), cue_end=matched.end()),
            prompt_hint=_focus_prompt_hint(category, expects_multiple_points=False),
            expects_multiple_points=False,
        )

    if (
        not profile.has_question_mark
        and not profile.has_structured_blocks
        and profile.semantic_token_count >= 2
        and (profile.possessive_split_count >= 1 or profile.token_count <= 8)
    ):
        return QuestionFocus(
            category=QuestionFocusType.ATTRIBUTE,
            cue_text='',
            slot_terms=_focus_slot_terms(query, cue_start=max(len(query) - 4, 0), cue_end=len(query)),
            prompt_hint=_focus_prompt_hint(QuestionFocusType.ATTRIBUTE, expects_multiple_points=expects_multiple_points),
            expects_multiple_points=expects_multiple_points,
        )

    return QuestionFocus(
        category=QuestionFocusType.DESCRIPTION,
        cue_text='',
        slot_terms=_focus_slot_terms(query, cue_start=0, cue_end=0),
        prompt_hint=_focus_prompt_hint(QuestionFocusType.DESCRIPTION, expects_multiple_points=expects_multiple_points),
        expects_multiple_points=expects_multiple_points,
    )


def build_structure_profile(query: str) -> QueryStructureProfile:
    """构建问题结构画像。

    结构画像是整套意图识别规则的基础输入，负责把“这个问题长什么样”
    转成一组可判定的统计特征，例如分句数量、连接词数量、是否含编号、
    是否像短片段检索词、是否存在结构化块等。

    后面的检索深度、证据需求和防御策略都直接依赖这份画像，因此这里偏向
    保守提取，不追求复杂分类模型，而追求行为可解释。
    """
    tagged_tokens = _tagged_tokens(query)
    token_count = len(tokenize_for_bm25(query))
    semantic_token_count = sum(1 for token in tagged_tokens if _is_semantic_token(token))
    possessive_split_count = query.count('的')
    identifier_count = len(extract_identifiers(query))
    clause_count = _count_clauses(query)
    quoted_anchor_count = len(QUOTED_SPAN_PATTERN.findall(query))
    connector_count = 0
    semantic_indices = [index for index, token in enumerate(tagged_tokens) if _is_semantic_token(token)]
    for left_index, right_index in zip(semantic_indices, semantic_indices[1:]):
        between = tagged_tokens[left_index + 1 : right_index]
        if not between:
            continue
        if any(token.flag.startswith(('c', 'p')) for token in between) and all(
            _is_functional_pos(token.flag) for token in between
        ):
            connector_count += 1
    has_question_mark = any(symbol in query for symbol in ('?', '？'))
    has_structured_blocks = len(LABELLED_BLOCK_PATTERN.findall(query)) >= 2
    is_fragment_like = (
        semantic_token_count <= 1
        and possessive_split_count == 0
        and clause_count == 1
        and not has_question_mark
    )
    return QueryStructureProfile(
        token_count=token_count,
        semantic_token_count=semantic_token_count,
        clause_count=clause_count,
        connector_count=connector_count,
        possessive_split_count=possessive_split_count,
        quoted_anchor_count=quoted_anchor_count,
        identifier_count=identifier_count,
        has_identifier=identifier_count > 0,
        has_question_mark=has_question_mark,
        has_structured_blocks=has_structured_blocks,
        is_fragment_like=is_fragment_like,
    )


def _prefers_multi_span(profile: QueryStructureProfile) -> bool:
    """判断是否更适合多片段作答。"""
    if profile.has_structured_blocks:
        return True
    if profile.clause_count >= 2:
        return True
    if profile.connector_count >= 1 and profile.semantic_token_count >= 3:
        return True
    if profile.quoted_anchor_count >= 2:
        return True
    if profile.has_identifier and profile.semantic_token_count >= 4:
        return True
    if profile.semantic_token_count >= 6:
        return True
    if profile.token_count >= 10:
        return True
    if profile.possessive_split_count >= 1 and not profile.has_question_mark and profile.semantic_token_count >= 2:
        return True
    return False


def infer_intent_type(query: str, profile: QueryStructureProfile) -> tuple[QueryIntentType, tuple[str, ...]]:
    """推断问题意图类型。

    这里给出的不是细粒度语义标签，而是问答链路真正关心的粗粒度工作模式：
    是偏“查一个对象/属性”的 lookup，还是偏事实问答 / 关系识别。
    分类保持克制，目的不是做学术意义上的意图识别，而是为后续检索和作答策略分流。
    """
    if profile.is_fragment_like:
        return QueryIntentType.LOOKUP, ('intent_lookup',)
    if profile.connector_count >= 1 and profile.clause_count <= 2 and profile.semantic_token_count <= 4:
        return QueryIntentType.RELATION, ('intent_relation',)
    return QueryIntentType.FACTOID, ('intent_factoid',)


def build_retrieval_depth(
    query: str,
    intent_type: QueryIntentType,
    profile: QueryStructureProfile,
) -> int:
    """生成当前问题的检索深度。

    检索深度本质上是在控制“要不要多拿几篇候选来覆盖复杂问题”。
    规则会综合编号命中、多分句结构、问题长度和锚点数量来抬高深度，
    但最终仍设置上限，避免因为少数长问题把后续重排和生成成本无限放大。
    """
    depth = DEFAULT_RETRIEVAL_DEPTH[intent_type]
    if profile.has_identifier:
        depth += 2
    if profile.token_count >= 10:
        depth += 2
    if _prefers_multi_span(profile):
        depth += 2
    if profile.quoted_anchor_count >= 2:
        depth += 1
    return min(depth, 16)


def infer_defense_profile(
    intent_type: QueryIntentType,
    profile: QueryStructureProfile,
) -> DefenseProfile:
    """推断当前问题的防御策略画像。

    防御策略描述的是回答时该有多保守。编号明确、目标单一的问题更适合严格防御，
    因为答错一个值就会非常明显；开放式、长文本、无显式锚点的问题则可以放松一点，
    否则系统会过于频繁地拒答。
    """
    if intent_type is QueryIntentType.LOOKUP:
        return DefenseProfile.STRICT
    if profile.has_identifier and not _prefers_multi_span(profile):
        return DefenseProfile.STRICT
    if profile.token_count >= 14 and not profile.has_identifier:
        return DefenseProfile.LOOSE
    return DefenseProfile.MODERATE


def infer_evidence_requirement(
    intent_type: QueryIntentType,
    profile: QueryStructureProfile,
) -> EvidenceRequirement:
    """推断当前问题的证据需求等级。

    这个判断直接影响后面证据如何组织。若问题只是查一个值，原子片段即可；
    若问题天然跨分项或跨片段，就应切换到 multi-span / full-document 取证思路，
    否则回答阶段即使模型能力足够，也会因为证据输入过窄而失真。
    """
    if profile.has_structured_blocks:
        return EvidenceRequirement.FULL_DOCUMENT
    if intent_type is QueryIntentType.LOOKUP:
        return EvidenceRequirement.ATOMIC_SPAN
    if _prefers_multi_span(profile):
        return EvidenceRequirement.MULTI_SPAN
    return EvidenceRequirement.ATOMIC_SPAN


def should_enable_judge(
    intent_type: QueryIntentType,
    *,
    evidence_requirement: EvidenceRequirement,
    defense_profile: DefenseProfile,
) -> bool:
    """判断当前问题是否应启用证据判断。"""
    if evidence_requirement is not EvidenceRequirement.ATOMIC_SPAN:
        return False
    if defense_profile is not DefenseProfile.STRICT:
        return False
    return intent_type in {QueryIntentType.LOOKUP, QueryIntentType.FACTOID, QueryIntentType.LOCATION}


class QueryIntentBuilder:
    @classmethod
    async def build(cls, query: str) -> QueryIntent:
        """构建供 RAG 主链路消费的完整问题意图。

        这是意图模块对外暴露的唯一入口，负责把原始用户问题逐步转换成：
        1. 规范化后的查询文本；
        2. 供词项检索使用的 keyword query；
        3. 检索深度、防御等级、证据需求等控制参数；
        4. 供日志和排障使用的 trace tags；
        5. 供最终回答使用的问题焦点提示。

        后端后续模块不会再重复分析一次问题，而是统一消费这里生成的 QueryIntent。
        """
        original_query = _collapse_spaces(query)
        normalized_query, normalization_tags = normalize_query(original_query)
        keyword_query = build_keyword_query(normalized_query)
        profile = build_structure_profile(normalized_query)
        intent_type, intent_tags = infer_intent_type(normalized_query, profile)
        question_focus = infer_question_focus(original_query, build_structure_profile(original_query))
        retrieval_depth = build_retrieval_depth(normalized_query, intent_type, profile)
        defense_profile = infer_defense_profile(intent_type, profile)
        evidence_requirement = infer_evidence_requirement(intent_type, profile)
        wants_short_answer = evidence_requirement is EvidenceRequirement.ATOMIC_SPAN
        needs_judge = should_enable_judge(
            intent_type,
            evidence_requirement=evidence_requirement,
            defense_profile=defense_profile,
        )

        trace_tags: list[str] = list(normalization_tags) + list(intent_tags)
        if profile.has_identifier:
            trace_tags.append('has_identifier')
        if profile.is_fragment_like:
            trace_tags.append('short_query')
        if profile.clause_count >= 2:
            trace_tags.append('multi_clause_query')
        if profile.connector_count >= 1:
            trace_tags.append('pair_connector_query')
        if profile.quoted_anchor_count:
            trace_tags.append('quoted_anchor_query')
        if profile.has_structured_blocks:
            trace_tags.append('structured_block_query')
        if keyword_query != normalized_query:
            trace_tags.append('keyword_query_compacted')
        trace_tags.append(f'question_focus_{question_focus.category.value}')
        if question_focus.expects_multiple_points:
            trace_tags.append('question_focus_multi_point')
        if question_focus.has_explicit_cue:
            trace_tags.append('question_focus_explicit')

        ordered_trace_tags: list[str] = []
        seen_tags: set[str] = set()
        for tag in trace_tags:
            if not tag or tag in seen_tags:
                continue
            seen_tags.add(tag)
            ordered_trace_tags.append(tag)

        return QueryIntent(
            original_query=original_query,
            normalized_query=normalized_query,
            keyword_query=keyword_query,
            intent_type=intent_type,
            retrieval_depth=retrieval_depth,
            defense_profile=defense_profile,
            evidence_requirement=evidence_requirement,
            wants_short_answer=wants_short_answer,
            needs_judge=needs_judge,
            trace_tags=tuple(ordered_trace_tags),
            question_focus=question_focus,
        )
