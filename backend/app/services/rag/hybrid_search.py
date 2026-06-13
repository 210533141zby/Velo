"""
=============================================================================
文件: hybrid_retrieval.py
描述: 将实验中的“自适应混合检索”落到后端 RAG 主链路

核心目标：
1. 复用主实验里的查询画像、自适应融合、RRF 和混合候选池思想。
2. 在 Chroma 向量召回之外，补一条文档级 BM25 词法检索链路。
3. 给 rerank 阶段提供更干净、更稳定的候选池，同时保留可解释分数。
=============================================================================
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from threading import Lock
from types import SimpleNamespace
from typing import Any, Sequence

from app.core.config import settings
from app.logger import logger

try:
    import jieba
except Exception:  # pragma: no cover - 依赖缺失时自动走正则分词兜底。
    jieba = None

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - 依赖缺失时自动走词法重叠兜底。
    BM25Okapi = None

STRUCTURED_IDENTIFIER_PATTERN = re.compile(
    r'[a-z]+[a-z0-9]*(?:[._-][a-z0-9]+)*\d+[a-z0-9._-]*|[a-z0-9]+(?:[-_][a-z0-9]+)+',
    re.IGNORECASE,
)
TITLE_ANCHOR_PATTERN = re.compile(r'《([^》]{1,40})》')
ASCII_ANCHOR_PATTERN = re.compile(r'(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9_-]{1,31}(?![A-Za-z0-9_-])')
KEYWORD_TRI_ROUTE_MAX_TOKENS = 5
KEYWORD_BRANCH_MAX_TOKENS = 8
KEYWORD_FILL_BASE_KEEP = 22

_index_lock = Lock()
_hybrid_index: 'HybridLexicalIndex | None' = None
_hybrid_index_signature: str | None = None
_hybrid_index_needs_refresh = True
_jieba_warning_emitted = False
_bm25_warning_emitted = False


@dataclass(frozen=True)
class IndexedDocument:
    doc_id: int
    title: str
    content_preview: str
    bm25_text: str
    tokens: list[str]
    identifier_tokens: set[str]


def clean_text(text: str, limit: int | None = None) -> str:
    """压缩空白、兜底空字符串，并在需要时执行长度裁剪。

    该函数贯穿索引构建和查询处理，目的是把不同来源的文本先压成可比较、
    可分词的稳定形态，避免后续词法统计受格式噪声影响。
    """
    normalized = re.sub(r'\s+', ' ', str(text or '').strip())
    if not normalized:
        normalized = '空白内容'
    if limit is not None:
        return normalized[:limit] or '空白内容'
    return normalized


def _normalize_identifier_token(token: str) -> str:
    """把编号、代号类词元归一化成适合精确匹配的形式。"""
    return str(token or '').strip().lower().replace('_', '-').strip('-')


def is_structured_identifier_token(token: str) -> bool:
    """判断一个词元是否像档案号、代号或带分隔符的结构化标识。"""
    normalized = _normalize_identifier_token(token)
    if not normalized:
        return False
    return bool(STRUCTURED_IDENTIFIER_PATTERN.fullmatch(normalized))


def _is_ascii_anchor_token(token: str) -> bool:
    """识别带大小写、数字或连接符的 ASCII 锚点词元。

    这类词在档案号、物料号、英文专名里很常见，通常比普通中文词更适合
    作为精确检索信号单独保留。
    """
    return any(character.isupper() for character in token) or any(character.isdigit() for character in token) or '-' in token or '_' in token


def extract_identifiers(text: str) -> set[str]:
    """从问题或文档中提取最适合做精确匹配的编号与锚点。

    除了通用的结构化编号，还会额外吸收书名号标题和 ASCII 专名，
    供后续关键词优先分支判断是否需要强化词法召回。
    """
    identifiers: set[str] = set()
    raw_text = str(text or '')
    lowered_text = raw_text.lower().replace('_', '-')
    for match in STRUCTURED_IDENTIFIER_PATTERN.finditer(lowered_text):
        normalized = _normalize_identifier_token(match.group(0))
        if normalized:
            identifiers.add(normalized)
    for title in TITLE_ANCHOR_PATTERN.findall(raw_text):
        normalized_title = clean_text(title).lower().replace(' ', '')
        if len(normalized_title) >= 2:
            identifiers.add(normalized_title)
    for token in ASCII_ANCHOR_PATTERN.findall(raw_text):
        normalized_token = _normalize_identifier_token(token)
        if normalized_token and _is_ascii_anchor_token(token):
            identifiers.add(normalized_token)
    return identifiers


def has_identifier(text: str) -> bool:
    """判断当前文本里是否出现了值得走精确检索分支的标识符。"""
    return bool(extract_identifiers(text))


def tokenize_for_bm25(text: str) -> list[str]:
    """把文本切成适合 BM25 使用的词元序列。

    这里不仅做普通分词，还会把标题锚点、ASCII 代号等高价值词元插回队首，
    让这些精确信号在词法检索里得到更稳定的权重。
    """
    global _jieba_warning_emitted

    raw_text = str(text or '')
    cleaned = clean_text(raw_text).lower()
    if jieba is not None:
        raw_tokens = [token.strip() for token in jieba.lcut(cleaned) if token.strip()]
    else:
        if not _jieba_warning_emitted:
            logger.warning('jieba 未安装，Hybrid 检索将回退到正则分词')
            _jieba_warning_emitted = True
        raw_tokens = re.findall(r'[0-9a-z]+|[\u4e00-\u9fff]+', cleaned)

    normalized: list[str] = []
    for token in raw_tokens:
        if re.fullmatch(r'[\u4e00-\u9fff]+', token):
            normalized.append(token)
        else:
            parts = re.findall(r'[0-9a-z]+|[\u4e00-\u9fff]+', token)
            normalized.extend(parts if parts else [token])
    anchor_tokens: list[str] = []
    for title in TITLE_ANCHOR_PATTERN.findall(raw_text):
        normalized_title = clean_text(title).lower().replace(' ', '')
        if len(normalized_title) >= 2:
            anchor_tokens.append(normalized_title)
    for token in ASCII_ANCHOR_PATTERN.findall(raw_text):
        normalized_token = _normalize_identifier_token(token)
        if normalized_token and _is_ascii_anchor_token(token):
            anchor_tokens.append(normalized_token)
    seen: set[str] = set()
    merged_tokens: list[str] = []
    for token in [*anchor_tokens, *normalized]:
        if token and token not in seen:
            merged_tokens.append(token)
            seen.add(token)
    return merged_tokens


def coverage_ratio(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    """计算候选文本覆盖了多少查询词，用作轻量词法相关性信号。"""
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    """把一批原始分数压到统一的 0 到 1 范围，便于后续融合。"""
    if not scores:
        return {}

    values = list(scores.values())
    low = min(values)
    high = max(values)
    if abs(high - low) <= 1e-12:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """对多路排序结果执行 RRF 融合，得到统一候选优先级。"""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def ranked_doc_ids(scores: dict[int, float]) -> list[int]:
    """把分数字典整理成按得分降序排列的文档编号列表。"""
    return [doc_id for doc_id, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def should_use_precise_query_route(query: str) -> bool:
    """根据问题长度和标识符特征判断是否启用精确查询路由。

    对短问题、带档案号的问题，纯向量召回容易把相近语义噪声一起带进来，
    这时更适合提升关键词信号的优先级。
    """
    tokens = tokenize_for_bm25(query)
    if not tokens:
        return False

    if has_identifier(query):
        return True

    compact_tokens = [token for token in tokens if token.strip()]
    if len(compact_tokens) <= 4:
        return True

    meaningful_tokens = [token for token in compact_tokens if len(token) >= 2]
    if not meaningful_tokens:
        return False
    average_token_length = sum(len(token) for token in meaningful_tokens) / len(meaningful_tokens)
    return len(compact_tokens) <= 6 and average_token_length >= 2.5


def compute_query_profile(query: str, idf_lookup: dict[str, float] | None = None) -> dict[str, float]:
    """根据问题特征估算词法检索与向量检索的融合权重。

    这个画像只使用轻量启发式信号，不依赖额外模型，适合在后端实时路由时
    快速决定当前问题更偏向“编号精确查找”还是“语义近邻召回”。
    """
    tokens = tokenize_for_bm25(query)
    token_set = set(tokens)
    idf_lookup = idf_lookup or {}
    idf_values = [float(idf_lookup.get(token, 0.0)) for token in token_set]
    avg_idf = (sum(idf_values) / len(idf_values)) if idf_values else 0.0
    max_idf = max(idf_values) if idf_values else 0.0

    lexical_weight = 0.36
    if has_identifier(query):
        lexical_weight += 0.18
    if len(tokens) <= 4:
        lexical_weight += 0.14
    if len(tokens) >= 10:
        lexical_weight -= 0.08
    if max_idf > max(avg_idf * 1.35, 2.5):
        lexical_weight += 0.10

    lexical_weight = max(0.24, min(0.78, lexical_weight))
    return {
        'lexical_weight': lexical_weight,
        'dense_weight': 1.0 - lexical_weight,
        'token_count': float(len(tokens)),
        'has_identifier': 1.0 if has_identifier(query) else 0.0,
    }


def build_index_signature(documents: Sequence[Any]) -> str:
    """根据文档数量、更新时间和长度生成索引版本签名。

    只要签名变化，就说明词法索引所依赖的文档集合已经不同，需要整体刷新。
    """
    digest = hashlib.sha1()
    for document in documents:
        updated_at = getattr(document, 'updated_at', None)
        updated_value = updated_at.isoformat() if updated_at is not None else ''
        digest.update(
            f'{getattr(document, "id", "")}|{updated_value}|{len(str(getattr(document, "title", "") or ""))}|'
            f'{len(str(getattr(document, "content", "") or ""))}\n'.encode('utf-8')
        )
    return f'n{len(documents)}_{digest.hexdigest()[:16]}'


def _build_indexed_document(document: Any) -> IndexedDocument | None:
    """把数据库文档整理成词法检索阶段使用的轻量索引对象。"""
    doc_id = getattr(document, 'id', None)
    title = clean_text(getattr(document, 'title', ''), limit=240)
    content = clean_text(getattr(document, 'content', ''), limit=settings.RERANK_MAX_INPUT_CHARS)
    if doc_id is None or not content:
        return None

    # 标题重复一次，提升精确标题和文档名查询在 BM25 中的权重。
    bm25_text = clean_text(f'{title}\n{title}\n{content}', limit=settings.RERANK_MAX_INPUT_CHARS * 2)
    tokens = tokenize_for_bm25(bm25_text)
    return IndexedDocument(
        doc_id=int(doc_id),
        title=title,
        content_preview=content,
        bm25_text=bm25_text,
        tokens=tokens,
        identifier_tokens=extract_identifiers(f'{title} {content[:500]}'),
    )


class HybridLexicalIndex:
    def __init__(self, documents: Sequence[IndexedDocument]) -> None:
        """基于一批文档建立 BM25 或词法覆盖率索引。

        如果 `rank_bm25` 可用，就构建正式 BM25；否则保留分词结果，
        退化为词元覆盖率排序，保证检索链路仍可运行。
        """
        global _bm25_warning_emitted

        self.documents = list(documents)
        self.by_id = {document.doc_id: document for document in self.documents}
        self._tokenized_corpus = [document.tokens or ['空白内容'] for document in self.documents]

        if self.documents and BM25Okapi is not None:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            self.idf_lookup = dict(getattr(self._bm25, 'idf', {}))
        else:
            self._bm25 = None
            self.idf_lookup = {}
            if self.documents and BM25Okapi is None and not _bm25_warning_emitted:
                logger.warning('rank_bm25 未安装，Hybrid 检索将回退到词法覆盖率排序')
                _bm25_warning_emitted = True

    def rank_bm25(self, query: str, top_k: int) -> tuple[list[int], dict[int, float], set[str]]:
        """对查询执行词法排序，并返回排序结果、原始分数和查询词集合。"""
        if not self.documents or top_k <= 0:
            return [], {}, set()

        query_tokens = tokenize_for_bm25(query)
        query_token_set = set(query_tokens)
        if not query_tokens:
            return [], {}, query_token_set

        raw_scores: dict[int, float] = {}
        if self._bm25 is not None:
            scores = self._bm25.get_scores(query_tokens)
            for index, score in enumerate(scores):
                raw_scores[self.documents[index].doc_id] = float(score)
        else:
            for document in self.documents:
                raw_scores[document.doc_id] = coverage_ratio(query_token_set, set(document.tokens))

        ranked_doc_ids = [
            doc_id
            for doc_id, _score in sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
            if _score > 0.0
        ][:top_k]
        return ranked_doc_ids, {doc_id: raw_scores[doc_id] for doc_id in ranked_doc_ids}, query_token_set


def hybrid_index_needs_refresh() -> bool:
    """判断全局词法索引缓存是否已经失效。"""
    with _index_lock:
        return _hybrid_index is None or _hybrid_index_needs_refresh


def get_hybrid_index() -> HybridLexicalIndex:
    """读取当前缓存中的混合词法索引；若不存在则返回空索引对象。"""
    with _index_lock:
        return _hybrid_index or HybridLexicalIndex([])


def ensure_hybrid_index(documents: Sequence[Any]) -> HybridLexicalIndex:
    """确保全局混合词法索引与当前文档集保持一致。

    这是后端词法检索的统一入口，会比较签名后决定复用缓存还是整体重建。
    """
    global _hybrid_index
    global _hybrid_index_signature
    global _hybrid_index_needs_refresh

    signature = build_index_signature(documents)
    with _index_lock:
        if _hybrid_index is not None and _hybrid_index_signature == signature and not _hybrid_index_needs_refresh:
            return _hybrid_index

        indexed_documents = []
        for document in documents:
            indexed = _build_indexed_document(document)
            if indexed is not None:
                indexed_documents.append(indexed)

        _hybrid_index = HybridLexicalIndex(indexed_documents)
        _hybrid_index_signature = signature
        _hybrid_index_needs_refresh = False
        logger.info(
            'Hybrid 词法索引已刷新',
            extra={
                'extra_data': {
                    'event': 'rag_hybrid_index_refreshed',
                    'document_count': len(indexed_documents),
                    'signature': signature,
                }
            },
        )
        return _hybrid_index


def invalidate_hybrid_index() -> None:
    """把混合词法索引标记为失效，等待下次检索时按需重建。"""
    global _hybrid_index_needs_refresh

    with _index_lock:
        _hybrid_index_needs_refresh = True


def _clone_doc(doc: Any) -> Any:
    """复制一份向量召回结果，避免在融合过程中污染原始对象。"""
    return SimpleNamespace(
        metadata=dict(getattr(doc, 'metadata', {}) or {}),
        page_content=str(getattr(doc, 'page_content', '') or ''),
    )


def _build_synthetic_doc(indexed_document: IndexedDocument) -> Any:
    """为纯词法命中的文档构造一个兼容向量结果格式的伪文档对象。"""
    return SimpleNamespace(
        metadata={
            'source': indexed_document.title,
            'doc_id': indexed_document.doc_id,
        },
        page_content=indexed_document.content_preview,
    )


def _collapse_vector_matches(scored_matches: Sequence[tuple[Any, float]]) -> list[tuple[int, Any, float]]:
    """对向量召回结果按文档去重，只保留每篇文档得分最高的片段。"""
    best_by_doc: dict[int, tuple[Any, float]] = {}
    for doc, score in scored_matches:
        doc_id = doc.metadata.get('doc_id')
        if doc_id is None:
            continue
        numeric_doc_id = int(doc_id)
        best = best_by_doc.get(numeric_doc_id)
        if best is None or float(score) > float(best[1]):
            best_by_doc[numeric_doc_id] = (doc, float(score))

    ranked = sorted(best_by_doc.items(), key=lambda item: item[1][1], reverse=True)
    return [(doc_id, doc, score) for doc_id, (doc, score) in ranked]


def _fill_candidate_ids(target: list[int], source: Sequence[int], limit: int) -> None:
    """按顺序把来源列表补入候选池，直到达到上限。"""
    for doc_id in source:
        if doc_id not in target:
            target.append(doc_id)
        if len(target) >= limit:
            return


def _should_use_keyword_branch(query: str, keyword_query: str, keyword_tokens: list[str]) -> bool:
    """判断关键词改写分支是否值得参与当前候选构造。"""
    if keyword_query.strip() == query.strip() or len(keyword_tokens) < 2:
        return False
    return has_identifier(query) or len(tokenize_for_bm25(query)) <= KEYWORD_BRANCH_MAX_TOKENS


def _build_keyword_fill_candidates(
    base_ranked: Sequence[int],
    keyword_ranked: Sequence[int],
    limit: int,
    base_keep: int = KEYWORD_FILL_BASE_KEEP,
) -> list[int]:
    """在基础 RRF 结果上插入关键词命中文档，形成补齐后的候选池。"""
    candidate_ids: list[int] = []
    for source in (base_ranked[:base_keep], keyword_ranked[:limit], base_ranked):
        for doc_id in source:
            if doc_id not in candidate_ids:
                candidate_ids.append(doc_id)
            if len(candidate_ids) >= limit:
                return candidate_ids
    return candidate_ids


def _build_lexical_first_candidates(
    raw_bm25_ranked: Sequence[int],
    keyword_bm25_ranked: Sequence[int],
    vector_ranked: Sequence[int],
    limit: int,
) -> list[int]:
    """优先保留强词法命中的文档，再用向量结果补足候选规模。"""
    candidate_ids: list[int] = []
    for source in (keyword_bm25_ranked, raw_bm25_ranked, vector_ranked):
        for doc_id in source:
            if doc_id not in candidate_ids:
                candidate_ids.append(doc_id)
            if len(candidate_ids) >= limit:
                return candidate_ids
    return candidate_ids


def _build_query_routed_keyword_candidates(
    query: str,
    keyword_query: str,
    keyword_tokens: list[str],
    vector_doc_ids: Sequence[int],
    raw_bm25_doc_ids: Sequence[int],
    keyword_bm25_doc_ids: Sequence[int],
    candidate_limit: int,
) -> tuple[list[int], dict[int, float], str]:
    """根据查询画像选择最合适的关键词候选构造方式。

    返回值同时包含候选列表、用于记录的先验分数和当前走到的路由名称，
    方便后续排查为什么某个问题会偏向词法或语义召回。
    """
    query_tokens = tokenize_for_bm25(query)
    base_rrf_scores = reciprocal_rank_fusion([list(vector_doc_ids), list(raw_bm25_doc_ids)])
    base_ranked = ranked_doc_ids(base_rrf_scores)
    if not _should_use_keyword_branch(query, keyword_query, keyword_tokens):
        return base_ranked[:candidate_limit], base_rrf_scores, 'baseline_rrf'

    tri_rrf_scores = reciprocal_rank_fusion([list(vector_doc_ids), list(raw_bm25_doc_ids), list(keyword_bm25_doc_ids)])
    tri_ranked = ranked_doc_ids(tri_rrf_scores)
    if has_identifier(query):
        lexical_prior = reciprocal_rank_fusion(
            [
                list(keyword_bm25_doc_ids),
                list(raw_bm25_doc_ids),
                list(keyword_bm25_doc_ids),
                list(raw_bm25_doc_ids),
                list(vector_doc_ids),
            ]
        )
        return (
            _build_lexical_first_candidates(
                raw_bm25_doc_ids,
                keyword_bm25_doc_ids,
                vector_doc_ids,
                candidate_limit,
            ),
            lexical_prior,
            'lexical_first',
        )
    if len(query_tokens) <= KEYWORD_TRI_ROUTE_MAX_TOKENS:
        return tri_ranked[:candidate_limit], tri_rrf_scores, 'tri_rrf'
    if len(query_tokens) <= KEYWORD_BRANCH_MAX_TOKENS:
        return (
            _build_keyword_fill_candidates(base_ranked, keyword_bm25_doc_ids, candidate_limit),
            tri_rrf_scores,
            'keyword_fill',
        )
    return base_ranked[:candidate_limit], base_rrf_scores, 'baseline_rrf'


def build_hybrid_candidates(
    query: str,
    vector_matches: Sequence[tuple[Any, float]],
    lexical_index: HybridLexicalIndex,
    bm25_query: str | None = None,
    vector_limit: int | None = None,
    bm25_limit: int | None = None,
    candidate_limit: int | None = None,
) -> list[tuple[Any, float]]:
    """把向量召回与词法召回融合成最终候选列表。

    这是 RAG 检索主链路进入 rerank 之前的最后一道候选整理步骤，
    会在不同检索模式之间切换，并为每个候选补齐可解释的中间分数字段。
    """
    vector_limit = vector_limit or settings.RAG_VECTOR_SEARCH_LIMIT
    bm25_limit = bm25_limit or settings.RAG_BM25_SEARCH_LIMIT
    candidate_limit = candidate_limit or settings.RAG_HYBRID_CANDIDATE_LIMIT

    collapsed_vector = _collapse_vector_matches(vector_matches)[:vector_limit]
    vector_doc_ids = [doc_id for doc_id, _doc, _score in collapsed_vector]
    vector_score_map = {doc_id: float(score) for doc_id, _doc, score in collapsed_vector}
    vector_doc_map = {doc_id: doc for doc_id, doc, _score in collapsed_vector}

    raw_bm25_doc_ids, raw_bm25_score_map, _ = lexical_index.rank_bm25(query, bm25_limit)
    keyword_query = clean_text(bm25_query or query)
    keyword_tokens = tokenize_for_bm25(keyword_query)
    keyword_bm25_doc_ids, keyword_bm25_score_map, _ = lexical_index.rank_bm25(keyword_query, bm25_limit)
    query_token_set = set(tokenize_for_bm25(query))
    if not vector_doc_ids and not raw_bm25_doc_ids:
        return []

    base_rrf_scores = reciprocal_rank_fusion([vector_doc_ids, raw_bm25_doc_ids])
    tri_rrf_scores = reciprocal_rank_fusion([vector_doc_ids, raw_bm25_doc_ids, keyword_bm25_doc_ids])
    base_ranked = ranked_doc_ids(base_rrf_scores)
    tri_ranked = ranked_doc_ids(tri_rrf_scores)
    retrieval_mode = (settings.RAG_RETRIEVAL_MODE or 'query_routed_keyword').strip().lower()
    keyword_enabled = _should_use_keyword_branch(query, keyword_query, keyword_tokens)
    vector_prior_scores = {doc_id: vector_score_map.get(doc_id, 0.0) for doc_id in vector_doc_ids}
    if retrieval_mode == 'dense':
        candidate_ids = vector_doc_ids[:candidate_limit]
        route_kind = 'dense'
        retrieval_prior = vector_prior_scores
    elif retrieval_mode == 'baseline_rrf':
        candidate_ids = base_ranked[:candidate_limit]
        route_kind = 'baseline_rrf'
        retrieval_prior = base_rrf_scores
    elif retrieval_mode == 'precise_route_dense_keyword':
        if not should_use_precise_query_route(query):
            candidate_ids = vector_doc_ids[:candidate_limit]
            route_kind = 'dense'
            retrieval_prior = vector_prior_scores
        else:
            candidate_ids, retrieval_prior, route_kind = _build_query_routed_keyword_candidates(
                query,
                keyword_query,
                keyword_tokens,
                vector_doc_ids,
                raw_bm25_doc_ids,
                keyword_bm25_doc_ids,
                candidate_limit,
            )
    elif retrieval_mode == 'keyword_fill':
        candidate_ids = (
            _build_keyword_fill_candidates(base_ranked, keyword_bm25_doc_ids, candidate_limit)
            if keyword_enabled
            else base_ranked[:candidate_limit]
        )
        route_kind = 'keyword_fill' if keyword_enabled else 'baseline_rrf'
        retrieval_prior = tri_rrf_scores if keyword_enabled else base_rrf_scores
    else:
        if not keyword_enabled:
            candidate_ids = base_ranked[:candidate_limit]
            route_kind = 'baseline_rrf'
        elif has_identifier(query):
            candidate_ids = _build_lexical_first_candidates(
                raw_bm25_doc_ids,
                keyword_bm25_doc_ids,
                vector_doc_ids,
                candidate_limit,
            )
            route_kind = 'lexical_first'
        elif len(tokenize_for_bm25(query)) <= 5:
            candidate_ids = tri_ranked[:candidate_limit]
            route_kind = 'tri_rrf'
        else:
            candidate_ids = _build_keyword_fill_candidates(base_ranked, keyword_bm25_doc_ids, candidate_limit)
            route_kind = 'keyword_fill'
        retrieval_prior = (
            reciprocal_rank_fusion(
                [
                    list(keyword_bm25_doc_ids),
                    list(raw_bm25_doc_ids),
                    list(keyword_bm25_doc_ids),
                    list(raw_bm25_doc_ids),
                    list(vector_doc_ids),
                ]
            )
            if keyword_enabled and route_kind == 'lexical_first'
            else tri_rrf_scores if keyword_enabled else base_rrf_scores
        )

    fused_doc_ids = list(candidate_ids)
    query_identifiers = extract_identifiers(query)
    prepared_docs: dict[int, Any] = {}

    for doc_id in fused_doc_ids:
        indexed_document = lexical_index.by_id.get(doc_id)
        if indexed_document is None:
            continue

        doc = _clone_doc(vector_doc_map[doc_id]) if doc_id in vector_doc_map else _build_synthetic_doc(indexed_document)
        candidate_source = route_kind
        if doc_id in vector_doc_map and max(raw_bm25_score_map.get(doc_id, 0.0), keyword_bm25_score_map.get(doc_id, 0.0)) > 0.0 and len(query_token_set) >= 2:
            chunk_token_coverage = coverage_ratio(query_token_set, set(tokenize_for_bm25(getattr(doc, 'page_content', '') or '')))
            if chunk_token_coverage < 0.30:
                doc = _build_synthetic_doc(indexed_document)
                candidate_source = 'hybrid_lexical_fallback'
        elif doc_id not in vector_doc_map and max(raw_bm25_score_map.get(doc_id, 0.0), keyword_bm25_score_map.get(doc_id, 0.0)) > 0.0:
            candidate_source = 'hybrid'
        coverage = coverage_ratio(query_token_set, set(indexed_document.tokens))
        identifier_overlap = coverage_ratio(query_identifiers, indexed_document.identifier_tokens)
        adaptive_score = retrieval_prior.get(doc_id, 0.0)

        doc.metadata.update(
            {
                'vector_score': vector_score_map.get(doc_id, 0.0),
                'bm25_score': max(raw_bm25_score_map.get(doc_id, 0.0), keyword_bm25_score_map.get(doc_id, 0.0)),
                'raw_bm25_score': raw_bm25_score_map.get(doc_id, 0.0),
                'keyword_bm25_score': keyword_bm25_score_map.get(doc_id, 0.0),
                'rrf_score': retrieval_prior.get(doc_id, 0.0),
                'adaptive_score': adaptive_score,
                'coverage_score': coverage,
                'identifier_overlap': identifier_overlap,
                'candidate_source': candidate_source,
            }
        )

        prepared_docs[doc_id] = doc

    return [(prepared_docs[doc_id], retrieval_prior.get(doc_id, 0.0)) for doc_id in candidate_ids if doc_id in prepared_docs]
