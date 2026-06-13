"""论文主线的串联版流程。

这个文件只保留论文最需要讲清的主链，并且顺序与第二章保持一致：

`Dense + BM25 -> RRF -> Rerank -> 分项重排 -> 覆盖取证 -> 按题作答`

如果老师现场要求“把整条 RAG 主线按代码顺着点一遍”，
优先从这个文件往下讲。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from m00_公共基础 import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    NO_CONTEXT_ANSWER,
    bm25_rank,
    contains_refusal,
    ensure_bm25_index,
    ensure_contextualized_texts,
    ensure_doc_embeddings,
    ensure_query_embeddings,
    ensure_query_rewrites,
    load_reranker,
    mmr_select_units,
    normalize_text,
    tokenize_text,
)
from m01_共享检索主干 import compute_dense_rankings, reciprocal_rank_fusion, rerank_documents
from m02_分项重排 import (
    PaperReorderBreakdown,
    extract_query_aspect_clauses,
    order_evidence_units_for_query,
    order_evidence_units_for_query_clauses,
    rerank_documents_by_paper_formula,
)
from m03_覆盖取证 import order_evidence_units_for_query_clauses_residual_v2
from m04_按题作答 import build_answer_prompt, generate_answer, infer_query_task_mode

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "04_算法实现"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from retrieval_pipeline.datasets import CorpusDoc, ExperimentCase  # noqa: E402


@dataclass(frozen=True)
class PipelineVariant:
    """实验链路配置。

    这里只保留论文主线答辩时最常提到的字段，不把所有证伪分支参数都堆进来。
    """

    key: str
    label: str
    retrieval_strategy: str = "raw"
    use_rerank: bool = True
    use_query_rewrite: bool = False
    rerank_mode: str = "default"
    answer_prompt_style: str = "aligned"
    evidence_selection: str = "topk"
    multi_snippet_count: int = 1
    retrieval_top_k: int = 50
    candidate_top_k: int = 20
    final_source_count: int = 3
    complex_source_count: int = 5
    selection_mode: str = "default"


@dataclass
class QueryFeatures:
    case_id: str
    query_tokens: list[str]
    raw_dense_indices: list[int]
    raw_bm25_indices: list[int]
    rewrite_dense_indices: list[int]
    rewrite_bm25_indices: list[int]
    contextual_dense_indices: list[int]
    contextual_bm25_indices: list[int]


@dataclass
class RetrievalCorpus:
    ids: list[str]
    texts: list[str]
    doc_embeddings: np.ndarray
    bm25: Any


@dataclass
class PreparedDataset:
    name: str
    cases: list[ExperimentCase]
    docs: list[CorpusDoc]
    case_map: dict[str, ExperimentCase]
    raw_corpus: RetrievalCorpus
    contextual_corpus: RetrievalCorpus
    query_features: dict[str, QueryFeatures]


@dataclass
class PipelineRunResult:
    case_id: str
    query: str
    response: str
    source_doc_ids: list[int]
    source_titles: list[str]
    retrieved_contexts: list[str]
    latency_ms: float
    predicted_refusal: bool
    reorder_breakdown_rows: list[dict[str, Any]]


class RagExperimentPipeline:
    """论文主线的阅读版执行器。"""

    def __init__(
        self,
        *,
        cache_root: Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
    ) -> None:
        self.cache_root = cache_root
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.reranker = load_reranker()

    def prepare_dataset(
        self,
        name: str,
        cases: Sequence[ExperimentCase],
        docs: Sequence[CorpusDoc],
        *,
        include_contextual: bool = True,
        include_query_rewrite: bool = True,
    ) -> PreparedDataset:
        """准备实验数据。

        这一步的作用不是跑算法，而是把“同一批题、同一批文档”整理成后面能反复复用的标准结构。
        """
        doc_ids = [str(doc.doc_id) for doc in docs]
        doc_titles = [doc.title for doc in docs]
        doc_texts = [doc.content for doc in docs]
        raw_doc_embeddings = ensure_doc_embeddings(self.cache_root, name, self.embedding_model, doc_ids, doc_texts)
        raw_bm25, _ = ensure_bm25_index(self.cache_root, name, doc_ids, doc_texts)
        raw_corpus = RetrievalCorpus(ids=doc_ids, texts=doc_texts, doc_embeddings=raw_doc_embeddings, bm25=raw_bm25)

        if include_contextual:
            contextual_texts = ensure_contextualized_texts(
                self.cache_root,
                name,
                doc_ids,
                doc_titles,
                doc_texts,
                model_name=self.llm_model,
            )
            contextual_doc_embeddings = ensure_doc_embeddings(
                self.cache_root,
                f"{name}_contextual",
                self.embedding_model,
                doc_ids,
                contextual_texts,
            )
            contextual_bm25, _ = ensure_bm25_index(self.cache_root, f"{name}_contextual", doc_ids, contextual_texts)
            contextual_corpus = RetrievalCorpus(
                ids=doc_ids,
                texts=contextual_texts,
                doc_embeddings=contextual_doc_embeddings,
                bm25=contextual_bm25,
            )
        else:
            contextual_corpus = raw_corpus

        case_list = list(cases)
        query_ids = [case.case_id for case in case_list]
        queries = [case.query for case in case_list]
        raw_query_embeddings = ensure_query_embeddings(
            self.cache_root,
            name,
            self.embedding_model,
            query_ids,
            queries,
            tag="raw",
        )
        raw_dense_rankings, _ = compute_dense_rankings(raw_corpus.doc_embeddings, raw_query_embeddings, top_k=50)

        if include_contextual:
            contextual_dense_rankings, _ = compute_dense_rankings(contextual_corpus.doc_embeddings, raw_query_embeddings, top_k=50)
        else:
            contextual_dense_rankings = raw_dense_rankings

        if include_query_rewrite:
            query_rewrites = ensure_query_rewrites(self.cache_root, name, query_ids, queries, model_name=self.llm_model)
            rewrite_dense_queries = [item["dense"] for item in query_rewrites]
            rewrite_query_embeddings = ensure_query_embeddings(
                self.cache_root,
                name,
                self.embedding_model,
                query_ids,
                rewrite_dense_queries,
                tag="rewrite_dense",
            )
            rewrite_dense_rankings, _ = compute_dense_rankings(raw_corpus.doc_embeddings, rewrite_query_embeddings, top_k=50)
        else:
            query_rewrites = [{"sparse": query, "dense": query} for query in queries]
            rewrite_dense_rankings = raw_dense_rankings

        query_features: dict[str, QueryFeatures] = {}
        for index, case in enumerate(case_list):
            query_tokens = tokenize_text(case.query)
            raw_bm25_indices, _ = bm25_rank(query_tokens, raw_corpus.bm25, top_k=50)
            if include_query_rewrite:
                rewrite_sparse_tokens = tokenize_text(query_rewrites[index]["sparse"])
                rewrite_bm25_indices, _ = bm25_rank(rewrite_sparse_tokens, raw_corpus.bm25, top_k=50)
            else:
                rewrite_bm25_indices = raw_bm25_indices
            if include_contextual:
                contextual_bm25_indices, _ = bm25_rank(query_tokens, contextual_corpus.bm25, top_k=50)
            else:
                contextual_bm25_indices = raw_bm25_indices
            query_features[case.case_id] = QueryFeatures(
                case_id=case.case_id,
                query_tokens=query_tokens,
                raw_dense_indices=raw_dense_rankings[index].tolist(),
                raw_bm25_indices=raw_bm25_indices,
                rewrite_dense_indices=rewrite_dense_rankings[index].tolist(),
                rewrite_bm25_indices=rewrite_bm25_indices,
                contextual_dense_indices=contextual_dense_rankings[index].tolist(),
                contextual_bm25_indices=contextual_bm25_indices,
            )
        return PreparedDataset(
            name=name,
            cases=case_list,
            docs=list(docs),
            case_map={case.case_id: case for case in case_list},
            raw_corpus=raw_corpus,
            contextual_corpus=contextual_corpus,
            query_features=query_features,
        )

    def _build_candidate_indices(self, features: QueryFeatures, variant: PipelineVariant) -> list[int]:
        """根据 dense、BM25、查询改写或上下文化视图生成候选文档列表。"""
        if variant.retrieval_strategy == "contextual":
            fused = reciprocal_rank_fusion(
                [features.contextual_dense_indices, features.contextual_bm25_indices],
                top_k=variant.retrieval_top_k,
            )
            ranked = [doc_index for doc_index, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]
            return ranked[: variant.candidate_top_k]
        dense_indices = features.rewrite_dense_indices if variant.use_query_rewrite else features.raw_dense_indices
        bm25_indices = features.rewrite_bm25_indices if variant.use_query_rewrite else features.raw_bm25_indices
        fused = reciprocal_rank_fusion([dense_indices, bm25_indices], top_k=variant.retrieval_top_k)
        ranked = [doc_index for doc_index, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]
        return ranked[: variant.candidate_top_k]

    def _expand_snippet_context(self, content: str, snippet: str, window_sentences: int = 0) -> str:
        """根据需要为片段补一点相邻句子上下文。主线实验默认不开窗。"""
        if window_sentences <= 0:
            return snippet
        return snippet

    def _build_evidence_units(
        self,
        *,
        reranked_doc_indices: Sequence[int],
        doc_snippets: dict[int, list[str]],
        doc_scores: dict[int, float],
        doc_reorder_scores: dict[int, float] | None,
        docs: Sequence[CorpusDoc],
        variant: PipelineVariant,
        source_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """把重排结果整理成统一证据单元。

        这一层很关键，因为后面的分项重排、覆盖取证、按题作答都只认这个统一结构。
        """
        candidate_units: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        effective_source_limit = max(1, source_limit if source_limit is not None else variant.final_source_count)
        doc_rank_map = {int(doc_index): rank for rank, doc_index in enumerate(reranked_doc_indices)}

        def unit_sort_key(item: dict[str, Any]) -> tuple[float, int, float]:
            """按论文分项重排后的文档优先级整理证据单元顺序。"""
            return (
                float(item.get("doc_reorder_score", item["score"])),
                -int(item.get("doc_reorder_rank", 10**9)),
                float(item["score"]),
            )

        for doc_index in reranked_doc_indices[:effective_source_limit]:
            doc = docs[doc_index]
            snippets = (doc_snippets.get(doc_index) or [doc.content[:360]])[: max(1, variant.multi_snippet_count)]
            for snippet in snippets:
                packed_text = self._expand_snippet_context(doc.content, snippet)
                normalized_text = normalize_text(packed_text)
                if not normalized_text or normalized_text in seen_texts:
                    continue
                seen_texts.add(normalized_text)
                candidate_units.append(
                    {
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "text": packed_text,
                        "tokens": tokenize_text(packed_text),
                        "score": float(doc_scores.get(doc_index, 0.0)),
                        # `score` 继续保留为上游相关性分，供覆盖取证等后续阶段使用；
                        # `doc_reorder_score` 则专门表示论文公式 2-1 产出的文档级分项重排结果。
                        "doc_reorder_score": float((doc_reorder_scores or {}).get(doc_index, doc_scores.get(doc_index, 0.0))),
                        "doc_reorder_rank": int(doc_rank_map.get(int(doc_index), 10**9)),
                    }
                )

        if variant.evidence_selection == "mmr":
            selected_units = mmr_select_units(candidate_units, limit=variant.final_source_count)
        else:
            candidate_units.sort(key=unit_sort_key, reverse=True)
            if variant.selection_mode == "aspect_cover_v2":
                candidate_pool_limit = min(
                    len(candidate_units),
                    max(effective_source_limit, effective_source_limit * max(1, variant.multi_snippet_count)),
                )
                selected_units = candidate_units[:candidate_pool_limit]
            else:
                selected_units = candidate_units[:effective_source_limit]
        selected_units.sort(key=unit_sort_key, reverse=True)
        return selected_units

    def _apply_paper_reorder_stage(
        self,
        *,
        query: str,
        reranked_doc_indices: Sequence[int],
        doc_snippets: dict[int, list[str]],
        doc_scores: dict[int, float],
        docs: Sequence[CorpusDoc],
    ) -> tuple[list[int], dict[int, float], list[dict[str, Any]]]:
        """执行论文公式 2-1 对应的分项重排阶段。

        这一步只做一件事：承接共享检索主干的 `Rerank` 结果，
        用论文中的 `R_base(d) + A(d) + C(d) + D(d)` 保守修正文档顺序。
        这样后面的覆盖取证就不再面对一串只偏向单一分项的候选文档。
        """

        reordered_doc_indices, breakdowns = rerank_documents_by_paper_formula(
            query=query,
            reranked_doc_indices=reranked_doc_indices,
            doc_snippets=doc_snippets,
            doc_scores=doc_scores,
            docs=docs,
        )
        reorder_score_map = {item.doc_index: float(item.reorder_score) for item in breakdowns}
        breakdown_rows: list[dict[str, Any]] = []
        for item in breakdowns:
            typed_item: PaperReorderBreakdown = item
            breakdown_rows.append(
                {
                    "doc_id": typed_item.doc_id,
                    "title": typed_item.title,
                    "R_base_raw": round(typed_item.rerank_score_raw, 4),
                    "R_base_norm": round(typed_item.rerank_score_norm, 4),
                    "A": round(typed_item.aspect_coverage, 4),
                    "C": round(typed_item.overall_coverage, 4),
                    "D": round(typed_item.detail_signal, 4),
                    "S_reorder": round(typed_item.reorder_score, 4),
                    "matched_aspects": list(typed_item.matched_aspects),
                }
            )
        return reordered_doc_indices, reorder_score_map, breakdown_rows

    def _apply_query_aligned_evidence_selection(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> list[dict[str, Any]]:
        """按论文主线选择证据：普通对齐、分项重排或覆盖取证。"""
        if variant.selection_mode == "aspect_cover_v2":
            clauses = extract_query_aspect_clauses(query)
            task_mode = infer_query_task_mode(query)
            unique_doc_ids = {int(unit.get("doc_id", 0) or 0) for unit in evidence_units}
            if (
                task_mode != "summary"
                and len(clauses) >= 2
                and len(evidence_units) > variant.final_source_count
                and len(unique_doc_ids) >= 2
            ):
                ordered = order_evidence_units_for_query_clauses_residual_v2(
                    query,
                    evidence_units,
                    limit=variant.final_source_count,
                )
                if ordered:
                    ordered_signatures = {
                        normalize_text(str(unit.get("text") or "")) for unit in ordered if normalize_text(str(unit.get("text") or ""))
                    }
                    tail = [
                        unit
                        for unit in evidence_units
                        if normalize_text(str(unit.get("text") or "")) not in ordered_signatures
                    ]
                    return ordered + tail
            ordered = order_evidence_units_for_query_clauses(query, evidence_units)
            return ordered[: len(evidence_units)] if ordered else list(evidence_units)
        if variant.selection_mode == "query_aligned":
            ordered = order_evidence_units_for_query(query_tokens, evidence_units)
            return ordered[: len(evidence_units)] if ordered else list(evidence_units)
        return list(evidence_units)

    def _resolve_prompt_style(self, variant: PipelineVariant, query: str, query_tokens: Sequence[str]) -> str:
        """解析当前问题最后要用哪种回答风格。"""
        if variant.answer_prompt_style in {"simple", "aligned", "task_aligned"}:
            return variant.answer_prompt_style
        if variant.answer_prompt_style == "task_router":
            task_mode = infer_query_task_mode(query)
            if task_mode == "summary":
                return "simple"
            if len(extract_query_aspect_clauses(query)) >= 2:
                return "task_aligned"
            return "aligned"
        return "simple"

    def _run_direct_answer(self, query: str, evidence_units: Sequence[dict[str, Any]], *, style: str) -> str:
        """在主线里执行单轮作答。"""
        if not evidence_units:
            return NO_CONTEXT_ANSWER
        prompt = build_answer_prompt(query, evidence_units, style=style)
        return generate_answer(prompt, model_name=self.llm_model)

    def run_case(self, prepared: PreparedDataset, case: ExperimentCase, variant: PipelineVariant) -> PipelineRunResult:
        """运行单条样例的论文主线流程。"""
        started = time.perf_counter()
        features = prepared.query_features[case.case_id]
        if variant.retrieval_strategy == "contextual":
            retrieval_corpus = prepared.contextual_corpus
        else:
            retrieval_corpus = prepared.raw_corpus
        candidate_indices = self._build_candidate_indices(features, variant)

        aspect_token_sets: list[set[str]] = []
        aggregation_mode = "default"
        if variant.rerank_mode in {"aspect_aware", "aspect_aware_conservative"}:
            aspect_clauses = extract_query_aspect_clauses(case.query, max_clauses=4)
            aspect_token_sets = [set(tokenize_text(clause)) for clause in aspect_clauses if tokenize_text(clause)]
            if variant.rerank_mode == "aspect_aware_conservative" and len(aspect_token_sets) >= 2:
                aggregation_mode = "aspect_aware_conservative"
            elif len(aspect_token_sets) >= 2:
                aggregation_mode = "aspect_aware"

        reranked_doc_indices, doc_snippets, doc_scores = rerank_documents(
            self.reranker,
            query=case.query,
            query_tokens=features.query_tokens,
            focus_tokens=features.query_tokens,
            candidate_doc_indices=candidate_indices,
            corpus_texts=retrieval_corpus.texts,
            multi_snippet_count=variant.multi_snippet_count,
            aggregation_mode=aggregation_mode,
            aspect_token_sets=aspect_token_sets,
        )
        reordered_doc_indices, reorder_score_map, reorder_breakdown_rows = self._apply_paper_reorder_stage(
            query=case.query,
            reranked_doc_indices=reranked_doc_indices,
            doc_snippets=doc_snippets,
            doc_scores=doc_scores,
            docs=prepared.docs,
        )

        routing_units = self._build_evidence_units(
            reranked_doc_indices=reordered_doc_indices,
            doc_snippets=doc_snippets,
            doc_scores=doc_scores,
            doc_reorder_scores=reorder_score_map,
            docs=prepared.docs,
            variant=variant,
            source_limit=max(variant.final_source_count, variant.complex_source_count),
        )
        routing_units = self._apply_query_aligned_evidence_selection(
            query=case.query,
            query_tokens=features.query_tokens,
            evidence_units=routing_units,
            variant=variant,
        )

        units = routing_units[: variant.final_source_count]
        answer = self._run_direct_answer(
            case.query,
            units,
            style=self._resolve_prompt_style(variant, case.query, features.query_tokens),
        )

        source_doc_ids: list[int] = []
        source_titles: list[str] = []
        for unit in units:
            if int(unit["doc_id"]) in source_doc_ids:
                continue
            source_doc_ids.append(int(unit["doc_id"]))
            source_titles.append(str(unit["title"]))
        retrieved_contexts = [str(unit["text"]) for unit in units]
        latency_ms = (time.perf_counter() - started) * 1000.0
        return PipelineRunResult(
            case_id=case.case_id,
            query=case.query,
            response=answer or NO_CONTEXT_ANSWER,
            source_doc_ids=source_doc_ids,
            source_titles=source_titles,
            retrieved_contexts=retrieved_contexts,
            latency_ms=latency_ms,
            predicted_refusal=contains_refusal(answer),
            reorder_breakdown_rows=reorder_breakdown_rows,
        )
