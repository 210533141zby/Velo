"""论文 RAG 主链路的真实执行实现。

这个模块负责把检索、重排、证据组织和最终作答串成一条可运行链路，
供消融实验、证伪实验和补充对比脚本直接调用。
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adaptive_evidence import (
    CompressedEvidence,
    EvidenceFact,
    NeedleAnnotatedEvidence,
    _sentence_fact_candidates,
    assess_complexity,
    build_slot_answer_prompt,
    build_slot_repair_prompt,
    build_slot_verification_prompt,
    build_distillation_prompt,
    build_grounded_synthesis_prompt,
    build_fact_repair_prompt,
    build_structured_evidence_answer_prompt,
    build_structured_plan_prompt,
    build_clause_support_plan,
    parse_distilled_facts,
    parse_slot_answers,
    parse_structured_plan,
    parse_supported_answer,
    parse_verified_answer,
    prune_answer_to_supported_sentences,
    repair_answer_with_supported_sentences,
    compress_evidence_units,
    extract_query_aligned_spans,
    build_edge_packed_evidence_layout,
    build_sentence_window_replacement_units,
    build_needled_evidence_units,
    group_evidence_units_by_query_aspects,
    order_evidence_units_for_query,
    refine_grounding_facts,
    render_fact_answer,
    build_title_structured_evidence_units,
    build_aspect_labeled_evidence_units,
    build_slot_packed_evidence_units,
    build_slot_packed_answer_prompt,
    build_plan_guided_answer_prompt,
    extract_query_aspect_clauses,
    build_clause_guided_answer_prompt,
    order_evidence_units_for_query_clauses,
    order_evidence_units_for_query_clauses_residual_v2,
)
from .common import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    NO_CONTEXT_ANSWER,
    bm25_rank,
    build_answer_prompt,
    build_aligned_answer_prompt_with_constraints,
    compute_dense_rankings,
    contains_refusal,
    ensure_bm25_index,
    ensure_contextualized_texts,
    ensure_doc_embeddings,
    ensure_query_embeddings,
    ensure_query_rewrites,
    generate_answer,
    generate_json_payload,
    infer_query_task_mode,
    load_reranker,
    mmr_select_units,
    normalize_text,
    rank_query_focused_snippets,
    reciprocal_rank_fusion,
    rerank_documents,
    tokenize_text,
    extract_tag_text,
)
from .datasets import CorpusDoc, ExperimentCase


@dataclass(frozen=True)
class PipelineVariant:
    key: str
    label: str
    retrieval_strategy: str = "raw"
    use_rerank: bool = True
    use_query_rewrite: bool = False
    rerank_mode: str = "default"
    synthesis_mode: str = "direct"
    answer_prompt_style: str = "aligned"
    evidence_selection: str = "topk"
    adaptive_prompt_min_query_tokens: int = 0
    multi_snippet_count: int = 1
    retrieval_top_k: int = 50
    candidate_top_k: int = 20
    final_source_count: int = 3
    complex_source_count: int = 5
    distilled_fact_limit: int = 6
    router_complexity_threshold: float = 0.50
    context_window_sentences: int = 0
    compression_max_units: int = 6
    support_pruning_threshold: float = 0.0
    span_constraint_limit: int = 6
    selection_mode: str = "default"
    rendering_mode: str = "default"


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
    child_dense_indices: list[int]
    child_bm25_indices: list[int]


@dataclass
class RetrievalCorpus:
    ids: list[str]
    texts: list[str]
    parent_doc_indices: list[int]
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
    child_corpus: RetrievalCorpus
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
    route_mode: str = "direct"
    complexity_score: float = 0.0
    distilled_fact_count: int = 0
    support_sentence_count: int = 0


class RagExperimentPipeline:
    def __init__(
        self,
        *,
        cache_root: Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
    ) -> None:
        """初始化实验流水线依赖。

        这个对象对应论文实验里“真正跑算法”的执行器。它不会保存具体数据集内容，
        只持有缓存目录、嵌入模型、回答模型和重排模型等跨实验共用的依赖，
        方便不同消融 / 证伪脚本复用同一条主链路。
        """
        self.cache_root = cache_root
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.reranker = load_reranker()

    def _build_child_corpus(self, name: str, docs: Sequence[CorpusDoc]) -> RetrievalCorpus:
        """构建父子块检索使用的子块语料。

        该视图对应检索侧的一条对照路线：先把文档切成更细的候选子块做召回，
        再把命中的子块映射回父文档参与作答。函数会同时构建子块文本、父文档映射、
        子块向量和子块 BM25 索引，供后续 parent-child 检索变体直接使用。
        """
        child_ids: list[str] = []
        child_texts: list[str] = []
        parent_doc_indices: list[int] = []
        for parent_index, doc in enumerate(docs):
            snippets = rank_query_focused_snippets([], [], doc.content)
            if not snippets:
                snippets = [doc.content[:360]]
            for snippet_index, snippet in enumerate(snippets):
                child_ids.append(f"{doc.doc_id}:{snippet_index}")
                child_texts.append(snippet)
                parent_doc_indices.append(parent_index)
        child_embeddings = ensure_doc_embeddings(
            self.cache_root,
            f"{name}_child",
            self.embedding_model,
            child_ids,
            child_texts,
        )
        child_bm25, _ = ensure_bm25_index(
            self.cache_root,
            f"{name}_child",
            child_ids,
            child_texts,
        )
        return RetrievalCorpus(
            ids=child_ids,
            texts=child_texts,
            parent_doc_indices=parent_doc_indices,
            doc_embeddings=child_embeddings,
            bm25=child_bm25,
        )

    def prepare_dataset(
        self,
        name: str,
        cases: Sequence[ExperimentCase],
        docs: Sequence[CorpusDoc],
        *,
        include_contextual: bool = True,
        include_parent_child: bool = True,
        include_query_rewrite: bool = True,
    ) -> PreparedDataset:
        """准备实验所需的数据集、检索视图和查询特征。

        这一步会把同一批文档整理成多种可对比的检索视图：
        原始文档视图、上下文化视图、父子块视图，以及是否使用查询改写的查询特征。
        这样后续不同实验变体共用同一个 `PreparedDataset` 即可，避免每个脚本各自重建数据。
        """
        doc_ids = [str(doc.doc_id) for doc in docs]
        doc_titles = [doc.title for doc in docs]
        doc_texts = [doc.content for doc in docs]
        raw_doc_embeddings = ensure_doc_embeddings(
            self.cache_root,
            name,
            self.embedding_model,
            doc_ids,
            doc_texts,
        )
        raw_bm25, _ = ensure_bm25_index(
            self.cache_root,
            name,
            doc_ids,
            doc_texts,
        )
        raw_corpus = RetrievalCorpus(
            ids=doc_ids,
            texts=doc_texts,
            parent_doc_indices=list(range(len(docs))),
            doc_embeddings=raw_doc_embeddings,
            bm25=raw_bm25,
        )
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
            contextual_bm25, _ = ensure_bm25_index(
                self.cache_root,
                f"{name}_contextual",
                doc_ids,
                contextual_texts,
            )
            contextual_corpus = RetrievalCorpus(
                ids=doc_ids,
                texts=contextual_texts,
                parent_doc_indices=list(range(len(docs))),
                doc_embeddings=contextual_doc_embeddings,
                bm25=contextual_bm25,
            )
        else:
            contextual_corpus = raw_corpus
        child_corpus = self._build_child_corpus(name, docs) if include_parent_child else raw_corpus

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
            contextual_dense_rankings, _ = compute_dense_rankings(
                contextual_corpus.doc_embeddings,
                raw_query_embeddings,
                top_k=50,
            )
        else:
            contextual_dense_rankings = raw_dense_rankings
        if include_query_rewrite:
            query_rewrites = ensure_query_rewrites(
                self.cache_root,
                name,
                query_ids,
                queries,
                model_name=self.llm_model,
            )
            rewrite_dense_queries = [item["dense"] for item in query_rewrites]
            rewrite_query_embeddings = ensure_query_embeddings(
                self.cache_root,
                name,
                self.embedding_model,
                query_ids,
                rewrite_dense_queries,
                tag="rewrite_dense",
            )
            rewrite_dense_rankings, _ = compute_dense_rankings(
                raw_corpus.doc_embeddings,
                rewrite_query_embeddings,
                top_k=50,
            )
        else:
            query_rewrites = [{"sparse": query, "dense": query} for query in queries]
            rewrite_dense_rankings = raw_dense_rankings
        if include_parent_child:
            child_dense_rankings, _ = compute_dense_rankings(
                child_corpus.doc_embeddings,
                raw_query_embeddings,
                top_k=50,
            )
        else:
            child_dense_rankings = raw_dense_rankings

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
            if include_parent_child:
                child_bm25_indices, _ = bm25_rank(query_tokens, child_corpus.bm25, top_k=50)
            else:
                child_bm25_indices = raw_bm25_indices
            query_features[case.case_id] = QueryFeatures(
                case_id=case.case_id,
                query_tokens=query_tokens,
                raw_dense_indices=raw_dense_rankings[index].tolist(),
                raw_bm25_indices=raw_bm25_indices,
                rewrite_dense_indices=rewrite_dense_rankings[index].tolist(),
                rewrite_bm25_indices=rewrite_bm25_indices,
                contextual_dense_indices=contextual_dense_rankings[index].tolist(),
                contextual_bm25_indices=contextual_bm25_indices,
                child_dense_indices=child_dense_rankings[index].tolist(),
                child_bm25_indices=child_bm25_indices,
            )
        return PreparedDataset(
            name=name,
            cases=case_list,
            docs=list(docs),
            case_map={case.case_id: case for case in case_list},
            raw_corpus=raw_corpus,
            contextual_corpus=contextual_corpus,
            child_corpus=child_corpus,
            query_features=query_features,
        )

    def _build_candidate_indices(self, features: QueryFeatures, variant: PipelineVariant) -> list[int]:
        """根据检索策略生成候选文档下标。

        这里负责把 dense、BM25、上下文化检索、父子块检索和查询改写这些变量
        统一折叠成“当前变体要从哪几个候选开始往下走”。它是检索视图和后续精排之间的接口层。
        """
        if variant.retrieval_strategy == "contextual":
            fused = reciprocal_rank_fusion(
                [features.contextual_dense_indices, features.contextual_bm25_indices],
                top_k=variant.retrieval_top_k,
            )
            ranked = [doc_index for doc_index, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]
            return ranked[: variant.candidate_top_k]
        if variant.retrieval_strategy == "parent_child":
            fused = reciprocal_rank_fusion(
                [features.child_dense_indices, features.child_bm25_indices],
                top_k=variant.retrieval_top_k,
            )
            ranked = [child_index for child_index, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]
            return ranked[: variant.candidate_top_k]
        dense_indices = features.rewrite_dense_indices if variant.use_query_rewrite else features.raw_dense_indices
        bm25_indices = features.rewrite_bm25_indices if variant.use_query_rewrite else features.raw_bm25_indices
        fused = reciprocal_rank_fusion(
            [dense_indices, bm25_indices],
            top_k=variant.retrieval_top_k,
        )
        ranked = [doc_index for doc_index, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]
        return ranked[: variant.candidate_top_k]

    def _expand_snippet_context(self, content: str, snippet: str, window_sentences: int) -> str:
        """为片段补回相邻句子作为上下文。"""
        if window_sentences <= 0:
            return snippet
        sentences = [segment.strip() for segment in re.split(r"(?<=[。！？!?；;])\s*", content) if segment.strip()]
        if len(sentences) <= 1:
            return snippet
        snippet_norm = re.sub(r"\s+", "", snippet)
        if not snippet_norm:
            return snippet

        best_index = -1
        best_overlap = 0
        for index, sentence in enumerate(sentences):
            sentence_norm = re.sub(r"\s+", "", sentence)
            if not sentence_norm:
                continue
            overlap = len(set(sentence_norm) & set(snippet_norm))
            if snippet_norm in sentence_norm:
                best_index = index
                best_overlap = len(snippet_norm) + len(sentence_norm)
                break
            if overlap > best_overlap:
                best_index = index
                best_overlap = overlap
        if best_index < 0:
            return snippet

        left = max(0, best_index - window_sentences)
        right = min(len(sentences), best_index + window_sentences + 1)
        return " ".join(sentences[left:right]).strip()

    def _build_evidence_units(
        self,
        *,
        reranked_doc_indices: Sequence[int],
        doc_snippets: dict[int, list[str]],
        doc_scores: dict[int, float],
        docs: Sequence[CorpusDoc],
        variant: PipelineVariant,
        source_limit: int | None = None,
        snippet_limit: int | None = None,
        context_window_sentences: int | None = None,
    ) -> list[dict[str, Any]]:
        """把重排结果整理成统一证据单元。

        证据单元是整条实验链路里的标准中间格式。无论候选来自普通文档、上下文化视图
        还是父子块聚合，到了这里都会被转换成统一的 `doc_id/title/text/tokens/score` 结构，
        方便后面的证据筛选、渲染和作答步骤复用。
        """
        candidate_units: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        effective_source_limit = max(1, source_limit if source_limit is not None else variant.final_source_count)
        effective_snippet_limit = max(1, snippet_limit if snippet_limit is not None else variant.multi_snippet_count)
        effective_context_window = variant.context_window_sentences if context_window_sentences is None else context_window_sentences
        for doc_index in reranked_doc_indices[:effective_source_limit]:
            doc = docs[doc_index]
            snippets = (doc_snippets.get(doc_index) or [doc.content[:360]])[:effective_snippet_limit]
            for snippet in snippets:
                packed_text = self._expand_snippet_context(doc.content, snippet, effective_context_window)
                normalized_text = re.sub(r"\s+", "", packed_text)
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
                    }
                )
        if variant.evidence_selection == "mmr":
            selected_units = mmr_select_units(candidate_units, limit=variant.final_source_count)
        else:
            candidate_units.sort(key=lambda item: float(item["score"]), reverse=True)
            if variant.selection_mode == "aspect_cover_v2":
                candidate_pool_limit = min(
                    len(candidate_units),
                    max(effective_source_limit, effective_source_limit * effective_snippet_limit),
                )
                selected_units = candidate_units[:candidate_pool_limit]
            else:
                selected_units = candidate_units[:effective_source_limit]
        selected_units.sort(key=lambda item: float(item["score"]), reverse=True)
        return selected_units

    def _run_direct_answer(self, query: str, evidence_units: Sequence[dict[str, Any]], *, style: str) -> str:
        """运行直接答案。"""
        if not evidence_units:
            return NO_CONTEXT_ANSWER
        prompt = build_answer_prompt(query, evidence_units, style=style)
        if style == "trace_structured":
            raw_answer = generate_answer(prompt, model_name=self.llm_model, num_predict=512)
            parsed_answer = extract_tag_text(raw_answer, "answer")
            return parsed_answer or raw_answer or NO_CONTEXT_ANSWER
        return generate_answer(prompt, model_name=self.llm_model)

    def _apply_query_aligned_evidence_selection(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> list[dict[str, Any]]:
        """按问题对齐策略筛选证据单元。

        这一层对应论文方案与多条消融路线之间最核心的差异点之一：
        是否仅按分数取前几条，还是按问题分项重新排序证据。
        不同 `selection_mode` 会在这里切换为基础排序、分项覆盖或最终的 residual v2 版本。
        """
        if variant.selection_mode == "aspect_cover":
            ordered = order_evidence_units_for_query_clauses(query, evidence_units)
            if not ordered:
                return list(evidence_units)
            return ordered[: len(evidence_units)]
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
            if not ordered:
                return list(evidence_units)
            return ordered[: len(evidence_units)]
        if variant.selection_mode != "query_aligned":
            return list(evidence_units)
        ordered = order_evidence_units_for_query(query_tokens, evidence_units)
        if not ordered:
            return list(evidence_units)
        return ordered[: len(evidence_units)]

    def _apply_detail_preserving_rendering(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> list[dict[str, Any]]:
        """按渲染策略保留证据中的关键细节。

        有些对照路线不改证据选择，只改“证据如何呈现给模型”。
        这里统一承接这些实验变量，例如句窗替换、标题结构化、分项标注等，
        便于区分问题究竟出在选证据，还是出在证据写给模型看的方式。
        """
        if variant.rendering_mode == "detail_window":
            rendered = build_sentence_window_replacement_units(
                query_tokens,
                evidence_units,
                max_units=len(evidence_units),
            )
            return rendered or list(evidence_units)
        if variant.rendering_mode == "title_structured":
            rendered = build_title_structured_evidence_units(
                query_tokens,
                evidence_units,
                max_units=len(evidence_units),
            )
            return rendered or list(evidence_units)
        if variant.rendering_mode == "aspect_labeled":
            rendered = build_aspect_labeled_evidence_units(
                query,
                query_tokens,
                evidence_units,
                max_units=len(evidence_units),
            )
            return rendered or list(evidence_units)
        return list(evidence_units)

    def _run_compressed_aligned_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, Sequence[dict[str, Any]], int]:
        """运行压缩证据后的对齐作答。

        这是“先压缩事实、再组织回答”的一条生成侧路线。函数会先判断问题是否值得走复杂路径，
        复杂时再调用证据压缩模块，把多条证据收束成更少但更密的事实单元，然后再执行对齐作答。
        """
        direct_units = evidence_units[: variant.final_source_count]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        if assessment.route != "complex":
            answer = self._run_direct_answer(query, direct_units, style="simple")
            return answer, direct_units, 0

        compressed = compress_evidence_units(
            query=query,
            query_tokens=query_tokens,
            evidence_units=evidence_units,
            reranker=getattr(self, "reranker", None),
            max_units=max(variant.final_source_count, variant.compression_max_units),
        )
        compressed_units = list(compressed.units)[: max(variant.final_source_count, variant.compression_max_units)]
        if not compressed_units:
            answer = self._run_direct_answer(query, direct_units, style="aligned")
            return answer, direct_units, 0
        answer = self._run_direct_answer(query, compressed_units, style="aligned")
        return answer, compressed_units, compressed.fact_count

    def _run_aligned_pruned_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, int]:
        """运行裁剪无支撑句后的对齐作答。"""
        direct_units = evidence_units[: variant.final_source_count]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        style = "aligned" if assessment.route == "complex" else "simple"
        answer = self._run_direct_answer(query, direct_units, style=style)
        if assessment.route != "complex" or variant.support_pruning_threshold <= 0.0:
            return answer, 0
        pruned = prune_answer_to_supported_sentences(
            answer,
            direct_units,
            reranker=getattr(self, "reranker", None),
            threshold=variant.support_pruning_threshold,
        )
        return pruned.text, pruned.support_sentence_count

    def _run_support_repair_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, int]:
        """运行带支撑修补的对齐作答。"""
        direct_units = evidence_units[: variant.final_source_count]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        answer = self._run_direct_answer(query, direct_units, style="aligned")
        if assessment.route != "complex" or variant.support_pruning_threshold <= 0.0:
            return answer, 0
        repaired = repair_answer_with_supported_sentences(
            answer,
            direct_units,
            reranker=getattr(self, "reranker", None),
            support_threshold=variant.support_pruning_threshold,
            rewrite_threshold=max(0.42, variant.support_pruning_threshold - 0.12),
        )
        return repaired.text, repaired.support_sentence_count

    def _run_ordered_aligned_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, Sequence[dict[str, Any]]]:
        """运行按证据顺序组织的对齐作答。"""
        direct_units = evidence_units[: variant.final_source_count]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        if assessment.route != "complex":
            return self._run_direct_answer(query, direct_units, style="simple"), direct_units
        ordered_units = order_evidence_units_for_query(query_tokens, evidence_units)
        final_units = ordered_units[: variant.final_source_count]
        return self._run_direct_answer(query, final_units, style="aligned"), final_units

    def _run_copy_constrained_aligned_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, int]:
        """运行带片段约束的对齐作答。"""
        direct_units = evidence_units[: variant.final_source_count]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        if assessment.route != "complex":
            return self._run_direct_answer(query, direct_units, style="simple"), 0
        spans = extract_query_aligned_spans(
            query_tokens,
            direct_units,
            max_spans=variant.span_constraint_limit,
        )
        prompt = build_aligned_answer_prompt_with_constraints(
            query,
            direct_units,
            [span.text for span in spans],
        )
        return generate_answer(prompt, model_name=self.llm_model), len(spans)

    def _run_needled_aligned_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, Sequence[dict[str, Any]], int]:
        """运行带关键句标注的对齐作答。"""
        direct_units = evidence_units[: variant.final_source_count]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        if assessment.route != "complex":
            return self._run_direct_answer(query, direct_units, style="simple"), direct_units, 0

        annotated = build_needled_evidence_units(
            query=query,
            query_tokens=query_tokens,
            evidence_units=direct_units,
            reranker=getattr(self, "reranker", None),
            max_needs=max(3, min(variant.distilled_fact_limit, 6)),
        )
        annotated_units = list(annotated.units)[: variant.final_source_count]
        if annotated.need_count <= 0 or not annotated_units:
            return self._run_direct_answer(query, direct_units, style="aligned"), direct_units, 0
        answer = self._run_direct_answer(query, annotated_units, style="needled")
        return answer, annotated_units, annotated.need_count

    def _run_grouped_aligned_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, Sequence[dict[str, Any]]]:
        """运行按分组证据组织的对齐作答。"""
        direct_units = evidence_units[: variant.final_source_count]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        if assessment.route != "complex":
            return self._run_direct_answer(query, direct_units, style="simple"), direct_units
        grouped_units = group_evidence_units_by_query_aspects(
            query_tokens,
            evidence_units,
            max_groups=variant.final_source_count,
        )
        final_units = grouped_units[: variant.final_source_count] or list(direct_units)
        return self._run_direct_answer(query, final_units, style="aligned"), final_units

    def _run_edge_packed_aligned_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, Sequence[dict[str, Any]]]:
        """运行首尾强化布局的对齐作答。"""
        direct_units = evidence_units[: variant.final_source_count]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        if assessment.route != "complex":
            return self._run_direct_answer(query, direct_units, style="simple"), direct_units
        final_units = build_edge_packed_evidence_layout(
            query_tokens,
            evidence_units,
            max_units=variant.final_source_count,
        )
        return self._run_direct_answer(query, final_units or direct_units, style="aligned"), final_units or list(direct_units)

    def _run_window_replaced_aligned_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, Sequence[dict[str, Any]]]:
        """运行句窗替换后的对齐作答。"""
        direct_units = evidence_units[: variant.final_source_count]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        if assessment.route != "complex":
            return self._run_direct_answer(query, direct_units, style="simple"), direct_units
        final_units = build_sentence_window_replacement_units(
            query_tokens,
            direct_units,
            max_units=variant.final_source_count,
        )
        return self._run_direct_answer(query, final_units or direct_units, style="aligned"), final_units or list(direct_units)

    def _score_fact_candidates(
        self,
        *,
        query: str,
        facts: Sequence[EvidenceFact],
    ) -> dict[str, float]:
        """计算事实候选项列表分数。"""
        if getattr(self, "reranker", None) is None:
            return {}
        unique_candidates: list[tuple[str, str]] = []
        seen_signatures: set[str] = set()
        for fact in facts:
            signature = normalize_text(fact.statement)
            if not signature or signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_candidates.append((signature, fact.statement))
        if not unique_candidates:
            return {}

        raw_scores = self.reranker.predict(
            [(query, statement) for _signature, statement in unique_candidates],
            batch_size=16,
            show_progress_bar=False,
        ).tolist()
        return {
            signature: 1.0 / (1.0 + math.exp(-float(score)))
            for (signature, _statement), score in zip(unique_candidates, raw_scores)
        }

    def _run_adaptive_evidence_synthesis(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, int, int]:
        """运行证据压缩与综合作答主流程。

        这是论文实验里最接近“重型综合方案”的主流程之一：
        先蒸馏事实，再筛选落地事实，最后基于事实综合答案。
        它主要用于验证“增加中间事实层”是否比直接作答更稳定，也是多条证伪路线的参照基线。
        """
        direct_units = evidence_units[: variant.final_source_count]
        distillation_prompt = build_distillation_prompt(
            query,
            evidence_units,
            max_facts=variant.distilled_fact_limit,
        )
        distillation_payload = generate_json_payload(
            distillation_prompt,
            model_name=self.llm_model,
            num_predict=640,
        )
        facts = parse_distilled_facts(
            distillation_payload,
            evidence_units,
            max_facts=variant.distilled_fact_limit,
        )
        if len(set(query_tokens)) > 8:
            candidate_facts = list(facts)
            candidate_facts.extend(_sentence_fact_candidates(query_tokens, evidence_units))
            semantic_scores_by_signature = self._score_fact_candidates(
                query=query,
                facts=candidate_facts,
            )
            facts = refine_grounding_facts(
                query_tokens,
                facts,
                evidence_units,
                max_facts=variant.distilled_fact_limit,
                semantic_scores_by_signature=semantic_scores_by_signature,
            )
        if not facts:
            fallback_answer = self._run_direct_answer(query, direct_units, style=variant.answer_prompt_style)
            return fallback_answer, 0, 0
        if len(facts) > max(3, variant.distilled_fact_limit):
            fallback_answer = self._run_direct_answer(query, direct_units, style=variant.answer_prompt_style)
            return fallback_answer, len(facts), 0

        synthesis_prompt = build_grounded_synthesis_prompt(query, facts)
        synthesis_payload = generate_json_payload(
            synthesis_prompt,
            model_name=self.llm_model,
            num_predict=512,
        )
        supported_answer = parse_supported_answer(synthesis_payload, facts)
        fact_backed_answer = render_fact_answer(facts)
        if normalize_text(supported_answer.text) == normalize_text(NO_CONTEXT_ANSWER):
            if normalize_text(fact_backed_answer.text) != normalize_text(NO_CONTEXT_ANSWER):
                return fact_backed_answer.text, len(facts), fact_backed_answer.support_sentence_count
            fallback_answer = self._run_direct_answer(query, direct_units, style=variant.answer_prompt_style)
            return fallback_answer, len(facts), supported_answer.support_sentence_count
        if 0 < len(supported_answer.covered_fact_ids) < len(facts):
            if normalize_text(fact_backed_answer.text) != normalize_text(NO_CONTEXT_ANSWER):
                return fact_backed_answer.text, len(facts), fact_backed_answer.support_sentence_count
        if len(set(query_tokens)) <= 8 and supported_answer.support_sentence_count != 1:
            # 短而原子的查询更适合由单个紧凑槽位答案来承接，
            # 没必要额外展开成多句综合回答。
            fallback_answer = self._run_direct_answer(query, direct_units, style=variant.answer_prompt_style)
            return fallback_answer, len(facts), supported_answer.support_sentence_count
        if len(facts) <= 2 and supported_answer.support_sentence_count == 0:
            fallback_answer = self._run_direct_answer(query, direct_units, style=variant.answer_prompt_style)
            return fallback_answer, len(facts), supported_answer.support_sentence_count
        if len(facts) >= 3 and supported_answer.support_sentence_count == 0:
            fallback_answer = self._run_direct_answer(query, direct_units, style=variant.answer_prompt_style)
            return fallback_answer, len(facts), supported_answer.support_sentence_count
        return supported_answer.text, len(facts), supported_answer.support_sentence_count

    def _run_slot_filling_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
        verify: bool,
    ) -> tuple[str, int, int]:
        """运行槽位填充式作答。"""
        direct_units = evidence_units[: variant.final_source_count]
        candidate_facts = _sentence_fact_candidates(query_tokens, evidence_units, max_candidates_per_unit=3)
        semantic_scores_by_signature = self._score_fact_candidates(
            query=query,
            facts=candidate_facts,
        )
        facts = refine_grounding_facts(
            query_tokens,
            candidate_facts,
            evidence_units,
            max_facts=variant.distilled_fact_limit,
            semantic_scores_by_signature=semantic_scores_by_signature,
        )
        if not facts:
            return self._run_direct_answer(query, direct_units, style=variant.answer_prompt_style), 0, 0

        plan_prompt = build_structured_plan_prompt(query, facts)
        plan_payload = generate_json_payload(
            plan_prompt,
            model_name=self.llm_model,
            num_predict=320,
        )
        plan = parse_structured_plan(plan_payload, facts)
        if not plan.slots:
            return self._run_direct_answer(query, direct_units, style=variant.answer_prompt_style), len(facts), 0

        slot_prompt = build_slot_answer_prompt(query, facts, plan)
        slot_payload = generate_json_payload(
            slot_prompt,
            model_name=self.llm_model,
            num_predict=420,
        )
        slot_answer = parse_slot_answers(slot_payload, plan, facts)
        if normalize_text(slot_answer.text) == normalize_text(NO_CONTEXT_ANSWER):
            return self._run_direct_answer(query, direct_units, style=variant.answer_prompt_style), len(facts), 0
        if not verify:
            return slot_answer.text, len(facts), slot_answer.support_sentence_count

        verification_prompt = build_slot_verification_prompt(query, slot_answer.text, facts)
        verification_payload = generate_json_payload(
            verification_prompt,
            model_name=self.llm_model,
            num_predict=420,
        )
        verified_answer = parse_verified_answer(verification_payload, facts)
        if normalize_text(verified_answer.text) == normalize_text(NO_CONTEXT_ANSWER):
            return slot_answer.text, len(facts), slot_answer.support_sentence_count
        return verified_answer.text, len(facts), verified_answer.support_sentence_count

    def _run_slot_repair_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, int, int]:
        """运行槽位修补式作答。"""
        direct_units = evidence_units[: variant.final_source_count]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        draft_answer = self._run_direct_answer(query, direct_units, style="aligned")
        if assessment.route != "complex":
            return draft_answer, 0, 0

        candidate_facts = _sentence_fact_candidates(query_tokens, evidence_units, max_candidates_per_unit=3)
        semantic_scores_by_signature = self._score_fact_candidates(
            query=query,
            facts=candidate_facts,
        )
        facts = refine_grounding_facts(
            query_tokens,
            candidate_facts,
            evidence_units,
            max_facts=variant.distilled_fact_limit,
            semantic_scores_by_signature=semantic_scores_by_signature,
        )
        if not facts:
            return draft_answer, 0, 0

        plan_prompt = build_structured_plan_prompt(query, facts)
        plan_payload = generate_json_payload(
            plan_prompt,
            model_name=self.llm_model,
            num_predict=320,
        )
        plan = parse_structured_plan(plan_payload, facts)
        if not plan.slots:
            return draft_answer, len(facts), 0

        repair_prompt = build_slot_repair_prompt(query, draft_answer, facts, plan)
        repair_payload = generate_json_payload(
            repair_prompt,
            model_name=self.llm_model,
            num_predict=420,
        )
        repaired_answer = parse_slot_answers(repair_payload, plan, facts)
        if normalize_text(repaired_answer.text) == normalize_text(NO_CONTEXT_ANSWER):
            return draft_answer, len(facts), 0
        return repaired_answer.text, len(facts), repaired_answer.support_sentence_count

    def _run_fact_repair_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, int, int]:
        """运行事实修补式作答。"""
        direct_units = evidence_units[: variant.final_source_count]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        draft_answer = self._run_direct_answer(query, direct_units, style="aligned")
        if assessment.route != "complex":
            return draft_answer, 0, 0

        candidate_facts = _sentence_fact_candidates(query_tokens, evidence_units, max_candidates_per_unit=3)
        semantic_scores_by_signature = self._score_fact_candidates(
            query=query,
            facts=candidate_facts,
        )
        facts = refine_grounding_facts(
            query_tokens,
            candidate_facts,
            evidence_units,
            max_facts=variant.distilled_fact_limit,
            semantic_scores_by_signature=semantic_scores_by_signature,
        )
        if not facts:
            return draft_answer, 0, 0

        repair_prompt = build_fact_repair_prompt(query, draft_answer, facts)
        repair_payload = generate_json_payload(
            repair_prompt,
            model_name=self.llm_model,
            num_predict=420,
        )
        repaired_answer = parse_supported_answer(repair_payload, facts)
        if normalize_text(repaired_answer.text) == normalize_text(NO_CONTEXT_ANSWER):
            return draft_answer, len(facts), 0
        return repaired_answer.text, len(facts), repaired_answer.support_sentence_count

    def _run_structured_support_repair_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, int, int]:
        """运行结构化支撑修补式作答。"""
        direct_units = evidence_units[: max(variant.final_source_count, variant.complex_source_count)]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        if assessment.route != "complex":
            answer = self._run_direct_answer(query, direct_units[: variant.final_source_count], style="aligned")
            return answer, 0, 0

        candidate_facts = _sentence_fact_candidates(query_tokens, direct_units, max_candidates_per_unit=3)
        semantic_scores_by_signature = self._score_fact_candidates(
            query=query,
            facts=candidate_facts,
        )
        facts = refine_grounding_facts(
            query_tokens,
            candidate_facts,
            direct_units,
            max_facts=variant.distilled_fact_limit,
            semantic_scores_by_signature=semantic_scores_by_signature,
        )
        if not facts:
            answer = self._run_direct_answer(query, direct_units[: variant.final_source_count], style="aligned")
            return answer, 0, 0

        prompt = build_structured_evidence_answer_prompt(query, direct_units[: variant.final_source_count], facts)
        answer = generate_answer(prompt, model_name=self.llm_model)
        repaired = repair_answer_with_supported_sentences(
            answer,
            direct_units[: variant.final_source_count],
            reranker=getattr(self, "reranker", None),
            support_threshold=max(0.50, variant.support_pruning_threshold or 0.55),
            rewrite_threshold=0.42,
        )
        final_answer = repaired.text if normalize_text(repaired.text) != normalize_text(NO_CONTEXT_ANSWER) else answer
        return final_answer, len(facts), repaired.support_sentence_count

    def _run_plan_packed_aligned_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, int]:
        """运行计划打包式对齐作答。"""
        direct_units = evidence_units[: max(variant.final_source_count, variant.complex_source_count)]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        if assessment.route != "complex":
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="simple"), 0

        candidate_facts = _sentence_fact_candidates(query_tokens, direct_units, max_candidates_per_unit=3)
        semantic_scores_by_signature = self._score_fact_candidates(
            query=query,
            facts=candidate_facts,
        )
        facts = refine_grounding_facts(
            query_tokens,
            candidate_facts,
            direct_units,
            max_facts=max(variant.distilled_fact_limit, variant.final_source_count),
            semantic_scores_by_signature=semantic_scores_by_signature,
        )
        if not facts:
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="aligned"), 0

        plan_prompt = build_structured_plan_prompt(query, facts)
        plan_payload = generate_json_payload(
            plan_prompt,
            model_name=self.llm_model,
            num_predict=320,
        )
        plan = parse_structured_plan(plan_payload, facts)
        if not plan.slots:
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="aligned"), len(facts)

        packed_units = build_slot_packed_evidence_units(
            plan,
            facts,
            direct_units,
            max_units=max(variant.final_source_count, min(len(plan.slots), variant.complex_source_count)),
        )
        if not packed_units:
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="aligned"), len(facts)

        prompt = build_slot_packed_answer_prompt(query, packed_units)
        answer = generate_answer(prompt, model_name=self.llm_model)
        if normalize_text(answer) == normalize_text(NO_CONTEXT_ANSWER):
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="aligned"), len(facts)
        return answer, len(facts)

    def _run_support_table_aligned_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, int]:
        """运行支持表驱动的对齐作答。"""
        direct_units = evidence_units[: max(variant.final_source_count, variant.complex_source_count)]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        if assessment.route != "complex":
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="simple"), 0

        candidate_facts = _sentence_fact_candidates(query_tokens, direct_units, max_candidates_per_unit=3)
        semantic_scores_by_signature = self._score_fact_candidates(
            query=query,
            facts=candidate_facts,
        )
        facts = refine_grounding_facts(
            query_tokens,
            candidate_facts,
            direct_units,
            max_facts=max(variant.distilled_fact_limit, variant.final_source_count),
            semantic_scores_by_signature=semantic_scores_by_signature,
        )
        if not facts:
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="task_aligned"), 0

        plan = build_clause_support_plan(
            query,
            facts,
            direct_units,
            max_slots=max(variant.final_source_count, min(variant.complex_source_count, 4)),
        )
        if len(plan.slots) < 2:
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="task_aligned"), len(facts)

        packed_units = build_slot_packed_evidence_units(
            plan,
            facts,
            direct_units,
            max_units=max(variant.final_source_count, min(len(plan.slots), variant.complex_source_count)),
        )
        if not packed_units:
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="task_aligned"), len(facts)

        prompt = build_slot_packed_answer_prompt(query, packed_units)
        answer = generate_answer(prompt, model_name=self.llm_model)
        if normalize_text(answer) == normalize_text(NO_CONTEXT_ANSWER):
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="task_aligned"), len(facts)
        return answer, len(facts)

    def _run_plan_guided_aligned_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, int]:
        """运行计划引导式对齐作答。"""
        direct_units = evidence_units[: max(variant.final_source_count, variant.complex_source_count)]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        if assessment.route != "complex":
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="simple"), 0

        candidate_facts = _sentence_fact_candidates(query_tokens, direct_units, max_candidates_per_unit=3)
        semantic_scores_by_signature = self._score_fact_candidates(
            query=query,
            facts=candidate_facts,
        )
        facts = refine_grounding_facts(
            query_tokens,
            candidate_facts,
            direct_units,
            max_facts=max(variant.distilled_fact_limit, variant.final_source_count),
            semantic_scores_by_signature=semantic_scores_by_signature,
        )
        if not facts:
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="aligned"), 0

        plan_prompt = build_structured_plan_prompt(query, facts)
        plan_payload = generate_json_payload(
            plan_prompt,
            model_name=self.llm_model,
            num_predict=320,
        )
        plan = parse_structured_plan(plan_payload, facts)
        if not plan.slots:
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="aligned"), len(facts)

        prompt = build_plan_guided_answer_prompt(
            query,
            direct_units[: variant.final_source_count],
            plan,
            facts,
        )
        answer = generate_answer(prompt, model_name=self.llm_model)
        if normalize_text(answer) == normalize_text(NO_CONTEXT_ANSWER):
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="aligned"), len(facts)
        return answer, len(facts)

    def _run_clause_guided_aligned_answer(
        self,
        *,
        query: str,
        query_tokens: Sequence[str],
        evidence_units: Sequence[dict[str, Any]],
        variant: PipelineVariant,
    ) -> tuple[str, int]:
        """运行子句引导式对齐作答。"""
        direct_units = evidence_units[: max(variant.final_source_count, variant.complex_source_count)]
        assessment = assess_complexity(
            query_tokens,
            evidence_units,
            threshold=variant.router_complexity_threshold,
        )
        if assessment.route != "complex":
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="simple"), 0

        clauses = extract_query_aspect_clauses(query)
        if len(clauses) < 2:
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="aligned"), 0

        clause_units: list[dict[str, Any]] = []
        for clause in clauses:
            clause_tokens = tokenize_text(clause)
            ordered_units = order_evidence_units_for_query(clause_tokens, direct_units)
            if ordered_units:
                clause_units.append(ordered_units[0])

        if not clause_units:
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="aligned"), 0

        prompt = build_clause_guided_answer_prompt(
            query,
            clauses,
            clause_units,
            direct_units[: variant.final_source_count],
        )
        answer = generate_answer(prompt, model_name=self.llm_model)
        if normalize_text(answer) == normalize_text(NO_CONTEXT_ANSWER):
            return self._run_direct_answer(query, direct_units[: variant.final_source_count], style="aligned"), len(clause_units)
        return answer, len(clause_units)

    def _select_documents_without_rerank(
        self,
        *,
        query_tokens: list[str],
        candidate_doc_indices: Sequence[int],
        docs: Sequence[CorpusDoc],
    ) -> tuple[list[int], dict[int, list[str]], dict[int, float]]:
        """在不重排时直接选择候选文档。"""
        ordered = list(candidate_doc_indices)
        doc_snippets: dict[int, list[str]] = {}
        doc_scores: dict[int, float] = {}
        for rank, doc_index in enumerate(ordered):
            ranked_snippets = rank_query_focused_snippets(query_tokens, query_tokens, docs[doc_index].content)
            doc_snippets[int(doc_index)] = ranked_snippets[:1] or [docs[doc_index].content[:360]]
            doc_scores[int(doc_index)] = 1.0 / float(rank + 1)
        return ordered, doc_snippets, doc_scores

    def _aggregate_child_candidates(
        self,
        *,
        candidate_child_indices: Sequence[int],
        child_corpus: RetrievalCorpus,
    ) -> tuple[list[int], dict[int, list[str]], dict[int, float]]:
        """把子块候选聚合回父文档。"""
        parent_scores: dict[int, float] = {}
        parent_snippets: dict[int, list[str]] = {}
        for rank, child_index in enumerate(candidate_child_indices):
            parent_doc_index = child_corpus.parent_doc_indices[child_index]
            score = 1.0 / float(rank + 1)
            snippet = child_corpus.texts[child_index]
            if score > parent_scores.get(parent_doc_index, float("-inf")):
                parent_scores[parent_doc_index] = score
                parent_snippets[parent_doc_index] = [snippet]
        ranked_parents = [
            parent_index for parent_index, _ in sorted(parent_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        return ranked_parents, parent_snippets, parent_scores

    def _rerank_fixed_snippets(
        self,
        *,
        query: str,
        candidate_doc_indices: Sequence[int],
        doc_snippets: dict[int, list[str]],
    ) -> tuple[list[int], dict[int, list[str]], dict[int, float]]:
        """对固定片段集合执行重排。"""
        if not candidate_doc_indices:
            return [], {}, {}
        pairs: list[tuple[str, str]] = []
        doc_order: list[int] = []
        for doc_index in candidate_doc_indices:
            snippet = (doc_snippets.get(doc_index) or [""])[:1][0]
            doc_order.append(int(doc_index))
            pairs.append((query, snippet))
        raw_scores = self.reranker.predict(pairs, batch_size=16, show_progress_bar=False).tolist()
        doc_scores = {doc_index: float(score) for doc_index, score in zip(doc_order, raw_scores)}
        ranked = [doc_index for doc_index, _ in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)]
        return ranked, doc_snippets, doc_scores

    def _select_retrieval_view(
        self,
        prepared: PreparedDataset,
        features: QueryFeatures,
        variant: PipelineVariant,
    ) -> tuple[list[int], RetrievalCorpus]:
        """选择当前实验要使用的检索视图。"""
        candidate_indices = self._build_candidate_indices(features, variant)
        if variant.retrieval_strategy == "contextual":
            return candidate_indices, prepared.contextual_corpus
        if variant.retrieval_strategy == "parent_child":
            return candidate_indices, prepared.child_corpus
        return candidate_indices, prepared.raw_corpus

    def run_case(self, prepared: PreparedDataset, case: ExperimentCase, variant: PipelineVariant) -> PipelineRunResult:
        """运行单条样例的完整实验流程。

        这是所有实验脚本最终都会调用的主入口。它把一次样例评测拆成：
        1. 选择检索视图并生成候选；
        2. 执行重排或父子块聚合；
        3. 构造统一证据单元；
        4. 进行证据选择与渲染；
        5. 根据变体进入不同作答路线；
        6. 汇总回答、来源、延迟和路由信息。

        因为消融实验、证伪实验和模型对比都共用这里，所以这段注释写得更偏“流水线地图”。
        """
        started = time.perf_counter()
        features = prepared.query_features[case.case_id]
        candidate_indices, retrieval_corpus = self._select_retrieval_view(prepared, features, variant)
        route_mode = "direct"
        complexity_score = 0.0
        distilled_fact_count = 0
        support_sentence_count = 0
        if variant.retrieval_strategy == "parent_child":
            reranked_doc_indices, doc_snippets, doc_scores = self._aggregate_child_candidates(
                candidate_child_indices=candidate_indices,
                child_corpus=retrieval_corpus,
            )
            if variant.use_rerank:
                reranked_doc_indices, doc_snippets, doc_scores = self._rerank_fixed_snippets(
                    query=case.query,
                    candidate_doc_indices=reranked_doc_indices,
                    doc_snippets=doc_snippets,
                )
        elif variant.use_rerank:
            aspect_token_sets: list[set[str]] = []
            aggregation_mode = "default"
            if variant.rerank_mode in {"aspect_aware", "aspect_aware_conservative"}:
                aspect_clauses = extract_query_aspect_clauses(case.query, max_clauses=4)
                aspect_token_sets = [set(tokenize_text(clause)) for clause in aspect_clauses if tokenize_text(clause)]
                if variant.rerank_mode == "aspect_aware_conservative":
                    explicit_multi_marker = any(marker in case.query for marker in ("以及", "并且", "同时", "分别"))
                    avg_clause_tokens = (
                        sum(len(tokens) for tokens in aspect_token_sets) / len(aspect_token_sets)
                        if aspect_token_sets
                        else 0.0
                    )
                    if explicit_multi_marker and len(aspect_token_sets) >= 2 and len(features.query_tokens) >= 14 and avg_clause_tokens >= 4.0:
                        aggregation_mode = "aspect_aware_conservative"
                elif len(aspect_token_sets) >= 2:
                    aggregation_mode = "aspect_aware"
            reranked_doc_indices, doc_snippets, doc_scores = rerank_documents(
                self.reranker,
                query=case.query,
                query_tokens=features.query_tokens,
                focus_tokens=features.query_tokens,
                candidate_doc_indices=candidate_indices,
                corpus_texts=[doc.content for doc in prepared.docs],
                multi_snippet_count=variant.multi_snippet_count,
                aggregation_mode=aggregation_mode,
                aspect_token_sets=aspect_token_sets,
            )
        else:
            reranked_doc_indices, doc_snippets, doc_scores = self._select_documents_without_rerank(
                query_tokens=features.query_tokens,
                candidate_doc_indices=candidate_indices,
                docs=prepared.docs,
            )
        routing_units = self._build_evidence_units(
            reranked_doc_indices=reranked_doc_indices,
            doc_snippets=doc_snippets,
            doc_scores=doc_scores,
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
        routing_units = self._apply_detail_preserving_rendering(
            query=case.query,
            query_tokens=features.query_tokens,
            evidence_units=routing_units,
            variant=variant,
        )

        if not routing_units:
            answer = NO_CONTEXT_ANSWER
            source_doc_ids: list[int] = []
            source_titles: list[str] = []
            retrieved_contexts: list[str] = []
        else:
            if variant.synthesis_mode == "adaptive_evidence":
                assessment = assess_complexity(
                    features.query_tokens,
                    routing_units,
                    threshold=variant.router_complexity_threshold,
                )
                route_mode = assessment.route
                complexity_score = assessment.score
                if assessment.route == "complex":
                    complex_units = routing_units[: max(variant.final_source_count, variant.complex_source_count)]
                    answer, distilled_fact_count, support_sentence_count = self._run_adaptive_evidence_synthesis(
                        query=case.query,
                        query_tokens=features.query_tokens,
                        evidence_units=complex_units,
                        variant=variant,
                    )
                    units = complex_units
                else:
                    direct_units = routing_units[: variant.final_source_count]
                    answer = self._run_direct_answer(
                        case.query,
                        direct_units,
                        style=self._resolve_prompt_style(variant, case.query, features.query_tokens),
                    )
                    units = direct_units
            elif variant.synthesis_mode in {"slot_fill", "slot_fill_verify"}:
                units = routing_units[: max(variant.final_source_count, variant.complex_source_count)]
                answer, distilled_fact_count, support_sentence_count = self._run_slot_filling_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                    verify=variant.synthesis_mode == "slot_fill_verify",
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0
            elif variant.synthesis_mode == "compressed_aligned":
                units = routing_units[: max(variant.final_source_count, variant.complex_source_count)]
                answer, units, distilled_fact_count = self._run_compressed_aligned_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if distilled_fact_count > 0 else 0.0
            elif variant.synthesis_mode == "aligned_pruned":
                units = routing_units[: variant.final_source_count]
                answer, support_sentence_count = self._run_aligned_pruned_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if support_sentence_count > 0 else 0.0
            elif variant.synthesis_mode == "support_repair":
                units = routing_units[: variant.final_source_count]
                answer, support_sentence_count = self._run_support_repair_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if support_sentence_count > 0 else 0.0
            elif variant.synthesis_mode == "slot_repair":
                units = routing_units[: max(variant.final_source_count, variant.complex_source_count)]
                answer, distilled_fact_count, support_sentence_count = self._run_slot_repair_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if support_sentence_count > 0 else 0.0
            elif variant.synthesis_mode == "fact_repair":
                units = routing_units[: max(variant.final_source_count, variant.complex_source_count)]
                answer, distilled_fact_count, support_sentence_count = self._run_fact_repair_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if support_sentence_count > 0 else 0.0
            elif variant.synthesis_mode == "structured_support_repair":
                units = routing_units[: max(variant.final_source_count, variant.complex_source_count)]
                answer, distilled_fact_count, support_sentence_count = self._run_structured_support_repair_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if support_sentence_count > 0 else 0.0
            elif variant.synthesis_mode == "plan_packed_aligned":
                units = routing_units[: max(variant.final_source_count, variant.complex_source_count)]
                answer, distilled_fact_count = self._run_plan_packed_aligned_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if distilled_fact_count > 0 else 0.0
            elif variant.synthesis_mode == "support_table_aligned":
                units = routing_units[: max(variant.final_source_count, variant.complex_source_count)]
                answer, distilled_fact_count = self._run_support_table_aligned_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if distilled_fact_count > 0 else 0.0
            elif variant.synthesis_mode == "plan_guided_aligned":
                units = routing_units[: max(variant.final_source_count, variant.complex_source_count)]
                answer, distilled_fact_count = self._run_plan_guided_aligned_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if distilled_fact_count > 0 else 0.0
            elif variant.synthesis_mode == "clause_guided_aligned":
                units = routing_units[: max(variant.final_source_count, variant.complex_source_count)]
                answer, distilled_fact_count = self._run_clause_guided_aligned_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if distilled_fact_count > 0 else 0.0
            elif variant.synthesis_mode == "ordered_aligned":
                units = routing_units[: max(variant.final_source_count, variant.complex_source_count)]
                answer, units = self._run_ordered_aligned_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if len(units) > 0 else 0.0
            elif variant.synthesis_mode == "copy_constrained_aligned":
                units = routing_units[: variant.final_source_count]
                answer, distilled_fact_count = self._run_copy_constrained_aligned_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if distilled_fact_count > 0 else 0.0
            elif variant.synthesis_mode == "needled_aligned":
                units = routing_units[: variant.final_source_count]
                answer, units, distilled_fact_count = self._run_needled_aligned_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if distilled_fact_count > 0 else 0.0
            elif variant.synthesis_mode == "grouped_aligned":
                units = routing_units[: max(variant.final_source_count, variant.complex_source_count)]
                answer, units = self._run_grouped_aligned_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if len(units) > 0 else 0.0
            elif variant.synthesis_mode == "edge_packed_aligned":
                units = routing_units[: max(variant.final_source_count, variant.complex_source_count)]
                answer, units = self._run_edge_packed_aligned_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if len(units) > 0 else 0.0
            elif variant.synthesis_mode == "window_replaced_aligned":
                units = routing_units[: variant.final_source_count]
                answer, units = self._run_window_replaced_aligned_answer(
                    query=case.query,
                    query_tokens=features.query_tokens,
                    evidence_units=units,
                    variant=variant,
                )
                route_mode = variant.synthesis_mode
                complexity_score = 1.0 if len(units) > 0 else 0.0
            else:
                units = routing_units[: variant.final_source_count]
                answer = self._run_direct_answer(
                    case.query,
                    units,
                    style=self._resolve_prompt_style(variant, case.query, features.query_tokens),
                )
            source_doc_ids = []
            source_titles = []
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
            route_mode=route_mode,
            complexity_score=complexity_score,
            distilled_fact_count=distilled_fact_count,
            support_sentence_count=support_sentence_count,
        )

    def _resolve_prompt_style(self, variant: PipelineVariant, query: str, query_tokens: Sequence[str]) -> str:
        """解析当前变体对应的提示词风格。

        某些实验变体会把 prompt 风格也做成动态路由，例如短问题更偏 simple，
        多分项问题更偏 aligned / task_aligned。这里集中处理这些映射，避免主流程里散落判断。
        """
        if variant.answer_prompt_style in {
            "simple",
            "neutral",
            "model_native",
            "aligned",
            "faithful",
            "task_aligned",
            "trace_structured",
        }:
            return variant.answer_prompt_style
        if variant.answer_prompt_style == "task_router":
            task_mode = infer_query_task_mode(query)
            if task_mode == "summary":
                return "simple"
            if len(extract_query_aspect_clauses(query)) >= 2:
                return "task_aligned"
            return "aligned"
        if variant.answer_prompt_style == "adaptive":
            if len(query_tokens) >= max(variant.adaptive_prompt_min_query_tokens, 1):
                return "faithful"
            return "simple"
        return "simple"
