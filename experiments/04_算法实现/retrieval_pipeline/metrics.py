"""论文实验指标计算模块。

这里负责把逐题结果汇总成论文表格中使用的命中率、相似度、
多文档整合质量等指标。
"""

from __future__ import annotations

import html
import importlib
import math
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

from ragas import EvaluationDataset, evaluate
from ragas.metrics import AnswerCorrectness, AnswerRelevancy, ContextPrecision, Faithfulness
from ragas.run_config import RunConfig

from .adaptive_evidence import extract_query_aspect_clauses, split_answer_sentences
from .common import DEFAULT_EMBEDDING_MODEL, contains_refusal, embed_texts
from .datasets import ExperimentCase
from .pipeline import PipelineRunResult


def normalize_text(text: str) -> str:
    """把文本压成适合匹配和相似度计算的稳定形式。"""
    lowered = html.unescape(str(text or "")).strip().lower()
    lowered = re.sub(r"\s+", "", lowered)
    return re.sub(r"[，,。.!！？?；;：:\"'“”‘’（）()【】《》<>]", "", lowered)


def match_any_answer(prediction: str, answers: Sequence[str]) -> bool:
    """判断预测答案是否命中任一参考答案或其紧邻变体。"""
    prediction_norm = normalize_text(prediction)
    if not prediction_norm:
        return False
    for answer in answers:
        answer_norm = normalize_text(answer)
        if not answer_norm:
            continue
        if prediction_norm == answer_norm:
            return True
        if answer_norm in prediction_norm or prediction_norm in answer_norm:
            return True
    return False


def retrieval_hits_positive_texts(source_texts: Sequence[str], positives: Sequence[str]) -> bool:
    """判断检索返回文本中是否至少命中一条正样本文本。"""
    normalized_sources = [normalize_text(text) for text in source_texts if text]
    normalized_positives = [normalize_text(text) for text in positives if text]
    for source in normalized_sources:
        for positive in normalized_positives:
            if not source or not positive:
                continue
            if source in positive or positive in source:
                return True
    return False


def text_similarity(left: str, right: str) -> float:
    """计算两段文本的字符级相似度。

    这是实验指标里的轻量字符串相似度基线，主要用于在语义相似度之外，
    再补一层对字面接近程度的衡量，方便生成综合质量分数。
    """
    a = normalize_text(left)
    b = normalize_text(right)
    if not a or not b:
        return 0.0
    if a == b or a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def semantic_similarity_scores(
    predictions: Sequence[str],
    answer_sets: Sequence[Sequence[str]],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[float]:
    """批量计算预测答案与参考答案集合之间的语义相似度。"""
    if not predictions:
        return []

    text_to_index: dict[str, int] = {}
    unique_texts: list[str] = []

    def index_text(text: str) -> int:
        """为文本建立临时索引表示。"""
        normalized = str(text or "").strip()
        if not normalized:
            return -1
        if normalized not in text_to_index:
            text_to_index[normalized] = len(unique_texts)
            unique_texts.append(normalized)
        return text_to_index[normalized]

    pair_indexes: list[tuple[int, list[int]]] = []
    for prediction, answers in zip(predictions, answer_sets):
        prediction_index = index_text(prediction)
        answer_indexes = [index_text(answer) for answer in answers if str(answer or "").strip()]
        pair_indexes.append((prediction_index, [index for index in answer_indexes if index >= 0]))

    if not unique_texts:
        return [0.0] * len(predictions)

    try:
        embeddings = embed_texts(model_name, unique_texts, batch_size=32)
    except Exception as exc:
        print(f"[SemanticMetric] degraded to string similarity: {exc}", file=sys.stderr, flush=True)
        return [
            max((text_similarity(prediction, answer) for answer in answers), default=0.0)
            for prediction, answers in zip(predictions, answer_sets)
        ]

    scores: list[float] = []
    for prediction_index, answer_indexes in pair_indexes:
        if prediction_index < 0 or not answer_indexes:
            scores.append(0.0)
            continue
        prediction_vector = embeddings[prediction_index]
        best_score = max(float(prediction_vector @ embeddings[answer_index]) for answer_index in answer_indexes)
        scores.append(max(0.0, min(1.0, best_score)))
    return scores


def summary_quality_score(semantic_similarity: float, text_similarity_score: float) -> float:
    """按论文口径把摘要任务的语义相似度与字面相似度压成单一分数。"""
    return max(0.0, min(1.0, 0.65 * float(semantic_similarity) + 0.35 * float(text_similarity_score)))


def integration_quality_score(
    semantic_similarity: float,
    text_similarity_score: float,
    focus_f1: float,
) -> float:
    """按论文口径把多文档整合质量压成单一综合指标。"""
    return max(
        0.0,
        min(
            1.0,
            0.40 * float(semantic_similarity) + 0.30 * float(text_similarity_score) + 0.30 * float(focus_f1),
        ),
    )


def _split_reference_nuggets(text: str) -> list[str]:
    """把参考答案拆成若干可独立比对的事实片段。"""
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    parts = [
        part.strip(" ，,；;。！？? ")
        for part in re.split(r"[。；;]|(?:以及|并且|同时|此外|且|并|而且)", cleaned)
        if part.strip(" ，,；;。！？? ")
    ]
    nuggets: list[str] = []
    seen: set[str] = set()
    for part in parts or [cleaned]:
        signature = normalize_text(part)
        if len(signature) < 6 or signature in seen:
            continue
        seen.add(signature)
        nuggets.append(part)
    return nuggets or [cleaned]


def multidoc_focus_scores(
    results: Sequence[PipelineRunResult],
    cases: Sequence[ExperimentCase],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, dict[str, float]]:
    """计算多文档问答的覆盖率、精确率和焦点 F1。

    这组指标专门服务复杂问答场景，用来判断模型是否真正覆盖了题目要求的
    多个信息点，而不是只答中了其中一部分。
    """
    case_map = {case.case_id: case for case in cases}
    case_payloads: list[tuple[str, list[str], list[str], list[str]]] = []
    unique_texts: list[str] = []
    text_to_index: dict[str, int] = {}

    def index_text(text: str) -> int:
        """为文本分配一个临时索引，便于复用同一份向量缓存。"""
        normalized = str(text or "").strip()
        if not normalized:
            return -1
        if normalized not in text_to_index:
            text_to_index[normalized] = len(unique_texts)
            unique_texts.append(normalized)
        return text_to_index[normalized]

    for result in results:
        case = case_map[result.case_id]
        if case.split not in {"questanswer_2docs", "questanswer_3docs"}:
            continue
        clauses = extract_query_aspect_clauses(case.query) or [case.query]
        reference_nuggets = _split_reference_nuggets(case.reference)
        response_sentences = split_answer_sentences(result.response) or [str(result.response or "").strip()]
        case_payloads.append((result.case_id, clauses, reference_nuggets, response_sentences))
        for text in clauses + reference_nuggets + response_sentences:
            index_text(text)

    if not case_payloads:
        return {}

    try:
        embeddings = embed_texts(model_name, unique_texts, batch_size=32)
    except Exception as exc:
        print(f"[MultidocFocus] degraded to empty scores: {exc}", file=sys.stderr, flush=True)
        return {case_id: {"coverage": 0.0, "precision": 0.0, "f1": 0.0} for case_id, *_ in case_payloads}

    def similarity(left: str, right: str) -> float:
        """在已缓存向量上直接计算两段文本的语义相似度。"""
        left_index = index_text(left)
        right_index = index_text(right)
        if left_index < 0 or right_index < 0:
            return 0.0
        return float(embeddings[left_index] @ embeddings[right_index])

    scores_by_case_id: dict[str, dict[str, float]] = {}
    for case_id, clauses, reference_nuggets, response_sentences in case_payloads:
        mapped_references: list[str] = []
        for clause in clauses:
            clause_tokens = set(normalize_text(token) for token in re.findall(r"[\w\u4e00-\u9fff]+", clause) if normalize_text(token))
            mapped_references.append(
                max(
                    reference_nuggets,
                    key=lambda nugget: (
                        len(clause_tokens & set(normalize_text(token) for token in re.findall(r"[\w\u4e00-\u9fff]+", nugget) if normalize_text(token))),
                        similarity(clause, nugget),
                    ),
                )
            )

        covered = 0
        for reference_nugget in mapped_references:
            best_sentence_similarity = max((similarity(sentence, reference_nugget) for sentence in response_sentences), default=0.0)
            if best_sentence_similarity >= 0.83:
                covered += 1
        coverage = covered / len(mapped_references) if mapped_references else 0.0

        focused_sentences = 0
        for sentence in response_sentences:
            best_clause_similarity = max((similarity(sentence, clause) for clause in clauses), default=0.0)
            best_reference_similarity = max((similarity(sentence, nugget) for nugget in reference_nuggets), default=0.0)
            if best_clause_similarity >= 0.72 or best_reference_similarity >= 0.82:
                focused_sentences += 1
        precision = focused_sentences / len(response_sentences) if response_sentences else 0.0
        f1 = (2.0 * coverage * precision / (coverage + precision)) if coverage + precision else 0.0
        scores_by_case_id[case_id] = {
            "coverage": round(coverage, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        }
    return scores_by_case_id


def percentile(values: list[float], ratio: float) -> float:
    """按离散样本近似计算分位数，用于汇总延迟指标。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio))))
    return float(ordered[index])


def evenly_spaced_case_ids(case_ids: Sequence[str], limit: int) -> list[str]:
    """从完整样例序列中按均匀间隔抽取一批代表性 case id。

    这主要用于 RAGAS 子评测，避免每次都只取头部样本，导致评测样本分布失真。
    """
    if limit <= 0 or len(case_ids) <= limit:
        return list(case_ids)
    step = len(case_ids) / limit
    return [case_ids[min(len(case_ids) - 1, int(index * step))] for index in range(limit)]


def _load_ragas_wrappers():
    """延迟加载 RAGAS 评测所需的 LlamaIndex 兼容封装。

    这里把导入和包装逻辑单独收口，是为了避免普通指标评测在不需要 RAGAS 时
    也强依赖整套额外组件，同时降低导入失败对主评测流程的影响。
    """
    module_prefix = "_".join(("llama", "index"))
    embedding_module = importlib.import_module(f"{module_prefix}.embeddings.ollama")
    llm_module = importlib.import_module(f"{module_prefix}.llms.ollama")
    ragas_embeddings = importlib.import_module("ragas.embeddings")
    ragas_llms = importlib.import_module("ragas.llms")

    embedding_cls = getattr(embedding_module, "OllamaEmbedding")
    llm_cls = getattr(llm_module, "Ollama")
    embedding_wrapper_cls = getattr(ragas_embeddings, "Llama" + "IndexEmbeddingsWrapper")
    llm_wrapper_cls = getattr(ragas_llms, "Llama" + "IndexLLMWrapper")

    llm = llm_wrapper_cls(
        llm_cls(
            model="qwen2.5:7b-instruct",
            temperature=0.0,
            base_url="http://127.0.0.1:11434",
            request_timeout=180.0,
        )
    )
    embeddings = embedding_wrapper_cls(
        embedding_cls(
            model_name="bge-m3:latest",
            base_url="http://127.0.0.1:11434",
        )
    )
    return llm, embeddings


def ragas_summary(
    rows: Sequence[dict[str, str]],
    *,
    ordered_case_ids: Sequence[str],
    enable_ragas: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """执行 RAGAS 评测，并兼容批量失败后的逐题降级统计。

    如果整批 RAGAS 调用失败，这里会自动回退到逐题评测，尽量保住可用结果，
    而不是让整组实验完全报废。
    """
    if not enable_ragas or not rows:
        return {
            "faithfulness": 0.0,
            "answer_correctness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "ragas_sample_count": 0,
        }, []

    llm, embeddings = _load_ragas_wrappers()
    run_config = RunConfig(timeout=180, max_retries=2, max_wait=10, max_workers=1)
    dataset = EvaluationDataset.from_list(list(rows))
    metrics = [Faithfulness(), AnswerCorrectness(), AnswerRelevancy(), ContextPrecision()]
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=False,
            show_progress=False,
            batch_size=1,
        )
        summary = {
            "faithfulness": round(float(result._repr_dict["faithfulness"]), 4),
            "answer_correctness": round(float(result._repr_dict["answer_correctness"]), 4),
            "answer_relevancy": round(float(result._repr_dict["answer_relevancy"]), 4),
            "context_precision": round(float(result._repr_dict["context_precision"]), 4),
            "ragas_sample_count": len(rows),
        }
        details = []
        for case_id, score_row in zip(ordered_case_ids, result.scores):
            details.append(
                {
                    "case_id": case_id,
                    "faithfulness": round(float(score_row.get("faithfulness", 0.0) or 0.0), 4),
                    "answer_correctness": round(float(score_row.get("answer_correctness", 0.0) or 0.0), 4),
                    "answer_relevancy": round(float(score_row.get("answer_relevancy", 0.0) or 0.0), 4),
                    "context_precision": round(float(score_row.get("context_precision", 0.0) or 0.0), 4),
                }
            )
        return summary, details
    except Exception as exc:
        print(f"[RAGAS] batch evaluation degraded: {exc}", file=sys.stderr, flush=True)

    detail_rows: list[dict[str, Any]] = []
    faithfulness_scores: list[float] = []
    answer_relevancy_scores: list[float] = []
    answer_correctness_scores: list[float] = []
    context_precision_scores: list[float] = []
    valid_count = 0
    for case_id, row in zip(ordered_case_ids, rows):
        try:
            single = evaluate(
                dataset=EvaluationDataset.from_list([row]),
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                run_config=run_config,
                raise_exceptions=False,
                show_progress=False,
                batch_size=1,
            )
            score_row = single.scores[0] if single.scores else {}
        except Exception as exc:
            print(f"[RAGAS] case failed ({case_id}): {exc}", file=sys.stderr, flush=True)
            score_row = {}
        faithfulness = score_row.get("faithfulness")
        answer_relevancy = score_row.get("answer_relevancy")
        answer_correctness = score_row.get("answer_correctness")
        context_precision = score_row.get("context_precision")
        detail_rows.append(
            {
                "case_id": case_id,
                "faithfulness": round(float(faithfulness), 4) if isinstance(faithfulness, (int, float)) and math.isfinite(float(faithfulness)) else "",
                "answer_correctness": round(float(answer_correctness), 4) if isinstance(answer_correctness, (int, float)) and math.isfinite(float(answer_correctness)) else "",
                "answer_relevancy": round(float(answer_relevancy), 4) if isinstance(answer_relevancy, (int, float)) and math.isfinite(float(answer_relevancy)) else "",
                "context_precision": round(float(context_precision), 4) if isinstance(context_precision, (int, float)) and math.isfinite(float(context_precision)) else "",
            }
        )
        if isinstance(faithfulness, (int, float)) and math.isfinite(float(faithfulness)):
            faithfulness_scores.append(float(faithfulness))
        if isinstance(answer_relevancy, (int, float)) and math.isfinite(float(answer_relevancy)):
            answer_relevancy_scores.append(float(answer_relevancy))
        if isinstance(answer_correctness, (int, float)) and math.isfinite(float(answer_correctness)):
            answer_correctness_scores.append(float(answer_correctness))
        if isinstance(context_precision, (int, float)) and math.isfinite(float(context_precision)):
            context_precision_scores.append(float(context_precision))
        if any(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (faithfulness, answer_correctness, answer_relevancy, context_precision)):
            valid_count += 1

    return {
        "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4) if faithfulness_scores else 0.0,
        "answer_correctness": round(sum(answer_correctness_scores) / len(answer_correctness_scores), 4) if answer_correctness_scores else 0.0,
        "answer_relevancy": round(sum(answer_relevancy_scores) / len(answer_relevancy_scores), 4) if answer_relevancy_scores else 0.0,
        "context_precision": round(sum(context_precision_scores) / len(context_precision_scores), 4) if context_precision_scores else 0.0,
        "ragas_sample_count": valid_count,
    }, detail_rows


@dataclass
class DatasetEvaluation:
    summary: dict[str, Any]
    detail_rows: list[dict[str, Any]]
    ragas_rows: list[dict[str, Any]]


def evaluate_rgb_results(
    variant_key: str,
    results: Sequence[PipelineRunResult],
    cases: Sequence[ExperimentCase],
    *,
    ragas_case_ids: Sequence[str],
    enable_ragas: bool,
) -> DatasetEvaluation:
    """汇总 RGB 任务结果，生成命中率、延迟和 RAGAS 指标。"""
    case_map = {case.case_id: case for case in cases}
    detail_rows: list[dict[str, Any]] = []
    positive_results = 0
    accuracy_hits = 0
    retrieval_hits_at_1 = 0
    retrieval_hits = 0
    latency_values: list[float] = []
    ragas_inputs: list[dict[str, str]] = []
    ragas_order: list[str] = []

    for result in results:
        case = case_map[result.case_id]
        accuracy_hit = int(match_any_answer(result.response, case.answers))
        retrieval_hit_at_1 = int(retrieval_hits_positive_texts(result.retrieved_contexts[:1], case.positive_texts))
        retrieval_hit = int(retrieval_hits_positive_texts(result.retrieved_contexts, case.positive_texts))
        positive_results += 1
        accuracy_hits += accuracy_hit
        retrieval_hits_at_1 += retrieval_hit_at_1
        retrieval_hits += retrieval_hit
        latency_values.append(result.latency_ms)
        detail_rows.append(
            {
                "variant": variant_key,
                "dataset": "rgb",
                "case_id": result.case_id,
                "query": result.query,
                "response": result.response,
                "route_mode": result.route_mode,
                "complexity_score": round(result.complexity_score, 4),
                "distilled_fact_count": result.distilled_fact_count,
                "support_sentence_count": result.support_sentence_count,
                "accuracy_hit": accuracy_hit,
                "retrieval_hit_at_1": retrieval_hit_at_1,
                "retrieval_hit_at_3": retrieval_hit,
                "predicted_refusal": int(result.predicted_refusal),
                "latency_ms": round(result.latency_ms, 2),
                "source_titles": " | ".join(result.source_titles),
            }
        )
        if result.case_id in ragas_case_ids:
            ragas_inputs.append(
                {
                    "user_input": case.query,
                    "response": result.response,
                    "retrieved_contexts": result.retrieved_contexts,
                    "reference": case.reference,
                }
            )
            ragas_order.append(result.case_id)

    ragas_metrics, ragas_rows = ragas_summary(ragas_inputs, ordered_case_ids=ragas_order, enable_ragas=enable_ragas)
    summary = {
        "variant": variant_key,
        "dataset": "rgb",
        **ragas_metrics,
        "accuracy": round(accuracy_hits / positive_results, 4) if positive_results else 0.0,
        "retrieval_hit_rate_at_1": round(retrieval_hits_at_1 / positive_results, 4) if positive_results else 0.0,
        "retrieval_hit_rate_at_3": round(retrieval_hits / positive_results, 4) if positive_results else 0.0,
        "latency_p50_ms": round(percentile(latency_values, 0.50), 2),
        "latency_p95_ms": round(percentile(latency_values, 0.95), 2),
        "sample_count": len(results),
    }
    return DatasetEvaluation(summary=summary, detail_rows=detail_rows, ragas_rows=ragas_rows)


def evaluate_crud_results(
    variant_key: str,
    results: Sequence[PipelineRunResult],
    cases: Sequence[ExperimentCase],
    *,
    ragas_case_ids: Sequence[str],
    qa_ragas_case_ids: Sequence[str],
    multidoc_ragas_case_ids: Sequence[str],
    enable_ragas: bool,
    semantic_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> DatasetEvaluation:
    """汇总 CRUD 任务结果，生成论文使用的复杂问答指标表。

    这里会同时计算准确率、检索命中率、多文档整合质量、拒答表现和延迟，
    是论文第三章表格的主要来源。
    """
    case_map = {case.case_id: case for case in cases}
    multidoc_focus_by_case_id = multidoc_focus_scores(results, cases, model_name=semantic_model_name)
    detail_rows: list[dict[str, Any]] = []
    latency_values: list[float] = []
    ragas_inputs: list[dict[str, str]] = []
    ragas_order: list[str] = []
    qa_ragas_inputs: list[dict[str, str]] = []
    qa_ragas_order: list[str] = []
    multidoc_ragas_inputs: list[dict[str, str]] = []
    multidoc_ragas_order: list[str] = []

    noise_scores: list[float] = []
    negative_hits = 0
    negative_count = 0
    integration_hits = 0
    integration_count = 0
    integration_string_similarity_scores: list[float] = []
    integration_semantic_similarity_scores: list[float] = []
    integration_focus_coverages: list[float] = []
    integration_focus_precisions: list[float] = []
    integration_focus_f1_scores: list[float] = []
    integration_quality_scores: list[float] = []
    accuracy_hits = 0
    positive_count = 0
    qa_hits = 0
    qa_count = 0
    summary_string_similarity_scores: list[float] = []
    summary_semantic_similarity_scores: list[float] = []
    qa_string_similarity_scores: list[float] = []
    qa_semantic_similarity_scores: list[float] = []
    overall_string_similarity_scores: list[float] = []
    overall_semantic_similarity_scores: list[float] = []
    complex_string_similarity_scores: list[float] = []
    complex_semantic_similarity_scores: list[float] = []
    complex_quality_scores: list[float] = []
    retrieval_hits_at_1 = 0
    retrieval_hits = 0

    semantic_case_ids: list[str] = []
    semantic_predictions: list[str] = []
    semantic_answer_sets: list[Sequence[str]] = []
    for result in results:
        case = case_map[result.case_id]
        if case.answers:
            semantic_case_ids.append(result.case_id)
            semantic_predictions.append(result.response)
            semantic_answer_sets.append(case.answers)
    semantic_similarity_by_case_id = {
        case_id: score
        for case_id, score in zip(
            semantic_case_ids,
            semantic_similarity_scores(
                semantic_predictions,
                semantic_answer_sets,
                model_name=semantic_model_name,
            ),
        )
    }

    for result in results:
        case = case_map[result.case_id]
        accuracy_hit = int(match_any_answer(result.response, case.answers)) if case.answers else 0
        retrieval_hit = int(any(doc_id in case.expected_doc_ids for doc_id in result.source_doc_ids)) if case.expected_doc_ids else 0
        retrieval_hit_at_1 = int(bool(result.source_doc_ids) and bool(case.expected_doc_ids) and result.source_doc_ids[0] in case.expected_doc_ids)
        string_similarity = max((text_similarity(result.response, reference) for reference in case.answers), default=0.0)
        semantic_similarity = semantic_similarity_by_case_id.get(result.case_id, 0.0)
        focus_scores = multidoc_focus_by_case_id.get(result.case_id, {})
        focus_f1 = float(focus_scores.get("f1", 0.0) or 0.0)
        if case.split in {"questanswer_2docs", "questanswer_3docs"}:
            quality_score = integration_quality_score(semantic_similarity, string_similarity, focus_f1)
        else:
            quality_score = summary_quality_score(semantic_similarity, string_similarity)
        latency_values.append(result.latency_ms)
        detail_rows.append(
            {
                "variant": variant_key,
                "dataset": "crud",
                "case_id": result.case_id,
                "split": case.split,
                "query": result.query,
                "response": result.response,
                "route_mode": result.route_mode,
                "complexity_score": round(result.complexity_score, 4),
                "distilled_fact_count": result.distilled_fact_count,
                "support_sentence_count": result.support_sentence_count,
                "accuracy_hit": accuracy_hit,
                "retrieval_hit_at_1": retrieval_hit_at_1,
                "retrieval_hit": retrieval_hit,
                "text_similarity": round(string_similarity, 4),
                "semantic_similarity": round(semantic_similarity, 4),
                "quality_score": round(quality_score, 4),
                "multidoc_focus_coverage": focus_scores.get("coverage", ""),
                "multidoc_focus_precision": focus_scores.get("precision", ""),
                "multidoc_focus_f1": focus_scores.get("f1", ""),
                "predicted_refusal": int(result.predicted_refusal),
                "expected_refusal": int(case.should_refuse),
                "latency_ms": round(result.latency_ms, 2),
                "source_titles": " | ".join(result.source_titles),
            }
        )
        if not case.should_refuse:
            positive_count += 1
            accuracy_hits += accuracy_hit
            retrieval_hits_at_1 += retrieval_hit_at_1
            retrieval_hits += retrieval_hit
            if case.split != "hallu_modified":
                overall_string_similarity_scores.append(string_similarity)
                overall_semantic_similarity_scores.append(semantic_similarity)
        if case.split.startswith("questanswer_"):
            qa_count += 1
            qa_hits += accuracy_hit
            qa_string_similarity_scores.append(string_similarity)
            qa_semantic_similarity_scores.append(semantic_similarity)
        if case.split == "event_summary":
            summary_string_similarity_scores.append(string_similarity)
            summary_semantic_similarity_scores.append(semantic_similarity)
            complex_string_similarity_scores.append(string_similarity)
            complex_semantic_similarity_scores.append(semantic_similarity)
            complex_quality_scores.append(quality_score)
        if case.split == "hallu_modified":
            noise_scores.append(string_similarity)
        if case.split == "negative_rejection":
            negative_count += 1
            negative_hits += int(result.predicted_refusal)
        if case.split in {"questanswer_2docs", "questanswer_3docs"}:
            integration_count += 1
            integration_hits += accuracy_hit
            integration_string_similarity_scores.append(string_similarity)
            integration_semantic_similarity_scores.append(semantic_similarity)
            if focus_scores:
                integration_focus_coverages.append(float(focus_scores["coverage"]))
                integration_focus_precisions.append(float(focus_scores["precision"]))
                integration_focus_f1_scores.append(focus_f1)
            else:
                integration_focus_f1_scores.append(0.0)
            integration_quality_scores.append(quality_score)
            complex_string_similarity_scores.append(string_similarity)
            complex_semantic_similarity_scores.append(semantic_similarity)
            complex_quality_scores.append(quality_score)
        if result.case_id in ragas_case_ids:
            ragas_inputs.append(
                {
                    "user_input": case.query,
                    "response": result.response,
                    "retrieved_contexts": result.retrieved_contexts,
                    "reference": case.reference,
                }
            )
            ragas_order.append(result.case_id)
        if result.case_id in qa_ragas_case_ids:
            qa_ragas_inputs.append(
                {
                    "user_input": case.query,
                    "response": result.response,
                    "retrieved_contexts": result.retrieved_contexts,
                    "reference": case.reference,
                }
            )
            qa_ragas_order.append(result.case_id)
        if result.case_id in multidoc_ragas_case_ids:
            multidoc_ragas_inputs.append(
                {
                    "user_input": case.query,
                    "response": result.response,
                    "retrieved_contexts": result.retrieved_contexts,
                    "reference": case.reference,
                }
            )
            multidoc_ragas_order.append(result.case_id)

    ragas_metrics, ragas_rows = ragas_summary(ragas_inputs, ordered_case_ids=ragas_order, enable_ragas=enable_ragas)
    qa_ragas_metrics, _ = ragas_summary(qa_ragas_inputs, ordered_case_ids=qa_ragas_order, enable_ragas=enable_ragas)
    multidoc_ragas_metrics, _ = ragas_summary(
        multidoc_ragas_inputs,
        ordered_case_ids=multidoc_ragas_order,
        enable_ragas=enable_ragas,
    )
    summary = {
        "variant": variant_key,
        "dataset": "crud",
        **ragas_metrics,
        "accuracy": round(accuracy_hits / positive_count, 4) if positive_count else 0.0,
        "qa_accuracy": round(qa_hits / qa_count, 4) if qa_count else 0.0,
        "retrieval_hit_rate_at_1": round(retrieval_hits_at_1 / positive_count, 4) if positive_count else 0.0,
        "retrieval_hit_rate_at_3": round(retrieval_hits / positive_count, 4) if positive_count else 0.0,
        "qa_faithfulness": qa_ragas_metrics["faithfulness"],
        "qa_answer_correctness": qa_ragas_metrics["answer_correctness"],
        "qa_answer_relevancy": qa_ragas_metrics["answer_relevancy"],
        "qa_context_precision": qa_ragas_metrics["context_precision"],
        "qa_ragas_sample_count": qa_ragas_metrics["ragas_sample_count"],
        "qa_similarity": round(sum(qa_semantic_similarity_scores) / len(qa_semantic_similarity_scores), 4) if qa_semantic_similarity_scores else 0.0,
        "qa_string_similarity": round(sum(qa_string_similarity_scores) / len(qa_string_similarity_scores), 4) if qa_string_similarity_scores else 0.0,
        "multidoc_faithfulness": multidoc_ragas_metrics["faithfulness"],
        "multidoc_answer_correctness": multidoc_ragas_metrics["answer_correctness"],
        "multidoc_answer_relevancy": multidoc_ragas_metrics["answer_relevancy"],
        "multidoc_context_precision": multidoc_ragas_metrics["context_precision"],
        "multidoc_ragas_sample_count": multidoc_ragas_metrics["ragas_sample_count"],
        "overall_similarity": round(sum(overall_semantic_similarity_scores) / len(overall_semantic_similarity_scores), 4) if overall_semantic_similarity_scores else 0.0,
        "overall_string_similarity": round(sum(overall_string_similarity_scores) / len(overall_string_similarity_scores), 4) if overall_string_similarity_scores else 0.0,
        "summary_similarity": round(sum(summary_semantic_similarity_scores) / len(summary_semantic_similarity_scores), 4) if summary_semantic_similarity_scores else 0.0,
        "summary_string_similarity": round(sum(summary_string_similarity_scores) / len(summary_string_similarity_scores), 4) if summary_string_similarity_scores else 0.0,
        "noise_robustness": round(sum(noise_scores) / len(noise_scores), 4) if noise_scores else 0.0,
        "negative_rejection": round(negative_hits / negative_count, 4) if negative_count else 0.0,
        "information_integration": round(integration_hits / integration_count, 4) if integration_count else 0.0,
        "integration_similarity": round(sum(integration_semantic_similarity_scores) / len(integration_semantic_similarity_scores), 4) if integration_semantic_similarity_scores else 0.0,
        "integration_string_similarity": round(sum(integration_string_similarity_scores) / len(integration_string_similarity_scores), 4) if integration_string_similarity_scores else 0.0,
        "integration_focus_coverage": round(sum(integration_focus_coverages) / len(integration_focus_coverages), 4) if integration_focus_coverages else 0.0,
        "integration_focus_precision": round(sum(integration_focus_precisions) / len(integration_focus_precisions), 4) if integration_focus_precisions else 0.0,
        "integration_focus_f1": round(sum(integration_focus_f1_scores) / len(integration_focus_f1_scores), 4) if integration_focus_f1_scores else 0.0,
        "integration_quality": round(sum(integration_quality_scores) / len(integration_quality_scores), 4) if integration_quality_scores else 0.0,
        "complex_similarity": round(sum(complex_semantic_similarity_scores) / len(complex_semantic_similarity_scores), 4) if complex_semantic_similarity_scores else 0.0,
        "complex_string_similarity": round(sum(complex_string_similarity_scores) / len(complex_string_similarity_scores), 4) if complex_string_similarity_scores else 0.0,
        "complex_quality": round(sum(complex_quality_scores) / len(complex_quality_scores), 4) if complex_quality_scores else 0.0,
        "latency_p50_ms": round(percentile(latency_values, 0.50), 2),
        "latency_p95_ms": round(percentile(latency_values, 0.95), 2),
        "sample_count": len(results),
    }
    return DatasetEvaluation(summary=summary, detail_rows=detail_rows, ragas_rows=ragas_rows)
