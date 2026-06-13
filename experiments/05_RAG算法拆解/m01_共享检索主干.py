"""论文 RAG 的共享检索主干。

这一层只回答一个问题：候选文档是怎么从“海量语料”一步步变成“可供后续证据选择的排序结果”的。
分项重排、覆盖取证、按题作答都建立在这一层输出之上。
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from m00_公共基础 import (
    QUERY_BATCH_SIZE,
    SNIPPET_MAX_SENTENCES,
    SNIPPET_WINDOW_CHARS,
    SNIPPET_WINDOW_STRIDE,
    clean_text,
    coverage_ratio,
    tokenize_text,
)


def reciprocal_rank_fusion(rankings: list[list[int]], *, top_k: int = 50, k: int = 60) -> dict[int, float]:
    """把 dense 和 BM25 等多路候选排序融合成一个统一分数。"""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_index in enumerate(ranking[:top_k], start=1):
            fused[doc_index] += 1.0 / (k + rank)
    return dict(fused)


def compute_dense_rankings(
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """批量计算稠密检索排序结果。

    这里的关键不是公式本身，而是它会自动根据环境切到 GPU 或 CPU，
    并按批处理查询向量，保证整批实验跑得动。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    doc_tensor = torch.tensor(doc_embeddings, dtype=torch.float32, device=device)
    effective_top_k = max(1, min(top_k, int(doc_embeddings.shape[0])))
    top_indices_batches: list[np.ndarray] = []
    top_scores_batches: list[np.ndarray] = []
    for start in range(0, len(query_embeddings), QUERY_BATCH_SIZE):
        query_tensor = torch.tensor(query_embeddings[start : start + QUERY_BATCH_SIZE], dtype=torch.float32, device=device)
        scores = query_tensor @ doc_tensor.T
        top_scores, top_indices = torch.topk(scores, k=effective_top_k, dim=1)
        top_indices_batches.append(top_indices.cpu().numpy())
        top_scores_batches.append(top_scores.cpu().numpy())
    return np.vstack(top_indices_batches), np.vstack(top_scores_batches)


def bm25_rank(query_tokens: list[str], bm25: BM25Okapi, *, top_k: int) -> tuple[list[int], list[float]]:
    """执行 BM25 召回，并返回文档下标和原始分数。"""
    scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float32)
    if len(scores) <= top_k:
        order = np.argsort(-scores)
    else:
        candidate = np.argpartition(-scores, top_k - 1)[:top_k]
        order = candidate[np.argsort(-scores[candidate])]
    return order.tolist(), scores[order].tolist()


def build_snippet_windows(text: str) -> list[str]:
    """把长文档切成适合精排的片段窗口。"""
    # 这里保留原文句子边界，方便后续把高相关局部片段送进 CrossEncoder 精排。
    cleaned = clean_text(text, limit=1600)
    if len(cleaned) <= SNIPPET_WINDOW_CHARS:
        return [cleaned]

    segments = [segment.strip() for segment in re.split(r"(?<=[。！？!?；;])|\n+", cleaned) if segment.strip()]
    windows: list[str] = []
    if segments:
        for start in range(len(segments)):
            current: list[str] = []
            current_length = 0
            for end in range(start, min(len(segments), start + SNIPPET_MAX_SENTENCES)):
                segment = segments[end]
                next_length = current_length + len(segment) + (1 if current else 0)
                if current and next_length > SNIPPET_WINDOW_CHARS:
                    break
                current.append(segment)
                current_length = next_length
                if current_length >= 80:
                    windows.append(" ".join(current))

    for start in range(0, len(cleaned), SNIPPET_WINDOW_STRIDE):
        window = cleaned[start : start + SNIPPET_WINDOW_CHARS]
        if window:
            windows.append(window)
        if start + SNIPPET_WINDOW_CHARS >= len(cleaned):
            break

    deduped: list[str] = []
    seen: set[str] = set()
    for window in windows:
        normalized = window.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped or [cleaned[:SNIPPET_WINDOW_CHARS]]


def score_snippet_window(
    query_tokens: list[str],
    focus_tokens: list[str],
    window: str,
    *,
    index: int,
    total_windows: int,
) -> float:
    """给一个片段窗口打分，综合查询覆盖、焦点覆盖、精确命中和位置权重。"""
    query_token_set = set(query_tokens)
    focus_token_set = set(focus_tokens)
    window_token_set = set(tokenize_text(window))
    coverage = coverage_ratio(query_token_set, window_token_set)
    focus_coverage = coverage_ratio(focus_token_set, window_token_set)
    exact_hits = 0
    lowered_window = window.lower()
    for token in focus_token_set:
        if len(token) >= 2 and token in lowered_window:
            exact_hits += 1
    position_bonus = max(0.0, 1.0 - (index / max(total_windows, 1))) * 0.08
    return coverage * 0.95 + focus_coverage * 1.15 + min(exact_hits, 4) * 0.08 + position_bonus


def rank_query_focused_snippets(query_tokens: list[str], focus_tokens: list[str], text: str) -> list[str]:
    """把长文档中的窗口按“更贴近问题”的程度排序。"""
    windows = build_snippet_windows(text)
    if len(windows) == 1:
        return windows
    scored_windows = [
        (
            score_snippet_window(query_tokens, focus_tokens, window, index=index, total_windows=len(windows)),
            index,
            window,
        )
        for index, window in enumerate(windows)
    ]
    scored_windows.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [window for _score, _index, window in scored_windows]


def rerank_documents(
    reranker: CrossEncoder,
    *,
    query: str,
    query_tokens: list[str],
    focus_tokens: list[str],
    candidate_doc_indices: list[int],
    corpus_texts: list[str],
    multi_snippet_count: int,
    aggregation_mode: str = "default",
    aspect_token_sets: Sequence[set[str]] | None = None,
) -> tuple[list[int], dict[int, list[str]], dict[int, float]]:
    """对候选文档做 CrossEncoder 精排，并保留每篇文档最有代表性的片段。

    这一层是共享主干的关键，因为论文后面的分项重排和覆盖取证，
    都建立在这里已经“先按相关性排好”之后。
    """
    if not candidate_doc_indices:
        return [], {}, {}

    query_token_set = set(query_tokens)
    effective_aspect_token_sets = [set(tokens) for tokens in (aspect_token_sets or []) if tokens]
    pair_doc_indices: list[int] = []
    pairs: list[tuple[str, str]] = []
    doc_snippets: dict[int, list[str]] = {}
    snippet_records: list[dict[str, Any]] = []

    # 先把每篇文档切成若干高相关片段，再把“问题-片段”喂给重排模型。
    for doc_index in candidate_doc_indices:
        ranked_snippets = rank_query_focused_snippets(query_tokens, focus_tokens, corpus_texts[doc_index])
        selected_snippets = ranked_snippets[: max(1, multi_snippet_count)]
        doc_snippets[int(doc_index)] = selected_snippets
        for snippet in selected_snippets:
            snippet_tokens = set(tokenize_text(snippet))
            base_coverage = coverage_ratio(query_token_set, snippet_tokens)
            detail_signal = 1.0 if re.search(r"\d|[%年月日:：\-]|[A-Z]{2,}", snippet) else 0.0
            covered_aspect_ids = {
                aspect_index
                for aspect_index, aspect_tokens in enumerate(effective_aspect_token_sets)
                if coverage_ratio(aspect_tokens, snippet_tokens) >= 0.14
            }
            pair_doc_indices.append(int(doc_index))
            pairs.append((query, snippet))
            snippet_records.append(
                {
                    "doc_index": int(doc_index),
                    "snippet": snippet,
                    "tokens": snippet_tokens,
                    "base_coverage": base_coverage,
                    "detail_signal": detail_signal,
                    "covered_aspect_ids": covered_aspect_ids,
                }
            )

    raw_scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
    if aggregation_mode in {"aspect_aware", "aspect_aware_conservative"} and len(effective_aspect_token_sets) >= 2:
        grouped_records: dict[int, list[dict[str, Any]]] = defaultdict(list)
        is_conservative = aggregation_mode == "aspect_aware_conservative"
        clause_threshold = 0.22 if is_conservative else 0.14
        for record, raw_score in zip(snippet_records, raw_scores.tolist()):
            ce_score = 1.0 / (1.0 + math.exp(-float(raw_score)))
            covered_aspect_ids = set(record["covered_aspect_ids"])
            aspect_ratio = len(covered_aspect_ids) / len(effective_aspect_token_sets)
            if is_conservative:
                covered_aspect_ids = {
                    aspect_index
                    for aspect_index, aspect_tokens in enumerate(effective_aspect_token_sets)
                    if coverage_ratio(aspect_tokens, record["tokens"]) >= clause_threshold
                }
                aspect_ratio = len(covered_aspect_ids) / len(effective_aspect_token_sets)
                blended_score = (
                    0.80 * ce_score
                    + 0.10 * float(record["base_coverage"])
                    + 0.06 * aspect_ratio
                    + 0.04 * float(record["detail_signal"])
                )
            else:
                blended_score = (
                    0.68 * ce_score
                    + 0.14 * float(record["base_coverage"])
                    + 0.12 * aspect_ratio
                    + 0.06 * float(record["detail_signal"])
                )
            grouped_records[int(record["doc_index"])].append(
                {
                    **record,
                    "covered_aspect_ids": covered_aspect_ids,
                    "ce_score": ce_score,
                    "blended_score": blended_score,
                }
            )
        doc_scores: dict[int, float] = {}
        for doc_index, records in grouped_records.items():
            records.sort(key=lambda item: item["blended_score"], reverse=True)
            top_score = records[0]["blended_score"]
            if len(records) > 1:
                top_score = top_score + 0.12 * sum(record["blended_score"] for record in records[1:]) / (len(records) - 1)
            doc_scores[doc_index] = float(top_score)
            doc_snippets[doc_index] = [record["snippet"] for record in records[: max(1, multi_snippet_count)]]
    else:
        doc_scores = {}
        for doc_index, score in zip(pair_doc_indices, raw_scores.tolist()):
            doc_scores[doc_index] = max(doc_scores.get(doc_index, float("-inf")), float(score))
        for doc_index in list(doc_snippets.keys()):
            doc_snippets[doc_index] = doc_snippets[doc_index][: max(1, multi_snippet_count)]

    ordered_doc_indices = [doc_index for doc_index, _ in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)]
    return ordered_doc_indices, doc_snippets, doc_scores
