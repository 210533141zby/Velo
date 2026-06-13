"""论文中“分项重排”的独立阅读版实现。

这一层只回答一个问题：在共享检索主干已经完成 `Rerank` 之后，
怎样把更贴近题目多个要求的候选文档往前提，再把这种排序偏好传递给后续证据阶段。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

from m00_公共基础 import clean_text, coverage_ratio, normalize_text, tokenize_text

# 论文公式 2-1（分项重排）：
# S_reorder(d) = 0.70 × R_base(d) + 0.16 × A(d) + 0.10 × C(d) + 0.04 × D(d)
#
# 在这份阅读版代码中的对应关系如下：
# 1. R_base(d): 上游共享检索主干经过 Rerank 后的基础相关性分数，
#    在代码里体现在 `build_document_reorder_breakdowns()` 里的 `rerank_score_norm`；
# 2. A(d): 文档对问题多个语义分项的覆盖情况，
#    在代码里体现在 `aspect_coverages` 与 `aspect_coverage`；
# 3. C(d): 文档对整道问题整体词项的覆盖程度，
#    在代码里体现在 `overall_coverage`；
# 4. D(d): 时间、数字、专名、范围限定等高价值细节信号，
#    在代码里体现在 `detail_signal`。
#
# 这里保留论文公式原样，是为了答辩时可以直接在代码里定位论文口径；
# 下面的实现属于“把公式思想拆成可执行步骤”的落地版本。
S_REORDER_PAPER_FORMULA = "S_reorder(d) = 0.70 × R_base(d) + 0.16 × A(d) + 0.10 × C(d) + 0.04 × D(d)"
_DETAIL_SIGNAL_PATTERN = re.compile(
    r"(?:约|近|逾|超|超过|至少|不足|将近)?\d+(?:\.\d+)?"
    r"(?:万|亿|千|百)?(?:%|％|年|月|日|元|美元|亿元|亿美元|人|名|个|件|项|次|架|枚|门|套|公里|米|吨|级|号)?"
    r"|[A-Z]{2,}[A-Za-z0-9\-:/]*"
)
_ASPECT_HIT_THRESHOLD = 0.18


@dataclass
class PaperReorderBreakdown:
    """论文公式 2-1 的单篇候选文档分解结果。

    0.70*精排结果具体落在 `rerank_score_norm` 与 `reorder_score` 即可。
    """

    doc_index: int
    doc_id: int
    title: str
    rerank_score_raw: float
    rerank_score_norm: float
    aspect_clauses: tuple[str, ...]
    aspect_coverages: tuple[float, ...]
    aspect_coverage: float
    overall_coverage: float
    detail_signal: float
    reorder_score: float
    matched_aspects: tuple[str, ...]
    representative_windows: tuple[str, ...]


def _compute_detail_signal(text: str) -> float:
    """计算细节信号强度，对应公式中的 D(d)。"""
    detail_hits = _DETAIL_SIGNAL_PATTERN.findall(text)
    return min(len(detail_hits), 3) / 3.0


def _min_max_normalize(values: Sequence[float]) -> list[float]:
    """把一组候选分数压到 0 到 1，便于和覆盖类特征放到同一公式里。"""
    if not values:
        return []
    lower = min(values)
    upper = max(values)
    if upper - lower <= 1e-8:
        return [1.0 if upper > 0.0 else 0.0 for _ in values]
    return [(value - lower) / (upper - lower) for value in values]


def build_document_reorder_breakdowns(
    query: str,
    reranked_doc_indices: Sequence[int],
    doc_snippets: dict[int, list[str]],
    doc_scores: dict[int, float],
    docs: Sequence[Any],
) -> list[PaperReorderBreakdown]:
    """把论文公式 2-1 拆成可执行、可讲解的文档级特征。

    这一层是 `05_RAG算法拆解` 里最直接对应论文分项重排公式的位置：
    1. 先承接共享检索主干给出的 `Rerank` 基础分；
    2. 再按窗口计算分项覆盖、整题覆盖和细节信号；
    3. 最后按论文权重合成为 `S_reorder(d)`。

    返回值不是简单分数列表，而是每篇文档一份完整分解，
    解释 `R_base(d)`、`A(d)`、`C(d)`、`D(d)` 的含义。
    """

    if not reranked_doc_indices:
        return []

    query_tokens = tokenize_text(query)
    query_token_set = set(query_tokens)
    aspect_clauses = extract_query_aspect_clauses(query, max_clauses=4)
    aspect_token_sets = [set(tokenize_text(clause)) for clause in aspect_clauses if tokenize_text(clause)]
    raw_rerank_scores = [float(doc_scores.get(doc_index, 0.0)) for doc_index in reranked_doc_indices]
    normalized_rerank_scores = _min_max_normalize(raw_rerank_scores)

    breakdowns: list[PaperReorderBreakdown] = []
    for order_index, doc_index in enumerate(reranked_doc_indices):
        doc = docs[doc_index]
        representative_windows = tuple(
            window
            for window in (doc_snippets.get(doc_index) or [])
            if clean_text(window, limit=360)
        ) or (clean_text(str(getattr(doc, "content", "") or ""), limit=360),)

        best_clause_coverages = [0.0] * len(aspect_token_sets)
        best_overall_coverage = 0.0
        best_detail_signal = 0.0

        # 论文里强调“不是直接拿整篇文档打分，而是先看局部窗口的覆盖情况”，
        # 因此这里逐窗口累积每篇文档最有代表性的覆盖信号。
        for window in representative_windows:
            window_tokens = set(tokenize_text(window))
            if not window_tokens:
                continue
            best_overall_coverage = max(best_overall_coverage, coverage_ratio(query_token_set, window_tokens))
            best_detail_signal = max(best_detail_signal, _compute_detail_signal(window))
            for aspect_index, aspect_tokens in enumerate(aspect_token_sets):
                best_clause_coverages[aspect_index] = max(
                    best_clause_coverages[aspect_index],
                    coverage_ratio(aspect_tokens, window_tokens),
                )

        if best_clause_coverages:
            # A(d) 不是“命中一个分项就算高分”，而是看多个分项总体覆盖得如何。
            aspect_coverage = sum(best_clause_coverages) / len(best_clause_coverages)
        else:
            aspect_coverage = best_overall_coverage

        rerank_score_norm = float(normalized_rerank_scores[order_index])
        reorder_score = (
            0.70 * rerank_score_norm
            + 0.16 * aspect_coverage
            + 0.10 * best_overall_coverage
            + 0.04 * best_detail_signal
        )
        matched_aspects = tuple(
            clause
            for clause, clause_coverage in zip(aspect_clauses, best_clause_coverages)
            if clause_coverage >= _ASPECT_HIT_THRESHOLD
        )
        breakdowns.append(
            PaperReorderBreakdown(
                doc_index=int(doc_index),
                doc_id=int(getattr(doc, "doc_id", doc_index)),
                title=str(getattr(doc, "title", f"doc_{doc_index}")),
                rerank_score_raw=float(doc_scores.get(doc_index, 0.0)),
                rerank_score_norm=rerank_score_norm,
                aspect_clauses=tuple(aspect_clauses),
                aspect_coverages=tuple(float(value) for value in best_clause_coverages),
                aspect_coverage=float(aspect_coverage),
                overall_coverage=float(best_overall_coverage),
                detail_signal=float(best_detail_signal),
                reorder_score=float(reorder_score),
                matched_aspects=matched_aspects,
                representative_windows=representative_windows,
            )
        )

    breakdowns.sort(
        key=lambda item: (
            item.reorder_score,
            item.rerank_score_norm,
            item.aspect_coverage,
            item.overall_coverage,
        ),
        reverse=True,
    )
    return breakdowns


def rerank_documents_by_paper_formula(
    query: str,
    reranked_doc_indices: Sequence[int],
    doc_snippets: dict[int, list[str]],
    doc_scores: dict[int, float],
    docs: Sequence[Any],
) -> tuple[list[int], list[PaperReorderBreakdown]]:
    """按论文公式 2-1 重新排列候选文档。

    这一步就是 `Rerank -> 分项重排` 的直接连接点：
    上游 `Rerank` 已经给出基础顺序，这里在不推翻它的前提下，
    用 `0.70 * R_base(d)` 主导、其余覆盖与细节信号做修正。
    """

    breakdowns = build_document_reorder_breakdowns(
        query=query,
        reranked_doc_indices=reranked_doc_indices,
        doc_snippets=doc_snippets,
        doc_scores=doc_scores,
        docs=docs,
    )
    reordered_doc_indices = [item.doc_index for item in breakdowns]
    return reordered_doc_indices, breakdowns


def _token_jaccard(left: set[str], right: set[str]) -> float:
    """计算两个词集合的 Jaccard 相似度，用来衡量证据之间有多重复。"""
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def extract_query_aspect_clauses(query: str, *, max_clauses: int = 4) -> list[str]:
    """把复杂问题拆成若干有顺序的分项子句。

    这是“分项重排”的前置步骤。它不追求语言学上特别细的切句，
    目标只是把用户问题中可以独立回答的几个要求先拆出来，
    这样后面才能判断哪条证据更贴近整道题的多个要求。
    """
    cleaned = clean_text(query, limit=240)
    if not cleaned:
        return []

    primary_parts = [
        segment.strip(" ，,；;。！？? ")
        for segment in re.split(
            r"(?:以及|并且|同时|并(?:说明|指出|给出|列举|列出|比较|分析|描述|判断)|[，,]\s*且| and | as well as )",
            cleaned,
        )
        if segment.strip(" ，,；;。！？? ")
    ]
    clauses: list[str] = []
    for part in primary_parts:
        secondary_parts = [
            segment.strip(" ，,；;。！？? ")
            for segment in re.split(r"[，,；;]", part)
            if segment.strip(" ，,；;。！？? ")
        ]
        clauses.extend(secondary_parts if secondary_parts else [part])

    normalized: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        if len(clause) < 6:
            continue
        signature = normalize_text(clause)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        normalized.append(clause)
        if len(normalized) >= max_clauses:
            break
    return normalized


def order_evidence_units_for_query(
    query_tokens: Sequence[str],
    evidence_units: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把分项重排思想继续下沉到证据单元顺序上。

    如果老师问“论文公式 2-1 的直接代码落点在哪里”，
    应优先看 `rerank_documents_by_paper_formula()`。
    本函数解决的是下一层问题：文档顺序已经调好以后，
    候选证据单元内部还要不要继续按贴题程度做一次阅读顺序整理。

    它仍然保留同样的保守思想：
    `rerank_score` 不会被丢掉，只是在其基础上叠加覆盖和细节信号。
    """

    query_token_set = set(query_tokens)
    if not evidence_units:
        return []

    # 先把每条证据整理成后面打分要用的几个基础量。
    prepared_units: list[dict[str, Any]] = []
    for index, unit in enumerate(evidence_units):
        text = str(unit.get("text") or "")
        tokens = [token for token in (unit.get("tokens") or tokenize_text(text)) if normalize_text(token)]
        token_set = set(tokens)
        # coverage: 这条证据对整道问题关键词的覆盖程度，对应论文里的整体覆盖信号。
        coverage = coverage_ratio(query_token_set, token_set)
        # detail_signal: 时间、数字、专名、范围限定等高价值细节信号，对应公式中的 D(d) 思路。
        detail_signal = min(_compute_detail_signal(text), 1.0)
        prepared_units.append(
            {
                "index": index,
                "unit": unit,
                "tokens": token_set,
                "coverage": coverage,
                "detail_signal": detail_signal,
                # rerank_score: 上游 Rerank 给出的基础相关性分数，对应论文公式里的 R_base(d)。
                "rerank_score": float(unit.get("score", 0.0)),
            }
        )

    ordered: list[dict[str, Any]] = []
    selected_token_sets: list[set[str]] = []
    covered_query_tokens: set[str] = set()
    remaining = prepared_units[:]

    # 这里采用贪心式重排：先放最像在回答问题的证据，后面再补更互补的证据。
    while remaining:
        current_coverage = coverage_ratio(query_token_set, covered_query_tokens)
        best_item: dict[str, Any] | None = None
        best_score = float("-inf")
        for item in remaining:
            token_set = item["tokens"]
            # base_coverage: 当前证据单独看时，对整题的覆盖程度。
            base_coverage = float(item["coverage"])
            # expanded_coverage / coverage_gain: 把当前证据加入后，整题覆盖能增加多少。
            expanded_coverage = coverage_ratio(query_token_set, covered_query_tokens | token_set)
            coverage_gain = max(0.0, expanded_coverage - current_coverage)
            # redundancy: 这条证据和前面已选证据有多像，用于避免头部结果全是同一类信息。
            redundancy = max((_token_jaccard(token_set, selected) for selected in selected_token_sets), default=0.0)
            detail_signal = float(item["detail_signal"])
            rerank_score = float(item["rerank_score"])

            if not ordered:
                # 第一条证据更强调“像不像直接在回答问题”。
                score = 0.38 * base_coverage + 0.22 * rerank_score + 0.40 * detail_signal
            else:
                # 后续证据更强调“能不能补前面还没补到的信息，同时不要重复”。
                score = (
                    0.34 * coverage_gain
                    + 0.24 * base_coverage
                    + 0.22 * rerank_score
                    + 0.20 * detail_signal
                    - 0.18 * redundancy
                )

            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None:
            break
        ordered.append(best_item["unit"])
        selected_token_sets.append(best_item["tokens"])
        covered_query_tokens |= best_item["tokens"]
        remaining = [item for item in remaining if item is not best_item]

    return ordered


def order_evidence_units_for_query_clauses(
    query: str,
    evidence_units: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按问题分项重排证据单元。

    这是把“分项”显式写出来的一版辅助函数：
    先把问题拆成子句，再给每个子句挑一条较合适的证据，最后补回整题高分证据。
    它更适合答辩时展示“为什么复杂问题不会只围着一个分项打转”。
    """
    clauses = extract_query_aspect_clauses(query)
    if len(clauses) < 2:
        return order_evidence_units_for_query(tokenize_text(query), evidence_units)

    ordered: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    seen_doc_ids: set[int] = set()

    def append_if_new(unit: dict[str, Any]) -> bool:
        """只有证据未出现过时，才把它加入当前结果。"""
        signature = normalize_text(str(unit.get("text") or ""))
        if not signature or signature in seen_signatures:
            return False
        seen_signatures.add(signature)
        doc_id = int(unit.get("doc_id", 0) or 0)
        if doc_id:
            seen_doc_ids.add(doc_id)
        ordered.append(unit)
        return True

    for clause in clauses:
        # 这里的做法很直观：每个分项先挑一条相对最贴近它的证据，避免所有头部证据都围着同一分项转。
        clause_tokens = tokenize_text(clause)
        clause_order = order_evidence_units_for_query(clause_tokens, evidence_units)
        preferred = next(
            (
                unit
                for unit in clause_order
                if int(unit.get("doc_id", 0) or 0) not in seen_doc_ids
                and normalize_text(str(unit.get("text") or "")) not in seen_signatures
            ),
            None,
        )
        if preferred is None:
            preferred = next(
                (unit for unit in clause_order if normalize_text(str(unit.get("text") or "")) not in seen_signatures),
                None,
            )
        if preferred is not None:
            append_if_new(preferred)

    for unit in order_evidence_units_for_query(tokenize_text(query), evidence_units):
        append_if_new(unit)
    return ordered
