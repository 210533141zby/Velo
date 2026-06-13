"""论文中“覆盖取证”的独立阅读版实现。

这一层只回答一个问题：前面已经排好的证据里，最后究竟留下哪几条进入回答模型。
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from m00_公共基础 import coverage_ratio, normalize_text, tokenize_text
from m02_分项重排 import extract_query_aspect_clauses, order_evidence_units_for_query, _token_jaccard

# 论文公式 2-2（覆盖取证）：
# S_select(u) = 0.34 × G_res(u) + 0.24 × H_un(u) + 0.14 × H_str(u)
#             + 0.10 × C_q(u) + 0.10 × R(u) + 0.08 × D(u) + 0.08 × B_new(u)
#             - 0.16 × Red(u) - 0.08 × P_same(u)
#
# 
# 1. G_res(u) -> residual_gain
# 2. H_un(u)  -> uncovered_hit_ratio
# 3. H_str(u) -> strong_fill_ratio
# 4. C_q(u)   -> overall_coverage
# 5. R(u)     -> rerank_score
# 6. D(u)     -> detail_signal
# 7. B_new(u) -> new_doc_bonus
# 8. Red(u)   -> redundancy
# 9. P_same(u)-> same_doc_penalty
S_SELECT_PAPER_FORMULA = (
    "S_select(u) = 0.34 × G_res(u) + 0.24 × H_un(u) + 0.14 × H_str(u) + "
    "0.10 × C_q(u) + 0.10 × R(u) + 0.08 × D(u) + 0.08 × B_new(u) - "
    "0.16 × Red(u) - 0.08 × P_same(u)"
)


def order_evidence_units_for_query_clauses_residual_v2(
    query: str,
    evidence_units: Sequence[dict[str, Any]],
    *,
    limit: int,
    clause_hit_threshold: float = 0.18,
    clause_strong_threshold: float = 0.30,
) -> list[dict[str, Any]]:
    """按问题分项和剩余覆盖收益继续重排证据单元。

    这是论文最终采用的覆盖取证版本。它和分项重排的区别在于：
    分项重排解决“谁先排前面”，这里解决“最后该留下哪几条”。
    文件顶部的
    `S_SELECT_PAPER_FORMULA`，再看下面 `residual_gain`、`uncovered_hit_ratio`、
    `strong_fill_ratio`、`new_doc_bonus`、`redundancy` 等量是怎样被用起来的。
    """
    clauses = extract_query_aspect_clauses(query)
    if len(clauses) < 2 or limit <= 0:
        return order_evidence_units_for_query(tokenize_text(query), evidence_units)[: max(limit, 0)]

    query_tokens = tokenize_text(query)
    query_token_set = set(query_tokens)
    clause_token_sets = [set(tokenize_text(clause)) for clause in clauses]
    global_order = order_evidence_units_for_query(query_tokens, evidence_units)
    if not global_order:
        return []

    evidence_signatures = {
        normalize_text(str(unit.get("text") or "")): unit for unit in evidence_units if normalize_text(str(unit.get("text") or ""))
    }
    anchor_signature = normalize_text(str(global_order[0].get("text") or ""))
    anchor_unit = evidence_signatures.get(anchor_signature)
    if anchor_unit is None:
        anchor_unit = global_order[0]

    # 先把每条候选证据整理成覆盖取证要用的核心量。
    prepared_items: list[dict[str, Any]] = []
    for unit in evidence_units:
        text = str(unit.get("text") or "")
        tokens = set(unit.get("tokens") or tokenize_text(text))
        if not tokens:
            continue
        # clause_coverages: 当前证据分别覆盖了每个问题分项多少。
        clause_coverages = [coverage_ratio(clause_tokens, tokens) for clause_tokens in clause_token_sets]
        # overall_coverage: 当前证据对整道问题的整体覆盖。
        overall_coverage = coverage_ratio(query_token_set, tokens)
        detail_signal = min(
            len(
                re.findall(
                    r"(?:约|近|逾|超|超过|至少|不足|将近)?\d+(?:\.\d+)?(?:万|亿|千|百)?(?:%|％|年|月|日|元|美元|亿元|亿美元|人|名|个|件|项|次|架|枚|门|套|公里|米|吨|级|号)?|[A-Z]{2,}[A-Za-z0-9:/-]*",
                    text,
                )
            ),
            3,
        ) / 3.0
        # best_clause_index: 这条证据最偏向支持哪一个分项。
        best_clause_index = max(range(len(clause_coverages)), key=lambda index: clause_coverages[index])
        prepared_items.append(
            {
                "unit": unit,
                "signature": normalize_text(text),
                "tokens": tokens,
                "doc_id": int(unit.get("doc_id", 0) or 0),
                # rerank_score: 上游主干传下来的基础相关性分数，对应论文公式里的 R(u)。
                "rerank_score": float(unit.get("score", 0.0)),
                "overall_coverage": overall_coverage,
                # detail_signal: 证据里是否含有数字、时间、专名等高价值细节，对应论文公式里的 D(u)。
                "detail_signal": detail_signal,
                "clause_coverages": clause_coverages,
                "best_clause_index": best_clause_index,
            }
        )

    if not prepared_items:
        return []

    prepared_by_signature = {item["signature"]: item for item in prepared_items if item["signature"]}
    anchor_item = prepared_by_signature.get(anchor_signature, prepared_items[0])
    selected: list[dict[str, Any]] = []
    selected_signatures: set[str] = set()
    selected_token_sets: list[set[str]] = []
    seen_doc_ids: set[int] = set()
    covered_clauses = [0.0] * len(clauses)

    def append_selected(item: dict[str, Any]) -> None:
        """把新证据追加到已选结果中，并同步更新当前分项覆盖状态。"""
        unit = dict(item["unit"])
        best_clause_index = int(item["best_clause_index"])
        unit["aligned_clause"] = clauses[best_clause_index]
        unit["aligned_clause_index"] = best_clause_index
        unit["aligned_clause_score"] = float(item["clause_coverages"][best_clause_index])
        unit["aligned_clause_coverages"] = tuple(float(value) for value in item["clause_coverages"])
        selected.append(unit)
        selected_signatures.add(item["signature"])
        selected_token_sets.append(set(item["tokens"]))
        if int(item["doc_id"]) > 0:
            seen_doc_ids.add(int(item["doc_id"]))
        for index, coverage in enumerate(item["clause_coverages"]):
            covered_clauses[index] = max(covered_clauses[index], float(coverage))

    append_selected(anchor_item)

    # 核心思想：不是永远拿静态总分最高的证据，而是持续追踪“还缺什么”。
    while len(selected) < limit:
        remaining_items = [item for item in prepared_items if item["signature"] not in selected_signatures]
        if not remaining_items:
            break
        unresolved_clause_exists = any(coverage < clause_hit_threshold for coverage in covered_clauses)
        best_item: dict[str, Any] | None = None
        best_score = float("-inf")
        for item in remaining_items:
            clause_coverages = [float(value) for value in item["clause_coverages"]]
            # residual_gain: 当前证据加入后，对各分项“尚未覆盖部分”能补上多少，对应 G_res(u)。
            residual_gains = [max(0.0, coverage - covered_clauses[index]) for index, coverage in enumerate(clause_coverages)]
            residual_gain = sum(residual_gains) / max(len(residual_gains), 1)
            # uncovered_hit_ratio: 当前证据是否命中了还没过阈值的分项，对应 H_un(u)。
            uncovered_hit_ratio = sum(
                1
                for index, coverage in enumerate(clause_coverages)
                if covered_clauses[index] < clause_hit_threshold and coverage >= clause_hit_threshold
            ) / max(len(clauses), 1)
            # strong_fill_ratio: 当前证据是否把覆盖偏弱的分项补强到更高水平，对应 H_str(u)。
            strong_fill_ratio = sum(
                1
                for index, coverage in enumerate(clause_coverages)
                if covered_clauses[index] < clause_strong_threshold and coverage >= clause_strong_threshold
            ) / max(len(clauses), 1)
            # redundancy: 与已选证据有多重复，对应 Red(u)。
            redundancy = max((_token_jaccard(item["tokens"], selected_tokens) for selected_tokens in selected_token_sets), default=0.0)
            if int(item["doc_id"]) not in seen_doc_ids and uncovered_hit_ratio > 0.0:
                # new_doc_bonus: 新文档且确实补到了未覆盖分项时，额外给奖励，对应 B_new(u)。
                new_doc_bonus = 1.0
            elif int(item["doc_id"]) not in seen_doc_ids:
                new_doc_bonus = 0.35
            else:
                new_doc_bonus = 0.0
            # same_doc_penalty: 还在同一篇文档里打转且没补到新分项时，要扣分，对应 P_same(u)。
            same_doc_penalty = 1.0 if int(item["doc_id"]) in seen_doc_ids and uncovered_hit_ratio <= 0.0 else 0.0
            score = (
                0.34 * residual_gain
                + 0.24 * uncovered_hit_ratio
                + 0.14 * strong_fill_ratio
                + 0.10 * float(item["overall_coverage"])
                + 0.10 * float(item["rerank_score"])
                + 0.08 * float(item["detail_signal"])
                + 0.08 * new_doc_bonus
                - 0.16 * redundancy
                - 0.08 * same_doc_penalty
            )
            # 只要当前仍有分项没补到，就进一步压低“没有新贡献”的证据。
            if unresolved_clause_exists and uncovered_hit_ratio <= 0.0:
                score -= 0.20
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None:
            break
        append_selected(best_item)

    ordered_signatures = set(selected_signatures)
    for unit in global_order:
        if len(selected) >= len(evidence_units):
            break
        signature = normalize_text(str(unit.get("text") or ""))
        if not signature or signature in ordered_signatures:
            continue
        fallback = dict(unit)
        fallback["aligned_clause"] = clauses[0]
        fallback["aligned_clause_index"] = 0
        fallback["aligned_clause_score"] = 0.0
        fallback["aligned_clause_coverages"] = ()
        selected.append(fallback)
        ordered_signatures.add(signature)

    return selected
