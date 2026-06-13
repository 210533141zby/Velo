"""论文中证据加工与答案组织相关的核心函数。

这里集中实现分项重排、覆盖取证、按题作答所依赖的证据排序、
事实压缩、槽位组织和后验修补逻辑。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import re
from typing import Any, Sequence

from .common import NO_CONTEXT_ANSWER, clean_text, coverage_ratio, normalize_text, tokenize_text


@dataclass(frozen=True)
class ComplexityAssessment:
    route: str
    score: float
    top1_share: float
    entropy: float
    supporting_docs: int
    top_gap: float = 0.0
    top1_coverage: float = 0.0
    marginal_gain: float = 0.0
    similarity: float = 0.0
    low_confidence: bool = False
    dominant_top1: bool = False
    short_atomic_query: bool = False
    redundant_high_coverage: bool = False


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    statement: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class SupportedAnswer:
    text: str
    support_sentence_count: int
    covered_fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompressedEvidence:
    units: tuple[dict[str, Any], ...]
    fact_count: int


@dataclass(frozen=True)
class NeedleAnnotatedEvidence:
    units: tuple[dict[str, Any], ...]
    need_count: int


@dataclass(frozen=True)
class SpanConstraint:
    text: str
    score: float


@dataclass(frozen=True)
class RepairedAnswer:
    text: str
    support_sentence_count: int
    repaired_sentence_count: int = 0


def extract_query_aspect_clauses(query: str, *, max_clauses: int = 4) -> list[str]:
    """把复杂问题拆成若干有顺序的分项子句。

    这是论文里“分项重排”和“按题作答”的起点。函数并不追求语言学上完美切句，
    而是尽量把用户问题中可独立回答的几个信息点拆出来，后面好据此检查证据覆盖面。
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


def split_answer_sentences(text: str) -> list[str]:
    """把答案拆成去重后的句子列表。"""
    cleaned = clean_text(text, limit=800)
    if not cleaned or normalize_text(cleaned) == normalize_text(NO_CONTEXT_ANSWER):
        return []
    segments = [segment.strip() for segment in re.split(r"(?<=[。！？!?；;])\s*|\n+", cleaned) if segment.strip()]
    sentences: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        sentence = re.sub(r"^\d+[\.\)、]\s*", "", segment).strip()
        signature = normalize_text(sentence)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        sentences.append(sentence)
    return sentences


def split_evidence_sentences(text: str) -> list[str]:
    """把证据文本拆成句子列表。"""
    cleaned = clean_text(text, limit=800)
    if not cleaned:
        return []
    segments = [segment.strip() for segment in re.split(r"(?<=[。！？!?；;])\s*|\n+", cleaned) if segment.strip()]
    return segments or [cleaned]


def order_evidence_units_for_query(
    query_tokens: Sequence[str],
    evidence_units: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按查询覆盖和细节信号重排证据单元。

    这是单维度版本的证据排序器。它综合查询覆盖率、细节信号、重排分和冗余惩罚，
    让排在前面的证据既要“像是在回答问题”，也要尽量保留数值、时间、专名这类高价值细节。
    当问题并不明显是多分项结构时，后续流程会优先使用这一版顺序。
    """
    query_token_set = set(query_tokens)
    if not evidence_units:
        return []

    prepared_units: list[dict[str, Any]] = []
    for index, unit in enumerate(evidence_units):
        text = str(unit.get("text") or "")
        tokens = [token for token in (unit.get("tokens") or tokenize_text(text)) if normalize_text(token)]
        token_set = set(tokens)
        coverage = coverage_ratio(query_token_set, token_set)
        detail_hits = re.findall(
            r"(?:约|近|逾|超|超过|至少|不足|将近)?\d+(?:\.\d+)?(?:万|亿|千|百)?(?:%|％|年|月|日|元|美元|亿元|亿美元|人|名|个|件|项|次|架|枚|门|套|公里|米|吨|级|号)?|[A-Z]{2,}[A-Za-z0-9\-:/]*",
            text,
        )
        detail_signal = min(len(detail_hits), 2) / 2.0
        prepared_units.append(
            {
                "index": index,
                "unit": unit,
                "tokens": token_set,
                "coverage": coverage,
                "detail_signal": detail_signal,
                "rerank_score": float(unit.get("score", 0.0)),
            }
        )

    ordered: list[dict[str, Any]] = []
    selected_token_sets: list[set[str]] = []
    covered_query_tokens: set[str] = set()
    remaining = prepared_units[:]

    # 这里不是一次性平铺打分，而是按“先放最能回答问题的证据，再补互补证据”的
    # 贪心策略排序，对应论文里的“分项重排”思路。
    while remaining:
        current_coverage = coverage_ratio(query_token_set, covered_query_tokens)
        best_item: dict[str, Any] | None = None
        best_score = float("-inf")
        for item in remaining:
            token_set = item["tokens"]
            base_coverage = float(item["coverage"])
            expanded_coverage = coverage_ratio(query_token_set, covered_query_tokens | token_set)
            coverage_gain = max(0.0, expanded_coverage - current_coverage)
            redundancy = max((_token_jaccard(token_set, selected) for selected in selected_token_sets), default=0.0)
            detail_signal = float(item["detail_signal"])
            rerank_score = float(item["rerank_score"])

            if not ordered:
                score = 0.38 * base_coverage + 0.22 * rerank_score + 0.40 * detail_signal
            else:
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

    该函数对应论文中较直观的一版“分项重排”：先把问题拆成子句，
    再为每个子句挑一条最能对应该分项的证据，最后补回全局高分证据。
    它解决的是多文档问答里“模型总盯着一个分项回答”的问题。
    """
    clauses = extract_query_aspect_clauses(query)
    if len(clauses) < 2:
        return order_evidence_units_for_query(tokenize_text(query), evidence_units)

    ordered: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    seen_doc_ids: set[int] = set()

    def append_if_new(unit: dict[str, Any]) -> bool:
        """仅在证据尚未出现时追加到结果列表。"""
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


def order_evidence_units_for_query_clauses_residual_v2(
    query: str,
    evidence_units: Sequence[dict[str, Any]],
    *,
    limit: int,
    clause_hit_threshold: float = 0.18,
    clause_strong_threshold: float = 0.30,
) -> list[dict[str, Any]]:
    """按问题分项和剩余覆盖收益继续重排证据单元。

    最终采用的版本。相比简单的逐子句挑证据，它会持续追踪：
    哪些分项已被覆盖、哪些分项仍缺口较大、引入新文档能否带来新增信息，
    然后用残余收益而不是静态相关度继续选证据，更贴近“覆盖取证”的目标。
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

    prepared_items: list[dict[str, Any]] = []
    for unit in evidence_units:
        text = str(unit.get("text") or "")
        tokens = set(unit.get("tokens") or tokenize_text(text))
        if not tokens:
            continue
        clause_coverages = [coverage_ratio(clause_tokens, tokens) for clause_tokens in clause_token_sets]
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
        best_clause_index = max(range(len(clause_coverages)), key=lambda index: clause_coverages[index])
        prepared_items.append(
            {
                "unit": unit,
                "signature": normalize_text(text),
                "tokens": tokens,
                "doc_id": int(unit.get("doc_id", 0) or 0),
                "rerank_score": float(unit.get("score", 0.0)),
                "overall_coverage": overall_coverage,
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
        """把新证据追加到已选结果中。"""
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

    while len(selected) < limit:
        remaining_items = [item for item in prepared_items if item["signature"] not in selected_signatures]
        if not remaining_items:
            break
        unresolved_clause_exists = any(coverage < clause_hit_threshold for coverage in covered_clauses)
        best_item: dict[str, Any] | None = None
        best_score = float("-inf")
        for item in remaining_items:
            clause_coverages = [float(value) for value in item["clause_coverages"]]
            residual_gains = [max(0.0, coverage - covered_clauses[index]) for index, coverage in enumerate(clause_coverages)]
            residual_gain = sum(residual_gains) / max(len(residual_gains), 1)
            uncovered_hit_ratio = sum(
                1
                for index, coverage in enumerate(clause_coverages)
                if covered_clauses[index] < clause_hit_threshold and coverage >= clause_hit_threshold
            ) / max(len(clauses), 1)
            strong_fill_ratio = sum(
                1
                for index, coverage in enumerate(clause_coverages)
                if covered_clauses[index] < clause_strong_threshold and coverage >= clause_strong_threshold
            ) / max(len(clauses), 1)
            redundancy = max((_token_jaccard(item["tokens"], selected_tokens) for selected_tokens in selected_token_sets), default=0.0)
            if int(item["doc_id"]) not in seen_doc_ids and uncovered_hit_ratio > 0.0:
                new_doc_bonus = 1.0
            elif int(item["doc_id"]) not in seen_doc_ids:
                new_doc_bonus = 0.35
            else:
                new_doc_bonus = 0.0
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


def group_evidence_units_by_query_aspects(
    query_tokens: Sequence[str],
    evidence_units: Sequence[dict[str, Any]],
    *,
    max_groups: int,
    max_sentences_per_group: int = 2,
) -> list[dict[str, Any]]:
    """按问题分项把证据整理成若干证据组。

    这条路线服务于一组对照实验：不直接把原片段送进回答，而是先把高相关句子
    重新聚合成几个“分项证据组”。这样可以观察显式分组是否比原始片段顺序
    更利于复杂问题整合。
    """
    query_token_set = set(query_tokens)
    if not evidence_units or max_groups <= 0:
        return []

    sentence_candidates: list[dict[str, Any]] = []
    for unit in evidence_units:
        doc_id = int(unit.get("doc_id", 0) or 0)
        title = str(unit.get("title") or "")
        unit_score = float(unit.get("score", 0.0))
        text = clean_text(str(unit.get("text") or ""), limit=420)
        if not text:
            continue
        sentences = split_answer_sentences(text) or [text]
        for position, sentence in enumerate(sentences):
            statement = clean_text(sentence, limit=220)
            if not statement:
                continue
            token_set = set(tokenize_text(statement))
            if not token_set:
                continue
            coverage = coverage_ratio(query_token_set, token_set)
            detail_signal = 1.0 if re.search(r"\d|[%年月日:：\-]|[A-Z]{2,}", statement) else 0.0
            if coverage < 0.08 and detail_signal <= 0.0 and len(token_set) < 6:
                continue
            position_bonus = max(0.0, 1.0 - (position / max(len(sentences), 1))) * 0.10
            score = 0.44 * coverage + 0.28 * unit_score + 0.18 * detail_signal + 0.10 * position_bonus
            sentence_candidates.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text": statement,
                    "tokens": token_set,
                    "coverage": coverage,
                    "detail_signal": detail_signal,
                    "score": score,
                    "unit_score": unit_score,
                }
            )

    if not sentence_candidates:
        return list(evidence_units[:max_groups])

    groups: list[dict[str, Any]] = []
    covered_query_tokens: set[str] = set()
    remaining = sentence_candidates[:]

    while remaining and len(groups) < max_groups:
        current_coverage = coverage_ratio(query_token_set, covered_query_tokens)
        best_candidate: dict[str, Any] | None = None
        best_score = float("-inf")
        for candidate in remaining:
            candidate_tokens = candidate["tokens"]
            expanded_coverage = coverage_ratio(query_token_set, covered_query_tokens | candidate_tokens)
            coverage_gain = max(0.0, expanded_coverage - current_coverage)
            redundancy = max(
                (_token_jaccard(candidate_tokens, group["group_tokens"]) for group in groups),
                default=0.0,
            )
            score = (
                0.40 * coverage_gain
                + 0.28 * float(candidate["coverage"])
                + 0.18 * float(candidate["unit_score"])
                + 0.14 * float(candidate["detail_signal"])
                - 0.14 * redundancy
            )
            if not groups:
                score += 0.10 * float(candidate["coverage"]) + 0.24 * float(candidate["detail_signal"])
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            break

        groups.append(
            {
                "members": [best_candidate],
                "group_tokens": set(best_candidate["tokens"]),
                "doc_ids": {int(best_candidate["doc_id"])},
                "titles": [str(best_candidate["title"])],
            }
        )
        covered_query_tokens |= set(best_candidate["tokens"])
        remaining = [candidate for candidate in remaining if candidate is not best_candidate]

    if not groups:
        return list(evidence_units[:max_groups])

    remaining.sort(key=lambda item: float(item["score"]), reverse=True)
    for candidate in remaining:
        best_group: dict[str, Any] | None = None
        best_score = float("-inf")
        for group in groups:
            similarity = _token_jaccard(candidate["tokens"], group["group_tokens"])
            same_doc = 1.0 if int(candidate["doc_id"]) in group["doc_ids"] else 0.0
            score = 0.58 * similarity + 0.24 * float(candidate["coverage"]) + 0.18 * same_doc
            if score > best_score:
                best_score = score
                best_group = group
        if best_group is None or best_score < 0.08:
            continue
        member_signatures = {normalize_text(member["text"]) for member in best_group["members"]}
        if normalize_text(candidate["text"]) in member_signatures:
            continue
        if len(best_group["members"]) >= max_sentences_per_group:
            continue
        best_group["members"].append(candidate)
        best_group["group_tokens"] |= set(candidate["tokens"])
        best_group["doc_ids"].add(int(candidate["doc_id"]))
        if str(candidate["title"]) not in best_group["titles"]:
            best_group["titles"].append(str(candidate["title"]))

    grouped_units: list[dict[str, Any]] = []
    for group in groups[:max_groups]:
        ordered_members = sorted(
            group["members"],
            key=lambda item: (float(item["coverage"]), float(item["unit_score"]), float(item["detail_signal"])),
            reverse=True,
        )
        texts: list[str] = []
        seen_texts: set[str] = set()
        for member in ordered_members:
            signature = normalize_text(member["text"])
            if not signature or signature in seen_texts:
                continue
            seen_texts.add(signature)
            texts.append(str(member["text"]))
        if not texts:
            continue
        grouped_text = " ".join(texts)
        grouped_units.append(
            {
                "doc_id": ordered_members[0]["doc_id"],
                "group_doc_ids": tuple(dict.fromkeys(int(member["doc_id"]) for member in ordered_members)),
                "title": " / ".join(title for title in group["titles"] if title),
                "text": grouped_text,
                "tokens": tokenize_text(grouped_text),
                "score": max(float(member["unit_score"]) for member in ordered_members),
            }
        )

    grouped_units.sort(
        key=lambda item: (
            1.0 if re.search(r"\d|[%年月日:：\-]|[A-Z]{2,}", str(item.get("text") or "")) else 0.0,
            float(item.get("score", 0.0)),
        ),
        reverse=True,
    )
    return grouped_units or list(evidence_units[:max_groups])


def build_edge_packed_evidence_layout(
    query_tokens: Sequence[str],
    evidence_units: Sequence[dict[str, Any]],
    *,
    max_units: int,
) -> list[dict[str, Any]]:
    """构造首尾证据更突出的证据布局。"""
    if not evidence_units or max_units <= 0:
        return []

    ordered = order_evidence_units_for_query(query_tokens, evidence_units)[:max_units]
    if len(ordered) <= 2:
        return ordered

    front = ordered[0]
    back = ordered[1]
    middle = ordered[2:]
    return [front] + middle + [back]


def build_needled_evidence_units(
    query: str,
    query_tokens: Sequence[str],
    evidence_units: Sequence[dict[str, Any]],
    *,
    reranker: Any,
    max_needs: int,
) -> NeedleAnnotatedEvidence:
    """为证据句添加关键句标注。

    这里实现的是“关键句针扎”式证据增强：先从证据中提炼潜在支撑事实，
    再把最值得模型注意的句子用标记包起来。对应的实验目的是验证
    显式强调关键句，能否替代更完整的覆盖取证与作答组织。
    """
    if not evidence_units or max_needs <= 0:
        return NeedleAnnotatedEvidence(units=tuple(evidence_units), need_count=0)

    candidate_facts = _sentence_fact_candidates(
        query_tokens,
        evidence_units,
        max_candidates_per_unit=3,
    )
    if not candidate_facts:
        return NeedleAnnotatedEvidence(units=tuple(evidence_units), need_count=0)

    semantic_scores_by_signature: dict[str, float] = {}
    if reranker is not None:
        unique_candidates: list[tuple[str, str]] = []
        seen_signatures: set[str] = set()
        for fact in candidate_facts:
            signature = normalize_text(fact.statement)
            if not signature or signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_candidates.append((signature, fact.statement))
        if unique_candidates:
            raw_scores = reranker.predict(
                [(query, statement) for _signature, statement in unique_candidates],
                batch_size=16,
                show_progress_bar=False,
            ).tolist()
            semantic_scores_by_signature = {
                signature: 1.0 / (1.0 + math.exp(-float(score)))
                for (signature, _statement), score in zip(unique_candidates, raw_scores)
            }

    needs = refine_grounding_facts(
        query_tokens,
        candidate_facts,
        evidence_units,
        max_facts=max_needs,
        semantic_scores_by_signature=semantic_scores_by_signature,
    )
    if not needs:
        return NeedleAnnotatedEvidence(units=tuple(evidence_units), need_count=0)

    query_token_set = set(query_tokens)
    sentence_pool: list[dict[str, Any]] = []
    for unit_index, unit in enumerate(evidence_units):
        raw_text = clean_text(str(unit.get("text") or ""), limit=800)
        if not raw_text:
            continue
        for sentence_index, sentence in enumerate(split_evidence_sentences(raw_text)):
            sentence_tokens = set(tokenize_text(sentence))
            if not sentence_tokens:
                continue
            sentence_pool.append(
                {
                    "unit_index": unit_index,
                    "sentence_index": sentence_index,
                    "sentence": sentence,
                    "tokens": sentence_tokens,
                    "unit_score": float(unit.get("score", 0.0)),
                }
            )

    if not sentence_pool:
        return NeedleAnnotatedEvidence(units=tuple(evidence_units), need_count=0)

    replacements_by_unit: dict[int, dict[str, str]] = {}
    used_sentences: set[tuple[int, int]] = set()
    applied_need_count = 0
    for need_index, fact in enumerate(needs, start=1):
        fact_tokens = set(tokenize_text(fact.statement))
        if not fact_tokens:
            continue
        best_candidate: dict[str, Any] | None = None
        best_score = float("-inf")
        for candidate in sentence_pool:
            candidate_key = (int(candidate["unit_index"]), int(candidate["sentence_index"]))
            if candidate_key in used_sentences:
                continue
            sentence_tokens = set(candidate["tokens"])
            fact_coverage = coverage_ratio(fact_tokens, sentence_tokens)
            query_coverage = coverage_ratio(query_token_set, sentence_tokens)
            if fact_coverage < 0.16 and query_coverage < 0.12:
                continue
            detail_signal = 1.0 if re.search(r"\d|[%年月日:：\-]|[A-Z]{2,}", str(candidate["sentence"])) else 0.0
            score = (
                0.52 * fact_coverage
                + 0.24 * query_coverage
                + 0.14 * float(candidate["unit_score"])
                + 0.10 * detail_signal
            )
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            continue
        used_sentences.add((int(best_candidate["unit_index"]), int(best_candidate["sentence_index"])))
        tagged_sentence = f"[need_{need_index}]{best_candidate['sentence']}[/need_{need_index}]"
        replacements_by_unit.setdefault(int(best_candidate["unit_index"]), {})[
            str(best_candidate["sentence"])
        ] = tagged_sentence
        applied_need_count += 1

    if applied_need_count <= 0:
        return NeedleAnnotatedEvidence(units=tuple(evidence_units), need_count=0)

    annotated_units: list[dict[str, Any]] = []
    for unit_index, unit in enumerate(evidence_units):
        raw_text = clean_text(str(unit.get("text") or ""), limit=800)
        replacements = replacements_by_unit.get(unit_index, {})
        annotated_text = raw_text
        for source_sentence, tagged_sentence in replacements.items():
            annotated_text = annotated_text.replace(source_sentence, tagged_sentence, 1)
        annotated_units.append(
            {
                **unit,
                "text": annotated_text,
                "tokens": tokenize_text(raw_text),
            }
        )

    return NeedleAnnotatedEvidence(units=tuple(annotated_units), need_count=applied_need_count)


def build_sentence_window_replacement_units(
    query_tokens: Sequence[str],
    evidence_units: Sequence[dict[str, Any]],
    *,
    max_units: int,
) -> list[dict[str, Any]]:
    """把长证据替换成更聚焦的句子窗口。"""
    query_token_set = set(query_tokens)
    if not evidence_units or max_units <= 0:
        return []

    replaced_units: list[dict[str, Any]] = []
    for unit in evidence_units[:max_units]:
        text = clean_text(str(unit.get("text") or ""), limit=420)
        if not text:
            continue
        sentences = split_answer_sentences(text) or [text]
        if len(sentences) <= 1:
            replaced_units.append(unit)
            continue

        scored: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            sentence_tokens = set(tokenize_text(sentence))
            coverage = coverage_ratio(query_token_set, sentence_tokens)
            detail_signal = 1.0 if re.search(r"\d|[%年月日:：\-]|[A-Z]{2,}", sentence) else 0.0
            score = 0.72 * coverage + 0.28 * detail_signal
            scored.append((score, index, sentence))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        best_score, best_index, _best_sentence = scored[0]
        if best_score < 0.08:
            replaced_units.append(unit)
            continue

        left = best_index
        right = best_index + 1
        if best_index > 0:
            prev_tokens = set(tokenize_text(sentences[best_index - 1]))
            if coverage_ratio(query_token_set, prev_tokens) >= 0.10:
                left = best_index - 1
        if best_index + 1 < len(sentences):
            next_tokens = set(tokenize_text(sentences[best_index + 1]))
            if coverage_ratio(query_token_set, next_tokens) >= 0.10:
                right = best_index + 2
        window_text = " ".join(sentences[left:right]).strip()
        replaced_units.append(
            {
                "doc_id": unit.get("doc_id"),
                "title": unit.get("title"),
                "text": window_text,
                "tokens": tokenize_text(window_text),
                "score": float(unit.get("score", 0.0)),
            }
        )

    return replaced_units or list(evidence_units[:max_units])


def build_title_structured_evidence_units(
    query_tokens: Sequence[str],
    evidence_units: Sequence[dict[str, Any]],
    *,
    max_units: int,
) -> list[dict[str, Any]]:
    """把标题和正文整理成结构化证据单元。"""
    query_token_set = set(query_tokens)
    if not evidence_units or max_units <= 0:
        return []

    rendered_units: list[dict[str, Any]] = []
    for unit in evidence_units[:max_units]:
        raw_title = clean_text(str(unit.get("title") or ""), limit=96)
        raw_text = clean_text(str(unit.get("text") or ""), limit=420)
        if not raw_text:
            continue
        sentences = split_answer_sentences(raw_text) or [raw_text]
        scored_sentences: list[tuple[float, int, str, float]] = []
        for index, sentence in enumerate(sentences):
            sentence_tokens = set(tokenize_text(sentence))
            coverage = coverage_ratio(query_token_set, sentence_tokens)
            detail_signal = 1.0 if re.search(r"\d|[%年月日:：\-]|[A-Z]{2,}", sentence) else 0.0
            lead_bonus = 0.10 if index == 0 else 0.0
            score = 0.58 * coverage + 0.30 * detail_signal + 0.12 * lead_bonus
            scored_sentences.append((score, index, sentence, detail_signal))
        scored_sentences.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        lead_sentence = sentences[0]
        key_sentence = scored_sentences[0][2]
        detail_sentence = ""
        for _score, _index, sentence, detail_signal in scored_sentences[1:]:
            if normalize_text(sentence) in {normalize_text(lead_sentence), normalize_text(key_sentence)}:
                continue
            if detail_signal > 0.0 or coverage_ratio(query_token_set, set(tokenize_text(sentence))) >= 0.10:
                detail_sentence = sentence
                break

        parts = [f"标题线索：{raw_title or '未命名文档'}"]
        if normalize_text(lead_sentence) != normalize_text(key_sentence):
            parts.append(f"导语：{lead_sentence}")
        parts.append(f"关键证据：{key_sentence}")
        if detail_sentence:
            parts.append(f"补充细节：{detail_sentence}")

        structured_text = "\n".join(parts)
        rendered_units.append(
            {
                **unit,
                "text": structured_text,
                "tokens": tokenize_text(f"{raw_title} {structured_text}"),
            }
        )

    return rendered_units or list(evidence_units[:max_units])


def build_aspect_labeled_evidence_units(
    query: str,
    query_tokens: Sequence[str],
    evidence_units: Sequence[dict[str, Any]],
    *,
    max_units: int,
) -> list[dict[str, Any]]:
    """为证据单元补充分项标签。"""
    clauses = extract_query_aspect_clauses(query)
    if len(clauses) < 2 or not evidence_units or max_units <= 0:
        return list(evidence_units[:max_units])

    query_token_set = set(query_tokens)
    rendered_units: list[dict[str, Any]] = []
    for unit in evidence_units[:max_units]:
        raw_text = clean_text(str(unit.get("text") or ""), limit=520)
        if not raw_text:
            continue
        best_clause = str(unit.get("aligned_clause") or "")
        if not best_clause:
            clause_scores = []
            for clause in clauses:
                clause_scores.append((coverage_ratio(set(tokenize_text(clause)), set(tokenize_text(raw_text))), clause))
            clause_scores.sort(key=lambda item: item[0], reverse=True)
            best_clause = clause_scores[0][1] if clause_scores else clauses[0]

        clause_token_set = set(tokenize_text(best_clause))
        sentences = split_answer_sentences(raw_text) or [raw_text]
        scored_sentences: list[tuple[float, str]] = []
        for index, sentence in enumerate(sentences):
            sentence_tokens = set(tokenize_text(sentence))
            clause_coverage = coverage_ratio(clause_token_set, sentence_tokens)
            query_coverage = coverage_ratio(query_token_set, sentence_tokens)
            detail_signal = 1.0 if re.search(r"\d|[%年月日:：-]|[A-Z]{2,}", sentence) else 0.0
            lead_bonus = 0.08 if index == 0 else 0.0
            score = 0.46 * clause_coverage + 0.26 * query_coverage + 0.20 * detail_signal + 0.08 * lead_bonus
            scored_sentences.append((score, sentence))
        scored_sentences.sort(key=lambda item: item[0], reverse=True)

        chosen_sentences: list[str] = []
        seen_signatures: set[str] = set()
        for score, sentence in scored_sentences:
            signature = normalize_text(sentence)
            if not signature or signature in seen_signatures:
                continue
            if chosen_sentences and score < 0.10:
                break
            chosen_sentences.append(sentence)
            seen_signatures.add(signature)
            if len(chosen_sentences) >= 2:
                break
        if not chosen_sentences:
            chosen_sentences = [sentences[0]]

        lines = [f"对应要求：{best_clause}", f"证据片段：{chosen_sentences[0]}"]
        if len(chosen_sentences) >= 2:
            lines.append(f"补充细节：{chosen_sentences[1]}")
        rendered_text = "\n".join(lines)
        rendered_units.append(
            {
                **unit,
                "text": rendered_text,
                "tokens": tokenize_text(f"{best_clause} {' '.join(chosen_sentences)}"),
            }
        )

    return rendered_units or list(evidence_units[:max_units])


@dataclass(frozen=True)
class StructuredSlot:
    slot_id: str
    prompt: str
    fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class StructuredPlan:
    slots: tuple[StructuredSlot, ...]


def build_clause_support_plan(
    query: str,
    facts: Sequence[EvidenceFact],
    evidence_units: Sequence[dict[str, Any]],
    *,
    max_slots: int,
    max_facts_per_slot: int = 2,
) -> StructuredPlan:
    """按问题子句构建支持信息计划。

    支持计划是生成侧重型对照路线里的中间表示。它会把问题子句和候选事实绑定成槽位，
    让模型先看到“这一问该由哪些事实支撑”，再进入后续的结构化回答阶段。
    """
    clauses = extract_query_aspect_clauses(query, max_clauses=max_slots)
    if not clauses:
        clauses = [clean_text(query, limit=160)]
    if not clauses or not facts:
        return StructuredPlan(slots=())

    query_token_set = set(tokenize_text(query))
    source_score_map = {
        f"S{index}": float(unit.get("score", 0.0))
        for index, unit in enumerate(evidence_units, start=1)
    }
    max_source_score = max(source_score_map.values(), default=0.0)
    used_fact_ids: set[str] = set()
    slots: list[StructuredSlot] = []

    for clause in clauses[:max_slots]:
        clause_text = clean_text(clause, limit=140)
        clause_tokens = set(tokenize_text(clause_text)) or query_token_set
        scored_candidates: list[tuple[float, str]] = []
        for fact in facts:
            fact_tokens = set(tokenize_text(fact.statement))
            if not fact_tokens:
                continue
            clause_coverage = coverage_ratio(clause_tokens, fact_tokens)
            query_coverage = coverage_ratio(query_token_set, fact_tokens)
            if clause_coverage < 0.10 and query_coverage < 0.18:
                continue
            source_scores = [source_score_map.get(source_id, 0.0) for source_id in fact.source_ids]
            if max_source_score > 0.0:
                source_strength = sum(source_scores) / max(1, len(source_scores)) / max_source_score
            else:
                source_strength = 0.0
            detail_signal = 1.0 if re.search(r"\d|[%年月日:：\-]|[A-Z]{2,}", fact.statement) else 0.0
            reuse_penalty = 0.10 if fact.fact_id in used_fact_ids else 0.0
            score = (
                0.48 * clause_coverage
                + 0.24 * query_coverage
                + 0.18 * source_strength
                + 0.10 * detail_signal
                - reuse_penalty
            )
            scored_candidates.append((score, fact.fact_id))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        selected_fact_ids: list[str] = []
        for score, fact_id in scored_candidates:
            if selected_fact_ids and score < 0.08:
                break
            if fact_id in selected_fact_ids:
                continue
            selected_fact_ids.append(fact_id)
            if len(selected_fact_ids) >= max_facts_per_slot:
                break
        if not selected_fact_ids:
            continue
        used_fact_ids.update(selected_fact_ids)
        slots.append(
            StructuredSlot(
                slot_id=f"SLOT{len(slots) + 1}",
                prompt=clause_text,
                fact_ids=tuple(selected_fact_ids),
            )
        )

    return StructuredPlan(slots=tuple(slots))


def build_slot_packed_evidence_units(
    plan: StructuredPlan,
    facts: Sequence[EvidenceFact],
    evidence_units: Sequence[dict[str, Any]],
    *,
    max_units: int,
    max_facts_per_slot: int = 2,
) -> list[dict[str, Any]]:
    """把整理后的事实按槽位打包成伪证据单元。

    这些伪证据单元并非原始检索片段，而是把同一槽位下的事实重新整理后的输入。
    它主要用于验证“增加一层显式结构化中间表示”是否能稳定提升最终答案质量。
    """
    if not plan.slots or not facts or max_units <= 0:
        return []

    fact_map = {fact.fact_id: fact for fact in facts}
    source_map = {
        f"S{index}": unit
        for index, unit in enumerate(evidence_units, start=1)
    }
    packed_units: list[dict[str, Any]] = []
    for slot in plan.slots[:max_units]:
        slot_facts: list[str] = []
        score_values: list[float] = []
        seen_signatures: set[str] = set()
        for fact_id in slot.fact_ids:
            fact = fact_map.get(fact_id)
            if fact is None:
                continue
            statement = clean_text(fact.statement, limit=220)
            signature = normalize_text(statement)
            if not statement or not signature or signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            slot_facts.append(statement)
            for source_id in fact.source_ids:
                source_unit = source_map.get(source_id)
                if source_unit is not None:
                    score_values.append(float(source_unit.get("score", 0.0)))
            if len(slot_facts) >= max_facts_per_slot:
                break

        if not slot_facts:
            continue

        packed_text = "\n".join(
            [f"需要覆盖：{slot.prompt}", "支持事实："]
            + [f"- {statement}" for statement in slot_facts]
        )
        packed_units.append(
            {
                "doc_id": len(packed_units) + 1,
                "title": f"{slot.slot_id} {slot.prompt}",
                "text": packed_text,
                "tokens": tokenize_text(f"{slot.prompt} {' '.join(slot_facts)}"),
                "score": max(score_values, default=0.0),
            }
        )
    return packed_units


def _softmax(values: Sequence[float]) -> list[float]:
    """计算一组分值的 softmax 结果。"""
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exps]


def _normalized_entropy(probabilities: Sequence[float]) -> float:
    """计算归一化熵值。"""
    if len(probabilities) <= 1:
        return 0.0
    entropy = 0.0
    for probability in probabilities:
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return entropy / math.log(len(probabilities))


def _token_jaccard(left: set[str], right: set[str]) -> float:
    """计算两个词集合的 Jaccard 相似度。"""
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _pairwise_similarity(token_sets: Sequence[set[str]]) -> float:
    """计算证据集合内部的平均相似度。"""
    similarities: list[float] = []
    for index, left_tokens in enumerate(token_sets):
        for right_tokens in token_sets[index + 1 :]:
            similarities.append(_token_jaccard(left_tokens, right_tokens))
    return sum(similarities) / len(similarities) if similarities else 0.0


def _support_ratio(reference_tokens: set[str], support_tokens: set[str]) -> float:
    """计算支撑文本覆盖了多少参考事实词元。

    这个比例专门用于衡量一条候选事实对目标事实的支撑程度，
    是覆盖取证阶段判断“是否真补上了缺失维度”的一个细粒度信号。
    """
    if not reference_tokens or not support_tokens:
        return 0.0
    return len(reference_tokens & support_tokens) / len(reference_tokens)


def assess_complexity(
    query_tokens: Sequence[str],
    evidence_units: Sequence[dict[str, Any]],
    *,
    threshold: float = 0.50,
) -> ComplexityAssessment:
    """评估当前问题是否需要走复杂问答路径。

    这个函数就是实验与论文里“路由式直接作答”的开关。它不直接看问题字面长短，
    而是结合候选证据的头部垄断程度、覆盖增益、相互相似度和支撑文档数，
    判断当前问题更像“单证据即可回答”还是“需要多证据综合”。
    """
    if len(evidence_units) <= 1:
        top_tokens = set(evidence_units[0].get("tokens") or tokenize_text(str(evidence_units[0].get("text") or ""))) if evidence_units else set()
        top1_coverage = coverage_ratio(set(query_tokens), top_tokens)
        return ComplexityAssessment(
            route="simple",
            score=0.0,
            top1_share=1.0,
            entropy=0.0,
            supporting_docs=len(evidence_units),
            top_gap=0.0,
            top1_coverage=round(top1_coverage, 4),
            marginal_gain=0.0,
            similarity=0.0,
            low_confidence=False,
            dominant_top1=True,
            short_atomic_query=len(set(query_tokens)) <= 8,
            redundant_high_coverage=top1_coverage >= 0.65,
        )

    top_units = list(evidence_units[:4])
    score_values = [float(unit.get("score", 0.0)) for unit in top_units]
    probabilities = _softmax(score_values)
    top1_share = probabilities[0] if probabilities else 1.0
    entropy = _normalized_entropy(probabilities)

    top_token_sets = [set(unit.get("tokens") or tokenize_text(str(unit.get("text") or ""))) for unit in top_units]
    query_token_set = set(query_tokens)
    query_length = len(query_token_set)
    top1_tokens = top_token_sets[0] if top_token_sets else set()
    top1_coverage = coverage_ratio(query_token_set, top1_tokens)
    similarity = _pairwise_similarity(top_token_sets[:3])

    union_tokens = set(top1_tokens)
    marginal_gain = 0.0
    supporting_docs = 1
    for token_set in top_token_sets[1:]:
        current_coverage = coverage_ratio(query_token_set, union_tokens)
        expanded_coverage = coverage_ratio(query_token_set, union_tokens | token_set)
        gain = max(0.0, expanded_coverage - current_coverage)
        marginal_gain += gain
        if gain >= 0.03:
            supporting_docs += 1
        union_tokens |= token_set
    supporting_docs = min(supporting_docs, len(top_units))

    top_score = score_values[0] if score_values else 0.0
    top_gap = top_score - score_values[1] if len(score_values) > 1 else top_score
    # 如果单篇文档已经足够回答问题，或者当前证据整体偏弱，
    # 就优先走直接作答路径，避免为了复杂综合额外引入噪声。
    low_confidence = top_score < 0.35
    dominant_top1 = top_gap >= 0.20
    short_atomic_query = query_length <= 8 and marginal_gain <= 0.35
    redundant_high_coverage = top1_coverage >= 0.65 and marginal_gain <= 0.20 and similarity >= 0.15

    gain_signal = min(marginal_gain / 0.18, 1.0)
    support_signal = min(max(supporting_docs - 1, 0) / 2.0, 1.0)
    dominance_penalty = 1.0 - min(max(top_gap, 0.0) / 0.20, 1.0)
    coverage_penalty = 1.0 - top1_coverage
    diversity_signal = 1.0 - similarity
    complexity_score = (
        0.45 * gain_signal
        + 0.25 * support_signal
        + 0.15 * coverage_penalty
        + 0.10 * dominance_penalty
        + 0.05 * diversity_signal
    )

    should_use_simple = low_confidence or dominant_top1 or short_atomic_query or redundant_high_coverage
    route = "complex" if not should_use_simple and supporting_docs >= 2 and complexity_score >= threshold else "simple"
    return ComplexityAssessment(
        route=route,
        score=round(complexity_score, 4),
        top1_share=round(top1_share, 4),
        entropy=round(entropy, 4),
        supporting_docs=supporting_docs,
        top_gap=round(top_gap, 4),
        top1_coverage=round(top1_coverage, 4),
        marginal_gain=round(marginal_gain, 4),
        similarity=round(similarity, 4),
        low_confidence=low_confidence,
        dominant_top1=dominant_top1,
        short_atomic_query=short_atomic_query,
        redundant_high_coverage=redundant_high_coverage,
    )


def build_distillation_prompt(
    query: str,
    evidence_units: Sequence[dict[str, Any]],
    *,
    max_facts: int,
) -> str:
    """生成“证据蒸馏为事实表”阶段的提示词。

    这一步对应重型生成路线里的第一层中间表示，把原始证据先压缩成原子事实，
    方便后续综合、修补或槽位填充阶段复用。
    """
    blocks: list[str] = []
    for index, unit in enumerate(evidence_units, start=1):
        blocks.append(f"[S{index}] {unit['title']}\n{unit['text']}")
    context = "\n\n".join(blocks)
    return (
        "你是一个严格的证据蒸馏器。请仅根据给定证据，为用户问题抽取最关键的原子事实，并只输出 JSON。\n"
        "输出格式严格为："
        '{"facts":[{"fact_id":"F1","statement":"...","source_ids":["S1"]}]}\n'
        "要求：\n"
        "1. 每条 fact 必须是完成回答所需的直接事实，不要输出背景评论。\n"
        "2. 每条 fact 必须自包含，保留专名、时间、数字、范围、条件等关键限定。\n"
        "3. 如果某个信息点依赖数值与其统计口径、时间边界或范围限定共同成立，必须把这些限定保留在同一条 fact 中，不要只保留主数值。\n"
        "4. 只保留能够直接回应问题中某个信息点的事实；若只是背景、评论或泛化描述，则不要输出。\n"
        "5. 不要编造证据中没有的信息，不要做开放性推断。\n"
        f"6. 最多输出 {max_facts} 条 fact，按对回答的重要性排序。\n"
        "7. source_ids 只能填写给定证据编号，例如 S1、S2。\n\n"
        f"用户问题：{query}\n\n"
        f"证据单元：\n{context}\n\n"
        "JSON："
    )


def fallback_facts(evidence_units: Sequence[dict[str, Any]], *, max_facts: int) -> list[EvidenceFact]:
    """在模型未产出可用事实时，从原始证据构造兜底事实列表。

    兜底策略故意非常保守，只截取前几条证据正文，目的是让实验链路继续可跑，
    同时清楚地区分“模型蒸馏成功”和“只是退回原证据”。
    """
    fallback: list[EvidenceFact] = []
    for index, unit in enumerate(evidence_units[:max_facts], start=1):
        statement = clean_text(str(unit.get("text") or ""), limit=220)
        if not statement:
            continue
        fallback.append(EvidenceFact(fact_id=f"F{index}", statement=statement, source_ids=(f"S{index}",)))
    return fallback


def _sentence_fact_candidates(
    query_tokens: Sequence[str],
    evidence_units: Sequence[dict[str, Any]],
    *,
    max_candidates_per_unit: int = 2,
) -> list[EvidenceFact]:
    """从证据句中抽取一批可供后续筛选的事实候选。

    与蒸馏模型直接输出的事实不同，这里完全基于原句切分和轻量打分，
    为后面的事实精炼过程提供一个“无模型也能回退”的候选池。
    """
    query_token_set = set(query_tokens)
    candidates: list[EvidenceFact] = []
    for index, unit in enumerate(evidence_units, start=1):
        text = clean_text(str(unit.get("text") or ""), limit=420)
        if not text:
            continue
        sentences = [segment.strip() for segment in re.split(r"(?<=[。！？!?；;])\s*|\n+", text) if segment.strip()]
        if not sentences:
            sentences = [text]
        scored: list[tuple[float, int, str]] = []
        for position, sentence in enumerate(sentences):
            statement = clean_text(sentence, limit=200)
            if not statement:
                continue
            token_set = set(tokenize_text(statement))
            if not token_set:
                continue
            base_coverage = coverage_ratio(query_token_set, token_set)
            if base_coverage < 0.08 and len(token_set) < 6:
                continue
            position_bonus = max(0.0, 1.0 - (position / max(len(sentences), 1))) * 0.08
            compactness = 1.0 / max(len(token_set), 1)
            score = base_coverage + position_bonus + min(compactness, 0.04)
            scored.append((score, position, statement))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        seen_statements: set[str] = set()
        selected = 0
        for _score, _position, statement in scored:
            signature = normalize_text(statement)
            if not signature or signature in seen_statements:
                continue
            seen_statements.add(signature)
            candidates.append(EvidenceFact(fact_id=f"F{len(candidates) + 1}", statement=statement, source_ids=(f"S{index}",)))
            selected += 1
            if selected >= max_candidates_per_unit:
                break
    return candidates


def parse_distilled_facts(
    payload: dict[str, Any],
    evidence_units: Sequence[dict[str, Any]],
    *,
    max_facts: int,
) -> list[EvidenceFact]:
    """解析模型返回的压缩事实列表。"""
    allowed_source_ids = {f"S{index}" for index in range(1, len(evidence_units) + 1)}
    raw_facts = payload.get("facts")
    if not isinstance(raw_facts, list):
        return fallback_facts(evidence_units, max_facts=max_facts)

    parsed: list[EvidenceFact] = []
    for index, item in enumerate(raw_facts[:max_facts], start=1):
        if not isinstance(item, dict):
            continue
        statement = clean_text(str(item.get("statement") or item.get("fact") or ""), limit=240)
        if not statement or normalize_text(statement) == normalize_text(NO_CONTEXT_ANSWER):
            continue
        raw_source_ids = item.get("source_ids") or item.get("sources") or []
        if isinstance(raw_source_ids, str):
            raw_source_ids = [raw_source_ids]
        source_ids = tuple(source_id for source_id in (str(value).strip() for value in raw_source_ids) if source_id in allowed_source_ids)
        if not source_ids:
            fallback_source = f"S{min(index, len(evidence_units))}"
            source_ids = (fallback_source,)
        parsed.append(EvidenceFact(fact_id=f"F{len(parsed) + 1}", statement=statement, source_ids=source_ids))

    return parsed or fallback_facts(evidence_units, max_facts=max_facts)


def refine_grounding_facts(
    query_tokens: Sequence[str],
    facts: Sequence[EvidenceFact],
    evidence_units: Sequence[dict[str, Any]],
    *,
    max_facts: int,
    semantic_scores_by_signature: dict[str, float] | None = None,
) -> list[EvidenceFact]:
    """结合证据覆盖情况筛选落地事实。

    无论事实来自蒸馏阶段还是句子候选阶段，最后都不能原样全送去回答。
    这里会综合查询覆盖、来源强度、语义匹配和去重结果筛掉不够稳的事实，
    让下游综合过程尽量只看到“既贴题、又有证据落点”的事实集合。
    """
    query_token_set = set(query_tokens)
    fallback_candidates = _sentence_fact_candidates(query_tokens, evidence_units)
    source_score_map = {
        f"S{index}": float(unit.get("score", 0.0))
        for index, unit in enumerate(evidence_units, start=1)
    }
    max_source_score = max(source_score_map.values(), default=0.0)
    candidate_pool: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    def add_candidate(fact: EvidenceFact, *, is_distilled: bool) -> None:
        """向候选结果集中加入一条新候选。"""
        statement = clean_text(fact.statement, limit=240)
        signature = normalize_text(statement)
        if not signature or signature == normalize_text(NO_CONTEXT_ANSWER) or signature in seen_signatures:
            return
        seen_signatures.add(signature)
        source_scores = [source_score_map.get(source_id, 0.0) for source_id in fact.source_ids]
        if max_source_score > 0.0:
            source_strength = sum(source_scores) / max(1, len(source_scores)) / max_source_score
        else:
            source_strength = 0.0
        semantic_score = 0.0
        if semantic_scores_by_signature is not None:
            semantic_score = float(semantic_scores_by_signature.get(signature, 0.0))
        candidate_pool.append(
            {
                "statement": statement,
                "source_ids": tuple(dict.fromkeys(fact.source_ids)),
                "tokens": set(tokenize_text(statement)),
                "is_distilled": is_distilled,
                "source_strength": source_strength,
                "semantic_score": semantic_score,
            }
        )

    for fact in facts:
        add_candidate(fact, is_distilled=True)
    for fact in fallback_candidates:
        add_candidate(fact, is_distilled=False)

    if not candidate_pool:
        return []

    selected: list[dict[str, Any]] = []
    selected_token_sets: list[set[str]] = []
    covered_query_tokens: set[str] = set()
    covered_sources: set[str] = set()

    while candidate_pool and len(selected) < max_facts:
        current_coverage = coverage_ratio(query_token_set, covered_query_tokens)
        best_candidate: dict[str, Any] | None = None
        best_score = float("-inf")

        for candidate in candidate_pool:
            candidate_tokens = candidate["tokens"]
            expanded_coverage = coverage_ratio(query_token_set, covered_query_tokens | candidate_tokens)
            coverage_gain = max(0.0, expanded_coverage - current_coverage)
            base_coverage = coverage_ratio(query_token_set, candidate_tokens)
            source_gain = len(set(candidate["source_ids"]) - covered_sources)
            source_bonus = min(source_gain, 2) / max(1, len(candidate["source_ids"]))
            source_strength = float(candidate["source_strength"])
            semantic_score = float(candidate["semantic_score"])
            redundancy = max((_token_jaccard(candidate_tokens, token_set) for token_set in selected_token_sets), default=0.0)
            length_penalty = max(0.0, (len(candidate_tokens) - 28) / 28.0)
            distilled_bonus = 0.06 if candidate["is_distilled"] else 0.0

            score = (
                0.28 * coverage_gain
                + 0.14 * base_coverage
                + 0.22 * source_strength
                + 0.14 * source_bonus
                + 0.20 * semantic_score
                + distilled_bonus
                - 0.16 * redundancy
                - 0.06 * length_penalty
            )
            if not selected:
                score += 0.08 * base_coverage + 0.08 * source_strength + 0.10 * semantic_score

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            break
        if selected and best_score < 0.05:
            break

        selected.append(best_candidate)
        selected_token_sets.append(best_candidate["tokens"])
        covered_query_tokens |= best_candidate["tokens"]
        covered_sources |= set(best_candidate["source_ids"])
        candidate_pool = [candidate for candidate in candidate_pool if candidate is not best_candidate]

    if not selected:
        selected = candidate_pool[:max_facts]

    return [
        EvidenceFact(
            fact_id=f"F{index}",
            statement=str(candidate["statement"]),
            source_ids=tuple(candidate["source_ids"]),
        )
        for index, candidate in enumerate(selected, start=1)
    ]


def compress_evidence_units(
    query: str,
    query_tokens: Sequence[str],
    evidence_units: Sequence[dict[str, Any]],
    *,
    reranker: Any,
    max_units: int,
    max_candidates_per_unit: int = 3,
) -> CompressedEvidence:
    """压缩证据单元，保留回答所需的核心事实。"""
    if not evidence_units:
        return CompressedEvidence(units=(), fact_count=0)

    candidate_facts = _sentence_fact_candidates(
        query_tokens,
        evidence_units,
        max_candidates_per_unit=max_candidates_per_unit,
    )
    if not candidate_facts:
        fallback_units = tuple(evidence_units[:max_units])
        return CompressedEvidence(units=fallback_units, fact_count=0)

    semantic_scores_by_signature: dict[str, float] = {}
    if reranker is not None:
        unique_candidates: list[tuple[str, str]] = []
        seen_signatures: set[str] = set()
        for fact in candidate_facts:
            signature = normalize_text(fact.statement)
            if not signature or signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_candidates.append((signature, fact.statement))
        if unique_candidates:
            raw_scores = reranker.predict(
                [(query, statement) for _signature, statement in unique_candidates],
                batch_size=16,
                show_progress_bar=False,
            ).tolist()
            semantic_scores_by_signature = {
                signature: 1.0 / (1.0 + math.exp(-float(score)))
                for (signature, _statement), score in zip(unique_candidates, raw_scores)
            }

    refined_facts = refine_grounding_facts(
        query_tokens,
        candidate_facts,
        evidence_units,
        max_facts=max_units,
        semantic_scores_by_signature=semantic_scores_by_signature,
    )
    if not refined_facts:
        fallback_units = tuple(evidence_units[:max_units])
        return CompressedEvidence(units=fallback_units, fact_count=0)

    source_map = {
        f"S{index}": unit
        for index, unit in enumerate(evidence_units, start=1)
    }
    compressed_units: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for fact in refined_facts:
        source_scores = [float(source_map.get(source_id, {}).get("score", 0.0)) for source_id in fact.source_ids]
        fact_score = max(source_scores, default=0.0)
        signature = normalize_text(fact.statement)
        if not signature or signature in seen_signatures:
            continue
        token_set = tokenize_text(fact.statement)
        query_coverage = coverage_ratio(set(query_tokens), set(token_set))
        semantic_score = float(semantic_scores_by_signature.get(signature, 0.0)) if semantic_scores_by_signature else 0.0
        if query_coverage < 0.12 and semantic_score < 0.55:
            continue
        seen_signatures.add(signature)
        compressed_units.append(
            {
                "doc_id": int(source_map.get(fact.source_ids[0], {}).get("doc_id", 0) or 0),
                "title": " / ".join(
                    dict.fromkeys(
                        str(source_map[source_id]["title"])
                        for source_id in fact.source_ids
                        if source_id in source_map
                    )
                ),
                "text": fact.statement,
                "tokens": token_set,
                "score": fact_score,
            }
        )
        if len(compressed_units) >= max_units:
            break

    if not compressed_units:
        fallback_units = tuple(evidence_units[:max_units])
        return CompressedEvidence(units=fallback_units, fact_count=len(refined_facts))
    return CompressedEvidence(units=tuple(compressed_units), fact_count=len(refined_facts))


def prune_answer_to_supported_sentences(
    answer: str,
    evidence_units: Sequence[dict[str, Any]],
    *,
    reranker: Any,
    threshold: float,
) -> SupportedAnswer:
    """裁剪掉答案中缺少支撑的句子。"""
    answer_sentences = split_answer_sentences(answer)
    if not answer_sentences or reranker is None:
        cleaned = clean_text(answer, limit=800) or NO_CONTEXT_ANSWER
        return SupportedAnswer(text=cleaned, support_sentence_count=0, covered_fact_ids=())

    evidence_sentences: list[str] = []
    seen_evidence: set[str] = set()
    for unit in evidence_units:
        for sentence in split_answer_sentences(str(unit.get("text") or "")):
            signature = normalize_text(sentence)
            if not signature or signature in seen_evidence:
                continue
            seen_evidence.add(signature)
            evidence_sentences.append(sentence)
    if not evidence_sentences:
        cleaned = clean_text(answer, limit=800) or NO_CONTEXT_ANSWER
        return SupportedAnswer(text=cleaned, support_sentence_count=0, covered_fact_ids=())

    pairs = [(sentence, evidence_sentence) for sentence in answer_sentences for evidence_sentence in evidence_sentences]
    raw_scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False).tolist()

    kept_sentences: list[str] = []
    best_sentence = answer_sentences[0]
    best_score = float("-inf")
    for index, sentence in enumerate(answer_sentences):
        start = index * len(evidence_sentences)
        end = start + len(evidence_sentences)
        sentence_scores = [1.0 / (1.0 + math.exp(-float(score))) for score in raw_scores[start:end]]
        score = max(sentence_scores) if sentence_scores else 0.0
        if score > best_score:
            best_score = score
            best_sentence = sentence
        keep_threshold = threshold if index > 0 else max(0.0, threshold - 0.06)
        if score >= keep_threshold:
            kept_sentences.append(sentence)

    if not kept_sentences:
        kept_sentences = [best_sentence]
    text = " ".join(kept_sentences).strip() or NO_CONTEXT_ANSWER
    return SupportedAnswer(
        text=text,
        support_sentence_count=len(kept_sentences),
        covered_fact_ids=tuple(),
    )


def repair_answer_with_supported_sentences(
    answer: str,
    evidence_units: Sequence[dict[str, Any]],
    *,
    reranker: Any,
    support_threshold: float,
    rewrite_threshold: float,
) -> RepairedAnswer:
    # 当原答案骨架本身已经有较好支撑时，尽量保留其整体结构；
    # 只把那些支撑边缘、表述偏泛的句子替换成更贴近证据的写法。
    """用有支撑的句子修补答案。"""
    answer_sentences = split_answer_sentences(answer)
    if not answer_sentences or reranker is None:
        cleaned = clean_text(answer, limit=800) or NO_CONTEXT_ANSWER
        return RepairedAnswer(text=cleaned, support_sentence_count=0, repaired_sentence_count=0)

    evidence_sentences: list[str] = []
    seen_evidence: set[str] = set()
    for unit in evidence_units:
        for sentence in split_answer_sentences(str(unit.get("text") or "")):
            signature = normalize_text(sentence)
            if not signature or signature in seen_evidence:
                continue
            seen_evidence.add(signature)
            evidence_sentences.append(sentence)
    if not evidence_sentences:
        cleaned = clean_text(answer, limit=800) or NO_CONTEXT_ANSWER
        return RepairedAnswer(text=cleaned, support_sentence_count=0, repaired_sentence_count=0)

    pairs = [(sentence, evidence_sentence) for sentence in answer_sentences for evidence_sentence in evidence_sentences]
    raw_scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False).tolist()

    revised_sentences: list[str] = []
    seen_sentences: set[str] = set()
    repaired_sentence_count = 0
    support_sentence_count = 0

    query_detail_pattern = re.compile(r"\d|[%年月日:：\-]|[A-Z]{2,}")
    for index, sentence in enumerate(answer_sentences):
        start = index * len(evidence_sentences)
        end = start + len(evidence_sentences)
        sentence_scores = [1.0 / (1.0 + math.exp(-float(score))) for score in raw_scores[start:end]]
        if not sentence_scores:
            continue
        best_index = max(range(len(sentence_scores)), key=sentence_scores.__getitem__)
        best_score = sentence_scores[best_index]
        best_evidence_sentence = evidence_sentences[best_index]
        keep_threshold = support_threshold if index > 0 else max(0.0, support_threshold - 0.05)
        rewrite_gate = min(keep_threshold, rewrite_threshold)
        answer_tokens = set(tokenize_text(sentence))
        evidence_tokens = set(tokenize_text(best_evidence_sentence))
        overlap = _support_ratio(answer_tokens, evidence_tokens)
        evidence_has_more_detail = len(evidence_tokens) >= len(answer_tokens) + 3

        final_sentence = sentence
        if best_score >= keep_threshold:
            support_sentence_count += 1
            if best_score < keep_threshold + 0.08 and overlap < 0.55 and evidence_has_more_detail:
                final_sentence = best_evidence_sentence
                repaired_sentence_count += 1
        elif best_score >= rewrite_gate:
            detail_mismatch = bool(query_detail_pattern.search(best_evidence_sentence)) and not bool(query_detail_pattern.search(sentence))
            if overlap >= 0.20 or detail_mismatch:
                final_sentence = best_evidence_sentence
                repaired_sentence_count += 1
                support_sentence_count += 1
            else:
                continue
        else:
            continue

        final_sentence = clean_text(final_sentence, limit=220)
        if not final_sentence:
            continue
        if final_sentence[-1:] not in "。！？!?；;":
            final_sentence = f"{final_sentence}。"
        signature = normalize_text(final_sentence)
        if not signature or signature in seen_sentences:
            continue
        seen_sentences.add(signature)
        revised_sentences.append(final_sentence)

    if not revised_sentences:
        best_sentence = max(
            answer_sentences,
            key=lambda sentence: max(
                [
                    1.0 / (1.0 + math.exp(-float(score)))
                    for score in raw_scores[
                        answer_sentences.index(sentence) * len(evidence_sentences):(answer_sentences.index(sentence) + 1) * len(evidence_sentences)
                    ]
                ]
                or [0.0]
            ),
        )
        fallback = clean_text(best_sentence, limit=220) or NO_CONTEXT_ANSWER
        if fallback[-1:] not in "。！？!?；;":
            fallback = f"{fallback}。"
        return RepairedAnswer(text=fallback, support_sentence_count=1, repaired_sentence_count=0)

    separator = "\n" if len(revised_sentences) >= 3 else " "
    return RepairedAnswer(
        text=separator.join(revised_sentences),
        support_sentence_count=support_sentence_count,
        repaired_sentence_count=repaired_sentence_count,
    )


def extract_query_aligned_spans(
    query_tokens: Sequence[str],
    evidence_units: Sequence[dict[str, Any]],
    *,
    max_spans: int,
) -> list[SpanConstraint]:
    """抽取与问题直接对齐的证据片段。"""
    query_token_set = set(query_tokens)
    candidates: list[tuple[float, str]] = []
    seen: set[str] = set()

    def add_candidate(text: str, score: float) -> None:
        """向候选结果集中加入一条新候选。"""
        cleaned = clean_text(text, limit=40)
        signature = normalize_text(cleaned)
        if not signature or signature in seen:
            return
        seen.add(signature)
        candidates.append((score, cleaned))

    for unit in evidence_units:
        for sentence in split_answer_sentences(str(unit.get("text") or "")):
            token_set = set(tokenize_text(sentence))
            coverage = coverage_ratio(query_token_set, token_set)
            if coverage < 0.10:
                continue
            for match in re.finditer(
                r"(?:约|近|逾|超|超过|至少|不足|将近)?\d+(?:\.\d+)?(?:万|亿|千|百)?(?:%|％|年|月|日|元|美元|亿元|亿美元|人|名|个|件|项|次|架|枚|门|套|公里|米|吨|级|号)?",
                sentence,
            ):
                span = match.group(0)
                if re.search(r"\d", span):
                    score = 0.58 * coverage + 0.30 + 0.12 * min(len(span), 12) / 12.0
                    add_candidate(span, score)

            sentence_tokens = [token for token in tokenize_text(sentence) if normalize_text(token)]
            token_count = len(sentence_tokens)
            for start in range(token_count):
                span_tokens: list[str] = []
                for end in range(start, min(token_count, start + 4)):
                    token = sentence_tokens[end]
                    if re.fullmatch(r"[，,。.!！？?；;：、]+", token):
                        break
                    span_tokens.append(token)
                    span_text = "".join(span_tokens).strip()
                    if len(span_text) < 2 or len(span_text) > 24:
                        continue
                    span_token_set = set(span_tokens)
                    fragment_coverage = coverage_ratio(query_token_set, span_token_set)
                    detail_signal = 1.0 if re.search(r"\d|[%年月日:：\-]|[A-Z]{2,}", span_text) else 0.0
                    if fragment_coverage < 0.08 and detail_signal <= 0.0:
                        continue
                    score = (
                        0.50 * coverage
                        + 0.30 * fragment_coverage
                        + 0.12 * detail_signal
                        + 0.08 * min(len(span_text), 12) / 12.0
                    )
                    add_candidate(span_text, score)

    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    selected: list[SpanConstraint] = []
    selected_signatures: list[str] = []
    for score, text in candidates:
        signature = normalize_text(text)
        if any(signature in chosen or chosen in signature for chosen in selected_signatures):
            continue
        selected.append(SpanConstraint(text=text, score=score))
        selected_signatures.append(signature)
        if len(selected) >= max_spans:
            break
    return selected


def build_grounded_synthesis_prompt(query: str, facts: Sequence[EvidenceFact]) -> str:
    """构建基于事实综合作答的提示词。"""
    fact_lines = [f"[{fact.fact_id}] {fact.statement} (sources: {', '.join(fact.source_ids)})" for fact in facts]
    fact_block = "\n".join(fact_lines)
    return (
        "你是一个严格的答案综合器。请仅根据给定事实回答问题，并只输出 JSON。\n"
        "输出格式严格为："
        '{"sentences":[{"text":"...","fact_ids":["F1","F2"]}]}\n'
        "要求：\n"
        "1. 先判断问题需要几个信息点，再逐句回答，每句只保留一个清晰信息点。\n"
        "2. 每句必须直接回应问题的一部分，并且 fact_ids 只能引用给定 fact。\n"
        "3. 如果需要列举多个并列要点，请把每个要点拆成单独 sentence，保持原有标点，不要把多项压成一整段。\n"
        "4. 先给出最小充分回答，再补充必要要点；不要把用户问题中的地点、机构或条件原样拷回答案，除非事实表明确包含它。\n"
        "5. 不要使用事实表之外的信息，不要添加来源说明。\n"
        f"6. 如果事实不足，请输出：{{\"sentences\":[{{\"text\":\"{NO_CONTEXT_ANSWER}\",\"fact_ids\":[]}}]}}\n\n"
        f"用户问题：{query}\n\n"
        f"事实表：\n{fact_block}\n\n"
        "JSON："
    )


def build_fact_repair_prompt(query: str, draft_answer: str, facts: Sequence[EvidenceFact]) -> str:
    """构建事实修补结果提示词。"""
    fact_lines = [f"[{fact.fact_id}] {fact.statement} (sources: {', '.join(fact.source_ids)})" for fact in facts]
    fact_block = "\n".join(fact_lines)
    return (
        "你是一个严格的答案修订器。请根据草稿答案和事实表，输出修订后的最终答案，并只输出 JSON。\n"
        "输出格式严格为："
        '{"sentences":[{"text":"...","fact_ids":["F1","F2"]}]}\n'
        "要求：\n"
        "1. 草稿答案中已经正确且贴合问题结构的内容尽量保留，不要无意义重写。\n"
        "2. 如果草稿遗漏了问题要求的关键维度、对象、条件或对比关系，必须补齐。\n"
        "3. 如果草稿把事实写得过于概括，应用事实表中的数值、名称、时间、地点和范围细节修正。\n"
        "4. 每句必须直接回答问题的一部分，并且 fact_ids 只能引用给定事实。\n"
        "5. 保持最小充分回答，不要添加背景总结、来源说明或与问题无关的解释。\n"
        f"6. 如果事实不足，请输出：{{\"sentences\":[{{\"text\":\"{NO_CONTEXT_ANSWER}\",\"fact_ids\":[]}}]}}\n\n"
        f"用户问题：{query}\n\n"
        f"草稿答案：\n{draft_answer}\n\n"
        f"事实表：\n{fact_block}\n\n"
        "JSON："
    )


def build_structured_evidence_answer_prompt(
    query: str,
    evidence_units: Sequence[dict[str, Any]],
    facts: Sequence[EvidenceFact],
) -> str:
    """构建结构化证据作答提示词。"""
    fact_lines = [f"[{fact.fact_id}] {fact.statement} (sources: {', '.join(fact.source_ids)})" for fact in facts]
    raw_blocks = []
    for index, unit in enumerate(evidence_units, start=1):
        raw_blocks.append(f"[{index}] {unit['title']}\n{unit['text']}")
    return (
        "你是一个严格的证据问答助手。请仅根据给定的结构化事实和原始证据回答问题。\n"
        "你需要先在心里根据结构化事实整理回答维度，再输出最终答案；不要展示思考过程。\n"
        "- 结构化事实是已经按问题筛过的一组高价值证据，应优先用它们覆盖回答的关键维度。\n"
        "- 原始证据只用于补齐结构化事实里未写全的对象、时间、数字、地点、范围或限定条件。\n"
        "- 如果问题包含多个对象、事件、地区、措施、结果或对比维度，不要把它们压成笼统总结；应保持一一对应。\n"
        "- 若事实中已经给出可直接填入问题的专名、数值、日期或结论，优先复用原词，不做无必要改写。\n"
        "- 回答保持最小充分，不要添加背景铺垫、来源说明或与问题无关的概括。\n"
        f"- 如果证据不足以得到可靠答案，原样回答：{NO_CONTEXT_ANSWER}\n\n"
        f"结构化事实：\n{chr(10).join(fact_lines)}\n\n"
        f"原始证据：\n{chr(10).join(raw_blocks)}\n\n"
        f"用户问题：{query}\n"
        "回答："
    )


def build_slot_packed_answer_prompt(
    query: str,
    packed_units: Sequence[dict[str, Any]],
) -> str:
    """构建按槽位打包证据的作答提示词。"""
    blocks = []
    for index, unit in enumerate(packed_units, start=1):
        blocks.append(f"[{index}] {unit['title']}\n{unit['text']}")
    context = "\n\n".join(blocks)
    return (
        "你是一个严格的证据问答助手。请仅根据下列回答槽位证据回答用户问题。\n"
        "每个证据块都对应问题中一个必须覆盖的回答维度。请先在心里检查每个维度是否都被覆盖，再输出最终答案；不要展示思考过程。\n"
        "- 优先按证据块对应的维度逐一覆盖；相邻维度可以自然合并成连贯句子，但不得遗漏任何有证据支持的维度。\n"
        "- 每个维度中的对象、事件、时间、数量、地点、范围、原因或措施要一一对应，不要跨维度混并。\n"
        "- 若证据中已经给出可直接填入问题的专名、数字、日期或结论，优先复用原词，不做无必要改写。\n"
        "- 回答保持最小充分，不要补充背景铺垫，不要输出编号、项目符号、JSON、来源说明，也不要提“槽位”或“证据块”。\n"
        f"- 如果证据不足以得到可靠答案，原样回答：{NO_CONTEXT_ANSWER}\n\n"
        f"回答槽位证据：\n{context}\n\n"
        f"用户问题：{query}\n"
        "回答："
    )


def build_plan_guided_answer_prompt(
    query: str,
    evidence_units: Sequence[dict[str, Any]],
    plan: StructuredPlan,
    facts: Sequence[EvidenceFact],
) -> str:
    """构建按计划引导作答的提示词。"""
    fact_map = {fact.fact_id: fact for fact in facts}
    slot_blocks: list[str] = []
    for slot in plan.slots:
        fact_lines = []
        for fact_id in slot.fact_ids[:2]:
            fact = fact_map.get(fact_id)
            if fact is not None:
                fact_lines.append(f"- {fact.statement}")
        if not fact_lines:
            continue
        slot_blocks.append(f"[{slot.slot_id}] {slot.prompt}\n" + "\n".join(fact_lines))

    evidence_blocks = []
    for index, unit in enumerate(evidence_units, start=1):
        evidence_blocks.append(f"[{index}] {unit['title']}\n{unit['text']}")

    return (
        "你是一个严格的证据问答助手。请仅根据下列回答维度提示和原始证据回答用户问题。\n"
        "回答维度提示列出了问题必须覆盖的几个信息点；你需要先在心里确认这些维度都被回答，再输出最终答案，不要展示思考过程。\n"
        "- 优先按回答维度逐一覆盖，但最终答案必须是自然、连贯的正文，不要输出编号、项目符号、JSON、来源说明，也不要提“维度”或“槽位”。\n"
        "- 每个维度中的对象、事件、时间、数量、地点、范围、原因或措施要一一对应，不要把不同维度的信息混成笼统概述。\n"
        "- 回答维度提示中的事实优先级高于自由概括；若原始证据给出更完整的限定信息，可用原始证据补足。\n"
        "- 若证据中已经给出可直接填入问题的专名、数字、日期或结论，优先复用原词，不做无必要改写。\n"
        "- 回答保持最小充分，不要添加背景铺垫或与问题无关的总结。\n"
        f"- 如果证据不足以得到可靠答案，原样回答：{NO_CONTEXT_ANSWER}\n\n"
        f"回答维度提示：\n{chr(10).join(slot_blocks)}\n\n"
        f"原始证据：\n{chr(10).join(evidence_blocks)}\n\n"
        f"用户问题：{query}\n"
        "回答："
    )


def build_clause_guided_answer_prompt(
    query: str,
    clauses: Sequence[str],
    clause_units: Sequence[dict[str, Any]],
    evidence_units: Sequence[dict[str, Any]],
) -> str:
    """构建按问题子句引导作答的提示词。"""
    clause_blocks = []
    for index, clause in enumerate(clauses, start=1):
        support_text = ""
        if index - 1 < len(clause_units):
            support_text = str(clause_units[index - 1].get("text") or "")
        clause_blocks.append(f"[C{index}] 需要覆盖：{clause}\n支持句：{support_text}")

    evidence_blocks = []
    for index, unit in enumerate(evidence_units, start=1):
        evidence_blocks.append(f"[{index}] {unit['title']}\n{unit['text']}")

    return (
        "你是一个严格的证据问答助手。请仅根据下列回答分句提示和原始证据回答用户问题。\n"
        "回答分句提示来自用户问题本身的并列结构。请按这些分句的顺序覆盖关键信息，再输出最终答案；不要展示思考过程。\n"
        "- 每个分句对应一个必须回答的信息点，不要遗漏，也不要把不同分句的信息混成笼统总结。\n"
        "- 如果原始证据中给出更完整的对象、时间、数量、地点、范围、原因或措施，可在对应分句下补足，但不要跨分句挪用。\n"
        "- 若证据中已经给出可直接填入问题的专名、数字、日期或结论，优先复用原词，不做无必要改写。\n"
        "- 回答保持最小充分，不要输出编号、项目符号、JSON、来源说明，也不要提“分句提示”。\n"
        f"- 如果证据不足以得到可靠答案，原样回答：{NO_CONTEXT_ANSWER}\n\n"
        f"回答分句提示：\n{chr(10).join(clause_blocks)}\n\n"
        f"原始证据：\n{chr(10).join(evidence_blocks)}\n\n"
        f"用户问题：{query}\n"
        "回答："
    )


def build_structured_plan_prompt(query: str, facts: Sequence[EvidenceFact]) -> str:
    """构建结构化回答计划提示词。"""
    fact_lines = [f"[{fact.fact_id}] {fact.statement}" for fact in facts]
    fact_block = "\n".join(fact_lines)
    return (
        "你是一个问题结构分析器。请根据问题和事实表，输出回答必须覆盖的信息槽位，并只输出 JSON。\n"
        "输出格式严格为："
        '{"slots":[{"slot_id":"S1","prompt":"需要回答的一个具体信息点","fact_ids":["F1"]}]}\n'
        "要求：\n"
        "1. 每个 slot 只对应一个明确的信息点，不要写背景介绍。\n"
        "2. slot 必须贴合用户问题的结构，不要凭空新增问题未要求的维度。\n"
        "3. fact_ids 只能引用给定事实，且至少引用一个。\n"
        "4. 槽位数量保持最小充分，一般为 2 到 5 个。\n\n"
        f"用户问题：{query}\n\n"
        f"事实表：\n{fact_block}\n\n"
        "JSON："
    )


def parse_structured_plan(payload: dict[str, Any], facts: Sequence[EvidenceFact]) -> StructuredPlan:
    """解析结构化计划。"""
    fact_map = {fact.fact_id: fact for fact in facts}
    raw_slots = payload.get("slots")
    if not isinstance(raw_slots, list):
        return StructuredPlan(slots=())

    slots: list[StructuredSlot] = []
    for index, item in enumerate(raw_slots, start=1):
        if not isinstance(item, dict):
            continue
        prompt = clean_text(str(item.get("prompt") or item.get("slot") or ""), limit=120)
        if not prompt:
            continue
        raw_fact_ids = item.get("fact_ids") or item.get("supports") or []
        if isinstance(raw_fact_ids, str):
            raw_fact_ids = [raw_fact_ids]
        fact_ids = tuple(str(value).strip() for value in raw_fact_ids if str(value).strip() in fact_map)
        if not fact_ids:
            continue
        slot_id = clean_text(str(item.get("slot_id") or f"S{index}"), limit=12)
        slots.append(StructuredSlot(slot_id=slot_id or f"S{index}", prompt=prompt, fact_ids=fact_ids))
    return StructuredPlan(slots=tuple(slots))


def build_slot_answer_prompt(query: str, facts: Sequence[EvidenceFact], plan: StructuredPlan) -> str:
    """构建槽位答案提示词。"""
    fact_lines = [f"[{fact.fact_id}] {fact.statement}" for fact in facts]
    slot_lines = [f"[{slot.slot_id}] {slot.prompt} (facts: {', '.join(slot.fact_ids)})" for slot in plan.slots]
    return (
        "你是一个严格的结构化答题器。请根据给定事实，逐槽位填写答案，并只输出 JSON。\n"
        "输出格式严格为："
        '{"answers":[{"slot_id":"S1","text":"...", "fact_ids":["F1"]}]}\n'
        "要求：\n"
        "1. 每个答案只回答对应槽位，不要加入背景概述。\n"
        "2. 必须保留数值、名称、日期、条件等关键细节。\n"
        "3. fact_ids 只能引用该槽位允许的事实。\n"
        "4. 不要输出槽位之外的信息。\n"
        f"5. 如果某个槽位无法回答，text 输出：{NO_CONTEXT_ANSWER}\n\n"
        f"用户问题：{query}\n\n"
        f"事实表：\n{chr(10).join(fact_lines)}\n\n"
        f"槽位表：\n{chr(10).join(slot_lines)}\n\n"
        "JSON："
    )


def build_slot_repair_prompt(query: str, draft_answer: str, facts: Sequence[EvidenceFact], plan: StructuredPlan) -> str:
    """构建槽位修补结果提示词。"""
    fact_lines = [f"[{fact.fact_id}] {fact.statement}" for fact in facts]
    slot_lines = [f"[{slot.slot_id}] {slot.prompt} (facts: {', '.join(slot.fact_ids)})" for slot in plan.slots]
    return (
        "你是一个答案修订器。请根据给定事实和槽位表修订草稿答案，并只输出 JSON。\n"
        "输出格式严格为："
        '{"answers":[{"slot_id":"S1","text":"...", "fact_ids":["F1"]}]}\n'
        "要求：\n"
        "1. 只修正草稿答案中未覆盖、错位或过于概括的部分；已正确且贴合问题的内容可保留。\n"
        "2. 每个 slot 都应对应一个直接回答该槽位的信息句；如果草稿把多个槽位混在一起，应拆开作答。\n"
        "3. 必须保留数值、名称、日期、地点、范围等关键细节，不要泛化成空洞概述。\n"
        "4. fact_ids 只能引用该槽位允许的事实。\n"
        "5. 不要输出槽位之外的背景总结，不要复述与问题无关的解释。\n"
        f"6. 如果某个槽位无法可靠回答，text 输出：{NO_CONTEXT_ANSWER}\n\n"
        f"用户问题：{query}\n\n"
        f"事实表：\n{chr(10).join(fact_lines)}\n\n"
        f"槽位表：\n{chr(10).join(slot_lines)}\n\n"
        f"草稿答案：\n{draft_answer}\n\n"
        "JSON："
    )


def parse_slot_answers(payload: dict[str, Any], plan: StructuredPlan, facts: Sequence[EvidenceFact]) -> SupportedAnswer:
    """解析槽位答案列表。"""
    fact_map = {fact.fact_id: fact for fact in facts}
    slot_map = {slot.slot_id: slot for slot in plan.slots}
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, list):
        return SupportedAnswer(text=NO_CONTEXT_ANSWER, support_sentence_count=0, covered_fact_ids=())

    accepted: list[str] = []
    covered_fact_ids: list[str] = []
    seen: set[str] = set()
    for item in raw_answers:
        if not isinstance(item, dict):
            continue
        slot_id = str(item.get("slot_id") or "").strip()
        slot = slot_map.get(slot_id)
        if slot is None:
            continue
        text = clean_text(str(item.get("text") or ""), limit=220)
        if not text or normalize_text(text) == normalize_text(NO_CONTEXT_ANSWER):
            continue
        raw_fact_ids = item.get("fact_ids") or []
        if isinstance(raw_fact_ids, str):
            raw_fact_ids = [raw_fact_ids]
        fact_ids = [fact_id for fact_id in (str(value).strip() for value in raw_fact_ids) if fact_id in fact_map and fact_id in slot.fact_ids]
        if not fact_ids:
            continue
        support_tokens = set()
        for fact_id in fact_ids:
            support_tokens.update(tokenize_text(fact_map[fact_id].statement))
        sentence_tokens = set(tokenize_text(text))
        if sentence_tokens and support_tokens:
            sentence_support = _support_ratio(sentence_tokens, support_tokens)
            fact_support = _support_ratio(support_tokens, sentence_tokens)
            if sentence_support < 0.45 and fact_support < 0.90:
                continue
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        accepted.append(text if text[-1:] in "。！？!?；;" else f"{text}。")
        for fact_id in fact_ids:
            if fact_id not in covered_fact_ids:
                covered_fact_ids.append(fact_id)

    if not accepted:
        return SupportedAnswer(text=NO_CONTEXT_ANSWER, support_sentence_count=0, covered_fact_ids=())
    separator = "\n" if len(accepted) >= 3 else " "
    return SupportedAnswer(
        text=separator.join(accepted),
        support_sentence_count=len(accepted),
        covered_fact_ids=tuple(covered_fact_ids),
    )


def build_slot_verification_prompt(query: str, draft_answer: str, facts: Sequence[EvidenceFact]) -> str:
    """构建槽位核验阶段的提示词。"""
    fact_lines = [f"[{fact.fact_id}] {fact.statement}" for fact in facts]
    return (
        "你是一个答案校验器。请检查草稿答案中的每个陈述是否被事实表支持，并只输出 JSON。\n"
        "输出格式严格为："
        '{"sentences":[{"text":"原句","verdict":"supported|unsupported|conflict","rewrite":"若需修正则填写修正版","fact_ids":["F1"]}]}\n'
        "要求：\n"
        "1. verdict 只能是 supported、unsupported、conflict 之一。\n"
        "2. unsupported 或 conflict 时，rewrite 必须改写为被事实表直接支持的表述；若无法修正则填空。\n"
        "3. supported 时，rewrite 为空字符串。\n"
        "4. fact_ids 只能引用给定事实。\n\n"
        f"用户问题：{query}\n\n"
        f"事实表：\n{chr(10).join(fact_lines)}\n\n"
        f"草稿答案：\n{draft_answer}\n\n"
        "JSON："
    )


def parse_verified_answer(payload: dict[str, Any], facts: Sequence[EvidenceFact]) -> SupportedAnswer:
    """解析已核验答案。"""
    fact_map = {fact.fact_id: fact for fact in facts}
    raw_sentences = payload.get("sentences")
    if not isinstance(raw_sentences, list):
        return SupportedAnswer(text=NO_CONTEXT_ANSWER, support_sentence_count=0, covered_fact_ids=())

    verified_lines: list[str] = []
    covered_fact_ids: list[str] = []
    for item in raw_sentences:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "").strip().lower()
        candidate = item.get("text") if verdict == "supported" else item.get("rewrite")
        text = clean_text(str(candidate or ""), limit=220)
        if not text:
            continue
        raw_fact_ids = item.get("fact_ids") or []
        if isinstance(raw_fact_ids, str):
            raw_fact_ids = [raw_fact_ids]
        fact_ids = [fact_id for fact_id in (str(value).strip() for value in raw_fact_ids) if fact_id in fact_map]
        if not fact_ids:
            continue
        if verdict not in {"supported", "unsupported", "conflict"}:
            continue
        support_tokens = set()
        for fact_id in fact_ids:
            support_tokens.update(tokenize_text(fact_map[fact_id].statement))
        sentence_tokens = set(tokenize_text(text))
        if sentence_tokens and support_tokens and _support_ratio(sentence_tokens, support_tokens) < 0.45:
            continue
        verified_lines.append(text if text[-1:] in "。！？!?；;" else f"{text}。")
        for fact_id in fact_ids:
            if fact_id not in covered_fact_ids:
                covered_fact_ids.append(fact_id)
    if not verified_lines:
        return SupportedAnswer(text=NO_CONTEXT_ANSWER, support_sentence_count=0, covered_fact_ids=())
    separator = "\n" if len(verified_lines) >= 3 else " "
    return SupportedAnswer(
        text=separator.join(verified_lines),
        support_sentence_count=len(verified_lines),
        covered_fact_ids=tuple(covered_fact_ids),
    )


def parse_supported_answer(payload: dict[str, Any], facts: Sequence[EvidenceFact]) -> SupportedAnswer:
    """解析有支撑答案。"""
    fact_map = {fact.fact_id: fact for fact in facts}
    raw_sentences = payload.get("sentences")
    if raw_sentences is None:
        raw_sentences = payload.get("answer_sentences")
    if isinstance(raw_sentences, dict):
        raw_sentences = [raw_sentences]
    elif not isinstance(raw_sentences, list):
        singleton_text = clean_text(
            str(
                payload.get("text")
                or payload.get("sentence")
                or payload.get("answer")
                or ""
            ),
            limit=220,
        )
        singleton_fact_ids = payload.get("fact_ids") or payload.get("supports") or []
        if singleton_text:
            raw_sentences = [{"text": singleton_text, "fact_ids": singleton_fact_ids}]
        else:
            raw_sentences = []

    accepted_sentences: list[str] = []
    seen_normalized: set[str] = set()
    covered_fact_ids: list[str] = []
    for item in raw_sentences:
        if not isinstance(item, dict):
            continue
        text = clean_text(str(item.get("text") or item.get("sentence") or ""), limit=220)
        if not text:
            continue
        if normalize_text(text) == normalize_text(NO_CONTEXT_ANSWER):
            return SupportedAnswer(text=NO_CONTEXT_ANSWER, support_sentence_count=0, covered_fact_ids=())
        raw_fact_ids = item.get("fact_ids") or item.get("supports") or []
        if isinstance(raw_fact_ids, str):
            raw_fact_ids = [raw_fact_ids]
        fact_ids = [str(value).strip() for value in raw_fact_ids if str(value).strip() in fact_map]
        if not fact_ids:
            continue
        support_token_set = set()
        for fact_id in fact_ids:
            support_token_set.update(tokenize_text(fact_map[fact_id].statement))
        sentence_token_set = set(tokenize_text(text))
        if sentence_token_set and support_token_set and _support_ratio(sentence_token_set, support_token_set) < 0.4:
            continue
        normalized = normalize_text(text)
        if not normalized or normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        accepted_sentences.append(text)
        for fact_id in fact_ids:
            if fact_id not in covered_fact_ids:
                covered_fact_ids.append(fact_id)

    if accepted_sentences:
        # 如果结果天然更像条目式输出，就保留这种结构，
        # 不强行压平成一整段，避免丢掉分项回答的可读性。
        short_fragments = [sentence for sentence in accepted_sentences if len(tokenize_text(sentence)) <= 16]
        if len(accepted_sentences) >= 3 and len(short_fragments) >= max(2, len(accepted_sentences) - 1):
            lines = [f"{index}. {sentence}" for index, sentence in enumerate(accepted_sentences, start=1)]
            return SupportedAnswer(
                text="\n".join(lines),
                support_sentence_count=len(accepted_sentences),
                covered_fact_ids=tuple(covered_fact_ids),
            )

        formatted_lines = [sentence if sentence[-1:] in "。！？!?；;" else f"{sentence}。" for sentence in accepted_sentences]
        separator = " " if len(formatted_lines) <= 2 else "\n"
        return SupportedAnswer(
            text=separator.join(formatted_lines),
            support_sentence_count=len(accepted_sentences),
            covered_fact_ids=tuple(covered_fact_ids),
        )
    return SupportedAnswer(text=NO_CONTEXT_ANSWER, support_sentence_count=0, covered_fact_ids=())


def render_fact_answer(facts: Sequence[EvidenceFact]) -> SupportedAnswer:
    """把事实列表整理成最终答案文本。"""
    statements = [clean_text(fact.statement, limit=220) for fact in facts if clean_text(fact.statement, limit=220)]
    if not statements:
        return SupportedAnswer(text=NO_CONTEXT_ANSWER, support_sentence_count=0, covered_fact_ids=())

    if len(statements) >= 3:
        lines = [f"{index}. {statement}" for index, statement in enumerate(statements, start=1)]
        return SupportedAnswer(
            text="\n".join(lines),
            support_sentence_count=len(statements),
            covered_fact_ids=tuple(fact.fact_id for fact in facts[: len(statements)]),
        )

    formatted_lines = [statement if statement[-1:] in "。！？!?；;" else f"{statement}。" for statement in statements]
    return SupportedAnswer(
        text=" ".join(formatted_lines),
        support_sentence_count=len(formatted_lines),
        covered_fact_ids=tuple(fact.fact_id for fact in facts[: len(formatted_lines)]),
    )
