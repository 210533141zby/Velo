"""比较少样例锚定、自由式思维展开等生成侧提示路线。"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = ROOT.parent
IMPL_ROOT = EXPERIMENTS_ROOT / "04_算法实现"
if str(IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL_ROOT))

from retrieval_pipeline.adaptive_evidence import _sentence_fact_candidates
from retrieval_pipeline.common import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    NO_CONTEXT_ANSWER,
    clean_text,
    contains_refusal,
    ensure_dir,
    generate_answer,
    generate_json_payload,
    normalize_text,
    request_completion,
    rerank_documents,
    tokenize_text,
)
from retrieval_pipeline.datasets import load_crud_cases
from retrieval_pipeline.metrics import evaluate_crud_results
from retrieval_pipeline.pipeline import PipelineRunResult, PipelineVariant, RagExperimentPipeline

OUTPUT_ROOT = ROOT / "results" / "02_生成侧补充对比"

COMPLEX_CASE_IDS = (
    "questanswer_2docs_002",
    "questanswer_2docs_003",
    "questanswer_2docs_005",
    "questanswer_2docs_006",
    "questanswer_2docs_008",
    "questanswer_3docs_003",
    "questanswer_3docs_004",
    "questanswer_3docs_006",
)

BASELINE_VARIANT = PipelineVariant(
    key="baseline_rrf_rerank_direct",
    label="基线直接作答",
    use_rerank=True,
    answer_prompt_style="simple",
    final_source_count=3,
    complex_source_count=5,
)

VARIANT_LABELS = {
    "baseline_prompt4": "基线直接作答",
    "fewshot_icl_direct": "少样例风格锚定",
    "scratchpad_direct": "自由式思维展开",
}

VARIANT_FILE_STEMS = {
    "baseline_prompt4": "生成侧_基线直接作答",
    "fewshot_icl_direct": "生成侧_少样例风格锚定",
    "scratchpad_direct": "生成侧_自由式思维展开",
}


def format_evidence_units(evidence_units: Sequence[dict[str, Any]]) -> str:
    """把证据单元列表渲染成提示词可直接拼接的文本块。

    多条生成侧路线都会共享同一批证据输入，因此这里先统一格式化，
    避免不同提示词模板在证据展示层面引入额外变量。
    """
    blocks = []
    for index, unit in enumerate(evidence_units, start=1):
        blocks.append(f"[S{index}] {unit['title']}\n{unit['text']}")
    return "\n\n".join(blocks)


def extract_answer_tag(text: str) -> str:
    """从带标签的模型输出中提取 `<answer>` 段落，兼容自由式思维展开路线。"""
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        answer = clean_text(match.group(1), limit=600)
        return answer or NO_CONTEXT_ANSWER
    return clean_text(text, limit=600) or NO_CONTEXT_ANSWER


def parse_claim_strings(payload: dict[str, Any]) -> list[str]:
    """从 JSON 载荷中抽取可继续核验的陈述列表。

    生成侧某些路线会先让模型输出 claim 数组，这里负责兼容字符串项和对象项，
    并过滤掉空值与拒答文案。
    """
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        return []
    claims: list[str] = []
    for item in raw_claims:
        if isinstance(item, str):
            text = clean_text(item, limit=220)
        elif isinstance(item, dict):
            text = clean_text(str(item.get("text") or item.get("claim") or ""), limit=220)
        else:
            text = ""
        if text and normalize_text(text) != normalize_text(NO_CONTEXT_ANSWER):
            claims.append(text)
    return claims


def extract_numeric_tokens(text: str) -> set[str]:
    """提取文本中的数字片段，用于比对 claim 与引用是否数值一致。"""
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def best_support_score(
    pipeline: RagExperimentPipeline,
    statement: str,
    candidates: Sequence[str],
) -> tuple[float, str]:
    """在候选支撑句中找出与目标陈述最匹配的一句。

    该函数主要用于支持性修补等路线，判断一条陈述是否真能被现有证据句支撑，
    并返回最佳支撑强度及对应句子。
    """
    if not candidates:
        return 0.0, ""
    raw_scores = pipeline.reranker.predict(
        [(statement, candidate) for candidate in candidates],
        batch_size=16,
        show_progress_bar=False,
    ).tolist()
    best_index = max(range(len(candidates)), key=lambda index: float(raw_scores[index]))
    best_score = 1.0 / (1.0 + math.exp(-float(raw_scores[best_index])))
    return best_score, candidates[best_index]


def build_verified_facts_prompt(query: str, facts: Sequence[str]) -> str:
    """把已经核验过的事实重新组织成二次作答提示词。"""
    bullet_lines = [f"- {fact}" for fact in facts]
    return (
        "你是一个严格依据事实作答的问答助手。请仅根据下面已经核验过的事实回答问题。\n"
        "- 第一句必须直接回答问题。\n"
        "- 保留关键数值、日期、名称、条件和并列关系。\n"
        "- 如果问题需要多个事实点，请用自然语言连贯组织，但不要扩写背景。\n"
        f"- 如果事实仍不足，请原样回答：{NO_CONTEXT_ANSWER}\n\n"
        f"已核验事实：\n{chr(10).join(bullet_lines)}\n\n"
        f"用户问题：{query}\n"
        "回答："
    )


def build_fewshot_prompt(query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """构建少样例风格锚定路线的提示词。

    这里刻意放入“直接回答 + 融合多文档细节”的样例，目的是观察示例模仿
    能否改善复杂问答的答案组织质量。
    """
    context = format_evidence_units(evidence_units)
    return (
        "你是一个多文档问答助手。请模仿下面优秀回答的整合方式：先直接回答，再把不同文档中的关键细节缝合成自然语言，不要分点，不要补背景。\n\n"
        "[样例1]\n"
        "文档A：某企业2023年营业收入同比增长12%，净利润同比增长8%。\n"
        "文档B：该企业2023年研发投入同比增长15%，并推出三项核心产品。\n"
        "问题：2023年这家企业的经营表现和研发进展如何？\n"
        "优秀回答：2023年这家企业营业收入同比增长12%，净利润同比增长8%；同时研发投入同比增长15%，并推出了三项核心产品。\n\n"
        "[样例2]\n"
        "文档A：A地遭遇强降雨后启动Ⅲ级应急响应。\n"
        "文档B：B地转移群众1200人，并关闭沿河景区。\n"
        "问题：两地分别采取了哪些防灾措施？\n"
        "优秀回答：针对强降雨，A地启动了Ⅲ级应急响应；B地则转移了1200名群众，并关闭了沿河景区。\n\n"
        "[当前任务]\n"
        f"证据单元：\n{context}\n\n"
        f"问题：{query}\n"
        "优秀回答："
    )


def build_scratchpad_prompt(query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """构建自由式思维展开路线的提示词，显式要求模型先写推演再给答案。"""
    context = format_evidence_units(evidence_units)
    return (
        "请根据以下文档回答问题。\n"
        f"文档：\n{context}\n\n"
        f"问题：{query}\n\n"
        "在给出最终答案前，请在 <thinking> 和 </thinking> 标签之间，用自然语言自由推演这些文档之间的关系、数字和线索。\n"
        f"如果证据不足，请在 <answer> 中原样输出：{NO_CONTEXT_ANSWER}\n"
        "推演结束后，在 <answer> 标签内输出完整、连贯的最终回答。\n"
    )


def build_citation_claim_prompt(query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """构建带引用 claim 抽取提示词，供后续支持性修补路线使用。"""
    context = format_evidence_units(evidence_units)
    return (
        "你是一个带引用的答案起草器。请仅根据证据回答问题，并只输出 JSON。\n"
        "输出格式严格为："
        '{"claims":[{"text":"一个关键陈述","source_id":"S1","quote":"证据中的原句"}]}\n'
        "要求：\n"
        "1. 每条 claim 只包含一个关键陈述。\n"
        "2. source_id 只能填写对应证据编号。\n"
        "3. quote 必须是证据中的原句或连续片段。\n"
        "4. 不要输出背景概述，不要编造引用。\n\n"
        f"证据单元：\n{context}\n\n"
        f"问题：{query}\n"
        "JSON："
    )


def parse_citation_claims(payload: dict[str, Any], evidence_units: Sequence[dict[str, Any]]) -> list[str]:
    """检查 claim、来源编号和原句引用是否一致，并留下可继续利用的事实。

    只有当 `quote` 真的出现在对应证据单元里时，才认为这条 claim 具备继续
    进入修补步骤的资格。
    """
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        return []
    source_map = {f"S{index}": str(unit["text"]) for index, unit in enumerate(evidence_units, start=1)}
    verified: list[str] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        claim = clean_text(str(item.get("text") or ""), limit=220)
        source_id = clean_text(str(item.get("source_id") or ""), limit=8)
        quote = clean_text(str(item.get("quote") or ""), limit=220)
        source_text = source_map.get(source_id, "")
        if not claim or not quote or not source_text:
            continue
        if quote not in source_text:
            continue
        claim_numbers = extract_numeric_tokens(claim)
        quote_numbers = extract_numeric_tokens(quote)
        if claim_numbers and not claim_numbers.issubset(quote_numbers):
            verified.append(quote)
            continue
        verified.append(claim)
    return verified


def build_decompose_prompt(answer: str) -> str:
    """把一段现成答案拆成原子事实列表，供后验核验路线继续处理。"""
    return (
        "请把下面答案拆成原子事实列表，并只输出 JSON。\n"
        '格式严格为 {"claims":["事实1","事实2"]}\n'
        "要求：每条 claim 只包含一个明确陈述，不要重复。\n\n"
        f"答案：{answer}\n"
        "JSON："
    )


def mine_additional_facts(
    pipeline: RagExperimentPipeline,
    query: str,
    query_tokens: Sequence[str],
    evidence_units: Sequence[dict[str, Any]],
    covered: set[str],
    *,
    limit: int = 2,
) -> list[str]:
    """在已有支撑事实之外，再补充少量高分候选事实。

    这个步骤用于避免修补路线只会机械保留原 claim，而无法从现有证据中
    顺手补回遗漏的重要事实点。
    """
    facts = _sentence_fact_candidates(query_tokens, evidence_units, max_candidates_per_unit=2)
    candidates = [fact.statement for fact in facts if normalize_text(fact.statement) not in covered]
    if not candidates:
        return []
    raw_scores = pipeline.reranker.predict(
        [(query, candidate) for candidate in candidates],
        batch_size=16,
        show_progress_bar=False,
    ).tolist()
    ranked = sorted(
        zip(candidates, raw_scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    selected: list[str] = []
    for candidate, score in ranked:
        if len(selected) >= limit:
            break
        if 1.0 / (1.0 + math.exp(-float(score))) < 0.62:
            continue
        selected.append(candidate)
    return selected


def build_baseline_evidence(
    pipeline: RagExperimentPipeline,
    prepared,
    case,
):
    """按基线检索与重排口径构造一份统一证据输入。

    生成侧对比不改动检索主干，因此不同提示路线都会共享这里产出的证据集合。
    """
    features = prepared.query_features[case.case_id]
    candidate_indices, _ = pipeline._select_retrieval_view(prepared, features, BASELINE_VARIANT)
    reranked_doc_indices, doc_snippets, doc_scores = rerank_documents(
        pipeline.reranker,
        query=case.query,
        query_tokens=features.query_tokens,
        focus_tokens=features.query_tokens,
        candidate_doc_indices=candidate_indices,
        corpus_texts=[doc.content for doc in prepared.docs],
        multi_snippet_count=BASELINE_VARIANT.multi_snippet_count,
    )
    routing_units = pipeline._build_evidence_units(
        reranked_doc_indices=reranked_doc_indices,
        doc_snippets=doc_snippets,
        doc_scores=doc_scores,
        docs=prepared.docs,
        variant=BASELINE_VARIANT,
        source_limit=max(BASELINE_VARIANT.final_source_count, BASELINE_VARIANT.complex_source_count),
    )
    units = routing_units[: BASELINE_VARIANT.final_source_count]
    return features, units


def run_baseline_answer(pipeline: RagExperimentPipeline, query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """运行直接作答基线，作为生成侧所有替代路线的比较起点。"""
    return pipeline._run_direct_answer(query, evidence_units, style=BASELINE_VARIANT.answer_prompt_style)


def run_fewshot_answer(query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """运行少样例风格锚定路线。"""
    prompt = build_fewshot_prompt(query, evidence_units)
    return generate_answer(prompt, model_name=DEFAULT_LLM_MODEL)


def run_scratchpad_answer(query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """运行自由式思维展开路线，并从完整输出中截取最终答案。"""
    prompt = build_scratchpad_prompt(query, evidence_units)
    raw = request_completion(prompt, model_name=DEFAULT_LLM_MODEL, num_predict=640)
    return extract_answer_tag(raw)


def run_citation_revise_answer(
    pipeline: RagExperimentPipeline,
    query: str,
    evidence_units: Sequence[dict[str, Any]],
) -> str:
    """运行支持性修补路线中的“引用 claim 核对后再作答”版本。"""
    payload = generate_json_payload(
        build_citation_claim_prompt(query, evidence_units),
        model_name=DEFAULT_LLM_MODEL,
        num_predict=480,
    )
    verified_claims = parse_citation_claims(payload, evidence_units)
    if not verified_claims:
        return run_baseline_answer(pipeline, query, evidence_units)
    prompt = build_verified_facts_prompt(query, verified_claims)
    return generate_answer(prompt, model_name=DEFAULT_LLM_MODEL)


def run_decompose_verify_answer(
    pipeline: RagExperimentPipeline,
    query: str,
    query_tokens: Sequence[str],
    evidence_units: Sequence[dict[str, Any]],
) -> str:
    """运行“先拆解答案再逐条核验”的后验修补路线。"""
    baseline_answer = run_baseline_answer(pipeline, query, evidence_units)
    payload = generate_json_payload(
        build_decompose_prompt(baseline_answer),
        model_name=DEFAULT_LLM_MODEL,
        num_predict=320,
    )
    claims = parse_claim_strings(payload)
    if not claims:
        return baseline_answer

    fact_candidates = [fact.statement for fact in _sentence_fact_candidates(query_tokens, evidence_units, max_candidates_per_unit=3)]
    kept_facts: list[str] = []
    covered: set[str] = set()
    for claim in claims:
        score, support = best_support_score(pipeline, claim, fact_candidates)
        if score < 0.60 or not support:
            continue
        signature = normalize_text(support)
        if signature in covered:
            continue
        covered.add(signature)
        kept_facts.append(support)
    kept_facts.extend(mine_additional_facts(pipeline, query, query_tokens, evidence_units, covered))
    if not kept_facts:
        return baseline_answer
    prompt = build_verified_facts_prompt(query, kept_facts)
    return generate_answer(prompt, model_name=DEFAULT_LLM_MODEL)


def run_variant_case(
    pipeline: RagExperimentPipeline,
    prepared,
    case,
    variant_key: str,
) -> PipelineRunResult:
    """对单条样例执行指定生成路线，并整理成统一评测结果结构。"""
    started = time.perf_counter()
    features, evidence_units = build_baseline_evidence(pipeline, prepared, case)
    if variant_key == "baseline_prompt4":
        answer = run_baseline_answer(pipeline, case.query, evidence_units)
    elif variant_key == "fewshot_icl_direct":
        answer = run_fewshot_answer(case.query, evidence_units)
    elif variant_key == "scratchpad_direct":
        answer = run_scratchpad_answer(case.query, evidence_units)
    else:
        raise ValueError(f"Unknown variant: {variant_key}")

    latency_ms = (time.perf_counter() - started) * 1000.0
    source_doc_ids: list[int] = []
    source_titles: list[str] = []
    for unit in evidence_units:
        doc_id = int(unit["doc_id"])
        if doc_id in source_doc_ids:
            continue
        source_doc_ids.append(doc_id)
        source_titles.append(str(unit["title"]))
    return PipelineRunResult(
        case_id=case.case_id,
        query=case.query,
        response=answer or NO_CONTEXT_ANSWER,
        source_doc_ids=source_doc_ids,
        source_titles=source_titles,
        retrieved_contexts=[str(unit["text"]) for unit in evidence_units],
        latency_ms=latency_ms,
        predicted_refusal=contains_refusal(answer),
        route_mode=variant_key,
    )


def main() -> None:
    """组织生成侧补充对比实验的主执行流程。"""
    parser = argparse.ArgumentParser(description="在 CRUD 复杂问答批次上运行生成侧提示路线对比。")
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_LABELS),
        required=True,
    )
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    ensure_dir(OUTPUT_ROOT)
    cache_root = ensure_dir(ROOT / ".cache")

    cases, docs = load_crud_cases(
        summary_samples=6,
        qa_1doc_samples=6,
        qa_2doc_samples=8,
        qa_3doc_samples=8,
        hallu_samples=4,
        negative_samples=6,
        distractor_count=600,
        seed=42,
    )
    selected_cases = [case for case in cases if case.case_id in COMPLEX_CASE_IDS]

    pipeline = RagExperimentPipeline(
        cache_root=cache_root,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        llm_model=DEFAULT_LLM_MODEL,
    )
    prepared = pipeline.prepare_dataset(
        "crud_generation_prompt_compare_batch",
        selected_cases,
        docs,
        include_contextual=False,
        include_parent_child=False,
        include_query_rewrite=False,
    )

    results: list[PipelineRunResult] = []
    total = len(selected_cases)
    for index, case in enumerate(selected_cases, start=1):
        if index == 1 or index == total:
            print(f"[crud-prompt4] {args.variant}: {index}/{total}", flush=True)
        results.append(run_variant_case(pipeline, prepared, case, args.variant))

    evaluation = evaluate_crud_results(
        args.variant,
        results,
        selected_cases,
        ragas_case_ids=(),
        qa_ragas_case_ids=(),
        multidoc_ragas_case_ids=(),
        enable_ragas=False,
    )
    summary = dict(evaluation.summary)
    summary["label"] = VARIANT_LABELS[args.variant]
    payload = {"case_ids": list(COMPLEX_CASE_IDS), "summaries": [summary]}
    stem = VARIANT_FILE_STEMS[args.variant]
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    (OUTPUT_ROOT / f"{stem}_结果汇总{suffix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / f"{stem}_逐题明细{suffix}.json").write_text(
        json.dumps(evaluation.detail_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
