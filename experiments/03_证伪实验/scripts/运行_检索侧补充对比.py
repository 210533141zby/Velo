"""比较检索侧替代路线在同一批复杂问答上的表现。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = ROOT.parent
IMPL_ROOT = EXPERIMENTS_ROOT / "04_算法实现"
if str(IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL_ROOT))

from retrieval_pipeline.common import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, ensure_dir
from retrieval_pipeline.datasets import load_crud_cases
from retrieval_pipeline.metrics import evaluate_crud_results
from retrieval_pipeline.pipeline import PipelineVariant, RagExperimentPipeline

OUTPUT_ROOT = ROOT / "results" / "03_检索侧补充对比"

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

VARIANTS = {
    "baseline": PipelineVariant(
        key="baseline_rrf_rerank_direct",
        label="基线",
        use_rerank=True,
        answer_prompt_style="simple",
        multi_snippet_count=2,
    ),
    "contextual": PipelineVariant(
        key="contextual_rrf_rerank_direct",
        label="上下文化检索",
        retrieval_strategy="contextual",
        use_rerank=True,
        answer_prompt_style="simple",
        multi_snippet_count=2,
    ),
    "rewrite": PipelineVariant(
        key="rewrite_rrf_rerank_direct",
        label="查询改写",
        use_rerank=True,
        use_query_rewrite=True,
        answer_prompt_style="simple",
        multi_snippet_count=2,
    ),
    "parent_child": PipelineVariant(
        key="parent_child_rrf_rerank_direct",
        label="父子块检索",
        retrieval_strategy="parent_child",
        use_rerank=True,
        answer_prompt_style="simple",
        multi_snippet_count=2,
    ),
}

SUMMARY_FILE_NAMES = {
    "baseline": "检索侧_基线_结果汇总.json",
    "contextual": "检索侧_上下文化检索_结果汇总.json",
    "rewrite": "检索侧_查询改写_结果汇总.json",
    "parent_child": "检索侧_父子块检索_结果汇总.json",
}

DETAIL_FILE_NAMES = {
    "baseline": "检索侧_基线_逐题明细.json",
    "contextual": "检索侧_上下文化检索_逐题明细.json",
    "rewrite": "检索侧_查询改写_逐题明细.json",
    "parent_child": "检索侧_父子块检索_逐题明细.json",
}


def run_variant_on_cases(pipeline: RagExperimentPipeline, prepared, variant: PipelineVariant):
    """在同一批复杂问答样例上运行某条检索侧替代路线。

    这里保持问题集合完全一致，确保不同检索策略之间的差异只来自方法本身，
    不来自样本波动。
    """
    results = []
    total = len(prepared.cases)
    for index, case in enumerate(prepared.cases, start=1):
        if index == 1 or index == total:
            print(f"[retrieval-compare] {variant.key}: {index}/{total}", flush=True)
        results.append(pipeline.run_case(prepared, case, variant))
    return results


def load_complex_cases():
    """加载论文检索侧对比使用的复杂问答题集，并校验样例是否齐全。"""
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
    if len(selected_cases) != len(COMPLEX_CASE_IDS):
        found = {case.case_id for case in selected_cases}
        missing = [case_id for case_id in COMPLEX_CASE_IDS if case_id not in found]
        raise RuntimeError(f"缺少复杂验证样例: {missing}")
    return selected_cases, docs


def main() -> None:
    """组织检索侧补充对比实验的主执行流程。"""
    parser = argparse.ArgumentParser(description="运行检索侧补充对比实验。")
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    args = parser.parse_args()

    ensure_dir(OUTPUT_ROOT)
    cache_root = ensure_dir(ROOT / ".cache")

    selected_cases, docs = load_complex_cases()
    variant = VARIANTS[args.variant]
    pipeline = RagExperimentPipeline(
        cache_root=cache_root,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        llm_model=args.llm_model,
    )
    prepared = pipeline.prepare_dataset(
        "crud_retrieval_compare_batch",
        selected_cases,
        docs,
        include_contextual=variant.retrieval_strategy == "contextual",
        include_parent_child=variant.retrieval_strategy == "parent_child",
        include_query_rewrite=variant.use_query_rewrite,
    )
    results = run_variant_on_cases(pipeline, prepared, variant)
    evaluation = evaluate_crud_results(
        variant.key,
        results,
        selected_cases,
        ragas_case_ids=(),
        qa_ragas_case_ids=(),
        multidoc_ragas_case_ids=(),
        enable_ragas=False,
    )
    summary = dict(evaluation.summary)
    summary["label"] = variant.label
    payload = {"case_ids": list(COMPLEX_CASE_IDS), "summaries": [summary]}
    suffix = args.variant
    (OUTPUT_ROOT / SUMMARY_FILE_NAMES[suffix]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / DETAIL_FILE_NAMES[suffix]).write_text(
        json.dumps(evaluation.detail_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
