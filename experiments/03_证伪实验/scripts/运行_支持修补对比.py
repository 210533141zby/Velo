"""比较 baseline、对齐作答与支持性修补三种生成路线。"""

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

VARIANTS = (
    PipelineVariant(
        key="baseline_simple",
        label="基线直接作答",
        use_rerank=True,
        answer_prompt_style="simple",
        multi_snippet_count=2,
    ),
    PipelineVariant(
        key="aligned_direct",
        label="对齐作答",
        use_rerank=True,
        answer_prompt_style="aligned",
        multi_snippet_count=2,
    ),
    PipelineVariant(
        key="support_repair",
        label="支持性修补",
        use_rerank=True,
        synthesis_mode="support_repair",
        answer_prompt_style="aligned",
        support_pruning_threshold=0.55,
        multi_snippet_count=2,
    ),
)


def run_variant_on_cases(pipeline: RagExperimentPipeline, prepared, variant: PipelineVariant):
    """在固定复杂问答样例集上运行某条生成路线。

    这组脚本只比较生成阶段差异，因此检索输入保持一致，避免混入检索侧变量。
    """
    results = []
    total = len(prepared.cases)
    for index, case in enumerate(prepared.cases, start=1):
        if index == 1 or index == total:
            print(f"[support-repair] {variant.key}: {index}/{total}", flush=True)
        results.append(pipeline.run_case(prepared, case, variant))
    return results


def main() -> None:
    """组织支持性修补对比实验的完整执行流程。"""
    parser = argparse.ArgumentParser(description="运行支持性修补对比实验。")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
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
    if len(selected_cases) != len(COMPLEX_CASE_IDS):
        found = {case.case_id for case in selected_cases}
        missing = [case_id for case_id in COMPLEX_CASE_IDS if case_id not in found]
        raise RuntimeError(f"缺少复杂验证样例: {missing}")

    summaries = []
    details = []
    for variant in VARIANTS:
        pipeline = RagExperimentPipeline(
            cache_root=cache_root,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            llm_model=args.llm_model,
        )
        prepared = pipeline.prepare_dataset(
            "crud_support_repair_batch",
            selected_cases,
            docs,
            include_contextual=False,
            include_parent_child=False,
            include_query_rewrite=False,
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
        summaries.append(summary)
        details.extend(evaluation.detail_rows)

    output = {
        "case_ids": list(COMPLEX_CASE_IDS),
        "llm_model": args.llm_model,
        "summaries": summaries,
    }
    (OUTPUT_ROOT / "生成侧_支持性修补对比_结果汇总.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "生成侧_支持性修补对比_逐题明细.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
