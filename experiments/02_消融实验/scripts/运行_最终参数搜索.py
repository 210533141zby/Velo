"""运行论文最终参数配置（4/6、5/7）的对比搜索。"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = ROOT.parent
IMPL_ROOT = EXPERIMENTS_ROOT / "04_算法实现"
if str(IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL_ROOT))

from retrieval_pipeline.common import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, ensure_dir
from retrieval_pipeline.datasets import load_crud_cases
from retrieval_pipeline.metrics import evaluate_crud_results
from retrieval_pipeline.pipeline import PipelineVariant, RagExperimentPipeline

OUTPUT_ROOT = ROOT / "results" / "02_最终参数搜索_20260517"
CRUD_SUBSET_SPLITS = {"questanswer_2docs": 797, "questanswer_3docs": 797}

VARIANTS = (
    PipelineVariant(
        key="base_router_4_6",
        label="当前最佳 4/6 + 按题作答",
        use_rerank=True,
        rerank_mode="aspect_aware_conservative",
        answer_prompt_style="task_router",
        multi_snippet_count=1,
        final_source_count=4,
        complex_source_count=6,
        selection_mode="aspect_cover_v2",
    ),
    PipelineVariant(
        key="task_aligned_4_6",
        label="当前最佳 4/6 + 强制对齐",
        use_rerank=True,
        rerank_mode="aspect_aware_conservative",
        answer_prompt_style="task_aligned",
        multi_snippet_count=1,
        final_source_count=4,
        complex_source_count=6,
        selection_mode="aspect_cover_v2",
    ),
    PipelineVariant(
        key="router_5_7",
        label="当前最佳 5/7 + 按题作答",
        use_rerank=True,
        rerank_mode="aspect_aware_conservative",
        answer_prompt_style="task_router",
        multi_snippet_count=1,
        final_source_count=5,
        complex_source_count=7,
        selection_mode="aspect_cover_v2",
    ),
    PipelineVariant(
        key="task_aligned_5_7",
        label="当前最佳 5/7 + 强制对齐",
        use_rerank=True,
        rerank_mode="aspect_aware_conservative",
        answer_prompt_style="task_aligned",
        multi_snippet_count=1,
        final_source_count=5,
        complex_source_count=7,
        selection_mode="aspect_cover_v2",
    ),
)

SUMMARY_FILE_NAMES = {
    "base_router_4_6": "参数搜索_当前最佳4_6加按题作答_汇总.json",
    "task_aligned_4_6": "参数搜索_当前最佳4_6加强制对齐_汇总.json",
    "router_5_7": "参数搜索_当前最佳5_7加按题作答_汇总.json",
    "task_aligned_5_7": "参数搜索_当前最佳5_7加强制对齐_汇总.json",
}


def run_variant_parallel(
    pipeline: RagExperimentPipeline,
    prepared,
    variant: PipelineVariant,
    *,
    workers: int,
) -> list[Any]:
    """并行运行指定实验变体。"""
    if workers <= 1:
        results = []
        total = len(prepared.cases)
        for index, case in enumerate(prepared.cases, start=1):
            if index == 1 or index % 10 == 0 or index == total:
                print(f"[param-sweep] {variant.key}: {index}/{total}", flush=True)
            results.append(pipeline.run_case(prepared, case, variant))
        return results

    total = len(prepared.cases)
    ordered_results: list[Any] = [None] * total
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(pipeline.run_case, prepared, case, variant): index
            for index, case in enumerate(prepared.cases)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            ordered_results[index] = future.result()
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == total:
                print(f"[param-sweep] {variant.key}: {completed}/{total}", flush=True)
    return ordered_results


def main() -> None:
    """组织当前脚本的主执行流程。"""
    parser = argparse.ArgumentParser(description="运行最终参数配置对比搜索。")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--qa-2doc-samples", type=int, default=20)
    parser.add_argument("--qa-3doc-samples", type=int, default=20)
    parser.add_argument("--distractor-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-root", default="", help="可选输出目录；为空时写入 results/02_最终参数搜索_20260517/")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve() if args.output_root else OUTPUT_ROOT
    ensure_dir(output_root)
    cache_root = ensure_dir(ROOT / ".cache")

    cases, docs = load_crud_cases(
        summary_samples=0,
        qa_1doc_samples=0,
        qa_2doc_samples=args.qa_2doc_samples,
        qa_3doc_samples=args.qa_3doc_samples,
        hallu_samples=0,
        negative_samples=0,
        distractor_count=args.distractor_count,
        seed=args.seed,
    )
    expected_count = args.qa_2doc_samples + args.qa_3doc_samples
    if len(cases) != expected_count:
        raise RuntimeError(f"CRUD 子样本当前评测批次数异常，期望 {expected_count}，实际 {len(cases)}")

    pipeline = RagExperimentPipeline(
        cache_root=cache_root,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
    )
    prepared = pipeline.prepare_dataset(
        "crud_param_sweep",
        cases,
        docs,
        include_contextual=False,
        include_parent_child=False,
        include_query_rewrite=False,
    )

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for variant in VARIANTS:
        results = run_variant_parallel(pipeline, prepared, variant, workers=max(1, args.workers))
        evaluation = evaluate_crud_results(
            variant.key,
            results,
            cases,
            ragas_case_ids=(),
            qa_ragas_case_ids=(),
            multidoc_ragas_case_ids=(),
            enable_ragas=False,
            semantic_model_name=args.embedding_model,
        )
        summary = dict(evaluation.summary)
        summary["label"] = variant.label
        summaries.append(summary)
        (output_root / SUMMARY_FILE_NAMES[variant.key]).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows.append(
            {
                "variant": variant.key,
                "label": variant.label,
                "hit1": summary["retrieval_hit_rate_at_1"],
                "hit3": summary["retrieval_hit_rate_at_3"],
                "integration_string_similarity": summary["integration_string_similarity"],
                "integration_focus_f1": summary["integration_focus_f1"],
                "integration_quality": summary["integration_quality"],
                "complex_quality": summary["complex_quality"],
                "p50": summary["latency_p50_ms"],
            }
        )

    manifest = {
        "crud_subset_total": sum(CRUD_SUBSET_SPLITS.values()),
        "crud_subset_splits": CRUD_SUBSET_SPLITS,
        "evaluation_batch_size": len(cases),
        "evaluation_batch_splits": {
            "questanswer_2docs": args.qa_2doc_samples,
            "questanswer_3docs": args.qa_3doc_samples,
        },
        "embedding_model": args.embedding_model,
        "llm_model": args.llm_model,
        "distractor_count": args.distractor_count,
        "seed": args.seed,
        "workers": args.workers,
        "case_ids": [case.case_id for case in cases],
        "variants": [variant.label for variant in VARIANTS],
    }
    (output_root / "参数搜索_路线总表.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "参数搜索_全部汇总.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "评测批次说明.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
