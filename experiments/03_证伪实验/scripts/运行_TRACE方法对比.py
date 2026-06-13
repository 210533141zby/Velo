"""在同一批复杂问答上比较 TRACE 方法与本文主线方案。"""

from __future__ import annotations

import argparse
import csv
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

OUTPUT_ROOT = ROOT / "results" / "05_TRACE方法对比"
CRUD_SUBSET_SPLITS = {"questanswer_2docs": 797, "questanswer_3docs": 797}

VARIANTS = (
    PipelineVariant(
        key="ours_router_4_6",
        label="Ours（分项重排 + 覆盖取证 + 按题作答）",
        use_rerank=True,
        rerank_mode="aspect_aware_conservative",
        answer_prompt_style="task_router",
        multi_snippet_count=1,
        final_source_count=4,
        complex_source_count=6,
        selection_mode="aspect_cover_v2",
    ),
    PipelineVariant(
        key="trace_method_4_6",
        label="TRACE 方法（透明三段式生成）",
        use_rerank=True,
        rerank_mode="aspect_aware_conservative",
        answer_prompt_style="trace_structured",
        multi_snippet_count=1,
        final_source_count=4,
        complex_source_count=6,
        selection_mode="aspect_cover_v2",
    ),
)


def run_variant_parallel(
    pipeline: RagExperimentPipeline,
    prepared,
    variant: PipelineVariant,
    *,
    workers: int,
) -> list[Any]:
    """在同一批 CRUD 子样本上运行某个 TRACE/Ours 对比方案。

    这里允许并行执行，是为了在不改变样本集合的前提下尽快拿到两条方法线的
    完整结果表。
    """
    total = len(prepared.cases)
    if workers <= 1:
        results = []
        for index, case in enumerate(prepared.cases, start=1):
            if index == 1 or index % 10 == 0 or index == total:
                print(f"[crud-trace-method] {variant.key}: {index}/{total}", flush=True)
            results.append(pipeline.run_case(prepared, case, variant))
        return results

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
                print(f"[crud-trace-method] {variant.key}: {completed}/{total}", flush=True)
    return ordered_results


def write_outputs(
    output_root: Path,
    metric_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    """把 TRACE 对比实验结果写成汇总表、逐题明细和批次说明。"""
    (output_root / "TRACE对比_汇总.json").write_text(
        json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "TRACE对比_指标表.json").write_text(
        json.dumps(metric_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "TRACE对比_逐题明细.json").write_text(
        json.dumps(detail_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_root / "TRACE对比_指标表.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "label",
                "sample_count",
                "retrieval_hit_rate_at_1",
                "retrieval_hit_rate_at_3",
                "integration_string_similarity",
                "integration_focus_f1",
                "integration_quality",
                "complex_quality",
                "latency_p50_ms",
                "latency_p95_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)
    (output_root / "评测批次说明.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """组织 TRACE 方法与论文主线方案的完整对比流程。"""
    parser = argparse.ArgumentParser(description="在同一批 CRUD 子样本评测批次上对比 TRACE 方法与 Ours。")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--qa-2doc-samples", type=int, default=20)
    parser.add_argument("--qa-3doc-samples", type=int, default=20)
    parser.add_argument("--distractor-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--output-root",
        default="",
        help="可选输出目录；为空时写入 results/05_TRACE方法对比/TRACE方法对比_20260520/",
    )
    args = parser.parse_args()

    output_root = (
        Path(args.output_root).resolve() if args.output_root else OUTPUT_ROOT / "TRACE方法对比_20260520"
    )
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
        "crud_trace_method_compare_batch",
        cases,
        docs,
        include_contextual=False,
        include_parent_child=False,
        include_query_rewrite=False,
    )

    summaries: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    manifest = {
        "dataset": "crud_trace_method_compare_batch",
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
        detail_rows.extend(evaluation.detail_rows)
        metric_rows.append(
            {
                "variant": variant.key,
                "label": variant.label,
                "sample_count": summary["sample_count"],
                "retrieval_hit_rate_at_1": summary["retrieval_hit_rate_at_1"],
                "retrieval_hit_rate_at_3": summary["retrieval_hit_rate_at_3"],
                "integration_string_similarity": summary["integration_string_similarity"],
                "integration_focus_f1": summary["integration_focus_f1"],
                "integration_quality": summary["integration_quality"],
                "complex_quality": summary["complex_quality"],
                "latency_p50_ms": summary["latency_p50_ms"],
                "latency_p95_ms": summary["latency_p95_ms"],
            }
        )
        write_outputs(output_root, metric_rows, summaries, detail_rows, manifest)

    print(json.dumps({"output_root": str(output_root), "metric_rows": metric_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
