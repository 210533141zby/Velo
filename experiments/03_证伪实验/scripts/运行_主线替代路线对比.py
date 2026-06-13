"""运行证伪实验中的主线替代路线对比。

这个脚本覆盖论文 3.3.1 中三类最直接的替代路线：
1. 双片段证据保留；
2. 候选证据重写与标注；
3. 结构化综合路线。
"""

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

RESULT_ROOT = ROOT / "results" / "01_主线替代路线"
CRUD_SUBSET_SPLITS = {"questanswer_2docs": 797, "questanswer_3docs": 797}

SUITES: dict[str, dict[str, Any]] = {
    "dual_snippet": {
        "output_root": RESULT_ROOT / "双片段路线_20260517",
        "variants": (
            PipelineVariant(
                key="baseline_rrf_rerank_direct",
                label="基线",
                use_rerank=True,
                answer_prompt_style="simple",
                multi_snippet_count=1,
            ),
            PipelineVariant(
                key="dual_snippet_direct",
                label="基线 + 双片段证据保留",
                use_rerank=True,
                answer_prompt_style="simple",
                multi_snippet_count=2,
            ),
            PipelineVariant(
                key="dual_snippet_aspect_rerank_direct",
                label="基线 + 双片段证据保留 + 分项重排",
                use_rerank=True,
                rerank_mode="aspect_aware_conservative",
                answer_prompt_style="simple",
                multi_snippet_count=2,
            ),
            PipelineVariant(
                key="dual_snippet_aspect_cover_direct",
                label="基线 + 双片段证据保留 + 分项重排 + 覆盖取证",
                use_rerank=True,
                rerank_mode="aspect_aware_conservative",
                answer_prompt_style="simple",
                multi_snippet_count=2,
                final_source_count=4,
                complex_source_count=6,
                selection_mode="aspect_cover_v2",
            ),
            PipelineVariant(
                key="dual_snippet_aspect_cover_router",
                label="基线 + 双片段证据保留 + 分项重排 + 覆盖取证 + 按题作答",
                use_rerank=True,
                rerank_mode="aspect_aware_conservative",
                answer_prompt_style="task_router",
                multi_snippet_count=2,
                final_source_count=4,
                complex_source_count=6,
                selection_mode="aspect_cover_v2",
            ),
        ),
    },
    "candidate": {
        "output_root": RESULT_ROOT / "候选证据路线_20260517",
        "variants": (
            PipelineVariant(
                key="aspect_cover_router",
                label="当前最佳",
                use_rerank=True,
                rerank_mode="aspect_aware_conservative",
                answer_prompt_style="task_router",
                multi_snippet_count=1,
                final_source_count=4,
                complex_source_count=6,
                selection_mode="aspect_cover_v2",
            ),
            PipelineVariant(
                key="aspect_cover_router_labeled",
                label="当前最佳 + 证据标注",
                use_rerank=True,
                rerank_mode="aspect_aware_conservative",
                answer_prompt_style="task_router",
                multi_snippet_count=1,
                final_source_count=4,
                complex_source_count=6,
                selection_mode="aspect_cover_v2",
                rendering_mode="aspect_labeled",
            ),
            PipelineVariant(
                key="aspect_cover_router_window",
                label="当前最佳 + 局部窗口重写",
                use_rerank=True,
                rerank_mode="aspect_aware_conservative",
                answer_prompt_style="task_router",
                multi_snippet_count=1,
                final_source_count=4,
                complex_source_count=6,
                selection_mode="aspect_cover_v2",
                rendering_mode="detail_window",
            ),
        ),
    },
    "structured": {
        "output_root": RESULT_ROOT / "结构化生成路线_20260517",
        "variants": (
            PipelineVariant(
                key="aspect_cover_router",
                label="当前最佳",
                use_rerank=True,
                rerank_mode="aspect_aware_conservative",
                answer_prompt_style="task_router",
                multi_snippet_count=1,
                final_source_count=4,
                complex_source_count=6,
                selection_mode="aspect_cover_v2",
            ),
            PipelineVariant(
                key="aspect_cover_clause_guided",
                label="当前最佳 + 模板引导",
                use_rerank=True,
                rerank_mode="aspect_aware_conservative",
                synthesis_mode="clause_guided_aligned",
                answer_prompt_style="aligned",
                multi_snippet_count=1,
                final_source_count=4,
                complex_source_count=6,
                selection_mode="aspect_cover_v2",
            ),
            PipelineVariant(
                key="aspect_cover_support_table",
                label="当前最佳 + 支持表综合",
                use_rerank=True,
                rerank_mode="aspect_aware_conservative",
                synthesis_mode="support_table_aligned",
                answer_prompt_style="task_aligned",
                multi_snippet_count=1,
                final_source_count=4,
                complex_source_count=6,
                selection_mode="aspect_cover_v2",
            ),
        ),
    },
}


def run_variant_parallel(
    pipeline: RagExperimentPipeline,
    prepared,
    variant: PipelineVariant,
    *,
    workers: int,
) -> list[Any]:
    """在当前 CRUD 评测批次上运行一个实验变体。

    当样例数量较多时可开启线程池并行，以缩短整组证伪实验的等待时间；
    若设置为单线程，则保持顺序执行，便于逐题排查异常。
    """
    if workers <= 1:
        results = []
        total = len(prepared.cases)
        for index, case in enumerate(prepared.cases, start=1):
            if index == 1 or index % 10 == 0 or index == total:
                print(f"[{prepared.name}] {variant.key}: {index}/{total}", flush=True)
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
                print(f"[{prepared.name}] {variant.key}: {completed}/{total}", flush=True)
    return ordered_results


def load_crud_batch(args: argparse.Namespace):
    """按论文证伪实验口径加载 CRUD 子样本评测批次。

    这里显式检查 2 文档题与 3 文档题的数量，避免因为采样配置变化
    导致结果表和论文描述不一致。
    """
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
    return cases, docs


def build_manifest(args: argparse.Namespace, cases, variants: tuple[PipelineVariant, ...], *, suite: str) -> dict[str, Any]:
    """汇总本次运行的样例范围、模型配置和方案列表，写成批次说明。"""
    return {
        "suite": suite,
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
        "variants": [variant.label for variant in variants],
    }


def write_ablation_like_outputs(
    output_root: Path,
    metric_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    """把主线替代路线结果写成论文与 notebook 都能直接消费的多种文件格式。"""
    (output_root / "主线替代路线_汇总.json").write_text(
        json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "主线替代路线_指标表.json").write_text(
        json.dumps(metric_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "主线替代路线_逐题明细.json").write_text(
        json.dumps(detail_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_root / "主线替代路线_指标表.csv").open("w", encoding="utf-8", newline="") as handle:
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


def _label_to_file_stem(label: str) -> str:
    """把中文方案名规整成稳定文件名前缀，便于批量落盘。"""
    return (
        label.replace(" + ", "_")
        .replace("+", "_")
        .replace("/", "_")
        .replace("（", "")
        .replace("）", "")
        .replace(" ", "")
    )


def write_rows_outputs(
    output_root: Path,
    summaries: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    """按单方案维度拆分输出文件，便于逐条展示某条替代路线的结果。"""
    rows = []
    for summary in summaries:
        file_stem = _label_to_file_stem(str(summary["label"]))
        (output_root / f"{file_stem}_汇总.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows.append(
            {
                "variant": summary["variant"],
                "label": summary["label"],
                "hit1": summary["retrieval_hit_rate_at_1"],
                "hit3": summary["retrieval_hit_rate_at_3"],
                "integration_string_similarity": summary["integration_string_similarity"],
                "integration_focus_f1": summary["integration_focus_f1"],
                "integration_quality": summary["integration_quality"],
                "complex_quality": summary["complex_quality"],
                "p50": summary["latency_p50_ms"],
            }
        )
    (output_root / "对比总表.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "评测批次说明.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_suite(args: argparse.Namespace, suite: str) -> Path:
    """执行整套实验评估。"""
    suite_config = SUITES[suite]
    output_root = suite_config["output_root"]
    variants: tuple[PipelineVariant, ...] = suite_config["variants"]
    ensure_dir(output_root)
    cache_root = ensure_dir(ROOT / ".cache")

    cases, docs = load_crud_batch(args)
    pipeline = RagExperimentPipeline(
        cache_root=cache_root,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
    )
    prepared = pipeline.prepare_dataset(
        f"crud_{suite}",
        cases,
        docs,
        include_contextual=False,
        include_parent_child=False,
        include_query_rewrite=False,
    )

    summaries: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for variant in variants:
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

    manifest = build_manifest(args, cases, variants, suite=suite)
    if suite == "dual_snippet":
        write_ablation_like_outputs(output_root, metric_rows, summaries, detail_rows, manifest)
    else:
        write_rows_outputs(output_root, summaries, manifest)
    return output_root


def main() -> None:
    """组织当前脚本的主执行流程。"""
    parser = argparse.ArgumentParser(description="运行证伪实验中的主线替代路线对比。")
    parser.add_argument("--suite", choices=("dual_snippet", "candidate", "structured", "all"), default="all")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--qa-2doc-samples", type=int, default=20)
    parser.add_argument("--qa-3doc-samples", type=int, default=20)
    parser.add_argument("--distractor-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    suites = tuple(SUITES) if args.suite == "all" else (args.suite,)
    output_roots = [str(evaluate_suite(args, suite)) for suite in suites]
    print(json.dumps({"output_roots": output_roots}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
