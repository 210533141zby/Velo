"""生成 RGB 300 稳定性支撑结果，并保留一份 CRUD 辅助对照表。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = ROOT.parent
IMPL_ROOT = EXPERIMENTS_ROOT / "04_算法实现"
if str(IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL_ROOT))

from retrieval_pipeline.common import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, ensure_dir
from retrieval_pipeline.datasets import load_crud_cases, load_rgb_cases
from retrieval_pipeline.metrics import DatasetEvaluation, evaluate_crud_results, evaluate_rgb_results, evenly_spaced_case_ids
from retrieval_pipeline.pipeline import PipelineVariant, RagExperimentPipeline

OUTPUT_ROOT = ROOT / "results" / "03_RGB300支撑结果"
NOTEBOOK_ROOT = ROOT / "results" / "_辅助notebook"
FONT_PATH = EXPERIMENTS_ROOT / ".assets" / "fonts" / "SourceHanSansSC-Regular.otf"

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "WenQuanYi Zen Hei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"


VARIANTS = (
    PipelineVariant(
        key="baseline_rrf_rerank_direct",
        label="基线（Dense + BM25 -> RRF -> Rerank -> Direct Answer）",
        use_rerank=True,
        answer_prompt_style="simple",
        multi_snippet_count=1,
    ),
    PipelineVariant(
        key="enhanced_dual_snippet_direct",
        label="增强检索（双片段证据保留）",
        use_rerank=True,
        answer_prompt_style="simple",
        multi_snippet_count=2,
    ),
    PipelineVariant(
        key="ours_task_router_aspect_cover_v2",
        label="主线方案（分项重排 + 覆盖取证 + 按题作答）",
        use_rerank=True,
        rerank_mode="aspect_aware_conservative",
        answer_prompt_style="task_router",
        multi_snippet_count=2,
        final_source_count=4,
        complex_source_count=6,
        selection_mode="aspect_cover_v2",
    ),
)


def load_cjk_font() -> font_manager.FontProperties | None:
    """按需注册中文字体，避免图表标题和坐标轴在新环境里出现乱码。

    这个脚本会输出论文可直接使用的 SVG 图，因此中文字体能否正确加载
    会直接影响最终图像质量。
    """
    if not FONT_PATH.exists():
        return None
    font_manager.fontManager.addfont(str(FONT_PATH))
    return font_manager.FontProperties(fname=str(FONT_PATH))


def to_markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    """把结果表转换成 Markdown 文本，便于写入说明 notebook。"""
    headers = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for _, row in df[list(columns)].iterrows():
        values: list[str] = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([headers, divider] + body)


def add_deltas(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """为相邻方案补充指标增减量，方便直接观察“加一项是否有收益”。"""
    rows: list[dict[str, Any]] = []
    previous_row: dict[str, Any] | None = None
    for _, row in df.iterrows():
        record = row.to_dict()
        for metric in metrics:
            delta_key = f"delta_{metric}_vs_prev"
            if previous_row is None:
                record[delta_key] = "baseline"
            else:
                record[delta_key] = f"{float(record[metric]) - float(previous_row[metric]):+.4f}"
        rows.append(record)
        previous_row = record
    return pd.DataFrame(rows)


def plot_dataset_metrics(df: pd.DataFrame, output_path: Path, *, dataset_label: str, metrics: Sequence[tuple[str, str]]) -> None:
    """把同一组方案的关键指标画成对比柱状图。

    输出图主要服务论文插图和 notebook 展示，因此这里统一控制标题、
    字体、坐标范围和导出格式。
    """
    cjk_font = load_cjk_font()
    figure, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 5.4))
    if len(metrics) == 1:
        axes = [axes]

    labels = df["label"].tolist()
    x = list(range(len(labels)))
    for axis, (metric_key, metric_label) in zip(axes, metrics):
        values = df[metric_key].astype(float).tolist()
        axis.bar(x, values, color=["#1f4e79", "#557c55", "#4c956c", "#2f855a"][: len(labels)], width=0.72)
        axis.set_xticks(x)
        if cjk_font is not None:
            axis.set_xticklabels(labels, rotation=16, ha="right", fontproperties=cjk_font)
            axis.set_title(f"{dataset_label}: {metric_label}", fontproperties=cjk_font)
        else:
            axis.set_xticklabels(labels, rotation=16, ha="right")
            axis.set_title(f"{dataset_label}: {metric_label}")
        upper = 1.05 if not metric_key.startswith("latency") else max(values) * 1.18
        axis.set_ylim(0.0, upper)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, format="svg")
    plt.close(figure)


def build_notebook(path: Path, config: dict[str, Any], rgb_df: pd.DataFrame, crud_df: pd.DataFrame) -> None:
    """生成一份带表格与图片链接的结果 notebook，方便答辩时现场展示。"""
    rgb_table = to_markdown_table(
        rgb_df,
        [
            "label",
            "candidate_top_k",
            "accuracy",
            "retrieval_hit_rate_at_1",
            "retrieval_hit_rate_at_3",
            "answer_relevancy",
            "delta_accuracy_vs_prev",
            "delta_retrieval_hit_rate_at_1_vs_prev",
            "delta_retrieval_hit_rate_at_3_vs_prev",
            "delta_answer_relevancy_vs_prev",
        ],
    )
    crud_table = to_markdown_table(
        crud_df,
        [
            "label",
            "candidate_top_k",
            "qa_accuracy",
            "qa_faithfulness",
            "qa_answer_correctness",
            "multidoc_faithfulness",
            "multidoc_answer_correctness",
            "negative_rejection",
            "delta_qa_accuracy_vs_prev",
            "delta_qa_faithfulness_vs_prev",
            "delta_multidoc_answer_correctness_vs_prev",
        ],
    )
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# RGB + CRUD 加法消融实验\n\n"
            "统一基线为 `Dense + BM25 -> RRF -> Rerank -> Direct Answer`，"
            "并与论文主线方案做正式对比。"
        ),
        nbf.v4.new_code_cell(
            "config = " + json.dumps(config, ensure_ascii=False, indent=2),
            outputs=[nbf.v4.new_output("execute_result", data={"text/plain": json.dumps(config, ensure_ascii=False, indent=2)}, execution_count=1)],
            execution_count=1,
        ),
        nbf.v4.new_markdown_cell("## RGB 结果\n\n" + rgb_table + "\n\n![RGB 消融图](../outputs/RGB300_柱状图.svg)"),
        nbf.v4.new_markdown_cell("## CRUD 结果\n\n" + crud_table + "\n\n![CRUD 消融图](../outputs/CRUD对照_柱状图.svg)"),
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
    }
    path.write_text(nbf.writes(notebook), encoding="utf-8")


def run_variant_on_cases(
    pipeline: RagExperimentPipeline,
    prepared: Any,
    variant: PipelineVariant,
) -> list[Any]:
    """在一组样例上顺序执行某个实验方案，并打印阶段性进度。"""
    results = []
    total = len(prepared.cases)
    for index, case in enumerate(prepared.cases, start=1):
        if index == 1 or index % 25 == 0 or index == total:
            print(f"[{prepared.name}] {variant.key}: {index}/{total}", flush=True)
        results.append(pipeline.run_case(prepared, case, variant))
    return results


def main() -> None:
    """组织 RGB300 与 CRUD 对照评测的完整执行流程。

    该入口负责读取参数、准备数据、运行各方案、汇总指标、导出图表，
    是这组加法消融实验的统一启动点。
    """
    parser = argparse.ArgumentParser(description="在 RGB 和 CRUD 上运行统一 baseline + 加法消融实验。")
    parser.add_argument("--dataset", choices=("both", "crud", "rgb"), default="both")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--output-root", default="", help="可选输出目录；为空时写入默认 outputs/。")
    parser.add_argument("--rgb-case-limit", type=int, default=0, help="RGB 快速验证样本数；0 为全量 300。")
    parser.add_argument("--rgb-ragas-sample-count", type=int, default=50)
    parser.add_argument("--crud-summary-samples", type=int, default=10)
    parser.add_argument("--crud-qa-1doc-samples", type=int, default=10)
    parser.add_argument("--crud-qa-2doc-samples", type=int, default=10)
    parser.add_argument("--crud-qa-3doc-samples", type=int, default=10)
    parser.add_argument("--crud-hallu-samples", type=int, default=0)
    parser.add_argument("--crud-negative-samples", type=int, default=16)
    parser.add_argument("--crud-distractor-count", type=int, default=1200)
    parser.add_argument("--crud-ragas-sample-count", type=int, default=24)
    parser.add_argument("--crud-seed", type=int, default=42)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve() if args.output_root else OUTPUT_ROOT
    notebook_root = output_root.parent / f"{output_root.name}_notebooks" if args.output_root else NOTEBOOK_ROOT

    ensure_dir(output_root)
    ensure_dir(notebook_root)
    cache_root = ensure_dir(ROOT / ".cache")
    run_rgb = args.dataset in {"both", "rgb"}
    run_crud = args.dataset in {"both", "crud"}

    pipeline = RagExperimentPipeline(
        cache_root=cache_root,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
    )
    include_contextual = any(variant.retrieval_strategy == "contextual" for variant in VARIANTS)
    include_parent_child = any(variant.retrieval_strategy == "parent_child" for variant in VARIANTS)
    include_query_rewrite = any(variant.use_query_rewrite for variant in VARIANTS)

    rgb_cases = []
    rgb_prepared = None
    rgb_ragas_case_ids: list[str] = []
    if run_rgb:
        rgb_cases, rgb_docs = load_rgb_cases(case_limit=args.rgb_case_limit)
        rgb_prepared = pipeline.prepare_dataset(
            "rgb",
            rgb_cases,
            rgb_docs,
            include_contextual=include_contextual,
            include_parent_child=include_parent_child,
            include_query_rewrite=include_query_rewrite,
        )
        rgb_ragas_case_ids = evenly_spaced_case_ids([case.case_id for case in rgb_cases], args.rgb_ragas_sample_count)

    crud_cases = []
    crud_prepared = None
    crud_ragas_case_ids: list[str] = []
    crud_qa_ragas_case_ids: list[str] = []
    crud_multidoc_ragas_case_ids: list[str] = []
    if run_crud:
        crud_cases, crud_docs = load_crud_cases(
            summary_samples=args.crud_summary_samples,
            qa_1doc_samples=args.crud_qa_1doc_samples,
            qa_2doc_samples=args.crud_qa_2doc_samples,
            qa_3doc_samples=args.crud_qa_3doc_samples,
            hallu_samples=args.crud_hallu_samples,
            negative_samples=args.crud_negative_samples,
            distractor_count=args.crud_distractor_count,
            seed=args.crud_seed,
        )
        crud_prepared = pipeline.prepare_dataset(
            "crud",
            crud_cases,
            crud_docs,
            include_contextual=include_contextual,
            include_parent_child=include_parent_child,
            include_query_rewrite=include_query_rewrite,
        )
        crud_ragas_case_ids = evenly_spaced_case_ids(
            [case.case_id for case in crud_cases if not case.should_refuse],
            args.crud_ragas_sample_count,
        )
        crud_qa_ragas_case_ids = evenly_spaced_case_ids(
            [case.case_id for case in crud_cases if case.split.startswith("questanswer_")],
            args.crud_ragas_sample_count,
        )
        crud_multidoc_ragas_case_ids = evenly_spaced_case_ids(
            [case.case_id for case in crud_cases if case.split in {"questanswer_2docs", "questanswer_3docs"}],
            args.crud_ragas_sample_count,
        )

    rgb_summaries: list[dict[str, Any]] = []
    crud_summaries: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    ragas_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        if run_rgb and rgb_prepared is not None:
            rgb_results = run_variant_on_cases(pipeline, rgb_prepared, variant)
            rgb_eval = evaluate_rgb_results(
                variant.key,
                rgb_results,
                rgb_cases,
                ragas_case_ids=rgb_ragas_case_ids,
                enable_ragas=not args.skip_ragas,
            )
            rgb_summary = dict(rgb_eval.summary)
            rgb_summary["label"] = variant.label
            rgb_summary["candidate_top_k"] = variant.candidate_top_k
            rgb_summaries.append(rgb_summary)
            detail_rows.extend(rgb_eval.detail_rows)
            ragas_rows.extend([{**row, "variant": variant.key, "dataset": "rgb"} for row in rgb_eval.ragas_rows])

        if run_crud and crud_prepared is not None:
            crud_results = run_variant_on_cases(pipeline, crud_prepared, variant)
            crud_eval = evaluate_crud_results(
                variant.key,
                crud_results,
                crud_cases,
                ragas_case_ids=crud_ragas_case_ids,
                qa_ragas_case_ids=crud_qa_ragas_case_ids,
                multidoc_ragas_case_ids=crud_multidoc_ragas_case_ids,
                enable_ragas=not args.skip_ragas,
                semantic_model_name=args.embedding_model,
            )
            crud_summary = dict(crud_eval.summary)
            crud_summary["label"] = variant.label
            crud_summary["candidate_top_k"] = variant.candidate_top_k
            crud_summaries.append(crud_summary)
            detail_rows.extend(crud_eval.detail_rows)
            ragas_rows.extend([{**row, "variant": variant.key, "dataset": "crud"} for row in crud_eval.ragas_rows])

    summary_payload: dict[str, Any] = {}
    rgb_df = pd.DataFrame()
    crud_df = pd.DataFrame()
    if run_rgb and rgb_summaries:
        rgb_df = add_deltas(
            pd.DataFrame(rgb_summaries),
            ["accuracy", "retrieval_hit_rate_at_1", "retrieval_hit_rate_at_3", "answer_relevancy"],
        )
        rgb_df.to_csv(output_root / "RGB300_指标表.csv", index=False, encoding="utf-8")
        summary_payload["rgb"] = rgb_df.to_dict(orient="records")
    if run_crud and crud_summaries:
        crud_df = add_deltas(
            pd.DataFrame(crud_summaries),
            [
                "qa_accuracy",
                "qa_faithfulness",
                "qa_answer_correctness",
                "multidoc_faithfulness",
                "multidoc_answer_correctness",
                "negative_rejection",
                "overall_similarity",
                "integration_similarity",
            ],
        )
        crud_df.to_csv(output_root / "CRUD对照_指标表.csv", index=False, encoding="utf-8")
        summary_payload["crud"] = crud_df.to_dict(orient="records")

    pd.DataFrame(detail_rows).to_csv(output_root / "支撑实验_逐题明细.csv", index=False, encoding="utf-8")
    pd.DataFrame(ragas_rows).to_csv(output_root / "支撑实验_RAGAS明细.csv", index=False, encoding="utf-8")

    config = {
        "dataset": args.dataset,
        "baseline": VARIANTS[0].label,
        "variants": [variant.__dict__ for variant in VARIANTS],
        "embedding_model": args.embedding_model,
        "llm_model": args.llm_model,
        "rgb_case_limit": args.rgb_case_limit,
        "rgb_ragas_sample_count": args.rgb_ragas_sample_count,
        "crud_summary_samples": args.crud_summary_samples,
        "crud_qa_1doc_samples": args.crud_qa_1doc_samples,
        "crud_qa_2doc_samples": args.crud_qa_2doc_samples,
        "crud_qa_3doc_samples": args.crud_qa_3doc_samples,
        "crud_hallu_samples": args.crud_hallu_samples,
        "crud_negative_samples": args.crud_negative_samples,
        "crud_distractor_count": args.crud_distractor_count,
        "crud_ragas_sample_count": args.crud_ragas_sample_count,
        "skip_ragas": args.skip_ragas,
    }
    (output_root / "支撑实验_实验配置.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "支撑实验_汇总.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if run_rgb and not rgb_df.empty:
        plot_dataset_metrics(
            rgb_df,
            output_root / "RGB300_柱状图.svg",
            dataset_label="RGB",
            metrics=(
                ("accuracy", "准确率"),
                ("retrieval_hit_rate_at_1", "命中率 Hit@1"),
                ("retrieval_hit_rate_at_3", "命中率 Hit@3"),
            ),
        )
    if run_crud and not crud_df.empty:
        plot_dataset_metrics(
            crud_df,
            output_root / "CRUD对照_柱状图.svg",
            dataset_label="CRUD",
            metrics=(
                ("qa_accuracy", "问答准确率"),
                ("qa_faithfulness", "答案忠实度"),
                ("multidoc_answer_correctness", "多文档答案正确率"),
            ),
        )

    if run_rgb and run_crud and not rgb_df.empty and not crud_df.empty:
        build_notebook(notebook_root / "rgb_crud_ablation.ipynb", config, rgb_df, crud_df)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
