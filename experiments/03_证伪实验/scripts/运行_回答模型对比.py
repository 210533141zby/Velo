"""固定检索链路，对比不同回答模型在同一批复杂问答上的表现。"""

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

from retrieval_pipeline.common import DEFAULT_EMBEDDING_MODEL, ensure_dir
from retrieval_pipeline.datasets import load_crud_cases
from retrieval_pipeline.metrics import evaluate_crud_results
from retrieval_pipeline.pipeline import PipelineVariant, RagExperimentPipeline

OUTPUT_ROOT = ROOT / "results" / "04_回答模型对比"

COMPLEX_CASE_IDS = (
    "questanswer_2docs_001",
    "questanswer_2docs_002",
    "questanswer_2docs_003",
    "questanswer_2docs_004",
    "questanswer_2docs_005",
    "questanswer_2docs_006",
    "questanswer_2docs_007",
    "questanswer_2docs_008",
    "questanswer_3docs_001",
    "questanswer_3docs_002",
    "questanswer_3docs_003",
    "questanswer_3docs_004",
    "questanswer_3docs_005",
    "questanswer_3docs_006",
    "questanswer_3docs_007",
    "questanswer_3docs_008",
)

NEUTRAL_DIRECT = PipelineVariant(
    key="neutral_direct",
    label="中性提示直接作答",
    use_rerank=True,
    answer_prompt_style="neutral",
    multi_snippet_count=2,
)

MODEL_NATIVE_DIRECT = PipelineVariant(
    key="model_native_direct",
    label="模型原生提示直接作答",
    use_rerank=True,
    answer_prompt_style="model_native",
    multi_snippet_count=2,
)


def main() -> None:
    # 这组结果只比较回答模型，不改检索主干和证据组织方式。
    """组织当前脚本的主执行流程。"""
    parser = argparse.ArgumentParser(description="固定检索，比较不同 answer model 在中性 prompt 与原生 prompt 下的表现。")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="要比较的 Ollama 模型列表，例如 qwen2.5:7b-instruct qwen3:8b llama3.1:8b",
    )
    parser.add_argument(
        "--variant",
        choices=("neutral_direct", "model_native_direct", "both"),
        default="both",
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
    if len(selected_cases) != len(COMPLEX_CASE_IDS):
        found = {case.case_id for case in selected_cases}
        missing = [case_id for case_id in COMPLEX_CASE_IDS if case_id not in found]
        raise RuntimeError(f"缺少复杂验证样例: {missing}")

    selected_variants = (
        (NEUTRAL_DIRECT, MODEL_NATIVE_DIRECT)
        if args.variant == "both"
        else ((NEUTRAL_DIRECT,) if args.variant == "neutral_direct" else (MODEL_NATIVE_DIRECT,))
    )

    summaries = []
    details = []
    for variant in selected_variants:
        for model_name in args.models:
            print(f"[crud-model-probe] variant={variant.key} model={model_name}", flush=True)
            pipeline = RagExperimentPipeline(
                cache_root=cache_root,
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                llm_model=model_name,
            )
            prepared = pipeline.prepare_dataset(
                "crud",
                selected_cases,
                docs,
                include_contextual=False,
                include_parent_child=False,
                include_query_rewrite=False,
            )
            results = []
            total = len(prepared.cases)
            for index, case in enumerate(prepared.cases, start=1):
                if index == 1 or index == total:
                    print(f"[crud-model-probe] {variant.key} {model_name}: {index}/{total}", flush=True)
                results.append(pipeline.run_case(prepared, case, variant))

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
            summary["llm_model"] = model_name
            summaries.append(summary)
            for row in evaluation.detail_rows:
                details.append({**row, "llm_model": model_name})

    output = {
        "case_ids": list(COMPLEX_CASE_IDS),
        "variant": args.variant,
        "summaries": summaries,
    }
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    (OUTPUT_ROOT / f"回答模型对比_结果汇总{suffix}.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / f"回答模型对比_逐题明细{suffix}.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
