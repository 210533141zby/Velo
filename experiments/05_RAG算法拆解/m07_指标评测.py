"""指标评测阅读版入口。"""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "04_算法实现"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from retrieval_pipeline.metrics import (  # noqa: E402
    evaluate_crud_results,
    integration_quality_score,
    multidoc_focus_scores,
)


def 指标层说明() -> str:
    """返回指标层的说明。"""
    return (
        "主指标不是单独 Accuracy，而是以 integration_focus_f1 和 integration_quality 为核心；"
        "前者衡量多分项有没有答全，后者把语义相似度、字符串相似度和分项覆盖综合起来。"
    )
