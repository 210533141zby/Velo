"""数据处理阅读版入口。

这个文件不再把 RGB / CRUD 的所有辅助函数铺满，而是把答辩时最常提到的入口、
数据结构和口径说明单独收口，方便快速定位。
"""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "04_算法实现"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from retrieval_pipeline.datasets import CorpusDoc, ExperimentCase, load_crud_cases, load_rgb_cases  # noqa: E402


def 数据层说明() -> str:
    """返回数据层。"""
    return (
        "RGB 用于简单事实问答稳定性检查；CRUD 子样本用于复杂多文档问答评测；"
        "论文统一使用 CRUD 子样本 1594 条这一口径，主表则从中固定抽取可复跑批次。"
    )
