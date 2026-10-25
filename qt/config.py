"""全局配置：数据路径、样本切分、测试段封存。

样本切分见 docs/DECISIONS.md。最终测试段（2022 年至今）在整个研究流程走完之前
不能被读取：所有取数都要经过 check_range()，未解锁时请求测试段会直接报错。
解锁方式是设置环境变量 QT_UNLOCK_TEST=1，只在最后做一次最终测试时使用。
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("QT_DATA_DIR", "~/.qlib/qlib_data/cn_data")).expanduser()

SEGMENTS: dict[str, tuple[str, str | None]] = {
    "train": ("2010-01-01", "2018-12-31"),
    "valid": ("2019-01-01", "2021-12-31"),
    "test": ("2022-01-01", None),
}

# 最长的前瞻窗口：IC 衰减最长看 20 天，加上 T+1 入场的 1 天
MAX_FORWARD_DAYS = 21


class TestSegmentLockedError(RuntimeError):
    """在测试段解锁之前请求了测试段的数据。"""

    __test__ = False  # 名字以 Test 开头，告诉 pytest 这不是测试类


def is_test_unlocked() -> bool:
    return os.environ.get("QT_UNLOCK_TEST") == "1"


def segment(name: str) -> tuple[str, str | None]:
    if name not in SEGMENTS:
        raise KeyError(f"未知的样本段: {name}，可选 {list(SEGMENTS)}")
    if name == "test" and not is_test_unlocked():
        raise TestSegmentLockedError("最终测试段已封存，只在最后一次最终测试时用 QT_UNLOCK_TEST=1 解锁")
    return SEGMENTS[name]


def check_range(start: str, end: str | None) -> None:
    """取数区间不能碰到测试段。end 为 None 表示取到最新，同样算碰到。"""
    if is_test_unlocked():
        return
    test_start = pd.Timestamp(SEGMENTS["test"][0])
    if end is None or pd.Timestamp(end) >= test_start or pd.Timestamp(start) >= test_start:
        raise TestSegmentLockedError(
            f"取数区间 {start} ~ {end} 碰到了最终测试段（{SEGMENTS['test'][0]} 起），测试段尚未解锁"
        )
