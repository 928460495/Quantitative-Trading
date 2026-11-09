"""计划 4 的因子合成（宽表）：类内等权 → 每类标准化 → 类间等权 → 标准化（docs/DECISIONS.md 2026-11-09）。

输入是每个因子中性化、按方向调整后的值（越大越好，每天横截面标准化过）。不估计任何权重。

每类先标准化再做类间平均：一类里的因子互相不太相关时，平均之后波动会变小，不标准化的话这一类在总分里的
分量就被悄悄压低了，"类间等权"名不副实。缺值的因子当天不参与本类平均；一只股票某类全缺（比如新股还没
有财报），这一类不参与它的类间平均。

和 qt/combine.py（沪深300 阶段的长表版、带权重估计）分开，避免混淆。
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from qt.wide_eval import zscore


def _mean_skipna(frames: list[pd.DataFrame]) -> pd.DataFrame:
    total = sum(f.fillna(0.0) for f in frames)
    count = sum(f.notna().astype(float) for f in frames)
    return (total / count).where(count > 0)


def category_scores(scores: Mapping[str, pd.DataFrame], categories: Mapping[str, str]) -> dict[str, pd.DataFrame]:
    """大类 → 本类因子的等权平均，再每天标准化。categories：因子名 → 大类，必须覆盖 scores 的每个因子。"""
    groups: dict[str, list[pd.DataFrame]] = {}
    for name, frame in scores.items():
        groups.setdefault(categories[name], []).append(frame)
    return {cat: zscore(_mean_skipna(frames)) for cat, frames in groups.items()}


def composite(cat_scores: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """类间等权平均后每天标准化。"""
    return zscore(_mean_skipna(list(cat_scores.values())))
