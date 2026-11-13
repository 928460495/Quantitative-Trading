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

from qt.wide_eval import neutralize, zscore


def _mean_skipna(frames: list[pd.DataFrame], weights: list[float] | None = None) -> pd.DataFrame:
    weights = weights or [1.0] * len(frames)
    total = sum(w * f.fillna(0.0) for f, w in zip(frames, weights))
    count = sum(w * f.notna().astype(float) for f, w in zip(frames, weights))
    return (total / count).where(count > 0)


def category_scores(scores: Mapping[str, pd.DataFrame], categories: Mapping[str, str]) -> dict[str, pd.DataFrame]:
    """大类 → 本类因子的等权平均，再每天标准化。categories：因子名 → 大类，必须覆盖 scores 的每个因子。"""
    groups: dict[str, list[pd.DataFrame]] = {}
    for name, frame in scores.items():
        groups.setdefault(categories[name], []).append(frame)
    return {cat: zscore(_mean_skipna(frames)) for cat, frames in groups.items()}


def composite(cat_scores: Mapping[str, pd.DataFrame], weights: Mapping[str, float] | None = None) -> pd.DataFrame:
    """类间加权平均（weights 里没写的大类权重为 1；不传就是等权）后每天标准化。"""
    w = weights or {}
    return zscore(_mean_skipna(list(cat_scores.values()), [w.get(c, 1.0) for c in cat_scores]))


LOWVOL = "低波动"


def neutral_composite(factor_scores: Mapping[str, pd.DataFrame], categories: Mapping[str, str], industry: pd.DataFrame,
                      log_mcap: pd.DataFrame, lowvol_weight: float = 1.0, smooth: int = 1) -> pd.DataFrame:
    """计划 3c 的合成分（docs/DECISIONS.md 2026-11-10「计划 3c 事先登记的规则」）：

    类内等权 → 每类标准化 → 类间加权（低波动类权重 lowvol_weight，其余 1）→ 每天对申万一级行业和对数流通市值
    中性化（不押大小盘）→ 过去 smooth 天平均（只用当天及以前，不足 smooth 天用已有天数）→ 每天标准化。
    """
    if smooth < 1:
        raise ValueError("平滑天数至少 1")
    cats = category_scores(factor_scores, categories)
    total = composite(cats, {LOWVOL: lowvol_weight})
    neutral = neutralize(total, industry, log_mcap, size=True)
    if smooth > 1:
        neutral = neutral.rolling(smooth, min_periods=1).mean().where(neutral.notna())   # 今天没分数 = 不在股票池
    return zscore(neutral)
