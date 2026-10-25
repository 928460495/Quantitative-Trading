"""横截面预处理：每个交易日在当天的股票之间去极值、标准化。

Qlib 自带的 RobustZScoreNorm 是在训练窗口上整体拟合的（时间序列方向），这里需要的是
每天横截面上做，所以自己实现，方便单独测试。中性化在第 3 个 PR 加入。
"""

from __future__ import annotations

import pandas as pd

MAD_SCALE = 1.4826  # 正态分布下 MAD × 1.4826 = 标准差


def _by_date(s: pd.Series):
    return s.groupby(level="datetime", group_keys=False)


def winsorize_mad(s: pd.Series, n: float = 3.0) -> pd.Series:
    """把超出 中位数 ± n × 1.4826 × MAD 的值截到边界上。"""

    def clip(x: pd.Series) -> pd.Series:
        med = x.median()
        mad = (x - med).abs().median() * MAD_SCALE
        return x if mad == 0 or pd.isna(mad) else x.clip(med - n * mad, med + n * mad)

    return _by_date(s).apply(clip)


def zscore(s: pd.Series) -> pd.Series:
    return _by_date(s).transform(lambda x: (x - x.mean()) / x.std())


def standardize(s: pd.Series) -> pd.Series:
    return zscore(winsorize_mad(s))
