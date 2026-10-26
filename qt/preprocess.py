"""横截面预处理：每个交易日在当天的股票之间去极值、标准化。

Qlib 自带的 RobustZScoreNorm 是在训练窗口上整体拟合的（时间序列方向），这里需要的是
每天横截面上做，所以自己实现，方便单独测试。Series 和 DataFrame（逐列处理）都可以传。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826  # 正态分布下 MAD × 1.4826 = 标准差


def _by_date(s):
    return s.groupby(level="datetime")


def winsorize_mad(s: pd.Series | pd.DataFrame, n: float = 3.0) -> pd.Series | pd.DataFrame:
    """把超出 中位数 ± n × 1.4826 × MAD 的值截到边界上。当天 MAD 为 0 时不截。"""
    med = _by_date(s).transform("median")
    mad = _by_date((s - med).abs()).transform("median") * MAD_SCALE
    width = (n * mad).where(mad > 0, np.inf)
    return s.clip(med - width, med + width)


def zscore(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    g = _by_date(s)
    return (s - g.transform("mean")) / g.transform("std")


def standardize(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return zscore(winsorize_mad(s))
