"""按交易所规则算涨跌停（docs/STRATEGY.md 第 9 节）。

- 主板 10%，ST 5%
- 创业板（300、301 开头）2020-08-24 起 20%（含 ST），之前同主板
- 科创板（688、689 开头）20%
- 新股上市初期的特殊规则不用管：股票池已剔除上市不满一年的股票

涨跌停价 = 前收盘价 × (1 ± 幅度)，四舍五入到分。前收盘价要用交易所的除权参考价（baostock 的 preclose），
除权除息日和前一天的收盘价不一样。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CHINEXT_REFORM = pd.Timestamp("2020-08-24")
TICK_TOLERANCE = 0.001


def limit_pct(instrument: pd.Index, date: pd.DatetimeIndex, is_st: pd.Series) -> pd.Series:
    """每行的涨跌幅限制，索引同 is_st。"""
    code = pd.Index(instrument).str
    chinext = (code.startswith("SZ300") | code.startswith("SZ301")) & (pd.DatetimeIndex(date) >= CHINEXT_REFORM)
    star = code.startswith("SH688") | code.startswith("SH689")
    st = is_st.to_numpy().astype(bool)
    pct = np.where(chinext | star, 0.20, np.where(st, 0.05, 0.10))
    return pd.Series(pct, index=is_st.index)


def _round_half_up(x: pd.Series) -> pd.Series:
    """四舍五入到分。价格两位小数、幅度是 5%/10%/20%，乘积最多四位小数，加一个极小量就能避开浮点误差。"""
    return np.floor(x * 100 + 0.5 + 1e-6) / 100


def limit_prices(preclose: pd.Series, pct: pd.Series) -> tuple[pd.Series, pd.Series]:
    return _round_half_up(preclose * (1 + pct)), _round_half_up(preclose * (1 - pct))


def limit_flags(close: pd.Series, preclose: pd.Series, pct: pd.Series) -> tuple[pd.Series, pd.Series]:
    """收盘是否涨停、跌停。前收盘价缺失时都算否。"""
    up, down = limit_prices(preclose, pct)
    return (close >= up - TICK_TOLERANCE).fillna(False) & up.notna(), (close <= down + TICK_TOLERANCE).fillna(False) & down.notna()
