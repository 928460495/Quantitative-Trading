"""股票池过滤：标出信号日之后哪些股票能买、哪些能卖。

当前能做的过滤（数据包里没有 ST 标记，ST 过滤等第 3 个 PR 补数据）：
- 能买：上市满一年（以数据包里第一次出现的日期近似上市日；2000-01-04 之前上市的不受影响）、
  不是北交所、入场日（T+1）没有停牌、入场日收盘没有涨停
- 能卖：入场日没有停牌、入场日收盘没有跌停

涨跌停统一按 9.5% 判断，对 20% 涨跌停的创业板和科创板偏保守。用 $change（未复权涨跌幅）判断。
"""

from __future__ import annotations

import pandas as pd

from qt.config import DATA_DIR

LIMIT_THRESHOLD = 0.095
MIN_LISTED_DAYS = 365

# 过滤需要的字段，和因子一起取，避免多次读数据
FILTER_EXPRS: dict[str, str] = {
    "_entry_close": "Ref($close,-1)",
    "_entry_volume": "Ref($volume,-1)",
    "_entry_change": "Ref($change,-1)",
}


def listing_dates() -> pd.Series:
    spans = pd.read_csv(DATA_DIR / "instruments" / "all.txt", sep="\t", header=None, names=["code", "start", "end"])
    return pd.to_datetime(spans.groupby("code")["start"].min())


def _entry_trading(df: pd.DataFrame) -> pd.Series:
    return df["_entry_close"].notna() & (df["_entry_volume"] > 0)


def tradable_mask(df: pd.DataFrame, listed: pd.Series | None = None) -> pd.Series:
    """能买入。df 需要包含 FILTER_EXPRS 的列。"""
    listed = listing_dates() if listed is None else listed
    dates = df.index.get_level_values("datetime")
    codes = df.index.get_level_values("instrument")

    seasoned = (dates - pd.DatetimeIndex(codes.map(listed))).days >= MIN_LISTED_DAYS
    not_bj = ~codes.str.startswith("BJ")
    entry_not_limit_up = ~(df["_entry_change"] >= LIMIT_THRESHOLD)
    return pd.Series(seasoned & not_bj, index=df.index) & _entry_trading(df) & entry_not_limit_up


def sellable_mask(df: pd.DataFrame) -> pd.Series:
    """能卖出。"""
    return _entry_trading(df) & ~(df["_entry_change"] <= -LIMIT_THRESHOLD)
