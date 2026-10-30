"""股票池过滤：标出信号日之后哪些股票能买、哪些能卖。

两套规则：
- tradable_mask / sellable_mask：沪深300 阶段用的，只靠 Qlib 数据包（没有 ST 标记，涨跌停统一按 9.5%）
  - 能买：上市满一年、不是北交所、入场日（T+1）没有停牌、入场日收盘没有涨停
  - 能卖：入场日没有停牌、入场日收盘没有跌停
- full_a_masks：全 A 用的（docs/STRATEGY.md 第 4 节），只交易主板；用 baostock 的交易状态、ST 标记和真实涨跌停，
  另外剔除主板里成交额最小的 20%
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qt.config import DATA_DIR
from qt.limits import limit_flags, limit_pct

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


# ---- 全 A（docs/STRATEGY.md 第 4 节）：用 baostock 的交易状态、ST 标记和真实涨跌停 ----

LIQUIDITY_CUT = 0.2  # 每天剔除主板里 20 日平均成交额最小的 20%
MAIN_BOARD_PREFIXES = ("SH600", "SH601", "SH603", "SH605", "SZ000", "SZ001", "SZ002", "SZ003")


def is_main_board(instruments: pd.Index) -> np.ndarray:
    """只交易主板（docs/DECISIONS.md 2026-10-27）：创业板、科创板、北交所都不纳入。"""
    return np.asarray(pd.Index(instruments).str.startswith(MAIN_BOARD_PREFIXES), dtype=bool)


def full_a_masks(df: pd.DataFrame, daily: pd.DataFrame, listed: pd.Series) -> tuple[pd.Series, pd.Series]:
    """信号日 T 的 (能买, 能卖)，索引同 df。

    df 需要 "_amount20"（Mean($amount,20)）；daily 是 load_daily(["preclose", "close", "tradestatus", "isST"])，
    要覆盖到 df 最后一天的下一个交易日。入场日（T+1）在 daily 里没有记录的，按停牌处理。
    """
    dates = df.index.get_level_values("datetime")
    codes = df.index.get_level_values("instrument")
    calendar = daily.index.get_level_values("datetime").unique().union(dates.unique()).sort_values()
    pos = calendar.get_indexer(dates) + 1
    entry_dates = pd.DatetimeIndex(np.where(pos < len(calendar), calendar[np.minimum(pos, len(calendar) - 1)], pd.NaT))
    entry = daily.reindex(pd.MultiIndex.from_arrays([entry_dates, codes]))
    entry.index = df.index

    pct = limit_pct(codes, entry_dates, entry["isST"].fillna(0))
    at_up, at_down = limit_flags(entry["close"], entry["preclose"], pct)
    trading = entry["tradestatus"] == 1

    seasoned = (dates - pd.DatetimeIndex(codes.map(listed))).days >= MIN_LISTED_DAYS
    main = is_main_board(codes)
    amount = df["_amount20"].where(main)
    liquid = amount.groupby(level="datetime").rank(pct=True) > LIQUIDITY_CUT
    static = pd.Series(seasoned & main, index=df.index)

    can_buy = static & liquid & trading & (entry["isST"] == 0) & ~at_up
    can_sell = trading & ~at_down
    return can_buy.fillna(False).astype(bool), can_sell.fillna(False).astype(bool)
