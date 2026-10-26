"""按真实账户记账的日频回测（docs/STRATEGY.md 第 5、6、8、9 节）。

和 qt/simulate.py 的近似模拟不同，这里按真实账户记账：10 万元、1 手取整、每笔最低 5 元佣金、
单股和行业上限、个股止损和冷却期、均线仓位控制、账户回撤兜底。

时间约定：T 日收盘出信号，T+1 收盘成交。第 i 天成交时，所有决定只用第 i-1 天（信号日）及以前的数据。
记账用复权价：持仓记"复权份额"，分红送转体现在复权价里（相当于分红再投资，不计红利税）；手数用未复权价算。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOT = 100
STAR_MIN_SHARES = 200  # 科创板最少 200 股，之后按 1 股加


@dataclass(frozen=True)
class AccountRules:
    cash: float = 100_000
    n_hold: int = 15
    sell_rank: int = 45
    max_stock_weight: float = 0.20
    max_industry_weight: float = 0.30
    stop_loss: float | None = 0.20
    cooldown_days: int = 20
    drawdown_stop: float | None = 0.20
    cost_multiplier: float = 1.0


@dataclass
class MarketData:
    """"日期 × 股票"的矩阵。score、can_buy、can_sell 按信号日对齐（can_buy 表示第二天能不能买）；价格是当天收盘。"""

    dates: pd.DatetimeIndex
    instruments: pd.Index
    score: np.ndarray
    adj_close: np.ndarray
    raw_close: np.ndarray
    can_buy: np.ndarray
    can_sell: np.ndarray
    industry: np.ndarray | None = None


def _wide(s: pd.Series, dates: pd.DatetimeIndex, instruments: pd.Index, fill, dtype) -> np.ndarray:
    return s.unstack("instrument").reindex(index=dates, columns=instruments).fillna(fill).to_numpy(dtype=dtype)


def market_data(score: pd.Series, adj_close: pd.Series, raw_close: pd.Series, can_buy: pd.Series,
                can_sell: pd.Series, industry: pd.Series | None = None) -> MarketData:
    """参数都是 (datetime, instrument) 索引的 Series。能买、能卖缺失按 False；价格不填充（停牌为 NaN）。"""
    dates = pd.DatetimeIndex(adj_close.index.get_level_values("datetime").unique().union(
        score.index.get_level_values("datetime").unique())).sort_values()
    instruments = pd.Index(sorted(set(adj_close.index.get_level_values("instrument"))
                                  | set(score.index.get_level_values("instrument"))))
    return MarketData(
        dates=dates,
        instruments=instruments,
        score=_wide(score.astype(float), dates, instruments, np.nan, float),
        adj_close=_wide(adj_close.astype(float), dates, instruments, np.nan, float),
        raw_close=_wide(raw_close.astype(float), dates, instruments, np.nan, float),
        can_buy=_wide(can_buy.astype(float), dates, instruments, 0.0, float) > 0,
        can_sell=_wide(can_sell.astype(float), dates, instruments, 0.0, float) > 0,
        industry=None if industry is None else _wide(industry.astype(object), dates, instruments, None, object),
    )


def lot_shares(budget: float, raw_price: float, instrument: str) -> int:
    """预算能买多少股：普通股票按 100 股取整；科创板不足 200 股为 0，否则按 1 股取整。"""
    if not raw_price > 0 or not budget > 0:
        return 0
    if instrument.startswith(("SH688", "SH689")):
        shares = math.floor(budget / raw_price)
        return shares if shares >= STAR_MIN_SHARES else 0
    return math.floor(budget / raw_price / LOT) * LOT
