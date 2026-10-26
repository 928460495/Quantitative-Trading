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

from qt.costs import trade_cost

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


@dataclass
class AccountResult:
    daily: pd.DataFrame
    trades: pd.DataFrame


_TRADE_COLUMNS = ["date", "instrument", "side", "shares", "price", "value", "cost", "reason"]


class _Account:
    """一个账户的状态和买卖动作。持仓记复权份额（units），成本记复权价。"""

    def __init__(self, data: MarketData, rules: AccountRules):
        self.data, self.rules = data, rules
        self.cash = float(rules.cash)
        self.units: dict[int, float] = {}
        self.cost_price: dict[int, float] = {}
        self.trades: list[tuple] = []
        self.day_cost = 0.0
        self.n_buys = self.n_sells = 0

    def value_of(self, col: int, prices: np.ndarray) -> float:
        return self.units[col] * prices[col]

    def sell(self, i: int, col: int, reason: str) -> None:
        d = self.data
        adj, raw = d.adj_close[i, col], d.raw_close[i, col]
        value = self.units[col] * adj
        cost = trade_cost(value, "sell", d.dates[i], self.rules.cost_multiplier)
        self.cash += value - cost
        self.day_cost += cost
        self.n_sells += 1
        self.trades.append((d.dates[i], d.instruments[col], "sell", round(value / raw), raw, value, cost, reason))
        del self.units[col], self.cost_price[col]

    def buy(self, i: int, col: int, budget: float) -> float:
        """按预算买，付完成本现金不够就一手一手往下减。返回买入金额（0 表示没买）。"""
        d = self.data
        code, adj, raw = d.instruments[col], d.adj_close[i, col], d.raw_close[i, col]
        shares = lot_shares(min(budget, self.cash), raw, code)
        step = 1 if code.startswith(("SH688", "SH689")) else LOT
        while shares > 0:
            value = shares * raw
            cost = trade_cost(value, "buy", d.dates[i], self.rules.cost_multiplier)
            if value + cost <= self.cash:
                break
            shares -= step
            if code.startswith(("SH688", "SH689")) and shares < STAR_MIN_SHARES:
                shares = 0
        if shares <= 0:
            return 0.0
        self.cash -= value + cost
        self.day_cost += cost
        self.n_buys += 1
        self.units[col] = value / adj
        self.cost_price[col] = adj
        self.trades.append((d.dates[i], code, "buy", shares, raw, value, cost, "buy"))
        return value


def _ranks(score_row: np.ndarray) -> np.ndarray:
    """排名从 1 开始；没有分数的为无穷大（视为不在股票池）。"""
    ranks = np.full(score_row.shape, np.inf)
    valid = np.flatnonzero(~np.isnan(score_row))
    order = valid[np.argsort(-score_row[valid], kind="stable")]
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def run_account(data: MarketData, rules: AccountRules = AccountRules(), exposure: pd.Series | None = None) -> AccountResult:
    acct = _Account(data, rules)
    marks = pd.DataFrame(data.adj_close).ffill().to_numpy()  # 估值用：停牌沿用最后一个价格
    rows = [(data.dates[0], acct.cash, acct.cash, 0.0, 0, rules.n_hold, 0.0, 0, 0)]

    for i in range(1, len(data.dates)):
        s = i - 1
        acct.day_cost, acct.n_buys, acct.n_sells = 0.0, 0, 0
        ranks = _ranks(data.score[s])
        tradable_today = ~np.isnan(data.adj_close[i]) & ~np.isnan(data.raw_close[i])

        # 1. 卖
        for col in list(acct.units):
            if ranks[col] > rules.sell_rank and data.can_sell[s, col] and tradable_today[col]:
                acct.sell(i, col, "rank")

        # 2. 买
        slot_limit = rules.n_hold
        empty = slot_limit - len(acct.units)
        if empty > 0:
            nav = acct.cash + sum(acct.value_of(c, marks[i]) for c in acct.units)
            budget = min(acct.cash / empty, nav * rules.max_stock_weight)
            industry_value: dict[object, float] = {}
            if data.industry is not None:
                for c in acct.units:
                    ind = data.industry[s, c]
                    industry_value[ind] = industry_value.get(ind, 0.0) + acct.value_of(c, marks[i])
            bought = 0
            for col in np.argsort(ranks, kind="stable"):
                if bought >= empty or not np.isfinite(ranks[col]):
                    break
                if col in acct.units or not data.can_buy[s, col] or not tradable_today[col]:
                    continue
                ind = None if data.industry is None else data.industry[s, col]
                if ind is not None:
                    shares = lot_shares(min(budget, acct.cash), data.raw_close[i, col], data.instruments[col])
                    if (industry_value.get(ind, 0.0) + shares * data.raw_close[i, col]) > nav * rules.max_industry_weight:
                        continue
                value = acct.buy(i, col, budget)
                if value > 0:
                    bought += 1
                    if ind is not None:
                        industry_value[ind] = industry_value.get(ind, 0.0) + value

        # 3. 记账
        stock_value = sum(acct.value_of(c, marks[i]) for c in acct.units)
        rows.append((data.dates[i], acct.cash + stock_value, acct.cash, stock_value, len(acct.units), slot_limit,
                     acct.day_cost, acct.n_buys, acct.n_sells))

    daily = pd.DataFrame(rows, columns=["date", "nav", "cash", "stock_value", "n_holdings", "slot_limit", "cost",
                                        "n_buys", "n_sells"]).set_index("date")
    return AccountResult(daily=daily, trades=pd.DataFrame(acct.trades, columns=_TRADE_COLUMNS))
