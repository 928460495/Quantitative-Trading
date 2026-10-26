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
        self.cooldown_until: dict[int, int] = {}  # 止损卖出后，到这一天（含）为止不能买回
        self.day_cost = 0.0
        self.n_buys = self.n_sells = 0

    def value_of(self, col: int, prices: np.ndarray) -> float:
        return self.units[col] * prices[col]

    def sell(self, i: int, col: int, reason: str, adj: float | None = None, raw: float | None = None) -> None:
        """按第 i 天收盘卖出；退市时传入最后的价格。"""
        d = self.data
        adj = d.adj_close[i, col] if adj is None else adj
        raw = d.raw_close[i, col] if raw is None else raw
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


def trend_exposure(index_close: pd.Series, window: int, low: float) -> pd.Series:
    """大盘趋势过滤（说明书第 6 节 A）：收盘价在 window 日均线上为 1，跌破为 low；均线还算不出时为 1。"""
    ma = index_close.rolling(window).mean()
    return pd.Series(np.where(ma.isna() | (index_close >= ma), 1.0, low), index=index_close.index)


def run_account(data: MarketData, rules: AccountRules = AccountRules(), exposure: pd.Series | None = None) -> AccountResult:
    """exposure：信号日 → 持股比例（缺失按 1），位置上限 = floor(n_hold × exposure)。

    账户回撤兜底：信号日净值比历史最高跌了 drawdown_stop，第二天起全部卖出、位置上限为 0；
    等 exposure 先出现 < 1 再出现 = 1 才恢复，最高点重置为恢复时的净值。不传 exposure 时触发后不会恢复。
    """
    acct = _Account(data, rules)
    marks = pd.DataFrame(data.adj_close).ffill().to_numpy()  # 估值用：停牌沿用最后一个价格
    raw_marks = pd.DataFrame(data.raw_close).ffill().to_numpy()
    has_price = ~np.isnan(data.adj_close)
    last_price_day = np.where(has_price.any(axis=0), len(data.dates) - 1 - np.argmax(has_price[::-1], axis=0), -1)
    exp = np.ones(len(data.dates)) if exposure is None else exposure.reindex(data.dates).fillna(1.0).to_numpy(float)
    navs = [acct.cash]
    peak, halted, seen_out = acct.cash, False, False
    rows = [(data.dates[0], acct.cash, acct.cash, 0.0, 0, rules.n_hold, 0.0, 0, 0)]

    for i in range(1, len(data.dates)):
        s = i - 1
        acct.day_cost, acct.n_buys, acct.n_sells = 0.0, 0, 0
        ranks = _ranks(data.score[s])
        tradable_today = ~np.isnan(data.adj_close[i]) & ~np.isnan(data.raw_close[i])

        # 0. 仓位：回撤兜底的状态和位置上限，都只看信号日
        if not halted:
            peak = max(peak, navs[s])
            if rules.drawdown_stop is not None and navs[s] <= peak * (1 - rules.drawdown_stop):
                halted, seen_out = True, False
        elif exp[s] < 1:
            seen_out = True
        elif seen_out:
            halted, peak = False, navs[s]
        slot_limit = 0 if halted else int(np.floor(rules.n_hold * exp[s] + 1e-9))

        # 1. 卖：退市 → 回撤兜底 → 止损 → 排名 → 仓位上限（多出来的按排名从低到高卖）
        for col in list(acct.units):
            if i > last_price_day[col]:  # 之后再也没有价格：按最后价格卖出（偏乐观）
                acct.sell(i, col, "delisted", adj=marks[i, col], raw=raw_marks[i, col])
                continue
            if not (data.can_sell[s, col] and tradable_today[col]):
                continue
            if halted:
                acct.sell(i, col, "drawdown")
            elif rules.stop_loss is not None and marks[s, col] <= acct.cost_price[col] * (1 - rules.stop_loss):
                acct.sell(i, col, "stop")
                acct.cooldown_until[col] = i + rules.cooldown_days
            elif ranks[col] > rules.sell_rank:
                acct.sell(i, col, "rank")
        for col in sorted(acct.units, key=lambda c: -ranks[c]):
            if len(acct.units) <= slot_limit:
                break
            if data.can_sell[s, col] and tradable_today[col]:
                acct.sell(i, col, "exposure")

        # 2. 买
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
                if acct.cooldown_until.get(col, -1) >= i:
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
        navs.append(acct.cash + stock_value)
        rows.append((data.dates[i], acct.cash + stock_value, acct.cash, stock_value, len(acct.units), slot_limit,
                     acct.day_cost, acct.n_buys, acct.n_sells))

    daily = pd.DataFrame(rows, columns=["date", "nav", "cash", "stock_value", "n_holdings", "slot_limit", "cost",
                                        "n_buys", "n_sells"]).set_index("date")
    return AccountResult(daily=daily, trades=pd.DataFrame(acct.trades, columns=_TRADE_COLUMNS))


TRADING_DAYS = 252


def account_metrics(result: AccountResult, bench_close: pd.Series) -> dict[str, float]:
    """复合年化收益、相对基准、最大回撤、波动、夏普（无风险利率按 0）、换手、成本、各类卖出次数、平均仓位。"""
    daily = result.daily
    nav = daily["nav"]
    n = len(nav) - 1
    bench = bench_close.reindex(nav.index).ffill()
    rets = nav.pct_change().iloc[1:]
    trades = result.trades
    sold = trades[trades["side"] == "sell"].groupby("date")["value"].sum().reindex(nav.index, fill_value=0.0)
    annual = (nav.iloc[-1] / nav.iloc[0]) ** (TRADING_DAYS / n) - 1
    bench_annual = (bench.iloc[-1] / bench.iloc[0]) ** (TRADING_DAYS / n) - 1
    reasons = trades["reason"].value_counts()
    return {
        "annual_return": annual,
        "bench_annual": bench_annual,
        "excess_annual": annual - bench_annual,
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
        "volatility": rets.std() * np.sqrt(TRADING_DAYS),
        "sharpe": rets.mean() / rets.std() * np.sqrt(TRADING_DAYS) if rets.std() > 0 else np.nan,
        "turnover": float((sold / nav).iloc[1:].mean()),
        "cost_annual": float((daily["cost"] / nav).iloc[1:].mean() * TRADING_DAYS),
        "n_trades": len(trades),
        "n_stops": int(reasons.get("stop", 0)),
        "n_delisted": int(reasons.get("delisted", 0)),
        "avg_exposure": float((daily["stock_value"] / nav).iloc[1:].mean()),
    }
