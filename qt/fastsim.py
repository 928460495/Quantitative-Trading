"""计划 3c 任务 4：快速模拟器，给"选"用（逐步加因子、216 个组合参数、CSCV）。

规则和 qt.account.run_account 相同（T 日信号、T+1 收盘成交、能买能卖、退市按最后价格卖、止损 −20% 冷却 20 天、
单股上限 20%、调仓间隔、跌出前 sell_mult×N 才卖、最小交易额、按回撤分档减仓），只有三点简化：

1. 不按手取整：买入金额 = 预算（付完成本现金不够就减到正好够）
2. 行业上限按只数算：每个行业最多 floor(industry_cap × N) 只
3. 不记逐笔交易明细，只返回日收益

成本仍用 qt.costs.trade_cost 按每笔金额算，所以最低 5 元佣金照样生效。
最终方案和验证段一律用 run_account 确认；两者在参考方案上的差距见 scripts/run_plan3c_train.py 第①步。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qt.account import TRADING_DAYS, AccountRules, MarketData, _ranks, buy_budget
from qt.costs import trade_cost
from qt.drawdown import DEFAULT_TIERS, DrawdownTiers

_DEFAULT = AccountRules()          # 止损、冷却、单股上限沿用账户默认值（DECISIONS 2026-11-10）


@dataclass(frozen=True)
class SimParams:
    n_hold: int = 20
    rebalance_every: int = 5
    sell_mult: int = 3
    industry_cap: float = 0.30
    tiers: bool = True
    min_trade: float = 2000.0
    cost_multiplier: float = 1.0


def sharpe(returns: pd.Series) -> float:
    """日收益的年化夏普（无风险利率 0）；波动为 0 时为 NaN。"""
    sd = returns.std()
    return float(returns.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def simulate(data: MarketData, p: SimParams, cash: float = 100_000) -> pd.Series:
    """返回日收益（index = data.dates，第一天为 0）。"""
    n_days = len(data.dates)
    adj = data.adj_close
    marks = pd.DataFrame(adj).ffill().to_numpy()
    has_price = ~np.isnan(adj)
    last_day = np.where(has_price.any(axis=0), n_days - 1 - np.argmax(has_price[::-1], axis=0), -1)
    sell_rank = p.sell_mult * p.n_hold
    cap_count = math.floor(p.industry_cap * p.n_hold + 1e-9)
    tiers = DrawdownTiers(DEFAULT_TIERS) if p.tiers else None
    stop, cooldown, max_weight = _DEFAULT.stop_loss, _DEFAULT.cooldown_days, _DEFAULT.max_stock_weight

    units: dict[int, float] = {}
    cost_price: dict[int, float] = {}
    cool: dict[int, int] = {}
    navs = np.empty(n_days)
    navs[0] = cash

    def sell(i: int, col: int, price: float) -> None:
        nonlocal cash
        value = units.pop(col) * price
        cost_price.pop(col)
        cash += value - trade_cost(value, "sell", data.dates[i], p.cost_multiplier)

    for i in range(1, n_days):
        s = i - 1
        fraction = 1.0 if tiers is None else tiers.update(navs[s])
        slot_limit = DrawdownTiers.slots(p.n_hold, fraction)
        rebalance = s % p.rebalance_every == 0
        tradable = has_price[i]
        ranks = None

        for col in list(units):
            if i > last_day[col]:
                sell(i, col, marks[i, col])
                continue
            if not (data.can_sell[s, col] and tradable[col]):
                continue
            if marks[s, col] <= cost_price[col] * (1 - stop):
                sell(i, col, adj[i, col])
                cool[col] = i + cooldown
            elif rebalance:
                if ranks is None:
                    ranks = _ranks(data.score[s])
                if ranks[col] > sell_rank:
                    sell(i, col, adj[i, col])
        if len(units) > slot_limit:
            if ranks is None:
                ranks = _ranks(data.score[s])
            for col in sorted(units, key=lambda c: -ranks[c]):
                if len(units) <= slot_limit:
                    break
                if data.can_sell[s, col] and tradable[col]:
                    sell(i, col, adj[i, col])

        empty = slot_limit - len(units) if rebalance else 0
        if empty > 0:
            if ranks is None:
                ranks = _ranks(data.score[s])
            nav = cash + sum(units[c] * marks[i, c] for c in units)
            budget = buy_budget(cash, empty, nav, p.n_hold, max_weight, fraction)
            if budget >= p.min_trade:
                counts: dict[object, int] = {}
                if data.industry is not None:
                    for c in units:
                        ind = data.industry[s, c]
                        counts[ind] = counts.get(ind, 0) + 1
                order = np.argsort(ranks, kind="stable")
                bought = 0
                for col in order:
                    if bought >= empty or not np.isfinite(ranks[col]):
                        break
                    if col in units or not data.can_buy[s, col] or not tradable[col] or cool.get(col, -1) >= i:
                        continue
                    ind = None if data.industry is None else data.industry[s, col]
                    if ind is not None and counts.get(ind, 0) >= cap_count:
                        continue
                    value = min(budget, cash)
                    cost = trade_cost(value, "buy", data.dates[i], p.cost_multiplier)
                    if value + cost > cash:
                        value = cash - cost
                        cost = trade_cost(value, "buy", data.dates[i], p.cost_multiplier)
                    if value <= 0:
                        break
                    cash -= value + cost
                    units[col] = value / adj[i, col]
                    cost_price[col] = adj[i, col]
                    bought += 1
                    if ind is not None:
                        counts[ind] = counts.get(ind, 0) + 1

        navs[i] = cash + sum(units[c] * marks[i, c] for c in units)

    out = pd.Series(navs, index=data.dates).pct_change()
    out.iloc[0] = 0.0
    return out
