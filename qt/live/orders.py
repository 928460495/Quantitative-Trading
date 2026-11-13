"""每个交易日要卖什么、买什么（计划 5 任务 1）。

规则必须和回测的账户引擎 qt/account.run_account 一模一样：tests/test_live_orders.py 的回放测试把回测账户
每天的状态喂给这里，要求产出和回测当天的成交完全相同。改了账户引擎的买卖逻辑，这里要一起改。

和回测唯一的差别在调用方：回测按 T+1 收盘价算股数；实盘出单时还不知道 T+1 的价格，传入 T 日收盘价当参考价。
只支持现行基线用到的规则；止损、回撤兜底、回撤分档在这里没有实现，传进来直接报错。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qt.account import LOT, STAR_MIN_SHARES, AccountRules, _ranks, buy_budget, lot_shares
from qt.costs import trade_cost


@dataclass(frozen=True)
class Buy:
    code: str
    shares: int
    price: float
    value: float
    cost: float


@dataclass
class Orders:
    sells: list[tuple[str, str]] = field(default_factory=list)      # (代码, 原因)，原因同回测：rank、exposure
    buys: list[Buy] = field(default_factory=list)
    cash_after: float = 0.0                                          # 按参考价全部成交后剩下的现金（估计）


def _check_rules(rules: AccountRules) -> None:
    if rules.stop_loss is not None or rules.drawdown_stop is not None or rules.drawdown_tiers is not None:
        raise NotImplementedError("出单只支持现行基线的规则：止损、回撤兜底、回撤分档没有实现")


def plan_orders(score: pd.Series, can_buy: pd.Series, can_sell: pd.Series, industry: pd.Series | None,
                tradable: pd.Series, holdings: dict[str, float], cash: float, price: pd.Series, slot_limit: int,
                rebalance: bool, rules: AccountRules, date: pd.Timestamp) -> Orders:
    """score、can_buy、can_sell、industry：信号日的值；tradable、price：成交日能不能交易、成交价（未复权）；
    holdings：现有持仓 {代码: 按成交价的市值}，顺序是买入的先后（排名相同时决定先卖哪只）。"""
    _check_rules(rules)
    codes = score.index
    ranks = pd.Series(_ranks(score.to_numpy(dtype=float)), index=codes)
    rank = lambda c: float(ranks.get(c, np.inf))
    flag = lambda s, c: bool(s.get(c, False))
    ind_of = lambda c: None if industry is None or pd.isna(industry.get(c, None)) else industry.get(c)
    held = dict(holdings)
    out = Orders()

    def sell(c: str, reason: str) -> None:
        nonlocal cash
        value = held.pop(c)
        cash += value - trade_cost(value, "sell", date, rules.cost_multiplier)
        out.sells.append((c, reason))

    can_trade = lambda c: flag(can_sell, c) and flag(tradable, c)
    for c in list(held):                                             # 1. 排名卖出（只在调仓日）
        if can_trade(c) and rebalance and rank(c) > rules.sell_rank:
            sell(c, "rank")
    for c in sorted(held, key=lambda c: -rank(c)):                   # 2. 超过持股只数上限：按排名从低到高卖
        if len(held) <= slot_limit:
            break
        if can_trade(c):
            sell(c, "exposure")

    empty = slot_limit - len(held) if rebalance else 0                # 3. 买
    if empty > 0:
        nav = cash + sum(held.values())
        budget = buy_budget(cash, empty, nav, rules.n_hold, rules.max_stock_weight)
        if rules.slot_budget is not None:
            budget = min(budget, nav * rules.slot_budget)
        if budget < rules.min_trade:
            empty = 0
        industry_value: dict[object, float] = {}
        for c, v in held.items():
            if ind_of(c) is not None:
                industry_value[ind_of(c)] = industry_value.get(ind_of(c), 0.0) + v
        bought = 0
        for j in np.argsort(ranks.to_numpy(), kind="stable"):
            c = codes[j]
            if bought >= empty or not np.isfinite(ranks.iloc[j]):
                break
            if c in held or not flag(can_buy, c) or not flag(tradable, c):
                continue
            p = float(price.get(c, np.nan))
            ind = ind_of(c)
            if ind is not None:
                shares = lot_shares(min(budget, cash), p, c)
                if industry_value.get(ind, 0.0) + shares * p > nav * rules.max_industry_weight:
                    continue
            shares = lot_shares(min(budget, cash), p, c)
            step = 1 if c.startswith(("SH688", "SH689")) else LOT
            while shares > 0:                                        # 付完成本现金不够就一手一手往下减
                value = shares * p
                cost = trade_cost(value, "buy", date, rules.cost_multiplier)
                if value + cost <= cash:
                    break
                shares -= step
                if c.startswith(("SH688", "SH689")) and shares < STAR_MIN_SHARES:
                    shares = 0
            if shares <= 0:
                continue
            cash -= value + cost
            out.buys.append(Buy(c, int(shares), p, value, cost))
            bought += 1
            if ind is not None:
                industry_value[ind] = industry_value.get(ind, 0.0) + value
    out.cash_after = cash
    return out
