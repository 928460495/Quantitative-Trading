"""交易成本假设（docs/DECISIONS.md、docs/STRATEGY.md 第 9 节）。

按比例的 buy_cost / sell_cost 给因子检验的近似模拟用；按金额的 trade_cost 给账户回测用，含每笔最低 5 元佣金。
"""

from __future__ import annotations

import pandas as pd

COMMISSION = 0.00025
SLIPPAGE = 0.001
MIN_COMMISSION = 5.0
STAMP_DUTY_CUT = pd.Timestamp("2023-08-28")


def stamp_duty(date: pd.Timestamp) -> float:
    return 0.0005 if date >= STAMP_DUTY_CUT else 0.001


def buy_cost(date: pd.Timestamp) -> float:
    return COMMISSION + SLIPPAGE


def sell_cost(date: pd.Timestamp) -> float:
    return COMMISSION + stamp_duty(date) + SLIPPAGE


def round_trip_cost(date: pd.Timestamp) -> float:
    return buy_cost(date) + sell_cost(date)


def trade_cost(value: float, side: str, date: pd.Timestamp, multiplier: float = 1.0) -> float:
    """一笔成交的总成本（元）：佣金（每笔最低 5 元）+ 滑点 + 卖出时的印花税，整体乘 multiplier。"""
    if side not in ("buy", "sell"):
        raise ValueError(f"side 只能是 buy 或 sell：{side}")
    cost = max(value * COMMISSION, MIN_COMMISSION) + value * SLIPPAGE
    if side == "sell":
        cost += value * stamp_duty(date)
    return cost * multiplier
