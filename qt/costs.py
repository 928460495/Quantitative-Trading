"""交易成本假设（docs/DECISIONS.md）。最低 5 元佣金只在回测里按金额处理，因子检验里不计。"""

from __future__ import annotations

import pandas as pd

COMMISSION = 0.00025
SLIPPAGE = 0.001
STAMP_DUTY_CUT = pd.Timestamp("2023-08-28")


def stamp_duty(date: pd.Timestamp) -> float:
    return 0.0005 if date >= STAMP_DUTY_CUT else 0.001


def buy_cost(date: pd.Timestamp) -> float:
    return COMMISSION + SLIPPAGE


def sell_cost(date: pd.Timestamp) -> float:
    return COMMISSION + stamp_duty(date) + SLIPPAGE


def round_trip_cost(date: pd.Timestamp) -> float:
    return buy_cost(date) + sell_cost(date)
