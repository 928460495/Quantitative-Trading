"""计划 4 任务 5：按真实账户规则回测合成分，加仓位控制 A（中证全指均线过滤）/ 对照 B（波动率控制）/ 兜底 C，
按事先写定的规则挑 A（docs/DECISIONS.md 2026-11-09）。

一段数据（seg）是 scripts/build_composite.load_segment 读回来的字典：composite、categories、close、raw_close、
can_buy、can_sell、industry（宽表，只含这一段的日期）和 index（中证全指 000985、沪深300 全收益 H00300，含预热期）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qt.account import AccountResult, AccountRules, account_metrics, index_volatility, run_account, trend_exposure, vol_exposure
from qt.composite import composite
from qt.wide_eval import _market_data

INDEX = "000985"          # 均线过滤、波动率控制看的指数
BENCH = "H00300"          # 比较基准：沪深300 全收益
A_WINDOWS = (60, 120, 250)
A_LOWS = (0.5, 0.0)
B_WINDOW = 20
MAX_DRAWDOWN = 0.20


def trend(seg: dict, window: int, low: float) -> pd.Series:
    """A：信号日 → 持股比例。均线用含预热期的指数算，再取这一段的日期。"""
    close = seg["index"][INDEX].dropna()
    return trend_exposure(close, window, low).reindex(seg["close"].index).fillna(1.0)


def vol_target(train: dict, window: int = B_WINDOW) -> float:
    """B 的目标波动率：训练段 20 日波动率的中位数（只用训练段，不看验证段）。"""
    sigma = index_volatility(train["index"][INDEX].dropna(), window)
    return float(sigma.reindex(train["close"].index).median())


def vol(seg: dict, target: float, window: int = B_WINDOW) -> pd.Series:
    close = seg["index"][INDEX].dropna()
    return vol_exposure(close, window, target).reindex(seg["close"].index).fillna(1.0)


def composite_without(seg: dict, category: str) -> pd.DataFrame:
    """去掉一个大类后其余各类等权（只报告用）。"""
    return composite({k: v for k, v in seg["categories"].items() if k != category})


def run_variant(seg: dict, score: pd.DataFrame, rules: AccountRules, exposure: pd.Series | None
                ) -> tuple[dict, AccountResult]:
    data = _market_data(score, seg["close"], seg["raw_close"], seg["can_buy"], seg["can_sell"], seg["industry"])
    result = run_account(data, rules, exposure)
    bench = seg["index"][BENCH].reindex(seg["close"].index).ffill()
    m = account_metrics(result, bench)
    trades = result.trades
    m["c_triggers"] = int(trades.loc[trades["reason"] == "drawdown", "date"].nunique())
    m["days_in_cash"] = float((result.daily["n_holdings"].iloc[1:] == 0).mean())
    return m, result


def yearly(result: AccountResult) -> pd.DataFrame:
    """分年度收益和年内最大回撤（从当年内的最高点算）。"""
    nav = result.daily["nav"]
    rows = {}
    for year, s in nav.groupby(nav.index.year):
        prev = nav[nav.index.year < year]
        start = prev.iloc[-1] if len(prev) else s.iloc[0]
        rows[year] = {"return": s.iloc[-1] / start - 1, "max_drawdown": float((s / s.cummax() - 1).min())}
    return pd.DataFrame.from_dict(rows, orient="index")


def choose_a(table: pd.DataFrame, limit: float = MAX_DRAWDOWN) -> str | None:
    """训练段和验证段最大回撤都不超过 limit 的组合里，选验证段年化收益最高的；都不达标返回 None（不放宽）。

    table：每行一个 A 组合，列 train_max_drawdown、valid_max_drawdown（负数）、valid_annual。
    """
    ok = table[(table["train_max_drawdown"] >= -limit - 1e-12) & (table["valid_max_drawdown"] >= -limit - 1e-12)]
    if ok.empty:
        return None
    return str(ok["valid_annual"].astype(float).idxmax())


def a_name(window: int, low: float) -> str:
    return f"A_{window}日_跌破后{int(round(low * 100))}%"
