"""成绩单和配对比较（docs/FRAMEWORK.md 第 5 节）。一个世代内冻结。

所有数都分三个制度阶段和全段报；账户按四个错开的调仓起点各跑一次，报平均，也报最差。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qt.account import TRADING_DAYS, AccountResult
from qt.diagnostics import BENCH, holdable_benchmark
from qt.evaluate import newey_west_t
from qt.lab import procedure as pc
from qt.lab.protocol import BENCH_EVERY, BIG_LOSS, ERAS, FULL, NW_LAG, OFFSETS, RISK_FREE, era_of

ROWS = [*ERAS, FULL]
COLUMNS = ["年化", "能持有等权", "沪深300全收益", "超额", "最差起点超额", "最大回撤", "日均换手", "年成本",
           "大亏只数", "最差五笔占比",
           # 框架 2.1：夏普（扣 2% 无风险收益）是第 8 节的判定口径；其余只报告
           "夏普", "最差起点夏普差", "等权夏普", "沪深300夏普", "信息比率", "主动风险", "α的t值", "M²", "索提诺", "卡尔玛"]
CASH = 100_000


def run_phases(dev: dict, proc: pc.Procedure, offsets=OFFSETS, cache: dict | None = None) -> dict[int, AccountResult]:
    cache = {} if cache is None else cache
    return {k: pc.run(dev, proc, k, cache) for k in offsets}


def daily_returns(results: dict[int, AccountResult]) -> pd.DataFrame:
    """每个起点一列的日收益，只留所有起点都已开始的日子（开始当天记 0）。"""
    out = pd.DataFrame({k: r.daily["nav"].pct_change() for k, r in results.items()})
    start = max(r.daily.index[0] for r in results.values())
    return out.loc[start:].fillna(0.0)


def bench_returns(dev: dict, results: dict[int, AccountResult]) -> tuple[pd.DataFrame, pd.Series]:
    """（每个起点对应的能持有等权日收益，沪深300 全收益日收益），日期和 daily_returns 一样。"""
    index = daily_returns(results).index
    hold = {}
    for k, r in results.items():
        close = dev["close"].loc[r.daily.index[0]:r.daily.index[-1]]
        hold[k] = holdable_benchmark(close, dev["can_buy"], BENCH_EVERY).reindex(index).fillna(0.0)
    hs = dev["index"][BENCH].reindex(dev["close"].index).ffill().pct_change().reindex(index).fillna(0.0)
    return pd.DataFrame(hold), hs


def closed_positions(result: AccountResult) -> pd.DataFrame:
    """每一笔已平仓的持仓：买入日、卖出日、扣成本后的收益率和盈亏（元）。没卖出的不算。"""
    rows, opened = [], {}
    for t in result.trades.itertuples():
        if t.side == "buy":
            opened[t.instrument] = (t.date, t.value + t.cost)
        elif t.instrument in opened:
            buy_date, paid = opened.pop(t.instrument)
            got = t.value - t.cost
            rows.append((t.instrument, buy_date, t.date, got / paid - 1, got - paid))
    out = pd.DataFrame(rows, columns=["instrument", "buy_date", "sell_date", "ret", "pnl"])
    return out.astype({"ret": float, "pnl": float})                         # 一笔都没有时也要是数值列


def _annual(r: pd.Series) -> float:
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1) if len(r) else float("nan")


def _max_drawdown(r: pd.Series) -> float:
    nav = np.concatenate([[1.0], (1 + r.to_numpy(dtype=float)).cumprod()])     # 从这段开始前的 1 算起
    return float((nav / np.maximum.accumulate(nav) - 1).min())


def risk_metrics(r: pd.Series, ew: pd.Series, hs: pd.Series) -> dict:
    """一段日收益的风险调整指标。ew = 同期能持有等权，hs = 沪深300 全收益（同一组日期）。

    夏普 = (r − rf) 均值 ÷ r 标准差 × √252，rf 每年 RISK_FREE；信息比率、主动风险相对能持有等权；
    α 的 t 值：(r − rf) 对 (ew − rf) 回归后的残差序列的 Newey-West t；M² = (夏普 − 等权夏普) × 等权年化波动。
    """
    rf = (1 + RISK_FREE) ** (1 / TRADING_DAYS) - 1
    root = np.sqrt(TRADING_DAYS)
    sharpe = lambda x: float((x - rf).mean() / x.std() * root) if x.std() > 0 else float("nan")
    active = r.to_numpy() - ew.to_numpy()
    a_sd = active.std(ddof=1)
    ex, bx = r.to_numpy() - rf, ew.to_numpy() - rf
    beta = float(np.cov(ex, bx, ddof=1)[0, 1] / bx.var(ddof=1)) if bx.var(ddof=1) > 0 else 0.0
    down = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    dd = _max_drawdown(r)
    s, s_ew = sharpe(r), sharpe(ew)
    return {"夏普": s, "等权夏普": s_ew, "沪深300夏普": sharpe(hs),
            "信息比率": float(active.mean() / a_sd * root) if a_sd > 0 else float("nan"), "主动风险": float(a_sd * root),
            "α的t值": float(newey_west_t(pd.Series(ex - beta * bx), NW_LAG)),
            "M²": float((s - s_ew) * ew.std() * root), "索提诺": float(ex.mean() / down * root) if down > 0 else float("nan"),
            "卡尔玛": float(_annual(r) / -dd) if dd < 0 else float("nan")}


def _slices(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    era = era_of(index)
    return {**{name: era == name for name in ERAS}, FULL: np.ones(len(index), dtype=bool)}


def scorecard(dev: dict, results: dict[int, AccountResult]) -> pd.DataFrame:
    rets = daily_returns(results)
    hold, hs = bench_returns(dev, results)
    positions = {k: closed_positions(r) for k, r in results.items()}
    rows = {}
    for name, mask in _slices(rets.index).items():
        dates = rets.index[mask]
        if not len(dates):
            continue
        ann = pd.Series({k: _annual(rets.loc[dates, k]) for k in rets})
        excess = ann - pd.Series({k: _annual(hold.loc[dates, k]) for k in rets})
        turn, cost, worst5, big = [], [], [], 0
        for k, r in results.items():
            d = r.daily.loc[dates[0]:dates[-1]]
            sold = r.trades[r.trades["side"] == "sell"].groupby("date")["value"].sum().reindex(d.index, fill_value=0.0)
            turn.append(float((sold / d["nav"]).mean()))
            cost.append(float((d["cost"] / d["nav"]).mean() * TRADING_DAYS))
            p = positions[k]
            p = p[(p["sell_date"] >= dates[0]) & (p["sell_date"] <= dates[-1])]
            big += int((p["ret"] <= BIG_LOSS).sum())
            worst5.append(float(p["pnl"].nsmallest(5).clip(upper=0).sum() / CASH))
        risk = pd.DataFrame({k: risk_metrics(rets.loc[dates, k], hold.loc[dates, k], hs.loc[dates]) for k in rets}).T
        rows[name] = {
            **risk.mean().to_dict(), "最差起点夏普差": float((risk["夏普"] - risk["等权夏普"]).min()),
            "年化": float(ann.mean()), "能持有等权": float((ann - excess).mean()), "沪深300全收益": _annual(hs.loc[dates]),
            "超额": float(excess.mean()), "最差起点超额": float(excess.min()),
            "最大回撤": float(np.mean([_max_drawdown(rets.loc[dates, k]) for k in rets])),
            "日均换手": float(np.mean(turn)), "年成本": float(np.mean(cost)), "大亏只数": float(big),
            "最差五笔占比": float(np.mean(worst5)),
        }
    return pd.DataFrame(rows).T[COLUMNS]


def paired(cand: dict[int, AccountResult], base: dict[int, AccountResult]) -> dict:
    """候选减基线的每日收益差（各起点平均）：分阶段和全段的年化差值，全段的 Newey-West t 值。"""
    a, b = daily_returns(cand), daily_returns(base)
    if list(a.columns) != list(b.columns):
        raise ValueError("候选和基线的调仓起点不一样，不能配对")
    index = a.index.intersection(b.index)
    d = (a.loc[index] - b.loc[index]).mean(axis=1)
    diff = pd.Series({name: float(d[mask].mean() * TRADING_DAYS) for name, mask in _slices(index).items() if mask.any()})
    return {"差值": diff, "t": float(newey_west_t(d, NW_LAG)), "daily": d}
