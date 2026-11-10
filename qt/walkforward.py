"""计划 3c 任务 9：训练段滚动前推（docs/DECISIONS.md 2026-11-10「计划 3c 事先登记的规则」）。

- 逐步加因子：参考组合下，从空集开始，每步加入使快速模拟器扣成本后夏普最高的因子；提高不到 0.05 停止
- 组合网格：选中因子下 216 个组合，挑夏普最高的
- 前推：每年只用 2010-01-04 到上一年末的数据做上面两步，跑下一年；每年结果存 json，可断点续跑

seg 是 build_composite.load_segment 读回来的一段（factors 是已中性化、调好方向的因子值；另有 log_mcap）。
pool 是 index = 因子名、带 category 列的表（scripts/build_pool.read_pool）。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from itertools import product
from pathlib import Path

import pandas as pd

from qt.account import AccountRules
from qt.composite import neutral_composite
from qt.drawdown import DEFAULT_TIERS
from qt.fastsim import SimParams, sharpe, simulate
from qt.wide_eval import _market_data

TRAIN_START = "2010-01-04"
MIN_GAIN = 0.05
REFERENCE = SimParams(n_hold=20, rebalance_every=5, sell_mult=3, industry_cap=0.30)
GRID: list[tuple[SimParams, int, float]] = [
    (SimParams(n_hold=n, rebalance_every=e, sell_mult=m, industry_cap=c), k, w)
    for n, e, k, m, c, w in product((15, 20), (1, 5, 10), (1, 5, 10), (3, 4, 6), (0.30, 0.20), (1.0, 2.0))
]

Logger = Callable[[str, dict, dict], None] | None
Setup = tuple[SimParams, int, float]


def label(p: SimParams, smooth: int, lowvol_weight: float) -> str:
    return f"N{p.n_hold}_e{p.rebalance_every}_k{smooth}_m{p.sell_mult}_ind{p.industry_cap:.2f}_lv{lowvol_weight:g}"


def to_dict(setup: Setup) -> dict:
    p, k, w = setup
    return {"n_hold": p.n_hold, "rebalance_every": p.rebalance_every, "sell_mult": p.sell_mult,
            "industry_cap": p.industry_cap, "smooth": k, "lowvol_weight": w}


def from_dict(d: dict) -> Setup:
    p = SimParams(n_hold=d["n_hold"], rebalance_every=d["rebalance_every"], sell_mult=d["sell_mult"],
                  industry_cap=d["industry_cap"])
    return p, int(d["smooth"]), float(d["lowvol_weight"])


def account_rules(p: SimParams) -> AccountRules:
    """快速模拟器参数对应的真实账户规则（行业上限在账户里按金额、在快速模拟器里按只数）。"""
    return AccountRules(n_hold=p.n_hold, sell_rank=p.sell_mult * p.n_hold, max_industry_weight=p.industry_cap,
                        rebalance_every=p.rebalance_every, min_trade=p.min_trade, drawdown_stop=None,
                        drawdown_tiers=DEFAULT_TIERS if p.tiers else None, cost_multiplier=p.cost_multiplier)


def _cut(frame: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
    return frame.loc[window[0]:window[1]]


def score_for(seg: dict, pool: pd.DataFrame, factors: Sequence[str], window: tuple[str, str], smooth: int = 1,
              lowvol_weight: float = 1.0) -> pd.DataFrame:
    """窗口内的合成分（先截窗口再算：中性化、平滑都只用窗口内、当天及以前的数据）。"""
    fs = {f: _cut(seg["factors"][f], window) for f in factors}
    return neutral_composite(fs, pool["category"].to_dict(), _cut(seg["industry"], window),
                             _cut(seg["log_mcap"], window), lowvol_weight=lowvol_weight, smooth=smooth)


def market(seg: dict, score: pd.DataFrame, window: tuple[str, str]):
    return _market_data(score, _cut(seg["close"], window), _cut(seg["raw_close"], window), _cut(seg["can_buy"], window),
                        _cut(seg["can_sell"], window), _cut(seg["industry"], window))


def run_setup(seg: dict, pool: pd.DataFrame, factors: Sequence[str], window: tuple[str, str], setup: Setup) -> pd.Series:
    p, k, w = setup
    return simulate(market(seg, score_for(seg, pool, factors, window, k, w), window), p)


def forward_select(seg: dict, pool: pd.DataFrame, window: tuple[str, str], min_gain: float = MIN_GAIN,
                   reference: SimParams = REFERENCE, log: Logger = None) -> list[str]:
    chosen: list[str] = []
    best = float("-inf")
    while len(chosen) < len(pool):
        scores = {}
        for cand in (f for f in pool.index if f not in chosen):
            trial = chosen + [cand]
            scores[cand] = sharpe(run_setup(seg, pool, trial, window, (reference, 1, 1.0)))
            if log:
                log("plan3c_step", {"window": list(window), "factors": sorted(trial)}, {"sharpe": scores[cand]})
        cand, value = max(scores.items(), key=lambda kv: (kv[1] if kv[1] == kv[1] else float("-inf")))
        if chosen and value - best < min_gain:
            break
        chosen.append(cand)
        best = value
    return chosen


def grid_select(seg: dict, pool: pd.DataFrame, factors: Sequence[str], window: tuple[str, str],
                grid: Sequence[Setup] = GRID, log: Logger = None) -> tuple[pd.DataFrame, Setup]:
    """网格里每个组合在窗口内的日收益（列名 = label）和夏普最高的组合。同样的平滑、低波动权重只算一次合成分。"""
    scores: dict[tuple[int, float], pd.DataFrame] = {}
    cols = {}
    for setup in grid:
        p, k, w = setup
        if (k, w) not in scores:
            scores[(k, w)] = score_for(seg, pool, factors, window, k, w)
        r = simulate(market(seg, scores[(k, w)], window), p)
        cols[label(*setup)] = r
        if log:
            log("plan3c_grid", {"window": list(window), "factors": sorted(factors), **to_dict(setup)}, {"sharpe": sharpe(r)})
    rets = pd.DataFrame(cols)
    best = max(grid, key=lambda s: sharpe(rets[label(*s)]))
    return rets, best


def walk_forward(seg: dict, pool: pd.DataFrame, years: Sequence[int] = range(2013, 2019), cache_dir: Path | None = None,
                 grid: Sequence[Setup] = GRID, reference: SimParams = REFERENCE, log: Logger = None) -> dict:
    """返回 {"years": {年: {"factors", "config"}}, "returns": 拼起来的前推日收益}。

    每年的选择只用 TRAIN_START 到上一年末；那一年的收益从上一年初开始跑（让持仓和回撤分档预热一年），只取那一年。
    """
    out_years, pieces = {}, []
    for year in years:
        window = (TRAIN_START, f"{year - 1}-12-31")
        path = None if cache_dir is None else Path(cache_dir) / f"wf_{year}.json"
        if path is not None and path.exists():
            choice = json.loads(path.read_text())
        else:
            factors = forward_select(seg, pool, window, reference=reference, log=log)
            _, best = grid_select(seg, pool, factors, window, grid=grid, log=log)
            choice = {"factors": factors, "config": to_dict(best)}
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(choice, ensure_ascii=False, indent=2))
        out_years[year] = choice
        run = (f"{year - 1}-01-01", f"{year}-12-31")
        r = run_setup(seg, pool, choice["factors"], run, from_dict(choice["config"]))
        pieces.append(r.loc[f"{year}-01-01":])
    return {"years": out_years, "returns": pd.concat(pieces)}
