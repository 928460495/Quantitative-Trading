"""单因子检验：IC、t 值、IC 衰减、分层、换手率、分年度稳健性、成本后收益。

约定：
- 主检验指标是 5 天持有期的 RankIC（对异常值不敏感）。
- 日频 IC 的标签互相重叠（5 天持有期，相邻两天有 4 天相同），所以 t 值用 Newey-West
  调整，滞后阶数取 持有期 - 1。
- 分层和换手率只在调仓日上算（每 5 个交易日一次），各期之间不重叠。
- 方向：先按 expected_sign 把因子统一成"越大越好"，再算分层、换手和成本后收益。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qt.costs import round_trip_cost

MIN_STOCKS = 30
PERIODS_PER_YEAR = 252 / 5


def ic_series(factor: pd.Series, ret: pd.Series, method: str = "spearman") -> pd.Series:
    """逐日横截面相关系数。当天有效样本少于 MIN_STOCKS 时记为 NaN。"""
    df = pd.DataFrame({"f": factor, "r": ret}).dropna()
    if method == "spearman":
        df = df.groupby(level="datetime").rank()
    elif method != "pearson":
        raise ValueError(f"未知的方法: {method}")

    def corr(g: pd.DataFrame) -> float:
        return g["f"].corr(g["r"]) if len(g) >= MIN_STOCKS else np.nan

    return df.groupby(level="datetime").apply(corr).rename(method)


def newey_west_t(x: pd.Series, lag: int) -> float:
    """均值的 Newey-West t 值（Bartlett 权重）。"""
    x = x.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 2:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(s / n) if s > 0 else np.nan


def ic_stats(ic: pd.Series, horizon: int) -> dict[str, float]:
    ic = ic.dropna()
    std = ic.std()
    return {
        "mean": ic.mean(),
        "std": std,
        "icir": ic.mean() / std if std > 0 else np.nan,
        "positive_ratio": (ic > 0).mean(),
        "t_nw": newey_west_t(ic, lag=horizon - 1),
        "n_days": len(ic),
    }


def rebalance_dates(index: pd.Index, every: int = 5) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(index.get_level_values("datetime").unique().sort_values())
    return dates[::every]


def _groups(x: pd.Series, n_groups: int) -> pd.Series:
    """把一天的横截面分成 n_groups 组，1 组最小。先排名再分组，避免并列值导致分组大小不均。"""
    return pd.qcut(x.rank(method="first"), n_groups, labels=range(1, n_groups + 1)).astype(int)


def quantile_excess_returns(score: pd.Series, ret: pd.Series, dates: pd.DatetimeIndex, n_groups: int = 5) -> pd.DataFrame:
    """每个调仓日按 score 分 n_groups 组（第 n_groups 组 score 最大），返回各组持有期收益减去全体等权均值。

    结果额外带一列 universe：当期全体等权平均收益。
    """
    rows = {}
    for d in dates:
        r = ret.xs(d, level="datetime").dropna()
        s = score.xs(d, level="datetime").reindex(r.index).dropna()
        if len(s) < MIN_STOCKS:
            continue
        g = _groups(s, n_groups)
        r = r.reindex(g.index)
        row = (r.groupby(g).mean() - r.mean()).to_dict()
        row["universe"] = r.mean()
        rows[d] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def top_group_turnover(score: pd.Series, dates: pd.DatetimeIndex, n_groups: int = 5) -> pd.Series:
    """相邻两个调仓日之间，最高组里被换掉的股票占比。第一期没有上一期，记为 1（全部新买）。"""
    prev: set[str] | None = None
    out = {}
    for d in dates:
        s = score.xs(d, level="datetime").dropna()
        if len(s) < MIN_STOCKS:
            continue
        g = _groups(s, n_groups)
        top = set(g[g == n_groups].index)
        out[d] = 1.0 if prev is None else 1 - len(top & prev) / len(top)
        prev = top
    return pd.Series(out).sort_index()


@dataclass
class FactorResult:
    name: str
    direction: int
    direction_source: str
    rank_ic: dict[int, dict[str, float]]
    pearson_ic_main: dict[str, float]
    rank_ic_daily: pd.Series
    yearly_rank_ic: pd.Series
    reversed_year_ratio: float
    quantiles: pd.DataFrame
    monotonicity: float
    turnover: pd.Series
    long_excess: pd.Series
    long_excess_net: pd.Series
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def evaluate_factor(
    name: str,
    raw: pd.Series,
    standardized: pd.Series,
    returns: dict[int, pd.Series],
    expected_sign: int | None,
    main_horizon: int = 5,
    n_groups: int = 5,
    t_threshold: float = 3.0,
) -> FactorResult:
    """returns：持有期 → 远期收益。raw 用来算 RankIC 和分层，standardized 用来算 Pearson IC。"""
    rank_ic = {h: ic_series(raw, r, "spearman") for h, r in returns.items()}
    main = rank_ic[main_horizon]
    if expected_sign is not None:
        direction, source = expected_sign, "文献"
    else:
        direction, source = (1 if main.mean() >= 0 else -1), "训练段实际 RankIC"

    rank_ic_stats = {h: ic_stats(ic * direction, h) for h, ic in rank_ic.items()}
    pearson = ic_stats(ic_series(standardized, returns[main_horizon], "pearson") * direction, main_horizon)

    yearly = main.groupby(main.index.year).mean() * direction
    reversed_ratio = float((yearly < 0).mean())

    score = raw * direction
    dates = rebalance_dates(raw.dropna().index, every=main_horizon)
    q = quantile_excess_returns(score, returns[main_horizon], dates, n_groups)
    group_cols = list(range(1, n_groups + 1))
    mono = pd.Series(q[group_cols].mean().values).corr(pd.Series(group_cols, dtype=float), method="spearman")

    turnover = top_group_turnover(score, dates, n_groups).reindex(q.index).fillna(1.0)
    long_excess = q[n_groups]
    cost = pd.Series([round_trip_cost(d) for d in q.index], index=q.index)
    long_excess_net = long_excess - turnover * cost

    checks = {
        f"t 值 > {t_threshold:g}": bool(rank_ic_stats[main_horizon]["t_nw"] > t_threshold),
        "方向反了的年份不超过一半": reversed_ratio <= 0.5,
        "扣成本后做多一侧超额 > 0": bool(long_excess_net.mean() > 0),
    }
    return FactorResult(
        name=name,
        direction=direction,
        direction_source=source,
        rank_ic=rank_ic_stats,
        pearson_ic_main=pearson,
        rank_ic_daily=main * direction,
        yearly_rank_ic=yearly,
        reversed_year_ratio=reversed_ratio,
        quantiles=q,
        monotonicity=float(mono),
        turnover=turnover,
        long_excess=long_excess,
        long_excess_net=long_excess_net,
        checks=checks,
    )
