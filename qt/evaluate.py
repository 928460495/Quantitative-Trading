"""单因子检验：IC、回归法、IC 衰减、分层、分年度稳健性、按实际交易规则模拟扣成本后的收益。

约定：
- 主检验指标是 5 天持有期的 RankIC（对异常值不敏感）。
- 日频 IC 和回归系数的标签互相重叠（5 天持有期，相邻两天有 4 天相同），所以 t 值用 Newey-West
  调整，滞后阶数取 持有期 - 1。
- 分层每 5 个交易日一次、各期不重叠，只用来看信号质量和单调性。
- 成本门槛用 qt/simulate.py 的日频缓冲组合（持仓 30、缓冲 30/90），超额相对第二天能买进的股票；
  相对全部有打分股票的超额作参考。
- 方向：先按方向把因子统一成"越大越好"，再算分层和组合。
- IC、回归、分层只用能买入的样本；组合模拟另外处理买不进、卖不掉。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qt.simulate import COMPARISON_RULES, PRIMARY_RULE, BufferRule, simulate_buffer, turnover

MIN_STOCKS = 30
TRADING_DAYS = 252

SCORING = "打分因子"
VETO = "排雷因子"
REJECTED = "淘汰"


def _demeaned(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    g = df.groupby(level="datetime")
    return df["f"] - g["f"].transform("mean"), df["r"] - g["r"].transform("mean"), g.size()


def ic_series(factor: pd.Series, ret: pd.Series, method: str = "spearman") -> pd.Series:
    """逐日横截面相关系数。当天有效样本少于 MIN_STOCKS 时记为 NaN。"""
    df = pd.DataFrame({"f": factor, "r": ret}).dropna()
    if method == "spearman":
        df = df.groupby(level="datetime").rank()
    elif method != "pearson":
        raise ValueError(f"未知的方法: {method}")
    d, e, n = _demeaned(df)
    by = d.index.get_level_values("datetime")
    num = (d * e).groupby(by).sum()
    den = np.sqrt((d * d).groupby(by).sum() * (e * e).groupby(by).sum())
    ic = (num / den).replace([np.inf, -np.inf], np.nan).where(n >= MIN_STOCKS)
    ic.index.name = "datetime"
    return ic.rename(method)


def factor_return_series(z: pd.Series, ret: pd.Series) -> pd.Series:
    """回归法：每天把远期收益对标准化后的因子做横截面回归（带截距），返回斜率，即因子收益率。"""
    d, e, n = _demeaned(pd.DataFrame({"f": z, "r": ret}).dropna())
    by = d.index.get_level_values("datetime")
    slope = ((d * e).groupby(by).sum() / (d * d).groupby(by).sum()).replace([np.inf, -np.inf], np.nan)
    slope.index.name = "datetime"
    return slope.where(n >= MIN_STOCKS).rename("factor_return")


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


def simulation_summary(sim: pd.DataFrame, rule: BufferRule) -> dict[str, float]:
    return {
        "excess_ann": sim["excess"].mean() * TRADING_DAYS,
        "cost_ann": sim["cost"].mean() * TRADING_DAYS,
        "net_ann": sim["net"].mean() * TRADING_DAYS,
        "net_t": newey_west_t(sim["net"], lag=0),
        "excess_ann_all": sim["excess_all"].mean() * TRADING_DAYS,
        "net_ann_all": sim["net_all"].mean() * TRADING_DAYS,
        "turnover": turnover(sim, rule.n_hold).mean(),
    }


@dataclass
class FactorResult:
    name: str
    direction: int
    direction_source: str
    main_horizon: int
    rank_ic: dict[int, dict[str, float]]
    pearson_ic_main: dict[str, float]
    factor_return_main: dict[str, float]
    rank_ic_daily: pd.Series
    yearly_rank_ic: pd.Series
    reversed_year_ratio: float
    quantiles: pd.DataFrame
    periods_per_year: float
    monotonicity: float
    bottom_excess_ann: float
    bottom_t: float
    simulations: dict[str, pd.DataFrame]
    sim_summary: dict[str, dict[str, float]]
    checks: dict[str, bool]
    veto_checks: dict[str, bool]
    category: str

    @property
    def passed(self) -> bool:
        return self.category == SCORING


def evaluate_factor(
    name: str,
    raw: pd.Series,
    standardized: pd.Series,
    returns: dict[int, pd.Series],
    can_buy: pd.Series,
    can_sell: pd.Series,
    expected_sign: int | None,
    main_horizon: int = 5,
    n_groups: int = 5,
    t_threshold: float = 3.0,
    rules: tuple[BufferRule, ...] = (PRIMARY_RULE, *COMPARISON_RULES),
) -> FactorResult:
    """returns：持有期 → 远期收益，必须包含 1 天（组合模拟用）和 main_horizon。

    raw 用来算 RankIC、分层和组合排序；standardized 用来算 Pearson IC 和回归法。
    IC、回归、分层只用 can_buy 为 True 的样本。
    """
    raw_ic = raw.where(can_buy)
    rank_ic = {h: ic_series(raw_ic, r, "spearman") for h, r in returns.items()}
    main = rank_ic[main_horizon]
    if expected_sign is not None:
        direction, source = expected_sign, "文献"
    else:
        direction, source = (1 if main.mean() >= 0 else -1), "训练段实际 RankIC"

    rank_ic_stats = {h: ic_stats(ic * direction, h) for h, ic in rank_ic.items()}
    z = standardized.where(can_buy)
    pearson = ic_stats(ic_series(z, returns[main_horizon], "pearson") * direction, main_horizon)
    fr = factor_return_series(z * direction, returns[main_horizon])
    fr_stats = {"mean": fr.mean(), "t_nw": newey_west_t(fr, lag=main_horizon - 1)}

    yearly = main.groupby(main.index.year).mean() * direction
    reversed_ratio = float((yearly < 0).mean())

    dates = rebalance_dates(raw_ic.dropna().index, every=main_horizon)
    q = quantile_excess_returns(raw_ic * direction, returns[main_horizon], dates, n_groups)
    groups = list(range(1, n_groups + 1))
    mono = pd.Series(q[groups].mean().values).corr(pd.Series(groups, dtype=float), method="spearman")
    periods_per_year = TRADING_DAYS / main_horizon
    bottom_ann = q[1].mean() * periods_per_year
    bottom_t = newey_west_t(q[1], lag=0)

    sims = {r.label: simulate_buffer(raw * direction, returns[1], can_buy, can_sell, r) for r in rules}
    sim_summary = {r.label: simulation_summary(sims[r.label], r) for r in rules}
    primary = sim_summary[rules[0].label]

    t_ok = bool(rank_ic_stats[main_horizon]["t_nw"] > t_threshold)
    stable = reversed_ratio <= 0.5
    checks = {
        f"t 值 > {t_threshold:g}": t_ok,
        "方向反了的年份不超过一半": stable,
        f"按实际交易规则（{rules[0].label}）扣成本后做多超额 > 0": bool(primary["net_ann"] > 0),
    }
    veto_checks = {
        f"t 值 > {t_threshold:g}": t_ok,
        "方向反了的年份不超过一半": stable,
        "最低组年化超额 < 0": bool(bottom_ann < 0),
        f"最低组超额 t 值 < -{t_threshold:g}": bool(bottom_t < -t_threshold),
    }
    if all(checks.values()):
        category = SCORING
    elif all(veto_checks.values()):
        category = VETO
    else:
        category = REJECTED

    return FactorResult(
        name=name,
        direction=direction,
        direction_source=source,
        main_horizon=main_horizon,
        rank_ic=rank_ic_stats,
        pearson_ic_main=pearson,
        factor_return_main=fr_stats,
        rank_ic_daily=main * direction,
        yearly_rank_ic=yearly,
        reversed_year_ratio=reversed_ratio,
        quantiles=q,
        periods_per_year=periods_per_year,
        monotonicity=float(mono),
        bottom_excess_ann=float(bottom_ann),
        bottom_t=float(bottom_t),
        simulations=sims,
        sim_summary=sim_summary,
        checks=checks,
        veto_checks=veto_checks,
        category=category,
    )
