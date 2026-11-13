"""计划 3d：四组因子等权、参数事先定死（docs/DECISIONS.md 2026-11-12「计划 3d 事先登记的规则」）。

和计划 4、3c 的区别：不让数据挑因子、挑参数。每一项按事先的理由定死；VARIANTS 里的单项改动只用来检查
（CHECKS 里的要高于基准）和报告，结果再好也不换主方案。

seg 是 scripts/build_composite.load_segment 读回来的一段（factors 是已中性化、调好方向的因子值），
另加两张"日期 × 股票"的布尔表：seg["is_st"]、seg["loss_small"]（scripts/build_plan3d_flags.py）。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qt.account import TRADING_DAYS, AccountResult, AccountRules, MarketData, run_account
from qt.composite import neutral_composite
from qt.diagnostics import BENCH, annualize, holdable_benchmark, max_drawdown
from qt.wide_eval import _market_data

GROUPS: dict[str, tuple[str, ...]] = {
    "便宜": ("ep_deduct", "bp", "sp_ttm"),
    "跌得多": ("ret60", "ret120", "high52w", "lit_high52w", "a158_MA60", "a158_QTLU60", "a158_ROC60", "a158_MAX60"),
    "冷门": ("turn_12m", "gtja100", "gtja081", "gtja132"),
    "业绩超预期": ("sue",),
}
VOL_FACTOR = "vol60"            # 缓存里的值已中性化、调好方向：越大波动越低
REVENUE_FLOOR = 3e8             # 现行退市规则：主板亏损且营业收入低于 3 亿元 → *ST
BENCH_EVERY = 20                # "能持有等权"每 20 个交易日调一次，所有版本都和它比
MAX_STOCK_WEIGHT, MAX_INDUSTRY_WEIGHT, MIN_TRADE, COOLDOWN_DAYS = 0.20, 0.30, 2000.0, 20


@dataclass(frozen=True)
class Spec:
    groups: tuple[str, ...] = tuple(GROUPS)
    n_hold: int = 15
    rebalance_every: int = 20
    sell_mult: int = 3
    stop_loss: float | None = None
    vol_cut: float | None = 0.20        # 波动最高的这一比例不买；None = 不剔除
    price_floor: float | None = 2.0     # 未复权收盘价低于它：不买，已持有的调仓日卖出；None = 不设
    drop_loss_small: bool = True        # 亏损且营收不足 3 亿元：同上
    cost_multiplier: float = 1.0
    offset: int = 0                     # 从第几个交易日开始跑（错开调仓起点）


MAIN = Spec()
_with = lambda **kw: dataclasses.replace(MAIN, **kw)
VARIANTS: dict[str, Spec] = {
    "主方案": MAIN,
    **{f"错开 {k} 天": _with(offset=k) for k in (5, 10, 15)},
    "成本翻倍": _with(cost_multiplier=2.0),
    **{f"去掉{g}": _with(groups=tuple(x for x in GROUPS if x != g)) for g in GROUPS},
    "持股 10 只": _with(n_hold=10),
    "持股 20 只": _with(n_hold=20),
    "每 10 天调仓": _with(rebalance_every=10),
    "每 40 天调仓": _with(rebalance_every=40),
    "跌出前 30 名卖": _with(sell_mult=2),
    "跌出前 60 名卖": _with(sell_mult=4),
    "加回止损": _with(stop_loss=0.20),
    "不剔除高波动": _with(vol_cut=None),
    "不排除低价和亏损小营收": _with(price_floor=None, drop_loss_small=False),
}
# 检查名 → 这些版本的年化都要高于同区间的"能持有等权"（第 1 条另外要高于沪深300 全收益）
CHECKS: dict[str, tuple[str, ...]] = {
    "1 主方案高于两个基准": ("主方案",),
    "2 错开调仓起点都高于能持有等权": ("主方案", "错开 5 天", "错开 10 天", "错开 15 天"),
    "3 成本翻倍仍高于能持有等权": ("成本翻倍",),
    "4 去掉任何一组仍高于能持有等权": tuple(f"去掉{g}" for g in GROUPS),
}
OFFSETS = CHECKS["2 错开调仓起点都高于能持有等权"]


# 主板"亏损且营收低于门槛"这条退市指标，按当时已经公开的规则（复查 R001 的 M1：不能把 2024 年的 3 亿元用到更早）。
# 2020-12-14 退市新规征求意见稿公开（营收 1 亿元）；2024-04-12 新规征求意见稿公开（主板 3 亿元）；之前没有这条指标。
REVENUE_FLOOR_HISTORY = ((pd.Timestamp("2020-12-14"), 1e8), (pd.Timestamp("2024-04-12"), 3e8))


def revenue_floor(dates: pd.DatetimeIndex) -> pd.Series:
    """每个交易日有效的营收门槛（元）；还没有这条规则时为 0，即永远不触发。"""
    dates = pd.DatetimeIndex(dates)
    out = pd.Series(0.0, index=dates)
    for start, floor in REVENUE_FLOOR_HISTORY:
        out[dates >= start] = floor
    return out


def loss_small_revenue(net_profit_ttm: pd.DataFrame, deduct_ttm: pd.DataFrame, revenue_ttm: pd.DataFrame,
                       floor: float | pd.Series = REVENUE_FLOOR) -> pd.DataFrame:
    """亏损（净利润、扣非净利润的滚动四季值有一个为负）且滚动四季营收低于 floor。数据缺失不算。

    floor 可以是一个数，也可以是按日期的序列（revenue_floor 的结果，按行对齐）。
    """
    loss = (net_profit_ttm < 0) | (deduct_ttm < 0)
    small = revenue_ttm.lt(floor.reindex(revenue_ttm.index), axis=0) if isinstance(floor, pd.Series) else revenue_ttm < floor
    return (loss & small).fillna(False).astype(bool)


def high_vol(vol_score: pd.DataFrame, cut: float) -> pd.DataFrame:
    """每天 vol_score 最低（= 波动最高）的 cut 比例。没有值的不算。"""
    return (vol_score.rank(axis=1, pct=True) <= cut).fillna(False).astype(bool)


def composite_score(seg: dict, groups: tuple[str, ...] = tuple(GROUPS)) -> pd.DataFrame:
    """组内等权 → 每组标准化 → 组间等权 → 对申万一级行业和对数流通市值中性化 → 标准化。"""
    group_of = {f: g for g in groups for f in GROUPS[g]}
    return neutral_composite({f: seg["factors"][f] for f in group_of}, group_of, seg["industry"], seg["log_mcap"])


def _flag(seg: dict, key: str, like: pd.DataFrame) -> pd.DataFrame:
    return seg[key].reindex(index=like.index, columns=like.columns).fillna(False).astype(bool)


def market(seg: dict, spec: Spec, cache: dict | None = None) -> MarketData:
    """主方案的打分和能买表：排除名单（ST、低价、亏损小营收）没有分数也不能买，高波动只是不能买。

    cache：同一个 seg 上重复调用时复用合成分（按 groups 存）。
    """
    cache = {} if cache is None else cache
    if spec.groups not in cache:
        cache[spec.groups] = composite_score(seg, spec.groups)
    score = cache[spec.groups]
    drop = _flag(seg, "is_st", score)
    if spec.drop_loss_small:
        drop = drop | _flag(seg, "loss_small", score)
    if spec.price_floor is not None:
        drop = drop | (seg["raw_close"].reindex(index=score.index, columns=score.columns) < spec.price_floor)
    no_buy = drop
    if spec.vol_cut is not None:
        vol = seg["factors"][VOL_FACTOR].reindex(index=score.index, columns=score.columns)
        no_buy = no_buy | high_vol(vol, spec.vol_cut)
    can_buy = _flag(seg, "can_buy", score) & ~no_buy
    cut = lambda x: x.iloc[spec.offset:]
    return _market_data(cut(score.mask(drop)), cut(seg["close"]), cut(seg["raw_close"]), cut(can_buy),
                        cut(seg["can_sell"]), cut(seg["industry"]))


def rules(spec: Spec) -> AccountRules:
    return AccountRules(n_hold=spec.n_hold, sell_rank=spec.sell_mult * spec.n_hold, max_stock_weight=MAX_STOCK_WEIGHT,
                        max_industry_weight=MAX_INDUSTRY_WEIGHT, stop_loss=spec.stop_loss, cooldown_days=COOLDOWN_DAYS,
                        drawdown_stop=None, drawdown_tiers=None, cost_multiplier=spec.cost_multiplier,
                        rebalance_every=spec.rebalance_every, min_trade=MIN_TRADE)


def run(seg: dict, spec: Spec, cache: dict | None = None) -> AccountResult:
    return run_account(market(seg, spec, cache), rules(spec))


def benchmarks(seg: dict, nav: pd.Series) -> dict[str, float]:
    """和 nav 同一段日期的两个基准年化。基准用原来的股票池，不受策略自己的排除名单影响。"""
    close = seg["close"].loc[nav.index[0]:nav.index[-1]]
    hold = (1 + holdable_benchmark(close, seg["can_buy"], BENCH_EVERY)).cumprod()
    hs300 = seg["index"][BENCH].reindex(nav.index).ffill().bfill()
    return {"能持有等权": annualize(hold), "沪深300全收益": annualize(hs300)}


Logger = Callable[[str, dict, dict], None] | None


def evaluate(seg: dict, variants: Mapping[str, Spec] = VARIANTS, log: Logger = None, tag: str = "train"
             ) -> tuple[pd.DataFrame, dict[str, AccountResult]]:
    """每个版本跑一次真实账户。返回（每个版本一行的表，各版本的回测结果）。"""
    cache: dict = {}
    rows, results = {}, {}
    for name, spec in variants.items():
        res = run(seg, spec, cache)
        nav, trades = res.daily["nav"], res.trades
        bench = benchmarks(seg, nav)
        sold = trades[trades["side"] == "sell"].groupby("date")["value"].sum().reindex(nav.index, fill_value=0.0)
        daily_ret = nav.pct_change().dropna()
        sd = daily_ret.std()
        rows[name] = {
            "年化": annualize(nav), **bench, "超额": annualize(nav) - bench["能持有等权"],
            "最大回撤": max_drawdown(nav), "日均换手": float((sold / nav).iloc[1:].mean()),
            "年成本": float((res.daily["cost"] / nav).iloc[1:].mean() * TRADING_DAYS), "交易笔数": float(len(trades)),
        }
        results[name] = res
        if log:
            log("plan3d", {"variant": name, "segment": tag, **dataclasses.asdict(spec)},
                {"annual_return": rows[name]["年化"], "max_drawdown": rows[name]["最大回撤"],
                 "sharpe": float(daily_ret.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")})
    return pd.DataFrame(rows).T, results


def check_table(table: pd.DataFrame, checks: Mapping[str, tuple[str, ...]] = CHECKS) -> pd.DataFrame:
    """每条检查一行：成员里超额最小的版本、它的超额、是否满足（每个成员都要高于基准，不是平均）。"""
    rows = {}
    first = next(iter(checks))
    for name, members in checks.items():
        excess = table.loc[list(members), "超额"].astype(float)
        ok = bool((excess > 0).all())
        if name == first:
            ok = ok and bool((table.loc[list(members), "年化"] > table.loc[list(members), "沪深300全收益"]).all())
        rows[name] = {"最差的版本": excess.idxmin(), "最小超额": float(excess.min()), "满足": ok}
    return pd.DataFrame(rows).T.astype({"最小超额": float, "满足": bool})
