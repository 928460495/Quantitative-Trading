"""流程：从因子族到每天持仓的每一步（docs/FRAMEWORK.md 第 3 节）。

目前流程里没有用数据来定的步骤（因子族、方向、权重都事先写死）。以后要加"用数据定"的步骤（比如选因子），
必须做成逐年只用过去数据重做的接口，不能直接把全段定出来的结果写进来（失效模式 F3）。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pandas as pd

from qt.account import AccountResult, AccountRules, MarketData, run_account
from qt.composite import neutral_composite
from qt.plan3d import GROUPS
from qt.wide_eval import _market_data

MAX_STOCK_WEIGHT, MAX_INDUSTRY_WEIGHT, MIN_TRADE, COOLDOWN_DAYS = 0.20, 0.30, 2000.0, 20


@dataclass(frozen=True)
class Procedure:
    families: tuple[tuple[str, tuple[str, ...]], ...]            # (族名, 因子们)；族内等权、族间等权
    n_hold: int = 15
    rebalance_every: int = 20
    sell_mult: int = 3                                           # 调仓日跌出前 sell_mult × n_hold 名才卖
    exclude: tuple[str, ...] = ("is_st", "loss_small", "cheap")  # 命中：不买，持有的要卖
    no_buy: tuple[str, ...] = ("high_vol",)                      # 命中：只是不买
    exit_next_day: bool = False                                  # exclude 命中的持仓：False 下个调仓日卖，True 第二个交易日卖
    stop_loss: float | None = None
    cost_multiplier: float = 1.0

    def __post_init__(self):
        # 规范写法：族按名字排、排除名单去重排序。这样写进文件再读回来、或者字典顺序不同，都还是同一个流程
        object.__setattr__(self, "families", tuple(sorted((n, tuple(fs)) for n, fs in self.families)))
        object.__setattr__(self, "exclude", tuple(sorted(set(self.exclude))))
        object.__setattr__(self, "no_buy", tuple(sorted(set(self.no_buy))))

    def factors(self) -> tuple[str, ...]:
        return tuple(f for _, fs in self.families for f in fs)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["families"] = {name: list(fs) for name, fs in self.families}
        d["exclude"], d["no_buy"] = list(self.exclude), list(self.no_buy)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Procedure":
        known = {f.name for f in dataclasses.fields(cls)}
        if set(d) - known:
            raise ValueError(f"流程里没有这些字段：{sorted(set(d) - known)}")
        d = dict(d)
        if "families" in d:
            d["families"] = tuple((name, tuple(fs)) for name, fs in d["families"].items())
        for key in ("exclude", "no_buy"):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)

    def replace(self, **changes) -> "Procedure":
        return Procedure.from_dict({**self.to_dict(), **changes})


PLAN3D = Procedure(families=tuple(GROUPS.items()))               # 计划 3d 的主方案，作为第一个基线


def composite(dev: dict, proc: Procedure) -> pd.DataFrame:
    group_of = {f: name for name, fs in proc.families for f in fs}
    return neutral_composite({f: dev["factors"][f] for f in group_of}, group_of, dev["industry"], dev["log_mcap"])


def _any(dev: dict, names: tuple[str, ...], like: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(False, index=like.index, columns=like.columns)
    for name in names:
        if name not in dev["flags"]:
            raise KeyError(f"没有这个标记：{name}（有的：{sorted(dev['flags'])}）")
        out = out | dev["flags"][name].reindex(index=like.index, columns=like.columns).fillna(False).astype(bool)
    return out


def market(dev: dict, proc: Procedure, offset: int = 0, cache: dict | None = None) -> MarketData:
    """offset：从第几个交易日开始跑（错开调仓起点）。cache：同一个 dev 上复用合成分（按 families 存）。"""
    cache = {} if cache is None else cache
    if proc.families not in cache:
        cache[proc.families] = composite(dev, proc)
    score = cache[proc.families]
    drop = _any(dev, proc.exclude, score)
    can_buy = dev["can_buy"].reindex(index=score.index, columns=score.columns).fillna(False).astype(bool)
    can_buy = can_buy & ~(drop | _any(dev, proc.no_buy, score))
    cut = lambda x: x.iloc[offset:]
    data = _market_data(cut(score.mask(drop)), cut(dev["close"]), cut(dev["raw_close"]), cut(can_buy),
                        cut(dev["can_sell"]), cut(dev["industry"]))
    if proc.exit_next_day:
        force = drop.reindex(index=data.dates, columns=data.instruments).fillna(False).to_numpy(dtype=bool)
        data = dataclasses.replace(data, force_exit=force)
    return data


def rules(proc: Procedure) -> AccountRules:
    return AccountRules(n_hold=proc.n_hold, sell_rank=proc.sell_mult * proc.n_hold, max_stock_weight=MAX_STOCK_WEIGHT,
                        max_industry_weight=MAX_INDUSTRY_WEIGHT, stop_loss=proc.stop_loss, cooldown_days=COOLDOWN_DAYS,
                        drawdown_stop=None, drawdown_tiers=None, cost_multiplier=proc.cost_multiplier,
                        rebalance_every=proc.rebalance_every, min_trade=MIN_TRADE)


def run(dev: dict, proc: Procedure, offset: int = 0, cache: dict | None = None) -> AccountResult:
    return run_account(market(dev, proc, offset, cache), rules(proc))
