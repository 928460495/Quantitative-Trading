"""全 A 因子的类内筛选（docs/DECISIONS.md 2026-11-01）。

每个大类里，通过检验的因子按 5 天 ICIR 从高到低（并列按名字）逐个看：和已留下的因子平均相关性的绝对值
> 0.7 就跳过，每类最多留 3 个；不足就有几个留几个，没有就空着。跨类不去重。

相关性 = 中性化、按方向调整后的因子值，每 5 个交易日一次的横截面 Spearman 相关（两个因子都有值的股票），
对时间取平均。所有因子用同一组抽样日（检验区间第一天起每 5 天），不按各自第一个有值的日子起算：
成长类要等 8 期财报才有值，起点不同会让抽样日错开、相关性算不出来（2026-11-09 修正）。只算贪心过程中真正要比较的那几对，不算整个相关矩阵：短周期价量一类可能有上百个候选。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from qt.wide_eval import rank_ic

MAX_PER_CATEGORY = 3
CORR_THRESHOLD = 0.7
FULL = f"已满 {MAX_PER_CATEGORY} 个，未比较"


def pair_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """两个因子逐日横截面 Spearman 相关的平均（不足 30 只的日子不算）。"""
    return float(rank_ic(a, b).mean())


def pick(table: pd.DataFrame, scores: Mapping[str, pd.DataFrame], max_n: int = MAX_PER_CATEGORY,
         threshold: float = CORR_THRESHOLD) -> pd.DataFrame:
    """table：一个大类里通过检验的因子（索引是名字，要有 icir 列）。scores：名字 → 调好方向的因子值。

    返回按考察顺序排列的表：kept、status（留下 / 和谁太像 / 已满）、dropped_by、corr。
    """
    cols = ["icir", "kept", "status", "dropped_by", "corr"]
    if table.empty:
        return pd.DataFrame(columns=cols)
    order = table.assign(_name=table.index).sort_values(["icir", "_name"], ascending=[False, True]).index
    kept: list[str] = []
    rows = []
    for name in order:
        row = {"factor": name, "icir": table.loc[name, "icir"], "kept": False, "dropped_by": None, "corr": np.nan}
        if len(kept) >= max_n:
            row["status"] = FULL
        else:
            corrs = {k: pair_corr(scores[name], scores[k]) for k in kept}
            clash = {k: r for k, r in corrs.items() if abs(r) > threshold}
            if clash:
                by = max(clash, key=lambda k: abs(clash[k]))
                row.update(status=f"和 {by} 太像", dropped_by=by, corr=clash[by])
            else:
                kept.append(name)
                row.update(kept=True, status="留下")
        rows.append(row)
    return pd.DataFrame(rows).set_index("factor")[cols]


def select_by_category(table: pd.DataFrame, scores: Mapping[str, pd.DataFrame], **kw) -> pd.DataFrame:
    """table 要有 icir、category 两列；每个大类各自 pick，结果拼在一起。"""
    parts = [pick(g, scores, **kw).assign(category=cat) for cat, g in table.groupby("category", sort=True)]
    return pd.concat(parts) if parts else pd.DataFrame(columns=["icir", "kept", "status", "dropped_by", "corr", "category"])


def corr_matrix(scores: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """选中因子之间的平均相关性矩阵（报告用，跨类不去重）。"""
    names = list(scores)
    m = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m.loc[a, b] = m.loc[b, a] = pair_corr(scores[a], scores[b])
    return m


# ---------- 单个因子的完整检验（通过条件三条，docs/DECISIONS.md 2026-11-01） ----------

import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

from qt.config import MAX_FORWARD_DAYS  # noqa: E402
from qt.wide_eval import (  # noqa: E402
    account_excess, forward_returns, ic_summary, neutral_score, quantile_excess, yearly_reversed_ratio,
)

SIZE_CATEGORY = "规模"          # 规模类只对行业中性化
T_THRESHOLD = 3.0
ACCOUNT_KEYS = ["annual_return", "bench_annual", "excess_annual", "excess_t", "max_drawdown", "turnover",
                "cost_annual", "avg_holdings", "n_trades", "n_stops"]


@dataclass
class ScreenContext:
    """检验用到的全部宽表（同样的日期和股票）。构造时算好远期收益，并确定检验用的日期：
    start 起（之前是因子预热），去掉最后 tail 个交易日（远期收益会用到段外的价格）。"""

    close: pd.DataFrame          # 复权收盘价
    raw_close: pd.DataFrame      # 未复权收盘价（按手取整用）
    can_buy: pd.DataFrame        # 信号日 T：T+1 能不能买
    can_sell: pd.DataFrame
    industry: pd.DataFrame       # 申万一级
    log_mcap: pd.DataFrame       # 对数流通市值
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    main_horizon: int = 5
    start: str | None = None
    tail: int = MAX_FORWARD_DAYS
    dates: pd.DatetimeIndex = field(init=False)
    rets: dict = field(init=False)

    def __post_init__(self):
        idx = self.close.index if self.start is None else self.close.index[self.close.index >= pd.Timestamp(self.start)]
        if len(idx) <= self.tail:
            raise ValueError(f"检验区间只有 {len(idx)} 个交易日，不够去掉 {self.tail} 天")
        self.dates = pd.DatetimeIndex(idx[:-self.tail] if self.tail else idx)   # tail=0：回测用，不需要远期收益
        self.rets = {h: r.loc[self.dates] for h, r in forward_returns(self.close, list(self.horizons)).items()}

    def trim(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.reindex(index=self.dates, columns=self.close.columns)


def evaluate_factor_wide(name: str, raw: pd.DataFrame, category: str, ctx: ScreenContext
                         ) -> tuple[dict, pd.DataFrame | None]:
    """返回 (summary 的一行, 通过时每 5 天一次的调好方向的中性化因子值；没通过为 None)。"""
    t0 = time.time()
    h = ctx.main_horizon
    size = category != SIZE_CATEGORY
    score = _neutral(raw, category, ctx)
    buy = ctx.trim(ctx.can_buy).fillna(False).astype(bool)

    from qt.wide_eval import rank_ic
    ics = {k: rank_ic(score, r, buy) for k, r in ctx.rets.items()}
    mean_main = ics[h].mean()
    direction = -1 if mean_main < 0 else 1
    row: dict = {"factor": name, "category": category, "direction": direction, "size_neutralized": size,
                 "coverage": float(score.where(buy).notna().to_numpy().sum() / max(buy.to_numpy().sum(), 1))}
    for k, ic in ics.items():
        st = ic_summary(ic * direction, k)
        row |= {f"rank_ic_{k}d": st["mean"], f"icir_{k}d": st["icir"], f"t_nw_{k}d": st["t_nw"],
                f"positive_ratio_{k}d": st["positive_ratio"], f"n_days_{k}d": st["n_days"]}
    row["reversed_year_ratio"] = yearly_reversed_ratio(ics[h], direction)

    signed = score * direction
    q = quantile_excess(signed.where(buy), ctx.rets[h], every=h)
    groups = [c for c in q.columns if c != "universe"]
    per_year = 252 / h
    for g in groups:
        row[f"q{g}_excess_ann"] = q[g].mean() * per_year
    row["monotonicity"] = float(pd.Series(q[groups].mean().to_numpy()).corr(pd.Series(groups, dtype=float),
                                                                          method="spearman")) if len(q) else np.nan

    row["pass_t"] = bool(row[f"t_nw_{h}d"] > T_THRESHOLD)
    row["pass_stable"] = bool(row["reversed_year_ratio"] <= 0.5)
    if row["pass_t"] and row["pass_stable"]:
        acct = account_excess(signed, ctx.trim(ctx.close), ctx.trim(ctx.raw_close), buy,
                              ctx.trim(ctx.can_sell).fillna(False).astype(bool), ctx.trim(ctx.industry))
        row |= {k: acct[k] for k in ACCOUNT_KEYS}
        row["pass_cost"] = bool(acct["excess_annual"] > 0)
    else:
        row |= {k: np.nan for k in ACCOUNT_KEYS}
        row["pass_cost"] = False
    row["passed"] = row["pass_t"] and row["pass_stable"] and row["pass_cost"]
    row["fail_reason"] = ("" if row["passed"] else
                          f"t 值 ≤ {T_THRESHOLD:g}" if not row["pass_t"] else
                          "方向反了的年份过半" if not row["pass_stable"] else
                          "扣成本后跑输能买股票等权")
    row["seconds"] = round(time.time() - t0, 1)
    sub = _sample(signed, ctx) if row["passed"] else None
    return row, sub


def _neutral(raw: pd.DataFrame, category: str, ctx: ScreenContext) -> pd.DataFrame:
    return neutral_score(ctx.trim(raw), ctx.trim(ctx.industry), ctx.trim(ctx.log_mcap),
                         size=category != SIZE_CATEGORY)


def _sample(signed: pd.DataFrame, ctx: ScreenContext) -> pd.DataFrame:
    """所有因子同一组抽样日：检验区间第一天起每 main_horizon 天一次。"""
    return signed.loc[ctx.dates[::ctx.main_horizon]].astype("float32")


def factor_score(raw: pd.DataFrame, category: str, direction: int, ctx: ScreenContext) -> pd.DataFrame:
    """每天的中性化、按方向调整后的因子值（计划 4 合成用），和检验时用的完全一样。"""
    return _neutral(raw, category, ctx) * direction


def screen_score(raw: pd.DataFrame, category: str, direction: int, ctx: ScreenContext) -> pd.DataFrame:
    """通过检验的因子留给相关性去重的值（和 evaluate_factor_wide 返回的一样），不重跑检验。"""
    return _sample(_neutral(raw, category, ctx) * direction, ctx)


def wide_masks(amount: pd.DataFrame, daily: pd.DataFrame, listed: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """qt.universe.full_a_masks 的宽表版本：(能买, 能卖)，信号日 T 对齐。

    amount 是 Qlib 成交额宽表；"_amount20" 同 Qlib 的 Mean($amount,20)（不足 20 天用已有的天数）。
    daily 是 load_daily(["preclose", "close", "tradestatus", "isST"])，要覆盖到最后一天的下一个交易日。
    """
    from qt.universe import full_a_masks

    a20 = amount.rolling(20, min_periods=1).mean().stack(future_stack=True)
    a20.index.names = ["datetime", "instrument"]
    buy, sell = full_a_masks(pd.DataFrame({"_amount20": a20}), daily, listed)
    unstack = lambda s: s.unstack("instrument").reindex(index=amount.index, columns=amount.columns).fillna(False).astype(bool)
    return unstack(buy), unstack(sell)
