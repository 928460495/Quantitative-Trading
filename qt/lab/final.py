"""最终测试的成绩单和判定（DECISIONS 2026-11-13「比较口径改为夏普比率」「最终测试的程序」）。

封存段不分制度阶段：只报全段和每个自然年。判定只看全段：四个起点平均的夏普高于沪深300 全收益的夏普，
并且高于能持有等权的夏普（能持有等权的夏普也是四个起点平均）。其余都只报告。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qt.account import AccountResult
from qt.lab import scorecard as sc

FULL = "全段"
COLUMNS = ["夏普", "最差起点夏普", "等权夏普", "沪深300夏普", "年化", "能持有等权", "沪深300全收益", "最大回撤",
           "信息比率", "主动风险", "α的t值", "M²", "索提诺", "卡尔玛", "年成本"]


def check_window(dates: pd.DatetimeIndex, start: str, end: str) -> None:
    dates = pd.DatetimeIndex(dates)
    if dates.min() < pd.Timestamp(start) or dates.max() > pd.Timestamp(end):
        raise ValueError(f"数据 {dates.min().date()} ~ {dates.max().date()} 超出登记的窗口 {start} ~ {end}")


def final_scorecard(dev: dict, results: dict[int, AccountResult]) -> pd.DataFrame:
    rets = sc.daily_returns(results)
    hold, hs = sc.bench_returns(dev, results)
    slices = {FULL: np.ones(len(rets), dtype=bool), **{str(y): rets.index.year == y for y in sorted(set(rets.index.year))}}
    rows = {}
    for name, mask in slices.items():
        dates = rets.index[mask]
        risk = pd.DataFrame({k: sc.risk_metrics(rets.loc[dates, k], hold.loc[dates, k], hs.loc[dates]) for k in rets}).T
        cost = [float((r.daily.loc[dates[0]:dates[-1], "cost"] / r.daily.loc[dates[0]:dates[-1], "nav"]).mean() * 252)
                for r in results.values()]
        rows[name] = {**risk.mean().to_dict(), "最差起点夏普": float(risk["夏普"].min()),
                      "年化": float(np.mean([sc._annual(rets.loc[dates, k]) for k in rets])),
                      "能持有等权": float(np.mean([sc._annual(hold.loc[dates, k]) for k in rets])),
                      "沪深300全收益": sc._annual(hs.loc[dates]),
                      "最大回撤": float(np.mean([sc._max_drawdown(rets.loc[dates, k]) for k in rets])),
                      "年成本": float(np.mean(cost))}
    return pd.DataFrame(rows).T[COLUMNS]


def verdict(card: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    row = card.loc[FULL]
    table = pd.DataFrame({"策略夏普": [row["夏普"]] * 2, "基准夏普": [row["沪深300夏普"], row["等权夏普"]]},
                         index=["夏普高于沪深300全收益", "夏普高于能持有等权"])
    table["满足"] = table["策略夏普"] > table["基准夏普"]
    return bool(table["满足"].all()), table
