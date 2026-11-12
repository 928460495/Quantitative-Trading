"""便宜的证据：不跑账户，直接看打分最顶端各档之后的收益（docs/FRAMEWORK.md 第 5 节，失效模式 F4）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qt.lab.protocol import ERAS, FULL, era_of
from qt.wide_eval import forward_returns

TOP_NS = (15, 50, 150)
HOLD = 20
CRASH = -0.20


def top_tail(score: pd.DataFrame, close: pd.DataFrame, can_buy: pd.DataFrame, every: int = HOLD, min_stocks: int = 100
             ) -> pd.DataFrame:
    """每 every 个交易日，在能买的股票里按 score 取前 N 名和最好的五分之一，看之后 HOLD 天的等权收益（不扣成本）。

    返回按阶段汇总的表：各档年化收益、全体年化、前 15 名里 HOLD 天内跌超 20% 的平均只数、随机挑 15 只的期望只数。
    """
    fwd = forward_returns(close, [HOLD])[HOLD]
    buy = can_buy.reindex(index=score.index, columns=score.columns).fillna(False).astype(bool)
    recs = {}
    for d in score.index[::every]:
        s = score.loc[d].where(buy.loc[d])
        r = fwd.loc[d].reindex(s.index)
        ok = s.notna() & r.notna()
        s, r = s[ok], r[ok]
        if len(s) < min_stocks:
            continue
        order = s.sort_values(ascending=False, kind="stable").index
        rec = {"全体": r.mean(), **{f"前{n}名": r[order[:n]].mean() for n in TOP_NS},
               "最好五分之一": r[order[:len(order) // 5]].mean(),
               "前15名跌超20%的只数": float((r[order[:15]] <= CRASH).sum()),
               "随机挑15只的期望": float((r <= CRASH).mean() * 15)}
        recs[d] = rec
    df = pd.DataFrame(recs).T
    era = era_of(df.index)
    rows = {}
    for name in [*ERAS, FULL]:
        g = df if name == FULL else df[era == name]
        if not len(g):
            continue
        counts = [c for c in g if "只" in c]
        ann = (1 + g.drop(columns=counts)).prod() ** (252 / HOLD / len(g)) - 1
        rows[name] = pd.concat([ann, g[counts].mean()])
    return pd.DataFrame(rows).T
