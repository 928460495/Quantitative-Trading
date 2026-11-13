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


def st_events(is_st: pd.DataFrame, close: pd.DataFrame, can_sell: pd.DataFrame, flags: dict[str, pd.DataFrame]
              ) -> pd.DataFrame:
    """每只股票第一次被戴 ST（之前至少有一天不是 ST）：之前 250、60 天的涨跌，之后连续卖不掉的天数，
    以及每个标记在戴帽前已经连续亮了多少天（戴帽当天没亮记缺失）。"""
    st = is_st.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    sell = can_sell.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool).to_numpy()
    px = close.ffill().to_numpy(dtype=float)
    rows = {}
    for j, code in enumerate(close.columns):
        s = st[code].to_numpy()
        starts = np.flatnonzero(s[1:] & ~s[:-1]) + 1
        if not len(starts):
            continue
        t = int(starts[0])
        back = lambda n: px[t, j] / px[t - n, j] - 1 if t >= n else np.nan
        locked = 0
        while t + locked < len(sell) and not sell[t + locked, j]:
            locked += 1
        row = {"date": close.index[t], "前250天涨跌": back(250), "前60天涨跌": back(60), "戴帽后连续卖不掉的天数": locked}
        traded = np.flatnonzero(~np.isnan(close[code].to_numpy(dtype=float)[:t]))   # 戴帽前有价格的日子（停牌日跳过）
        for name, f in flags.items():
            on = f[code].reindex(close.index).fillna(False).to_numpy(dtype=bool) if code in f else np.zeros(len(s), bool)
            lead = 0
            while lead < len(traded) and on[traded[-1 - lead]]:
                lead += 1
            row[f"{name} 提前天数"] = lead if lead else np.nan
        rows[code] = row
    return pd.DataFrame(rows).T.infer_objects()


def flag_split(score: pd.DataFrame, close: pd.DataFrame, can_buy: pd.DataFrame, flags: dict[str, pd.DataFrame],
               top: int = 50, every: int = HOLD) -> pd.DataFrame:
    """每 every 天取能买的股票里打分前 top 名：带某个标记的占多少，带和不带的之后 HOLD 天收益、跌超 20% 的比例。

    索引是（阶段, 标记），阶段含"全段"。
    """
    fwd = forward_returns(close, [HOLD])[HOLD]
    buy = can_buy.reindex(index=score.index, columns=score.columns).fillna(False).astype(bool)
    recs = []
    for d in score.index[::every]:
        s = score.loc[d].where(buy.loc[d]).dropna()
        r = fwd.loc[d].reindex(s.index)
        s, r = s[r.notna()], r[r.notna()]
        if len(s) < top:
            continue
        pick = s.sort_values(ascending=False, kind="stable").index[:top]
        for name, f in flags.items():
            on = f.loc[d].reindex(pick).fillna(False).astype(bool).to_numpy() if d in f.index else np.zeros(top, bool)
            rp = r[pick].to_numpy()
            recs.append({"date": d, "标记": name, "n_on": on.sum(), "n": top,
                         "ret_on": rp[on].sum(), "ret_off": rp[~on].sum(),
                         "crash_on": (rp[on] <= CRASH).sum(), "crash_off": (rp[~on] <= CRASH).sum()})
    df = pd.DataFrame(recs)
    df["阶段"] = era_of(pd.DatetimeIndex(df["date"]))
    out = {}
    for (era, name), g in [*df.groupby(["阶段", "标记"]), *((("全段", n), g) for n, g in df.groupby("标记"))]:
        on, off = g["n_on"].sum(), (g["n"] - g["n_on"]).sum()
        out[(era, name)] = {"顶端带标记的比例": on / g["n"].sum(),
                            "带标记的20天收益": g["ret_on"].sum() / on if on else np.nan,
                            "不带标记的20天收益": g["ret_off"].sum() / off if off else np.nan,
                            "带标记的跌超20%比例": g["crash_on"].sum() / on if on else np.nan,
                            "不带标记的跌超20%比例": g["crash_off"].sum() / off if off else np.nan}
    return pd.DataFrame(out).T.sort_index()
