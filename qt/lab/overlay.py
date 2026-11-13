"""估值 + 趋势仓位（DECISIONS 2026-11-13「仓位管理定为估值 + 趋势」）。

    sig = monthly_signals(load_monthly())          # 每月：估值百分位、趋势
    f = daily_target(sig, days)                     # 每天的目标股票比例（只用日期严格早于当天的月份）
    k = slot_counts(f, n_hold=15, high=0.75)        # 每天的持股只数上限：每 20 天更新，差不超过 2 只不动

数据：上证市场平均市盈率和上证指数（乐咕乐股，月度，data/cache/external/sse_pe_legulegu.csv）。
未解锁时，2022-01-01 起的行（封存段）一律丢掉。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qt.config import SEGMENTS, is_test_unlocked

ROOT = Path(__file__).resolve().parents[2]
PE_FILE = ROOT / "data" / "cache" / "external" / "sse_pe_legulegu.csv"
VAL_MONTHS, TREND_MONTHS = 60, 10
LOW, HIGH, EVERY, BAND = 0.25, 0.75, 20, 2


def load_monthly(path: Path = PE_FILE) -> pd.DataFrame:
    """index = 月份日期，列 ep（1 ÷ 平均市盈率）、index（上证指数）。未解锁时只留封存段之前的行。"""
    df = pd.read_csv(path)
    df["日期"] = pd.to_datetime(df["日期"])
    if not is_test_unlocked():
        df = df[df["日期"] < pd.Timestamp(SEGMENTS["test"][0])]
    df = df.sort_values("日期").set_index("日期")
    return pd.DataFrame({"ep": 1 / df["平均市盈率"].astype(float), "index": df["指数"].astype(float)})


def monthly_signals(m: pd.DataFrame) -> pd.DataFrame:
    """valuation：当月 E/P 在过去 60 个月（含当月）里的百分位；trend：指数 ≥ 过去 10 个月平均为 1，否则 0。不够长为空。"""
    ep = m["ep"].to_numpy()
    val = [np.nan if i + 1 < VAL_MONTHS else float((ep[i + 1 - VAL_MONTHS:i + 1] <= ep[i]).mean()) for i in range(len(ep))]
    ma = m["index"].rolling(TREND_MONTHS, min_periods=TREND_MONTHS).mean()
    trend = (m["index"] >= ma).astype(float).where(ma.notna())
    return pd.DataFrame({"valuation": val, "trend": trend}, index=m.index)


def daily_target(sig: pd.DataFrame, days: pd.DatetimeIndex, low: float = LOW, high: float = HIGH) -> pd.Series:
    """第 t 天的目标股票比例：用日期严格早于 t 的最近一个月的两个信号。"""
    days = pd.DatetimeIndex(days)
    pos = sig.index.searchsorted(days, side="left") - 1
    take = lambda col: np.where(pos >= 0, sig[col].to_numpy()[np.clip(pos, 0, None)], np.nan)
    combo = 0.5 * take("valuation") + 0.5 * take("trend")
    return pd.Series(low + (high - low) * combo, index=days)


def slot_counts(f: pd.Series, n_hold: int, high: float = HIGH, every: int = EVERY, band: int = BAND) -> pd.Series:
    """每天的持股只数上限。第 0、every、2×every … 天按 round(n_hold × f ÷ high) 重算；和现在差不超过 band 只就不动。"""
    want = np.rint(n_hold * f.to_numpy() / high)
    if np.isnan(want).any():
        raise ValueError("目标比例有缺失（估值或趋势的历史不够长）")
    out = np.empty(len(want))
    cur = want[0]
    for j, w in enumerate(want):
        if j % every == 0 and abs(w - cur) > band:
            cur = w
        out[j] = cur
    return pd.Series(np.clip(out, 0, n_hold).astype(int), index=f.index)
