"""宽表版的单因子检验：行是日期、列是股票。全 A 主板约 2800 只 × 2400 天 × 500 个因子，长表版太慢。

和长表版 qt/evaluate.py（沪深300 上已经核对过）的约定完全一样，测试里逐值对比：
- 远期收益：T 日收盘出信号，T+1 收盘买入，持有 h 天后收盘卖出
- RankIC：每天在因子和收益都有值（且能买）的股票里，排名后算相关；有效样本少于 30 只记缺失
- 分层：每 5 个交易日一次，先排名再分组（并列按先后），各组收益减全体等权均值
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qt.evaluate import MIN_STOCKS, _groups, ic_stats

ic_summary = ic_stats   # 均值、标准差、ICIR、正值比例、Newey-West t（滞后 持有期 − 1）、天数


def forward_returns(close: pd.DataFrame, horizons: list[int]) -> dict[int, pd.DataFrame]:
    """持有期 → 远期收益宽表。close 是复权收盘价，停牌为缺失（对应的收益也缺失）。"""
    entry = close.shift(-1)
    out = {}
    for h in horizons:
        if h < 1:
            raise ValueError("持有期至少 1 天")
        out[h] = close.shift(-(h + 1)) / entry - 1
    return out


def drop_tail(wide: pd.DataFrame, n: int) -> pd.DataFrame:
    """去掉最后 n 个交易日：这几天的远期收益要用到段外的价格（同 qt.data.drop_embargo）。"""
    if len(wide) <= n:
        raise ValueError(f"样本只有 {len(wide)} 个交易日，不够去掉 {n} 天")
    return wide.iloc[:-n]


def rank_ic(factor: pd.DataFrame, ret: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.Series:
    """逐日横截面 Spearman 相关。mask 为 False 的格子不参与（比如第二天买不进）。"""
    ret = ret.reindex(index=factor.index, columns=factor.columns)
    valid = factor.notna() & ret.notna()
    if mask is not None:
        valid &= mask.reindex(index=factor.index, columns=factor.columns, fill_value=False).astype(bool)
    fr = factor.where(valid).rank(axis=1).to_numpy(dtype=float)
    rr = ret.where(valid).rank(axis=1).to_numpy(dtype=float)
    n = valid.to_numpy().sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = fr - np.nanmean(fr, axis=1, keepdims=True)
        e = rr - np.nanmean(rr, axis=1, keepdims=True)
        num = np.nansum(d * e, axis=1)
        den = np.sqrt(np.nansum(d * d, axis=1) * np.nansum(e * e, axis=1))
        ic = num / den
    ic[~np.isfinite(ic) | (n < MIN_STOCKS)] = np.nan
    return pd.Series(ic, index=pd.DatetimeIndex(factor.index, name="datetime"), name="spearman")


def yearly_reversed_ratio(ic: pd.Series, direction: int) -> float:
    """按方向调整后，年均 RankIC 小于 0 的年份占比。"""
    yearly = ic.groupby(ic.index.year).mean() * direction
    return float((yearly < 0).mean())


def quantile_excess(score: pd.DataFrame, ret: pd.DataFrame, every: int = 5, n_groups: int = 5) -> pd.DataFrame:
    """每 every 个交易日按 score 分 n_groups 组（第 n_groups 组最大），各组持有期收益减去全体等权均值。

    score 里不能买的格子应先置为缺失。调仓日从 score 第一个有值的日子开始数。结果额外带一列 universe。
    """
    ret = ret.reindex(index=score.index, columns=score.columns)
    days = score.index[score.notna().any(axis=1)][::every]
    rows = {}
    for d in days:
        r = ret.loc[d].dropna()
        s = score.loc[d].reindex(r.index).dropna()
        if len(s) < MIN_STOCKS:
            continue
        g = _groups(s, n_groups)
        r = r.reindex(g.index)
        row = (r.groupby(g).mean() - r.mean()).to_dict()
        row["universe"] = r.mean()
        rows[d] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()
