"""宽表版的单因子检验：行是日期、列是股票。全 A 主板约 2800 只 × 2400 天 × 500 个因子，长表版太慢。

和长表版 qt/evaluate.py（沪深300 上已经核对过）的约定完全一样，测试里逐值对比：
- 远期收益：T 日收盘出信号，T+1 收盘买入，持有 h 天后收盘卖出
- RankIC：每天在因子和收益都有值（且能买）的股票里，排名后算相关；有效样本少于 30 只记缺失
- 分层：每 5 个交易日一次，先排名再分组（并列按先后），各组收益减全体等权均值
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qt.account import TRADING_DAYS, AccountRules, MarketData, run_account
from qt.evaluate import MIN_STOCKS, _groups, ic_stats, newey_west_t
from qt.neutralize import MIN_INDUSTRY_SIZE
from qt.preprocess import MAD_SCALE

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


# ---------- 预处理和中性化：和 qt/preprocess.py、qt/neutralize.py 的长表版一致 ----------

def winsorize_mad(x: pd.DataFrame, n: float = 3.0) -> pd.DataFrame:
    """每天把超出 中位数 ± n × 1.4826 × MAD 的值截到边界上；当天 MAD 为 0 时不截。"""
    med = x.median(axis=1)
    mad = x.sub(med, axis=0).abs().median(axis=1) * MAD_SCALE
    width = (n * mad).where(mad > 0, np.inf)
    return x.clip(med - width, med + width, axis=0)


def zscore(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)


def standardize(x: pd.DataFrame) -> pd.DataFrame:
    return zscore(winsorize_mad(x))


def neutralize(x: pd.DataFrame, industry: pd.DataFrame, log_mcap: pd.DataFrame, size: bool = True,
               min_industry_size: int = MIN_INDUSTRY_SIZE) -> pd.DataFrame:
    """每天对申万一级行业哑变量（和标准化后的对数流通市值）回归取残差，再标准化。

    当天不足 min_industry_size 只的行业并入"其他"，"其他"不设哑变量；有效股票少于 30 只的日子整行缺失。
    size=False 只对行业中性化（规模类因子用，否则规模被中性化掉了）。缺行业（或 size=True 时缺市值）的
    股票当天不参与。
    """
    ind = industry.reindex(index=x.index, columns=x.columns).to_numpy(dtype=object)
    xv = x.to_numpy(dtype=float)
    mv = log_mcap.reindex(index=x.index, columns=x.columns).to_numpy(dtype=float)
    out = np.full(xv.shape, np.nan)
    for i in range(len(xv)):
        valid = ~np.isnan(xv[i]) & pd.notna(ind[i])
        if size:
            valid &= ~np.isnan(mv[i])
        if valid.sum() < MIN_STOCKS:
            continue
        codes, labels = pd.factorize(ind[i][valid])
        counts = np.bincount(codes)
        keep = np.flatnonzero(counts >= min_industry_size)         # 小行业并入"其他"：不设哑变量
        cols = [(codes == k).astype(float) for k in keep]
        if size:
            m = mv[i][valid]
            sd = m.std(ddof=1)
            cols.append((m - m.mean()) / sd if sd > 0 else m * 0)
        y = xv[i][valid]
        if cols:
            design = np.column_stack(cols)
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            y = y - design @ beta
        out[i, valid] = y
    return zscore(pd.DataFrame(out, index=x.index, columns=x.columns))


def neutral_score(raw: pd.DataFrame, industry: pd.DataFrame, log_mcap: pd.DataFrame, size: bool = True) -> pd.DataFrame:
    """在当天全部股票上去极值、标准化再中性化（同 qt.pipeline.neutral_score）：已持有但第二天买不进的股票
    也要有排名，才能决定留还是卖。"""
    return neutralize(standardize(raw), industry, log_mcap, size=size)


# ---------- 按实际账户规则扣成本（通过条件第三条） ----------

# docs/DECISIONS.md 2026-11-01：账户规则用默认值，不用仓位控制 A（不传 exposure）、不用回撤兜底 C
SCREEN_RULES = AccountRules(drawdown_stop=None)


def buyable_benchmark(adj_close: pd.DataFrame, can_buy: pd.DataFrame) -> pd.Series:
    """能买的股票每天等权的收益：第 t 天 = 信号日 t−2 能买（t−1 收盘买进）的股票 t−1 → t 的平均收益。

    和账户同样的时间约定，不扣成本，每天重新等权。前两天没有持仓，记 0。
    """
    r = adj_close / adj_close.shift(1) - 1
    held = can_buy.reindex(index=adj_close.index, columns=adj_close.columns, fill_value=False).astype(bool)
    held = held.shift(2, fill_value=False)
    return r.where(held).mean(axis=1).fillna(0.0)


def _market_data(score, adj_close, raw_close, can_buy, can_sell, industry) -> MarketData:
    idx, cols = adj_close.index, adj_close.columns
    al = lambda x, fill: x.reindex(index=idx, columns=cols, fill_value=fill)
    ind = None
    if industry is not None:
        ind = al(industry, None).to_numpy(dtype=object, copy=True)   # pandas 3 可能返回只读数组
        ind[pd.isna(ind)] = None                     # 缺行业不受行业上限限制（同 qt.account._industry）
    return MarketData(
        dates=pd.DatetimeIndex(idx), instruments=pd.Index(cols),
        score=al(score, np.nan).to_numpy(dtype=float),
        adj_close=adj_close.to_numpy(dtype=float),
        raw_close=al(raw_close, np.nan).to_numpy(dtype=float),
        can_buy=al(can_buy, False).to_numpy(dtype=bool),
        can_sell=al(can_sell, False).to_numpy(dtype=bool),
        industry=ind,
    )


def account_excess(score: pd.DataFrame, adj_close: pd.DataFrame, raw_close: pd.DataFrame, can_buy: pd.DataFrame,
                   can_sell: pd.DataFrame, industry: pd.DataFrame | None, rules: AccountRules = SCREEN_RULES) -> dict:
    """按因子打分、按实际账户规则回测，和能买股票的等权收益比。score 越大越好（方向先调好），按信号日对齐。

    年化都是复合年化，区间相同。excess_annual > 0 是通过条件第三条。
    """
    result = run_account(_market_data(score, adj_close, raw_close, can_buy, can_sell, industry), rules)
    daily = result.daily
    nav = daily["nav"]
    n = len(nav) - 1
    bench = buyable_benchmark(adj_close, can_buy).reindex(nav.index).fillna(0.0)
    bench_nav = (1 + bench).cumprod()
    annual = (nav.iloc[-1] / nav.iloc[0]) ** (TRADING_DAYS / n) - 1
    bench_annual = (bench_nav.iloc[-1] / bench_nav.iloc[0]) ** (TRADING_DAYS / n) - 1
    trades = result.trades
    sold = trades[trades["side"] == "sell"].groupby("date")["value"].sum().reindex(nav.index, fill_value=0.0)
    excess_daily = nav.pct_change().iloc[1:] - bench.iloc[1:]
    reasons = trades["reason"].value_counts()
    return {
        "annual_return": float(annual),
        "bench_annual": float(bench_annual),
        "excess_annual": float(annual - bench_annual),
        "excess_t": float(newey_west_t(excess_daily, lag=0)),
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
        "turnover": float((sold / nav).iloc[1:].mean()),
        "cost_annual": float((daily["cost"] / nav).iloc[1:].mean() * TRADING_DAYS),
        "avg_holdings": float(daily["n_holdings"].iloc[1:].mean()),
        "n_trades": int(len(trades)),
        "n_stops": int(reasons.get("stop", 0)),
        "n_drawdown_sells": int(reasons.get("drawdown", 0)),
    }
