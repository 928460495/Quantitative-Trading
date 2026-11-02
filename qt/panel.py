"""因子计算的输入宽表（行是交易日，列是股票）。

    panel = load_panel("2010-01-01", "2018-12-31")
    panel["close"], panel["turn"], panel["net_profit_ttm"], ...

来源：
- Qlib 数据包：开高低收、均价、成交量、成交额（千元）、复权因子；价格都是复权价，`raw_close` 是未复权收盘价
- 全 A 补充数据（baostock）：换手率（%）、估值（市盈率 TTM、市净率、市销率 TTM、市现率 TTM）、ST 标记；
  流通股本由成交量 / 换手率反推，流通市值 = 未复权收盘价 × 流通股本
- 季度财务：按首次公告日的下一个交易日起可用（qt/fundamentals.py），每天沿用最新一期
- 申万行业：一级、二级、三级（6 位代码的前 2、4、6 位），按变动日期取当时的分类
- 沪深300 指数：`benchmark_close`、`benchmark_open`，每天各列相同（国泰君安公式里的基准指数）

只取主板股票（沪市 600、601、603、605，深市 000、001、002、003）。只取 [start, end]；
需要回看窗口的，调用方把 start 往前多取一段。估值字段原样给出，用的时候要错开一天（docs/DATA_NOTES.md）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qt.config import DATA_DIR, FULL_A_DIR, check_range
from qt.external import _point_in_time, load_external
from qt.universe import is_main_board

PRICE_FIELDS = {"open": "$open", "high": "$high", "low": "$low", "close": "$close", "vwap": "$vwap",
                "volume": "$volume", "amount": "$amount", "factor": "$factor"}
DAILY_FIELDS = {"turn": "turn", "pe_ttm": "peTTM", "pb_mrq": "pbMRQ", "ps_ttm": "psTTM", "pcf_ttm": "pcfNcfTTM",
                "is_st": "isST", "bs_volume": "volume"}
BENCHMARK = "SH000300"
# 季度财务：原始字段 + 由累计值换算出的滚动四季、单季、单季同比
FUND_FIELDS = ["net_profit", "revenue", "eps", "bps", "roe", "gross_margin", "ocf_per_share", "total_assets",
               "total_liab", "total_equity", "op_cash_flow", "deduct_net_profit", "operate_cost"]
TTM_FIELDS = ["net_profit", "revenue", "op_cash_flow", "deduct_net_profit", "operate_cost"]
YOY_FIELDS = ["net_profit", "revenue", "deduct_net_profit"]


def main_board_instruments(start: str, end: str, data_dir: Path = DATA_DIR) -> list[str]:
    """[start, end] 期间上市过的主板股票。"""
    spans = pd.read_csv(data_dir / "instruments" / "all.txt", sep="\t", header=None, names=["code", "start", "end"])
    spans = spans[is_main_board(pd.Index(spans["code"]))]
    alive = (pd.to_datetime(spans["start"]) <= pd.Timestamp(end)) & (pd.to_datetime(spans["end"]) >= pd.Timestamp(start))
    return sorted(spans.loc[alive, "code"].unique())


def industry_levels(index: pd.MultiIndex, industry: pd.DataFrame) -> dict[str, pd.Series]:
    """每个 (交易日, 股票) 当时的申万一、二、三级行业代码。"""
    code = _point_in_time(index, industry, "start_date", "industry_code").astype(object)
    return {f"industry_l{lv}": code.map(lambda c, n=n: c[:n] if isinstance(c, str) else np.nan)
            for lv, n in ((1, 2), (2, 4), (3, 6))}


def _qlib_wide(instruments: list[str], fields: dict[str, str], start: str, end: str) -> dict[str, pd.DataFrame]:
    from qlib.data import D

    from qt.data import init_qlib

    init_qlib()
    df = D.features(instruments, list(fields.values()), start_time=start, end_time=end)
    df.columns = list(fields)
    df = df.swaplevel().sort_index()
    df.index.names = ["datetime", "instrument"]
    df = df.replace([np.inf, -np.inf], np.nan).astype(np.float64)  # Qlib 给的是 float32，长链条运算误差会放大到 1e-7
    return {k: df[k].unstack("instrument") for k in fields}


def _fundamental_wide(dates: pd.DatetimeIndex, instruments: list[str], directory: Path) -> dict[str, pd.DataFrame]:
    from qt.fundamentals import align_to_days, level_yoy, sue, tidy_reports, ttm, yoy

    raw = {n: pd.read_parquet(directory / "reports" / f"{n}.parquet") for n in ["yjbb", "income", "balance", "cashflow"]}
    rec = tidy_reports(raw)
    rec = rec[rec["instrument"].isin(instruments)].reset_index(drop=True)
    for f in TTM_FIELDS:
        rec[f"{f}_ttm"] = ttm(rec, f)
    for f in YOY_FIELDS:
        rec[f"{f}_yoy"] = yoy(rec, f)
    rec["net_profit_sue"] = sue(rec, "net_profit")
    rec["total_assets_yoy"] = level_yoy(rec, "total_assets")
    cols = (FUND_FIELDS + [f"{f}_ttm" for f in TTM_FIELDS] + [f"{f}_yoy" for f in YOY_FIELDS]
            + ["net_profit_sue", "total_assets_yoy"])
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    aligned = align_to_days(rec, cols, index, dates)
    out = {c: aligned[c].astype(float).unstack("instrument") for c in cols}
    out["report_date"] = aligned["report_date"].unstack("instrument")
    return out


def load_panel(start: str, end: str, instruments: list[str] | None = None,
               full_a_dir: Path = FULL_A_DIR) -> dict[str, pd.DataFrame]:
    check_range(start, end)
    instruments = instruments or main_board_instruments(start, end)
    panel = _qlib_wide(instruments, PRICE_FIELDS, start, end)
    close = panel["close"]
    dates, cols = close.index, close.columns
    panel["returns"] = close / close.shift(1) - 1
    panel["raw_close"] = close / panel["factor"]

    bench = _qlib_wide([BENCHMARK], {"benchmark_close": "$close", "benchmark_open": "$open"}, start, end)
    for k, v in bench.items():
        s = v[BENCHMARK].reindex(dates)
        panel[k] = pd.DataFrame(np.repeat(s.to_numpy()[:, None], len(cols), axis=1), index=dates, columns=cols)

    index = pd.MultiIndex.from_product([dates, cols], names=["datetime", "instrument"])
    for k, s in industry_levels(index, load_external()[0]).items():
        panel[k] = s.unstack("instrument").reindex(index=dates, columns=cols)

    from qt.full_a import load_daily

    daily = load_daily(list(DAILY_FIELDS.values()), start, end, full_a_dir)
    for k, f in DAILY_FIELDS.items():
        panel[k] = daily[f].unstack("instrument").reindex(index=dates, columns=cols).astype(float)
    shares = (panel["bs_volume"] / (panel["turn"] / 100)).where(panel["turn"] > 0).ffill()
    panel["float_mcap"] = panel["raw_close"] * shares

    panel.update(_fundamental_wide(dates, list(cols), full_a_dir))
    for k in list(panel):
        panel[k] = panel[k].reindex(index=dates, columns=cols)
    return panel
