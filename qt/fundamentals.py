"""按发布日期对齐的季度财务数据（docs/STRATEGY.md 第 10 节）。

数据来自东方财富数据中心（scripts/collect_full_a.py）。实测（docs/DATA_NOTES.md）：
- 业绩报表（yjbb）的 NOTICE_DATE 是首次公告日，可以当作"什么时候知道"
- 利润表、资产负债表、现金流量表的 NOTICE_DATE 是最后一次披露这期数字的日子（往往是第二年同期报告），
  中位数比报告期晚 396 天，不能用；这三张表只取数值，按（股票，报告期）接到业绩报表上，可用日期一律用业绩报表的

公告日当天收盘后才算知道，下一个交易日起可用。累计值（年初至今）在这里换算成单季、滚动四季。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 原始列名 → 统一列名
COLUMN_MAP: dict[str, dict[str, str]] = {
    "yjbb": {
        "NOTICE_DATE": "notice_date",
        "PARENT_NETPROFIT": "net_profit",        # 归母净利润，年初至今累计
        "TOTAL_OPERATE_INCOME": "revenue",       # 营业总收入，累计
        "BASIC_EPS": "eps",                      # 基本每股收益，累计
        "BPS": "bps",                            # 每股净资产
        "WEIGHTAVG_ROE": "roe",                  # 加权净资产收益率（%），累计
        "XSMLL": "gross_margin",                 # 销售毛利率（%）
        "MGJYXJJE": "ocf_per_share",             # 每股经营现金流，累计
        "YSTZ": "revenue_yoy_reported",          # 营收同比（%），按累计口径
        "SJLTZ": "profit_yoy_reported",          # 净利润同比（%），按累计口径
    },
    "income": {"DEDUCT_PARENT_NETPROFIT": "deduct_net_profit", "OPERATE_COST": "operate_cost"},
    "balance": {"TOTAL_ASSETS": "total_assets", "TOTAL_LIABILITIES": "total_liab", "TOTAL_EQUITY": "total_equity"},
    "cashflow": {"NETCASH_OPERATE": "op_cash_flow"},
}
KEYS = ["instrument", "report_date"]


def tidy_reports(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """每个（股票，报告期）一行。只要沪深 A 股；多次公告只留最早的公告日。"""
    yj = raw["yjbb"]
    yj = yj[(yj["SECURITY_TYPE"] == "A股") & yj["instrument"].str.startswith(("SH", "SZ"))]
    base = yj[KEYS + list(COLUMN_MAP["yjbb"])].rename(columns=COLUMN_MAP["yjbb"])
    base = base.assign(report_date=pd.to_datetime(base["report_date"]),
                       notice_date=pd.to_datetime(base["notice_date"]).dt.normalize())
    base = base.sort_values("notice_date").drop_duplicates(KEYS, keep="first")
    for name in ("income", "balance", "cashflow"):
        if name not in raw:
            continue
        extra = raw[name][KEYS + list(COLUMN_MAP[name])].rename(columns=COLUMN_MAP[name])
        extra = extra.assign(report_date=pd.to_datetime(extra["report_date"])).drop_duplicates(KEYS, keep="last")
        base = base.merge(extra, on=KEYS, how="left")
    return base.sort_values(KEYS).reset_index(drop=True)


def _wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """报告期 × 股票，报告期补齐成连续的季度（缺的季度为 NaN）。"""
    w = df.pivot(index="report_date", columns="instrument", values=col)
    return w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="QE"))


def _back(w: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """宽表的值取回到 df 的每一行。"""
    stacked = w.stack(future_stack=True)
    return pd.Series(stacked.reindex(pd.MultiIndex.from_arrays([df["report_date"], df["instrument"]])).to_numpy(),
                     index=df.index)


def _single_quarter_wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    w = _wide(df, col)
    q1 = pd.Series(w.index.month == 3, index=w.index)
    return (w - w.shift(1)).mask(q1, w, axis=0)


def single_quarter(df: pd.DataFrame, col: str) -> pd.Series:
    """累计值换算成单季：一季度就是本身，其余季度减上一季度（上一季度缺失就是 NaN）。"""
    return _back(_single_quarter_wide(df, col), df)


def ttm(df: pd.DataFrame, col: str) -> pd.Series:
    """滚动四季合计，四个单季缺任何一个就是 NaN。"""
    return _back(_single_quarter_wide(df, col).rolling(4, min_periods=4).sum(), df)


def yoy(df: pd.DataFrame, col: str) -> pd.Series:
    """单季同比；去年同季的单季值 ≤ 0 时为 NaN。"""
    sq = _single_quarter_wide(df, col)
    base = sq.shift(4)
    return _back((sq / base - 1).where(base > 0), df)


def align_to_days(records: pd.DataFrame, cols: list[str], index: pd.MultiIndex, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """把财务记录对齐到 (datetime, instrument)：公告日之后的第一个交易日起可用，一直沿用到下一份更新的报告。

    晚公告的旧报告期（比如一季报先出、年报后出）不覆盖已经可用的更新报告期。返回 cols 加上 report_date。
    """
    rec = records[["instrument", "report_date", "notice_date", *cols]].astype({"instrument": str})
    pos = calendar.searchsorted(rec["notice_date"].to_numpy(), side="right")
    rec = rec[pos < len(calendar)].assign(available=calendar[pos[pos < len(calendar)]])
    rec = rec.sort_values(["instrument", "available", "report_date"])
    newest = rec.groupby("instrument")["report_date"].cummax()
    rec = rec[rec["report_date"] >= newest].drop_duplicates(["instrument", "available"], keep="last")

    days = pd.DataFrame({"datetime": index.get_level_values("datetime"),
                         "instrument": index.get_level_values("instrument").astype(str)})
    days["_order"] = np.arange(len(days))
    merged = pd.merge_asof(days.sort_values("datetime"), rec.sort_values("available"),
                           left_on="datetime", right_on="available", by="instrument", direction="backward")
    merged = merged.sort_values("_order")
    out = merged[cols + ["report_date"]]
    out.index = index
    return out


def level_yoy(df: pd.DataFrame, col: str) -> pd.Series:
    """时点值（如总资产）和去年同季相比的增长率；基数 ≤ 0 时为 NaN。"""
    w = _wide(df, col)
    base = w.shift(4)
    return _back((w / base - 1).where(base > 0), df)


def sue(df: pd.DataFrame, col: str, quarters: int = 8) -> pd.Series:
    """标准化预期外盈余（季节性随机游走）：单季值减去年同季，除以最近 quarters 个这种变化的标准差。"""
    sq = _single_quarter_wide(df, col)
    change = sq - sq.shift(4)
    std = change.rolling(quarters, min_periods=quarters).std()
    return _back((change / std).where(std > 0), df)
