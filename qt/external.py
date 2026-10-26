"""外部数据：申万行业变动历史、巨潮股本变动。由 scripts/collect_external.py 在本地采集，存在 data/external/。

两份数据都按"当时能知道的版本"对齐到每个交易日（point-in-time）：
- 行业：取计入日期不晚于当天的最近一条记录
- 股本：生效日期取 公告日期 和 变动日期 中较晚的一个，再取不晚于当天的最近一条
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = Path(os.environ.get("QT_EXTERNAL_DIR", ROOT / "data" / "external"))
INDUSTRY_FILE = "sw_industry_hist.csv"
SHARES_FILE = "share_changes.csv"


def to_symbol(instrument: str) -> str:
    """Qlib 代码转 6 位代码：SH600000 → 600000。"""
    return instrument[2:]


def industry_level1(code: str) -> str:
    """申万 2021 行业代码的前两位是一级行业。"""
    return str(code)[:2]


SW_INDUSTRY_URL = "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"
_SW_COLUMNS = {"股票代码": "symbol", "计入日期": "start_date", "行业代码": "industry_code", "更新日期": "update_time"}


def read_sw_industry_file(path: Path) -> pd.DataFrame:
    """读手动下载的申万行业分类表（SW_INDUSTRY_URL），列名改成和 AkShare 一样。支持 .xls / .xlsx / .csv。"""
    dtype = {"股票代码": str, "行业代码": str}
    path = Path(path)
    raw = pd.read_csv(path, dtype=dtype) if path.suffix.lower() == ".csv" else pd.read_excel(path, dtype=dtype)
    missing = set(_SW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} 里缺少列 {sorted(missing)}，实际的列是 {list(raw.columns)}")
    return raw.rename(columns=_SW_COLUMNS)


def tidy_industry(raw: pd.DataFrame) -> pd.DataFrame:
    """整理 AkShare stock_industry_clf_hist_sw() 的返回值。"""
    df = raw[["symbol", "start_date", "industry_code"]].copy()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["industry_code"] = df["industry_code"].astype(str)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    return df.dropna(subset=["start_date"]).sort_values(["symbol", "start_date"]).reset_index(drop=True)


def tidy_share_changes(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """整理 AkShare stock_share_change_cninfo() 的返回值。

    流通 A 股用"人民币普通股"，缺失时退回"已流通股份"。单位沿用巨潮原始口径（取对数后不影响中性化）。
    """
    if raw.empty:
        return pd.DataFrame(columns=["symbol", "declare_date", "change_date", "effective_date", "total_shares", "float_a_shares"])
    float_a = raw["人民币普通股"] if "人民币普通股" in raw else pd.Series(np.nan, index=raw.index)
    if "已流通股份" in raw:
        float_a = float_a.fillna(raw["已流通股份"])
    df = pd.DataFrame(
        {
            "symbol": str(symbol).zfill(6),
            "declare_date": pd.to_datetime(raw["公告日期"], errors="coerce"),
            "change_date": pd.to_datetime(raw["变动日期"], errors="coerce"),
            "total_shares": pd.to_numeric(raw["总股本"], errors="coerce"),
            "float_a_shares": pd.to_numeric(float_a, errors="coerce"),
        }
    )
    df["effective_date"] = df[["declare_date", "change_date"]].max(axis=1)
    df = df.dropna(subset=["effective_date", "float_a_shares"])
    df = df[df["float_a_shares"] > 0]
    return df.sort_values(["effective_date", "declare_date"]).reset_index(drop=True)


def _point_in_time(index: pd.MultiIndex, table: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    left = pd.DataFrame(
        {
            "datetime": index.get_level_values("datetime"),
            "instrument": index.get_level_values("instrument"),
        }
    )
    left["symbol"] = left["instrument"].map(to_symbol)
    left["_order"] = np.arange(len(left))
    right = table[["symbol", date_col, value_col]].dropna().sort_values(date_col)
    merged = pd.merge_asof(
        left.sort_values("datetime"), right, left_on="datetime", right_on=date_col, by="symbol", direction="backward"
    )
    out = merged.sort_values("_order")[value_col].to_numpy()
    return pd.Series(out, index=index, name=value_col)


def industry_on(index: pd.MultiIndex, industry: pd.DataFrame) -> pd.Series:
    """每个 (交易日, 股票) 当时所属的申万一级行业代码；没有记录的为 NaN。"""
    table = industry.assign(level1=industry["industry_code"].map(industry_level1))
    return _point_in_time(index, table, "start_date", "level1")


def float_shares_on(index: pd.MultiIndex, shares: pd.DataFrame) -> pd.Series:
    """每个 (交易日, 股票) 当时已公告生效的流通 A 股股数；没有记录的为 NaN。"""
    # 同一生效日有多条记录时，保留最后公告的那条
    table = shares.sort_values(["effective_date", "declare_date"]).drop_duplicates(["symbol", "effective_date"], keep="last")
    return _point_in_time(index, table, "effective_date", "float_a_shares")


def load_external(directory: Path = EXTERNAL_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    ind_path, sh_path = directory / INDUSTRY_FILE, directory / SHARES_FILE
    missing = [p.name for p in (ind_path, sh_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{directory} 里缺少 {missing}，先在本地运行 python scripts/collect_external.py")
    industry = pd.read_csv(ind_path, dtype={"symbol": str, "industry_code": str}, parse_dates=["start_date"])
    shares = pd.read_csv(sh_path, dtype={"symbol": str}, parse_dates=["declare_date", "change_date", "effective_date"])
    return industry, shares
