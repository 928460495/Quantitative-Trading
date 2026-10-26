"""读取全 A 补充数据（scripts/download_full_a.py 解包到 FULL_A_DIR）：baostock 日线、股票列表、指数。

所有按日期取数的函数都先经过 check_range()，测试段未解锁时拿不到 2022 年以后的数据。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qt.config import FULL_A_DIR, check_range


def load_daily(columns: list[str], start: str, end: str, directory: Path = FULL_A_DIR) -> pd.DataFrame:
    """baostock 日线的指定列，以 (datetime, instrument) 为索引。"""
    check_range(start, end)
    df = pd.read_parquet(directory / "daily.parquet", columns=["instrument", "date", *columns],
                         filters=[("date", ">=", pd.Timestamp(start)), ("date", "<=", pd.Timestamp(end))])
    df = df.rename(columns={"date": "datetime"}).astype({"instrument": str})
    return df.set_index(["datetime", "instrument"]).sort_index()[columns]


def listing_dates_full_a(directory: Path = FULL_A_DIR) -> pd.Series:
    stocks = pd.read_parquet(directory / "stocks.parquet")
    return pd.to_datetime(stocks.set_index("instrument")["ipo_date"])


def float_shares(daily: pd.DataFrame) -> pd.Series:
    """流通股本 = 成交量（股）/ 换手率（%）× 100。停牌（换手率为 0 或缺失）的日子沿用前一个有效值。"""
    shares = (daily["volume"] / (daily["turn"] / 100)).where(daily["turn"] > 0)
    return shares.groupby(level="instrument").ffill()


def load_index(symbol: str, start: str, end: str, directory: Path = FULL_A_DIR) -> pd.Series:
    check_range(start, end)
    df = pd.read_parquet(directory / "index.parquet")
    df = df[df["symbol"] == symbol].assign(date=lambda x: pd.to_datetime(x["date"]))
    s = df.set_index("date")["close"].sort_index()
    return s.loc[pd.Timestamp(start):pd.Timestamp(end)].rename(symbol)
