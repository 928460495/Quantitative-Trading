"""全 A 数据采集：在本地电脑上运行（云端连不上 baostock 和东方财富），结果打包后上传到 GitHub Release。

采集内容（docs/STRATEGY.md 第 10 节）：
- baostock 日线：前收盘价、换手率、交易状态、ST 标记、估值（市盈率、市净率、市销率、市现率）
- 东方财富数据中心：利润表、资产负债表、现金流量表、业绩报表，每个报告期一次取全市场
- 中证指数官网：沪深300 全收益指数、中证全指

原始结果缓存在 data/raw/full_a/（不进仓库）：
    stocks.csv                 股票列表（含退市股）
    daily/<instrument>.csv     每只股票一个文件
    reports/<表名>/<报告期>.csv
    index/<指数代码>.csv
中途断了直接重跑，已经下载过的会跳过。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "full_a"

DAILY_FIELDS = "date,code,preclose,close,volume,amount,turn,tradestatus,isST,peTTM,pbMRQ,psTTM,pcfNcfTTM"

_EXCHANGE_BY_FIRST_DIGIT = {"6": "SH", "9": "SH", "0": "SZ", "2": "SZ", "3": "SZ", "4": "BJ", "8": "BJ"}


def to_instrument(code: str) -> str:
    """各种写法统一成 Qlib 的写法：sh.600000、600000、SH600000 → SH600000。"""
    code = code.strip()
    if "." in code:
        exchange, digits = code.split(".")
        return exchange.upper() + digits
    if code[:2].upper() in ("SH", "SZ", "BJ"):
        return code[:2].upper() + code[2:]
    return _EXCHANGE_BY_FIRST_DIGIT[code[0]] + code


def to_bs_code(instrument: str) -> str:
    return instrument[:2].lower() + "." + instrument[2:]


def _read_all(rs) -> pd.DataFrame:
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def stock_list(bs) -> pd.DataFrame:
    """全部股票（含退市），排除指数、基金和北交所。"""
    raw = _read_all(bs.query_stock_basic())
    raw = raw[raw["type"] == "1"]
    out = pd.DataFrame({
        "instrument": raw["code"].map(to_instrument),
        "name": raw["code_name"],
        "ipo_date": raw["ipoDate"],
        "out_date": raw["outDate"],
    })
    return out[~out["instrument"].str.startswith("BJ")].reset_index(drop=True)


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _fetch_daily(bs, instrument: str, start: str, end: str | None) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(to_bs_code(instrument), DAILY_FIELDS, start_date=start, end_date=end,
                                      frequency="d", adjustflag="3")
    if rs.error_code != "0":
        bs.login()  # 常见原因是登录过期
        raise ConnectionError(f"baostock 返回 {rs.error_code} {rs.error_msg}")
    return _read_all(rs)


def collect_daily(bs, instruments: list[str], out_dir: Path, start: str = "2009-01-01", end: str | None = None,
                  retries: int = 3, pause: float = 0.0) -> dict[str, str]:
    """每只股票一个文件，已存在就跳过；没有数据的写一个只有表头的文件，重跑时也跳过。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    for instrument in instruments:
        path = out_dir / f"{instrument}.csv"
        if path.exists():
            status[instrument] = "ok" if len(pd.read_csv(path, nrows=1)) else "empty"
            continue
        for attempt in range(1, retries + 1):
            try:
                df = _fetch_daily(bs, instrument, start, end)
                break
            except Exception as e:  # 网络中断、登录过期、返回格式异常都重试
                if attempt == retries:
                    status[instrument] = f"failed: {type(e).__name__}: {e}"
                    df = None
                else:
                    time.sleep(pause * 2**attempt)
        if df is None:
            continue
        if df.empty:
            df = pd.DataFrame(columns=DAILY_FIELDS.split(","))
        _write_atomic(df, path)
        status[instrument] = "ok" if len(df) else "empty"
        time.sleep(pause)
    return status
