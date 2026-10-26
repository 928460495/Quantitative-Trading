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


EASTMONEY_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_A_SHARE_FILTER = '(SECURITY_TYPE_CODE in ("058001001","058001008"))(TRADE_MARKET_CODE!="069001017")'
# 表名 → （接口里的报表名，报告期字段名）；三张 DMSK 表的筛选条件沿用 AkShare
REPORTS: dict[str, tuple[str, str]] = {
    "income": ("RPT_DMSK_FN_INCOME", "REPORT_DATE"),
    "balance": ("RPT_DMSK_FN_BALANCE", "REPORT_DATE"),
    "cashflow": ("RPT_DMSK_FN_CASHFLOW", "REPORT_DATE"),
    "yjbb": ("RPT_LICO_FN_CPD", "REPORTDATE"),
}


def report_dates(start: str = "2008-03-31", end: str | None = None) -> list[str]:
    """start 到 end 之间的季末日期。"""
    quarter_ends = pd.date_range(start, end or pd.Timestamp.today().normalize(), freq="QE")
    return [d.strftime("%Y-%m-%d") for d in quarter_ends]


class EmptyResult(Exception):
    """接口说这个报告期没有数据：不是网络问题，不重试。"""


def _fetch_report(session, name: str, report_date: str) -> pd.DataFrame:
    report, date_field = REPORTS[name]
    flt = f"({date_field}='{report_date}')"
    if name != "yjbb":
        flt = _A_SHARE_FILTER + flt
    params = {"sortColumns": "SECURITY_CODE", "sortTypes": "1", "pageSize": "500", "pageNumber": "1",
              "reportName": report, "columns": "ALL", "filter": flt}
    pages, frames, page = 1, [], 1
    while page <= pages:
        payload = session.get(EASTMONEY_URL, params=params | {"pageNumber": str(page)}, timeout=30).json()
        result = payload.get("result")
        if result is None:
            if page == 1:
                raise EmptyResult(payload.get("message", ""))
            raise ConnectionError(f"第 {page} 页没有数据：{payload.get('message', '')}")
        pages = int(result["pages"])
        frames.append(pd.DataFrame(result["data"]))
        page += 1
    return pd.concat(frames, ignore_index=True)


def collect_reports(session, names: list[str], dates: list[str], out_dir: Path, retries: int = 3,
                    pause: float = 0.5) -> dict[str, str]:
    """每个（表，报告期）一个文件，保留接口返回的全部原始列；已存在就跳过。"""
    status: dict[str, str] = {}
    for name in names:
        (out_dir / name).mkdir(parents=True, exist_ok=True)
        for report_date in dates:
            key, path = f"{name}/{report_date}", out_dir / name / f"{report_date}.csv"
            if path.exists():
                status[key] = "ok"
                continue
            for attempt in range(1, retries + 1):
                try:
                    _write_atomic(_fetch_report(session, name, report_date), path)
                    status[key] = "ok"
                    break
                except EmptyResult:
                    status[key] = "empty"
                    break
                except Exception as e:
                    if attempt == retries:
                        status[key] = f"failed: {type(e).__name__}: {e}"
                    else:
                        time.sleep(pause * 2**attempt)
            time.sleep(pause)
    return status


INDEXES = ("H00300", "000985")  # 沪深300 全收益、中证全指


def collect_indexes(ak, out_dir: Path, start: str = "20090101", end: str | None = None) -> dict[str, str]:
    """指数日收盘价，存 index/<代码>.csv（列 date、close）。历史比 start 短时照实记录起始日期。"""
    (out_dir / "index").mkdir(parents=True, exist_ok=True)
    end = end or pd.Timestamp.today().strftime("%Y%m%d")
    status: dict[str, str] = {}
    for symbol in INDEXES:
        try:
            raw = ak.stock_zh_index_hist_csindex(symbol=symbol, start_date=start, end_date=end)
            df = pd.DataFrame({"date": pd.to_datetime(raw["日期"]).dt.strftime("%Y-%m-%d"), "close": raw["收盘"]})
            df = df.sort_values("date")
            _write_atomic(df, out_dir / "index" / f"{symbol}.csv")
            status[symbol] = f"ok {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}"
        except Exception as e:
            status[symbol] = f"failed: {type(e).__name__}: {e}"
    return status
