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

用法（先 pip install -r requirements-collect.txt）：
    python scripts/collect_full_a.py               # 全部：股票列表 → 日线 → 财务 → 指数 → 打包
    python scripts/collect_full_a.py --part daily  # 只跑某一步（list、daily、reports、index、pack）

日线要逐只股票请求，约 5500 只，大约 1–3 小时；财务按报告期请求，约 300 次。
跑完会在 data/ 下生成 full_a_<日期>.tar，按 data/external/README.md 上传到 GitHub Release。
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
                  retries: int = 3, pause: float = 0.0, progress_every: int = 50) -> dict[str, str]:
    """每只股票一个文件，已存在就跳过；没有数据的写一个只有表头的文件，重跑时也跳过。每 progress_every 只打印一次进度。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    started = time.time()
    for n, instrument in enumerate(instruments, 1):
        if n % progress_every == 0:
            failed = sum(v.startswith("failed") for v in status.values())
            print(f"  日线 {n}/{len(instruments)}，失败 {failed}，已用 {(time.time() - started) / 60:.0f} 分钟", flush=True)
        path = out_dir / f"{instrument}.csv"
        if path.exists():
            status[instrument] = "ok" if len(pd.read_csv(path, nrows=1)) else "empty"
            continue
        for attempt in range(1, retries + 1):
            try:
                df = _fetch_daily(bs, instrument, start, end)
                break
            except Exception as e:  # 网络中断、超时、登录过期、返回格式异常都重试
                if attempt == retries:
                    status[instrument] = f"failed: {type(e).__name__}: {e}"
                    df = None
                else:
                    time.sleep(pause * 2**attempt)
                    if not isinstance(e, ConnectionError) or "baostock 返回" not in str(e):
                        try:  # 连接断了要重新登录才能继续（返回错误码的情况 _fetch_daily 已经登录过）
                            bs.login()
                        except Exception:
                            pass
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


_FLOAT_COLS = ["preclose", "close", "volume", "amount", "turn", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
_FLAG_COLS = ["tradestatus", "isST"]


def _read_daily(path: Path) -> pd.DataFrame:
    """一只股票的日线，读的时候就转成紧凑的类型：全 A 约 1300 万行，全按字符串读要占十几 GB 内存。"""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = pd.DataFrame({"instrument": path.stem, "date": pd.to_datetime(df["date"])})
    for c in _FLOAT_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for c in _FLAG_COLS:  # 缺失按 0：不交易、不是 ST
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")
    return out


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """读本地缓存：股票列表、全部日线（统一 dtype）、各张财务表（加上 instrument、report_date 两列）。"""
    stocks = pd.read_csv(raw_dir / "stocks.csv", dtype=str, keep_default_na=False)
    frames = [_read_daily(p) for p in sorted((raw_dir / "daily").glob("*.csv"))]
    daily = pd.concat([f for f in frames if len(f)], ignore_index=True)
    daily["instrument"] = daily["instrument"].astype("category")

    reports = {}
    for name in REPORTS:
        files = sorted((raw_dir / "reports" / name).glob("*.csv"))
        if not files:
            continue
        parts = []
        for p in files:
            df = pd.read_csv(p, dtype={"SECURITY_CODE": str})
            df.insert(0, "report_date", p.stem)
            df.insert(0, "instrument", df["SECURITY_CODE"].map(to_instrument))
            parts.append(df)
        reports[name] = pd.concat(parts, ignore_index=True)
    return stocks, daily, reports


def coverage(stocks: pd.DataFrame, daily: pd.DataFrame, reports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """每年：当年上市过的股票数，其中有日线的、每张财务表有当年报告期的。退市股没有数据会在这里露出来。"""
    ipo = pd.to_datetime(stocks["ipo_date"], errors="coerce")
    out = pd.to_datetime(stocks["out_date"].replace("", None), errors="coerce")
    years = range(daily["date"].dt.year.min(), daily["date"].dt.year.max() + 1)
    rows = {}
    for y in years:
        listed = (ipo <= pd.Timestamp(y, 12, 31)) & (out.isna() | (out >= pd.Timestamp(y, 1, 1)))
        row = {"listed": int(listed.sum()),
               "daily": daily.loc[daily["date"].dt.year == y, "instrument"].nunique()}
        for name, df in reports.items():
            row[name] = df.loc[df["report_date"].str[:4] == str(y), "instrument"].nunique()
        rows[y] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def _status_summary(status: dict[str, dict[str, str]]) -> tuple[dict, dict]:
    counts, failures = {}, {}
    for part, items in status.items():
        kinds = pd.Series([v.split(":")[0].split(" ")[0] for v in items.values()]).value_counts()
        counts[part] = {k: int(v) for k, v in kinds.items()}
        failures[part] = {k: v for k, v in items.items() if v.startswith("failed")}
    return counts, failures


def pack(raw_dir: Path, out_dir: Path, collected_on: str) -> Path:
    """把本地缓存打成一个 tar：parquet 已经压缩过，tar 本身不再压缩。"""
    import json
    import shutil
    import tarfile

    stocks, daily, reports = load_raw(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"full_a_{collected_on}.tar"
    tmp = path.with_name(f".{path.name}.tmp")
    staging = out_dir / f".staging_{collected_on}"
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "reports").mkdir(parents=True)
    try:
        stocks.to_parquet(staging / "stocks.parquet", index=False)
        daily.to_parquet(staging / "daily.parquet", index=False, compression="zstd")
        for name, df in reports.items():
            df.to_parquet(staging / "reports" / f"{name}.parquet", index=False, compression="zstd")
        index_files = sorted((raw_dir / "index").glob("*.csv"))
        index = pd.concat([pd.read_csv(p).assign(symbol=p.stem) for p in index_files], ignore_index=True)
        index[["symbol", "date", "close"]].to_parquet(staging / "index.parquet", index=False)

        status_file = raw_dir / "status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        counts, failures = _status_summary(status)
        cov = coverage(stocks, daily, reports)
        meta = {"collected_on": collected_on, "status": counts, "failures": failures,
                "coverage": {str(y): {k: int(v) for k, v in row.items()} for y, row in cov.iterrows()},
                "rows": {"daily": len(daily), **{k: len(v) for k, v in reports.items()}}}
        (staging / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        with tarfile.open(tmp, "w") as tar:
            for p in sorted(staging.rglob("*")):
                if p.is_file():
                    tar.add(p, arcname=str(p.relative_to(staging)))
        os.replace(tmp, path)
        return path
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        tmp.unlink(missing_ok=True)


def _save_status(raw_dir: Path, part: str, status: dict[str, str]) -> None:
    import json

    path = raw_dir / "status.json"
    allstatus = json.loads(path.read_text()) if path.exists() else {}
    allstatus[part] = allstatus.get(part, {}) | status
    path.write_text(json.dumps(allstatus, ensure_ascii=False, indent=2))
    counts, _ = _status_summary({part: allstatus[part]})
    print(f"{part}：{counts[part]}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all", choices=["all", "list", "daily", "reports", "index", "pack"])
    ap.add_argument("--start", default="2009-01-01", help="日线起始日期")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--pause", type=float, default=0.0, help="每次请求之间停几秒")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = ap.parse_args()
    raw_dir: Path = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    parts = ["list", "daily", "reports", "index", "pack"] if args.part == "all" else [args.part]

    bs = None
    if {"list", "daily"} & set(parts):
        import socket

        import baostock

        socket.setdefaulttimeout(60)  # baostock 不设超时，服务器不响应时会一直卡住；超时后按失败重试
        bs = baostock
        lg = bs.login()
        if lg.error_code != "0":
            raise SystemExit(f"baostock 登录失败：{lg.error_code} {lg.error_msg}")

    if "list" in parts:
        stocks = stock_list(bs)
        _write_atomic(stocks, raw_dir / "stocks.csv")
        print(f"股票列表：{len(stocks)} 只（含退市）")
    if "daily" in parts:
        stocks = pd.read_csv(raw_dir / "stocks.csv", dtype=str, keep_default_na=False)
        gone_early = pd.to_datetime(stocks["out_date"].replace("", None), errors="coerce") < pd.Timestamp(args.start)
        todo = list(stocks.loc[~gone_early, "instrument"])
        print(f"日线：{len(todo)} 只（{args.start} 前已退市的 {int(gone_early.sum())} 只不请求）")
        _save_status(raw_dir, "daily", collect_daily(bs, todo, raw_dir / "daily", start=args.start,
                                                     retries=args.retries, pause=args.pause))
    if bs is not None:
        bs.logout()
    if "reports" in parts:
        import requests

        with requests.Session() as session:
            _save_status(raw_dir, "reports", collect_reports(session, list(REPORTS), report_dates(),
                                                             raw_dir / "reports", retries=args.retries,
                                                             pause=max(args.pause, 0.5)))
    if "index" in parts:
        import akshare

        _save_status(raw_dir, "index", collect_indexes(akshare, raw_dir))
    if "pack" in parts:
        path = pack(raw_dir, ROOT / "data", collected_on=pd.Timestamp.today().strftime("%Y%m%d"))
        size = path.stat().st_size / 2**20
        import json
        import tarfile

        with tarfile.open(path) as tar:
            meta = json.load(tar.extractfile("meta.json"))
        print("每年覆盖情况（上市股票数 / 有日线 / 各张财务表有数据的股票数）：")
        print(pd.DataFrame.from_dict(meta["coverage"], orient="index").to_string())
        print(f"打包完成：{path}（{size:.0f} MB）")
        print("下一步：按 data/external/README.md 把这个文件上传到 GitHub Release，然后把 tag 告诉 Claude")


if __name__ == "__main__":
    main()
