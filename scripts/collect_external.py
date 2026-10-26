"""在本地采集申万行业变动历史和巨潮股本变动，写到 data/external/。

需要能访问申万宏源研究（swsresearch.com）和巨潮资讯（webapi.cninfo.com.cn）。另外装 AkShare：
    pip install -r requirements-collect.txt

用法：
    python scripts/collect_external.py                   # 沪深300 历史上的所有成分股
    python scripts/collect_external.py --market csi800   # 以后扩股票池时用
    python scripts/collect_external.py --only industry   # 只重下行业
    python scripts/collect_external.py --only industry --industry-file ~/Downloads/StockClassifyUse_stock.xls
                                                         # 行业表下载失败时，用浏览器手动下载后读本地文件

股本要逐只股票请求，沪深300 历史成分约 950 只，按默认间隔大约要 20–30 分钟。中途断了直接重跑，
已经下载过的股票会跳过（缓存在 data/raw/share_changes/，不进仓库）。失败的股票记在
data/external/collect_meta.json 里，重跑会自动补。

跑完把 data/external/ 下的三个文件提交到仓库。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qt.config import DATA_DIR  # noqa: E402
from qt.external import (  # noqa: E402
    EXTERNAL_DIR,
    SW_INDUSTRY_URL,
    INDUSTRY_FILE,
    SHARES_FILE,
    industry_level1,
    read_sw_industry_file,
    tidy_industry,
    tidy_share_changes,
    to_symbol,
)

CACHE_DIR = ROOT / "data" / "raw" / "share_changes"
META_FILE = "collect_meta.json"


def market_symbols(market: str) -> list[str]:
    path = DATA_DIR / "instruments" / f"{market}.txt"
    if not path.exists():
        sys.exit(f"找不到 {path}，先运行 python scripts/download_data.py")
    codes = pd.read_csv(path, sep="\t", header=None, names=["code", "start", "end"])["code"].unique()
    return sorted({to_symbol(c) for c in codes if not c.startswith("BJ")})


def _retry(fn, label: str, retries: int, pause: float):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:  # 网络错误、接口返回异常格式等都重试
            if attempt == retries:
                raise
            wait = pause * 2**attempt
            print(f"  {label} 第 {attempt} 次失败（{type(e).__name__}: {e}），{wait:.0f} 秒后重试")
            time.sleep(wait)
    raise AssertionError("unreachable")


def collect_industry(ak, retries: int = 3, pause: float = 1.0, file: Path | None = None) -> pd.DataFrame | None:
    """下载失败时返回 None，不中断后面的股本采集。"""
    if file is not None:
        print(f"读取本地申万行业表 {file} ...")
        raw = read_sw_industry_file(file)
    else:
        print("下载申万行业变动历史 ...")
        try:
            raw = _retry(ak.stock_industry_clf_hist_sw, "申万行业", retries, pause)
        except Exception as e:
            print(f"申万行业下载失败：{type(e).__name__}: {e}")
            print("可以用浏览器打开下面的地址手动下载，再用 --industry-file 指定下载好的文件：")
            print(f"  {SW_INDUSTRY_URL}")
            print("  python scripts/collect_external.py --only industry --industry-file <下载好的文件路径>")
            return None
    df = tidy_industry(raw)
    df.to_csv(EXTERNAL_DIR / INDUSTRY_FILE, index=False)
    level1 = df["industry_code"].map(industry_level1)
    print(f"  {len(df)} 条记录，{df['symbol'].nunique()} 只股票，"
          f"{df['start_date'].min().date()} ~ {df['start_date'].max().date()}")
    print(f"  行业代码长度分布：{df['industry_code'].str.len().value_counts().to_dict()}")
    print(f"  按前两位归类得到 {level1.nunique()} 个一级行业（申万 2021 版应为 31 个）：{sorted(level1.unique())}")
    return df


def _fetch_one(ak, symbol: str, end: str, retries: int, pause: float) -> pd.DataFrame:
    return _retry(lambda: ak.stock_share_change_cninfo(symbol=symbol, start_date="19900101", end_date=end), symbol, retries, pause)


def collect_shares(ak, symbols: list[str], pause: float, retries: int) -> tuple[pd.DataFrame, dict[str, str]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    end = date.today().strftime("%Y%m%d")
    failed: dict[str, str] = {}
    todo = [s for s in symbols if not (CACHE_DIR / f"{s}.csv").exists()]
    print(f"下载股本变动：共 {len(symbols)} 只，已缓存 {len(symbols) - len(todo)} 只，本次下载 {len(todo)} 只")
    for i, symbol in enumerate(todo, 1):
        try:
            raw = _fetch_one(ak, symbol, end, retries, pause)
            tidy_share_changes(raw, symbol).to_csv(CACHE_DIR / f"{symbol}.csv", index=False)
        except Exception as e:
            failed[symbol] = f"{type(e).__name__}: {e}"
        if i % 50 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}，失败 {len(failed)}")
        time.sleep(pause)

    frames = [pd.read_csv(CACHE_DIR / f"{s}.csv", dtype={"symbol": str}) for s in symbols if (CACHE_DIR / f"{s}.csv").exists()]
    shares = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    shares.to_csv(EXTERNAL_DIR / SHARES_FILE, index=False)
    empty = sorted(set(symbols) - set(shares["symbol"]) - set(failed)) if not shares.empty else []
    print(f"  共 {len(shares)} 条记录，覆盖 {shares['symbol'].nunique() if not shares.empty else 0} 只；"
          f"失败 {len(failed)} 只；返回为空 {len(empty)} 只")
    return shares, failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default="csi300", help="按哪个股票池的历史成分采集股本，默认 csi300")
    ap.add_argument("--only", choices=["industry", "shares"], help="只采集其中一项")
    ap.add_argument("--pause", type=float, default=1.0, help="两次请求之间等几秒，默认 1")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--industry-file", type=Path, help="手动下载的申万行业表（.xls），不联网下载")
    args = ap.parse_args()

    try:
        import akshare as ak
    except ImportError:
        sys.exit("没有安装 AkShare：pip install -r requirements-collect.txt")

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = EXTERNAL_DIR / META_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta |= {"akshare_version": ak.__version__, "market": args.market}

    if args.only in (None, "industry"):
        file = args.industry_file.expanduser() if args.industry_file else None
        ind = collect_industry(ak, args.retries, args.pause, file)
        if ind is not None:
            meta["industry"] = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": len(ind),
                                "source": file.name if file else "akshare"}
    if args.only in (None, "shares"):
        symbols = market_symbols(args.market)
        shares, failed = collect_shares(ak, symbols, args.pause, args.retries)
        meta["shares"] = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "symbols": len(symbols),
            "rows": len(shares),
            "failed": failed,
        }
        if failed:
            print(f"有 {len(failed)} 只失败，重跑这个脚本会只补这些")

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    if not (EXTERNAL_DIR / INDUSTRY_FILE).exists():
        print(f"注意：还没有 {INDUSTRY_FILE}，行业表需要按上面的提示手动下载后再跑一次 --only industry")
    print(f"完成，结果在 {EXTERNAL_DIR}。请把 {INDUSTRY_FILE}、{SHARES_FILE}、{META_FILE} 提交到仓库")


if __name__ == "__main__":
    main()
