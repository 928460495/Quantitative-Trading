"""在训练段（2010–2018）的全 A 主板上算出因子库的全部因子，做体检，写出 reports/factor_library/。

用法：
    python scripts/build_factor_library.py               # 全部 500 个
    python scripts/build_factor_library.py --save        # 另外把因子值存到 data/cache/factors/（不进仓库；每个约 20 MB，全部约 11 GB）
    python scripts/build_factor_library.py --only wq001 gtja003

**只体检、不检验**：覆盖率、无穷值比例、横截面全相同的天数、每天不同取值的比例、计算耗时。
不算 IC、不看收益、不记试验日志（docs/DECISIONS.md 2026-10-30：先全部纳入，后筛选）。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qt.config import FULL_A_DIR, segment  # noqa: E402
from qt.formula import evaluate  # noqa: E402
from qt.library import HELD, FactorDef, all_factors  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
CACHE_DIR = ROOT / "data" / "cache" / "factors"
OUT_DIR = ROOT / "reports" / "factor_library"
WARMUP_START = "2009-01-05"  # 比训练段早一年，给 250 日窗口预热；baostock 日线从 2009 年开始
QLIB_BATCH = 20


def health(f: pd.DataFrame, close: pd.DataFrame) -> dict[str, float]:
    """只在有价格的格子上统计。"""
    vals = f.reindex_like(close).where(close.notna())
    n = int(close.notna().to_numpy().sum())
    arr = vals.to_numpy(dtype=float)
    finite = np.isfinite(arr)
    clean = vals.where(finite)
    cnt = clean.count(axis=1)
    std = clean.std(axis=1)
    distinct = (clean.nunique(axis=1) / cnt)[cnt > 0]
    return {
        "coverage": round(float(finite.sum() / n), 4),
        "inf_share": round(float(np.isinf(arr).sum() / n), 4),
        "constant_days": round(float(((std == 0) & (cnt >= 2)).sum() / len(close)), 4),
        "distinct_share": round(float(distinct.median()), 4) if len(distinct) else np.nan,
    }


_CTX: dict = {}


def _one(name: str) -> tuple[str, dict, pd.DataFrame | None]:
    """在子进程里算一个因子。只传名字：函数因子是闭包，不能序列化，子进程 fork 时已经继承了登记表。"""
    panel, start, save = _CTX["panel"], _CTX["start"], _CTX["save"]
    f = _CTX["factors"][name]
    t0 = time.time()
    try:
        out = evaluate(f.formula, panel, f.dialect) if f.formula else f.func(panel)
        err = ""
    except Exception as e:  # 体检要把失败的因子列出来，不中断
        out, err = None, f"{type(e).__name__}: {e}"
    secs = round(time.time() - t0, 2)
    close = panel["close"].loc[start:]
    if out is None:
        return f.name, {"seconds": secs, "error": err}, None
    out = out.loc[start:]
    stats = health(out, close) | {"seconds": secs, "error": ""}
    if save:
        out.astype("float32").to_parquet(CACHE_DIR / f"{f.name}.parquet", compression="zstd")
    return f.name, stats, None


def _qlib_batches(factors: list[FactorDef], codes: list[str], start: str, end: str, close: pd.DataFrame, save: bool):
    from qt.panel import _qlib_wide

    rows = {}
    for i in range(0, len(factors), QLIB_BATCH):
        batch = factors[i:i + QLIB_BATCH]
        t0 = time.time()
        frames = _qlib_wide(codes, {f.name: f.formula for f in batch}, WARMUP_START, end)
        secs = round((time.time() - t0) / len(batch), 2)
        for f in batch:
            out = frames[f.name].reindex(index=close.index, columns=close.columns).loc[start:]
            rows[f.name] = health(out, close.loc[start:]) | {"seconds": secs, "error": ""}
            if save:
                out.astype("float32").to_parquet(CACHE_DIR / f"{f.name}.parquet", compression="zstd")
        print(f"  Qlib 表达式 {i + len(batch)}/{len(factors)}", flush=True)
    return rows


def report_only() -> None:
    """不重算，只用已有的 health.csv 重写 README。"""
    table = pd.read_csv(OUT_DIR / "health.csv", index_col="factor", dtype={"error": str})
    table["error"] = table["error"].fillna("")
    _write_report(table, argparse.Namespace(segment="train"), len(main_board_codes()))


def main_board_codes() -> list[str]:
    from qt.panel import main_board_instruments

    return main_board_instruments(*segment("train"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", default="train", choices=["train", "valid"])
    ap.add_argument("--only", nargs="*", help="只算这几个因子")
    ap.add_argument("--save", action="store_true", help="因子值存到 data/cache/factors/")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--report-only", action="store_true", help="不重算，只用 health.csv 重写 README")
    args = ap.parse_args()
    if args.report_only:
        report_only()
        return

    from qt.data import init_qlib
    from qt.panel import load_panel, main_board_instruments

    init_qlib(kernels=args.jobs)
    start, end = segment(args.segment)
    factors = [f for f in all_factors() if not args.only or f.name in args.only]
    if args.save:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    codes = main_board_instruments(start, end)
    print(f"读取 {len(codes)} 只主板股票的宽表（{WARMUP_START} ~ {end}）...", flush=True)
    t0 = time.time()
    panel = load_panel(WARMUP_START, end, instruments=codes)
    print(f"  用时 {time.time() - t0:.0f} 秒", flush=True)

    rows = {}
    qlib = [f for f in factors if f.dialect == "qlib"]
    if qlib:
        rows |= _qlib_batches(qlib, codes, start, end, panel["close"], args.save)
    rest = [f for f in factors if f.dialect != "qlib"]
    _CTX.update(panel=panel, start=start, save=args.save, factors={f.name: f for f in rest})
    with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context("fork")) as pool:
        for k, (name, stats, _) in enumerate(pool.map(_one, [f.name for f in rest], chunksize=4), 1):
            rows[name] = stats
            if k % 25 == 0:
                print(f"  其他因子 {k}/{len(rest)}", flush=True)

    meta = {f.name: f for f in factors}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.insert(0, "family", [meta[n].family for n in table.index])
    table.insert(1, "category", [meta[n].category for n in table.index])
    table.index.name = "factor"
    table = table.loc[[f.name for f in factors]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "health.csv")
    _write_report(table, args, len(codes))
    print(table.groupby("family")[["coverage", "seconds"]].median())


def _flags(r: pd.Series) -> str:
    out = []
    if r.get("error"):
        out.append("计算失败")
    if r.get("coverage", 1) < 0.5:
        out.append("覆盖率 < 50%")
    if r.get("constant_days", 0) > 0.05:
        out.append("超过 5% 的天横截面全相同")
    if r.get("inf_share", 0) > 0:
        out.append("有无穷值")
    if r.get("distinct_share", 1) < 0.1:
        out.append("取值很少（不到一成）")
    return "、".join(out)


def _write_report(table: pd.DataFrame, args, n_codes: int) -> None:
    table = table.assign(flags=table.apply(_flags, axis=1))
    by_family = table.groupby("family").agg(因子数=("coverage", "size"), 覆盖率中位数=("coverage", "median"),
                                            有问题的=("flags", lambda s: int((s != "").sum())),
                                            平均耗时秒=("seconds", "mean"))
    by_cat = table.groupby("category").size().rename("因子数")
    problems = table[table["flags"] != ""][["family", "category", "coverage", "constant_days", "distinct_share", "flags"]]
    version = (FULL_A_DIR / ".version").read_text().strip() if (FULL_A_DIR / ".version").exists() else "未知"
    lines = [
        "# 因子库体检",
        "",
        f"- 样本：{args.segment} 段的全 A 主板（{n_codes} 只，含退市）；宽表从 {WARMUP_START} 开始算，给长窗口预热",
        f"- 全 A 补充数据版本：`{version}`",
        f"- 因子：{len(table)} 个；暂缓纳入 {len(HELD)} 个（原因见 `qt/formulas/*.tsv`）",
        "- **只体检，不检验**：没有算 IC、没有看收益、没有记试验日志",
        "- 覆盖率 = 有价格的格子里因子有值的比例；横截面全相同 = 当天所有股票取值一样",
        "- 时间序列窗口里最多允许缺 1/4（停牌），用有值的日子算（`qt/ops.py`）",
        "- \"取值很少\"多是天然离散的因子（上涨天数、高低点位置、比较大小得到的 0/1），不是错误；筛选时注意它们分组会不均匀",
        "- 用到 baostock 日线（换手率、估值、流通市值）的因子覆盖率偏低：这一版补充数据的日线只有部分股票，补齐后重跑",
        "",
        "## 按来源",
        "",
        by_family.to_markdown(floatfmt=".3f"),
        "",
        "## 按大类",
        "",
        by_cat.to_frame().to_markdown(),
        "",
        f"## 需要留意的因子（{len(problems)} 个）",
        "",
        problems.to_markdown(floatfmt=".3f") if len(problems) else "无",
        "",
        "全部因子的体检结果在 `health.csv`。",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
