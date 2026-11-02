"""全 A 主板单因子检验与筛选（计划 3b，规则见 docs/DECISIONS.md 2026-11-01，运行前写定）。

用法（在有数据的电脑上）：
    python scripts/run_full_a_screen.py --jobs 4                 # 全部 500 个，约 1–2 小时
    python scripts/run_full_a_screen.py --factors wq001 rev20    # 只跑几个（调试用，不写最终名单、不记试验日志）

输出：
    reports/single_factor/full_a_train_neutral/summary.csv、README.md   每个因子的检验结果
    reports/screening/full_a_train_neutral/selected.csv、README.md    每类选中的因子、去掉的原因、选中因子的相关性
    reports/trials.jsonl                                              每个因子一条试验记录

只用训练段（2010–2018）；验证段、测试段不碰。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qt.library import FactorDef  # noqa: E402
from qt.screen_wide import (  # noqa: E402
    ACCOUNT_KEYS, ScreenContext, corr_matrix, evaluate_factor_wide, select_by_category,
)
from qt.trials import TRIALS_FILE, count_unique, log_trial  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
SINGLE_DIR = ROOT / "reports" / "single_factor" / "full_a_train_neutral"
SELECT_DIR = ROOT / "reports" / "screening" / "full_a_train_neutral"
CACHE_ROOT = ROOT / "data" / "cache" / "screen_full_a"   # 不进仓库
WARMUP_START = "2009-01-05"
CATEGORY_ORDER = ["动量/反转", "价值", "规模", "低波动", "成长", "质量", "流动性", "短周期价量"]
QLIB_BATCH = 20

# 子进程通过 fork 继承：检验上下文、因子登记、怎么算因子
_CTX: dict = {}


def _evaluate(name: str, raw: pd.DataFrame | None) -> tuple[dict, pd.DataFrame | None]:
    f: FactorDef = _CTX["factors"][name]
    base = {"factor": name, "family": f.family, "category": f.category}
    try:
        if raw is None:
            raw = _CTX["compute"](f)
        row, sub = evaluate_factor_wide(name, raw, f.category, _CTX["ctx"])
        return base | row | {"error": ""}, sub
    except Exception as e:  # 一个因子出错不中断全部；报告里列出来
        return base | {"passed": False, "fail_reason": "计算出错", "error": f"{type(e).__name__}: {e}"}, None


def _evaluate_star(args):
    return _evaluate(*args)


def _save(cache_dir: Path, row: dict, sub: pd.DataFrame | None) -> None:
    """一个因子算完就存下来：中途被打断，重跑时跳过已经算完的。先写临时文件再改名，不会留下半个文件。"""
    name = row["factor"]
    if sub is not None:
        tmp = cache_dir / f".{name}.parquet"
        sub.to_parquet(tmp)
        tmp.replace(cache_dir / f"{name}.parquet")
    tmp = cache_dir / f".{name}.json"
    tmp.write_text(json.dumps(row, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(cache_dir / f"{name}.json")


def _load(cache_dir: Path, names: list[str]) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    rows, subs = [], {}
    for name in names:
        path = cache_dir / f"{name}.json"
        if not path.exists():
            continue
        rows.append(json.loads(path.read_text()))
        if (cache_dir / f"{name}.parquet").exists():
            subs[name] = pd.read_parquet(cache_dir / f"{name}.parquet")
    return rows, subs


def run_screen(ctx: ScreenContext, factors: list[FactorDef], compute: Callable[[FactorDef], pd.DataFrame],
               jobs: int = 1, parent_compute: Callable[[list[FactorDef]], dict] | None = None,
               on_result: Callable[[dict], None] | None = None, cache_dir: Path | None = None
               ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """逐个检验。compute 在子进程里算一个因子；parent_compute 在主进程里成批算（Qlib 表达式，不能在子进程里调）。

    cache_dir：每个因子算完就存进去；里面已有的因子直接读，不重算、不重记试验日志（出错的不存，重跑会再试）。
    返回 (summary：每个因子一行, 通过的因子每 5 天一次的调好方向的中性化值)。
    """
    _CTX.update(ctx=ctx, factors={f.name: f for f in factors}, compute=compute)
    rows, subs = ([], {}) if cache_dir is None else _load(cache_dir, [f.name for f in factors])
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if rows:
            print(f"  已经算完 {len(rows)} 个（{cache_dir}），跳过", flush=True)
    done = {r["factor"] for r in rows}
    todo = [f for f in factors if f.name not in done]
    in_parent = [f for f in todo if parent_compute and f.dialect == "qlib"]
    in_child = [f for f in todo if f not in in_parent]
    tasks = [[(f.name, None) for f in in_child]]
    t0 = time.time()

    def batches():
        for i in range(0, len(in_parent), QLIB_BATCH):
            batch = in_parent[i:i + QLIB_BATCH]
            raws = parent_compute(batch)
            yield [(f.name, raws[f.name]) for f in batch]

    def consume(results):
        for row, sub in results:
            rows.append(row)
            if sub is not None:
                subs[row["factor"]] = sub
            if cache_dir is not None and not row.get("error"):
                _save(cache_dir, row, sub)
            if on_result:
                on_result(row)
            if len(rows) % 25 == 0:
                print(f"  {len(rows)}/{len(factors)}，通过 {sum(r.get('passed', False) for r in rows)} 个，"
                      f"已用 {(time.time() - t0) / 60:.0f} 分钟", flush=True)

    if jobs > 1:
        with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as pool:
            for task in [*tasks, *batches()]:
                consume(pool.map(_evaluate_star, task, chunksize=1))
    else:
        for task in [*tasks, *batches()]:
            consume(map(_evaluate_star, task))

    order = {f.name: i for i, f in enumerate(factors)}
    summary = pd.DataFrame(rows).set_index("factor")
    summary = summary.loc[sorted(summary.index, key=order.get)]
    return summary, subs


def select(summary: pd.DataFrame, subs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """类内筛选和选中因子的相关性矩阵。"""
    passed = summary[summary["passed"].fillna(False).astype(bool)]
    table = passed[["category"]].assign(icir=passed["icir_5d"])
    selection = select_by_category(table, subs)
    chosen = [n for n in selection.index[selection["kept"].astype(bool)]]
    return selection, corr_matrix({n: subs[n] for n in chosen})


# ---------- 报告 ----------

def _fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(floatfmt=".3f") if len(df) else "（无）"


def write_reports(summary: pd.DataFrame, selection: pd.DataFrame, corr: pd.DataFrame, context: str,
                  single_dir: Path = SINGLE_DIR, select_dir: Path = SELECT_DIR) -> None:
    single_dir.mkdir(parents=True, exist_ok=True)
    select_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(single_dir / "summary.csv", float_format="%.6f")

    s = summary.copy()
    for c in ("pass_t", "pass_stable", "pass_cost", "passed"):
        s[c] = s.get(c, False).fillna(False).astype(bool)
    by_cat = s.groupby("category").agg(因子数=("passed", "size"), t值过3=("pass_t", "sum"),
                                       再加方向稳定=("pass_stable", lambda x: int((x & s.loc[x.index, "pass_t"]).sum())),
                                       三条都过=("passed", "sum"))
    by_cat = by_cat.reindex([c for c in CATEGORY_ORDER if c in by_cat.index])
    errors = s[s.get("error", "").fillna("") != ""] if "error" in s else s.iloc[:0]
    cols = ["family", "direction", "rank_ic_5d", "icir_5d", "t_nw_5d", "reversed_year_ratio", "excess_annual",
            "turnover", "fail_reason"]
    lines = ["# 全 A 主板单因子检验（训练段）", "", context, "",
             "规则见 `docs/DECISIONS.md` 2026-11-01。通过 = ① 5 天 RankIC 的 NW t 值 > 3 ② 方向反了的年份不超过一半 "
             "③ 按实际账户规则扣成本后跑赢能买股票等权（只对 ①② 都过的因子回测）。", "",
             "## 按大类", "", _fmt(by_cat), ""]
    if len(errors):
        lines += [f"## 计算出错（{len(errors)} 个）", "", _fmt(errors[["family", "category", "error"]]), ""]
    lines += ["## 每类按 ICIR 前 10", ""]
    for cat in by_cat.index:
        top = s[s["category"] == cat].sort_values("icir_5d", ascending=False).head(10)
        lines += [f"### {cat}", "", _fmt(top[[c for c in cols if c in top]]), ""]
    lines += ["全部结果在 `summary.csv`。"]
    (single_dir / "README.md").write_text("\n".join(lines) + "\n")

    kept = selection[selection["kept"].astype(bool)] if len(selection) else selection
    detail = kept.join(summary[["family", "direction", "rank_ic_5d", "t_nw_5d", "reversed_year_ratio",
                                *ACCOUNT_KEYS]], how="left") if len(kept) else kept
    detail.to_csv(select_dir / "selected.csv", float_format="%.6f")
    selection.to_csv(select_dir / "all_candidates.csv", float_format="%.6f")
    corr.to_csv(select_dir / "corr_selected.csv", float_format="%.4f")
    empty = [c for c in CATEGORY_ORDER if c not in set(kept.get("category", []))]
    lines = ["# 全 A 主板因子筛选（训练段）", "", context, "",
             "每类：通过三条检验的因子按 5 天 ICIR 从高到低，和已留下的因子平均相关性绝对值 > 0.7 的跳过，最多 3 个；"
             "选不出就空着。跨类不去重。", "", f"## 选中（{len(kept)} 个）", "",
             _fmt(detail[["category", "family", "icir", "rank_ic_5d", "t_nw_5d", "excess_annual", "turnover"]]
                  if len(detail) else detail), "",
             "## 空着的大类", "", "、".join(empty) if empty else "（无）", "",
             "## 每类的候选和去留", ""]
    for cat in [c for c in CATEGORY_ORDER if c in set(selection.get("category", []))]:
        part = selection[selection["category"] == cat]
        lines += [f"### {cat}", "", _fmt(part[["icir", "status", "corr"]]), ""]
    lines += ["## 选中因子之间的相关性（跨类，留给计划 4）", "", _fmt(corr), ""]
    (select_dir / "README.md").write_text("\n".join(lines) + "\n")


def trial_logger(data_version: str, path: Path = TRIALS_FILE) -> Callable[[dict], None]:
    def log(row: dict) -> None:
        if row.get("error"):
            return
        log_trial("single_factor",
                  {"factor": row["factor"], "family": row["family"], "category": row["category"],
                   "direction": int(row["direction"]), "market": "full_a_main", "segment": "train", "horizon": 5,
                   "neutralized": True, "size_neutralized": bool(row["size_neutralized"]),
                   "rule": "account_15_45", "benchmark": "buyable_ew", "data_version": data_version},
                  {k: row.get(k) for k in ("rank_ic_5d", "icir_5d", "t_nw_5d", "reversed_year_ratio",
                                           "excess_annual", "turnover", "passed", "fail_reason")},
                  path=path)
    return log


# ---------- 读真实数据 ----------

def load_context(start: str, end: str, jobs: int):
    from qt.config import DATA_DIR, FULL_A_DIR
    from qt.data import init_qlib
    from qt.full_a import listing_dates_full_a, load_daily
    from qt.panel import load_panel, main_board_instruments
    from qt.screen_wide import wide_masks

    init_qlib(kernels=jobs)
    codes = main_board_instruments(start, end)
    print(f"读取 {len(codes)} 只主板股票的宽表（{WARMUP_START} ~ {end}）...", flush=True)
    panel = load_panel(WARMUP_START, end, instruments=codes)
    daily = load_daily(["preclose", "close", "tradestatus", "isST"], WARMUP_START, end)
    can_buy, can_sell = wide_masks(panel["amount"], daily, listing_dates_full_a())
    ctx = ScreenContext(close=panel["close"], raw_close=panel["raw_close"], can_buy=can_buy, can_sell=can_sell,
                        industry=panel["industry_l1"], log_mcap=np.log(panel["float_mcap"].where(panel["float_mcap"] > 0)),
                        start=start)
    v1 = (DATA_DIR / ".qt_data_version").read_text().strip() if (DATA_DIR / ".qt_data_version").exists() else "unknown"
    v2 = (FULL_A_DIR / ".version").read_text().strip() if (FULL_A_DIR / ".version").exists() else "unknown"
    return panel, ctx, codes, f"{v1}+{v2}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=1, help="并行进程数")
    ap.add_argument("--factors", nargs="*", help="只跑这几个（调试用：不写最终名单、不记试验日志）")
    ap.add_argument("--fresh", action="store_true", help="清掉已经算完的中间结果，从头算")
    args = ap.parse_args()

    from qt.config import segment
    from qt.formula import evaluate
    from qt.library import all_factors
    from qt.panel import _qlib_wide

    start, end = segment("train")
    factors = [f for f in all_factors() if not args.factors or f.name in args.factors]
    t0 = time.time()
    panel, ctx, codes, version = load_context(start, end, args.jobs)
    buy_share = ctx.trim(ctx.can_buy).to_numpy().mean()
    print(f"  用时 {time.time() - t0:.0f} 秒；检验 {ctx.dates[0].date()} ~ {ctx.dates[-1].date()}，"
          f"{len(ctx.dates)} 天，能买的格子占 {buy_share:.1%}", flush=True)

    def compute(f: FactorDef) -> pd.DataFrame:
        return evaluate(f.formula, panel, f.dialect) if f.formula else f.func(panel)

    def parent_compute(batch: list[FactorDef]) -> dict:
        frames = _qlib_wide(codes, {f.name: f.formula for f in batch}, WARMUP_START, end)
        return {f.name: frames[f.name].reindex(index=ctx.close.index, columns=ctx.close.columns) for f in batch}

    debug = bool(args.factors)
    cache_dir = None if debug else CACHE_ROOT / version.replace("/", "_")
    if cache_dir is not None and args.fresh:
        shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"检验 {len(factors)} 个因子，{args.jobs} 个进程 ...", flush=True)
    try:
        summary, subs = run_screen(ctx, factors, compute, args.jobs, parent_compute,
                                   on_result=None if debug else trial_logger(version), cache_dir=cache_dir)
    except BrokenProcessPool:
        sys.exit(f"有一个子进程被系统强制结束了，多半是内存不够。已经算完的保存在 {cache_dir}，"
                 f"把 --jobs 改小（比如 {max(args.jobs // 2, 1)}）重跑同一条命令，会接着算剩下的。")
    context = (f"样本：全 A 主板 {len(codes)} 只（含退市），训练段 {ctx.dates[0].date()} ~ {ctx.dates[-1].date()}"
               f"（段尾去掉 {ctx.tail} 个交易日），{len(ctx.dates)} 个交易日；能买的格子占 {buy_share:.1%}；"
               f"数据版本 `{version}`")
    if debug:
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(summary[["category", "direction", "rank_ic_5d", "icir_5d", "t_nw_5d", "reversed_year_ratio",
                           "fail_reason"]].round(4).to_string())
            print("\n账户回测（只有前两条都过的因子才跑）：")
            print(summary[["annual_return", "bench_annual", "excess_annual", "turnover", "cost_annual",
                           "n_trades", "avg_holdings", "n_stops", "max_drawdown"]].round(4).to_string())
        return
    selection, corr = select(summary, subs)
    write_reports(summary, selection, corr, context)
    print(f"通过 {int(summary['passed'].fillna(False).sum())} 个，选中 {int(selection['kept'].sum()) if len(selection) else 0} 个；"
          f"报告在 {SINGLE_DIR.relative_to(ROOT)} 和 {SELECT_DIR.relative_to(ROOT)}")
    print(f"试验日志里不同配置的单因子试验共 {count_unique(kind='single_factor')} 个")


if __name__ == "__main__":
    main()
