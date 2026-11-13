"""计划 3c 任务 6：59 个已通过三条检验的因子，每天的中性化、调好方向的值，连同回测要用的行情、对数流通市值一起存下来。

用法（在有数据的电脑上，单进程，约 1–2 小时）：
    python scripts/build_pool.py                  # 训练段和验证段都算
    python scripts/build_pool.py --segment train

规则见 docs/DECISIONS.md 2026-11-10「计划 3c 事先登记的规则」：候选是 reports/screening/full_a_train_neutral/
all_candidates.csv 的 59 个，方向、大类取 reports/single_factor/full_a_train_neutral/summary.csv，中性化同 3b。
结果在 data/cache/pool/<数据版本>/<段>/（不进仓库），结构同 data/cache/composite/，多一个 log_mcap.parquet。
composite.parquet 是 59 个因子等权的合成分，计划 3c 不用它（只是沿用 build_segment 的输出）。

结束时打印"能买的股票里一手金额超过 nav/N 的比例"（N = 15、20）；N=20 时某年超过 20% 就以退出码 2 结束：
按规则停下来讨论（会让持股达不到 N）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_composite import INDEXES, build_segment, warmup_start  # noqa: E402
from qt.account import LOT  # noqa: E402

CANDIDATES = ROOT / "reports" / "screening" / "full_a_train_neutral" / "all_candidates.csv"
SUMMARY = ROOT / "reports" / "single_factor" / "full_a_train_neutral" / "summary.csv"
CACHE_ROOT = ROOT / "data" / "cache" / "pool"
NAV = 100_000
MAX_UNAFFORDABLE = 0.20


def read_pool(candidates: Path = CANDIDATES, summary: Path = SUMMARY) -> pd.DataFrame:
    """index = 因子名，列 category、direction。"""
    names = pd.read_csv(candidates)["factor"]
    info = pd.read_csv(summary, index_col="factor")
    pool = info.loc[names, ["category", "direction"]].astype({"direction": int})
    pool.index.name = "factor"
    return pool


def lot_feasibility(raw_close: pd.DataFrame, can_buy: pd.DataFrame, nav: float, n_hold: int) -> pd.Series:
    """每年：能买的股票里，一手金额超过 nav / n_hold 的比例。"""
    buy = can_buy.reindex(index=raw_close.index, columns=raw_close.columns, fill_value=False).astype(bool)
    too_dear = (raw_close * LOT > nav / n_hold) & buy
    by_year = lambda x: x.groupby(x.index.year).sum()
    return (by_year(too_dear.sum(axis=1)) / by_year(buy.sum(axis=1))).fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", choices=["train", "valid", "test", "live", "both"], default="both",
                    help="test：封存段（最终测试用）；live：纸面跟单的最新数据")
    args = ap.parse_args()

    from qt.config import parse_segments, segment_range
    from qt.formula import evaluate
    from qt.full_a import load_index
    from qt.library import all_factors
    from qt.panel import _qlib_wide
    from run_full_a_screen import load_context

    pool = read_pool()
    defs = {f.name: f for f in all_factors()}
    stop = False
    for seg in parse_segments(args.segment):
        start, end = segment_range(seg)
        warm = warmup_start(start)
        print(f"== {seg}：{start} ~ {end}（从 {warm} 开始算），{len(pool)} 个因子", flush=True)
        panel, ctx, codes, version = load_context(start, end, 1, warmup=warm, tail=0)

        def compute(name: str) -> pd.DataFrame:
            f = defs[name]
            if f.dialect == "qlib":
                frame = _qlib_wide(codes, {name: f.formula}, warm, end)[name]
                return frame.reindex(index=ctx.close.index, columns=ctx.close.columns)
            return evaluate(f.formula, panel, f.dialect) if f.formula else f.func(panel)

        index = {s: load_index(s, warm, end) for s in INDEXES}
        out_dir = CACHE_ROOT / version.replace("/", "_") / seg
        build_segment(ctx, pool, compute, out_dir, index, meta={"segment": seg, "data_version": version},
                      save_log_mcap=True)
        print(f"  存在 {out_dir}", flush=True)
        if seg == "train":
            raw, buy = ctx.trim(ctx.raw_close), ctx.trim(ctx.can_buy).fillna(False)
            table = pd.DataFrame({f"N={n}": lot_feasibility(raw, buy, NAV, n) for n in (15, 20)})
            print("能买的股票里一手买不起的比例（训练段）：")
            print(table.round(3).to_string())
            stop = bool((table["N=20"] > MAX_UNAFFORDABLE).any())
        del panel, ctx
    if stop:
        print(f"N=20 时有年份超过 {MAX_UNAFFORDABLE:.0%}：按规则停下来讨论（DECISIONS 2026-11-10）")
        sys.exit(2)


if __name__ == "__main__":
    main()
