"""计划 4 任务 4：算 21 个因子的合成分，连同回测要用的行情、能买能卖、行业、指数一起存下来。

用法（在有数据的电脑上，单进程）：
    python scripts/build_composite.py                  # 训练段和验证段都算
    python scripts/build_composite.py --segment valid  # 只算一段

规则见 docs/DECISIONS.md 2026-11-09：因子名单、方向、大类读 reports/screening/full_a_train_neutral/selected.csv；
中性化和计划 3b 完全一样（规模类只对行业）；类内等权 → 每类标准化 → 类间等权 → 标准化。
训练段、验证段分开算，各自提前一年开始给长窗口预热；取数到 2021-12-31 为止，不碰测试段。

结果存在 data/cache/composite/<数据版本>/<段>/（不进仓库）：
    factors/<因子>.parquet   每个因子每天的中性化、调好方向的值
    categories/<大类>.parquet
    composite.parquet        总分
    close / raw_close / can_buy / can_sell / industry .parquet   回测用
    index.parquet            中证全指（000985）、沪深300 全收益（H00300），含预热期
    meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt.composite import category_scores, composite  # noqa: E402
from qt.screen_wide import ScreenContext, factor_score  # noqa: E402

SELECTED = ROOT / "reports" / "screening" / "full_a_train_neutral" / "selected.csv"
CACHE_ROOT = ROOT / "data" / "cache" / "composite"
EARLIEST = pd.Timestamp("2009-01-05")   # baostock 日线从这天开始
INDEXES = ("000985", "H00300")
MARKET = ("close", "raw_close", "can_buy", "can_sell", "industry")


def read_selected(path: Path = SELECTED) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df[["category", "direction"]].astype({"direction": int})


def warmup_start(start: str) -> str:
    return str(max(pd.Timestamp(start) - pd.DateOffset(years=1), EARLIEST).date())


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.columns = out.columns.astype(str)
    out.to_parquet(path)


def build_segment(ctx: ScreenContext, selected: pd.DataFrame, compute: Callable[[str], pd.DataFrame], out_dir: Path,
                  index: dict[str, pd.Series], meta: dict | None = None) -> pd.DataFrame:
    """compute(因子名) -> 原始因子宽表（含预热期）。返回总分（只含 ctx.dates）。"""
    scores = {}
    t0 = time.time()
    for k, (name, row) in enumerate(selected.iterrows(), 1):
        scores[name] = factor_score(compute(name), row["category"], int(row["direction"]), ctx).astype("float32")
        _save(scores[name], out_dir / "factors" / f"{name}.parquet")
        print(f"  因子 {k}/{len(selected)} {name}，已用 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    cats = category_scores(scores, selected["category"].to_dict())
    for cat, frame in cats.items():
        _save(frame.astype("float32"), out_dir / "categories" / f"{cat.replace('/', '_')}.parquet")
    total = composite(cats)
    _save(total, out_dir / "composite.parquet")
    for name in MARKET:
        frame = ctx.trim(getattr(ctx, name))
        if name in ("can_buy", "can_sell"):
            frame = frame.fillna(False).astype(bool)
        elif name == "industry":
            frame = frame.astype("string")
        _save(frame, out_dir / f"{name}.parquet")
    idx = pd.DataFrame({k: s.reindex(ctx.close.index) for k, s in index.items()})
    idx.to_parquet(out_dir / "index.parquet")
    info = {"factors": list(selected.index), "categories": sorted(cats), "start": str(ctx.dates[0].date()),
            "end": str(ctx.dates[-1].date()), **(meta or {})}
    (out_dir / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return total


def load_segment(seg_dir: Path) -> dict:
    """读 build_segment 存下的东西。大类名里的 / 存成 _，读回来时还原（按 meta.json 里的名字）。"""
    meta = json.loads((seg_dir / "meta.json").read_text())
    read = lambda p: pd.read_parquet(p)
    out = {"meta": meta, "composite": read(seg_dir / "composite.parquet"),
           "factors": {n: read(seg_dir / "factors" / f"{n}.parquet") for n in meta["factors"]},
           "categories": {c: read(seg_dir / "categories" / f"{c.replace('/', '_')}.parquet") for c in meta["categories"]},
           "index": read(seg_dir / "index.parquet")}
    for name in MARKET:
        out[name] = read(seg_dir / f"{name}.parquet")
    out["industry"] = out["industry"].astype(object).where(out["industry"].notna(), None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", choices=["train", "valid", "both"], default="both")
    args = ap.parse_args()

    from qt.config import segment
    from qt.formula import evaluate
    from qt.full_a import load_index
    from qt.library import all_factors
    from qt.panel import _qlib_wide
    from run_full_a_screen import load_context

    selected = read_selected()
    defs = {f.name: f for f in all_factors()}
    for seg in (["train", "valid"] if args.segment == "both" else [args.segment]):
        start, end = segment(seg)
        warm = warmup_start(start)
        print(f"== {seg}：{start} ~ {end}（从 {warm} 开始算，给长窗口预热）", flush=True)
        panel, ctx, codes, version = load_context(start, end, 1, warmup=warm, tail=0)

        def compute(name: str) -> pd.DataFrame:
            f = defs[name]
            if f.dialect == "qlib":
                frame = _qlib_wide(codes, {name: f.formula}, warm, end)[name]
                return frame.reindex(index=ctx.close.index, columns=ctx.close.columns)
            return evaluate(f.formula, panel, f.dialect) if f.formula else f.func(panel)

        index = {s: load_index(s, warm, end) for s in INDEXES}
        out_dir = CACHE_ROOT / version.replace("/", "_") / seg
        build_segment(ctx, selected, compute, out_dir, index, meta={"segment": seg, "data_version": version})
        print(f"  存在 {out_dir}", flush=True)
        del panel, ctx


if __name__ == "__main__":
    main()
