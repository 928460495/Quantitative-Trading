"""实验室：给 data/cache/pool/<版本>/<段>/ 补两张财务困境标记表（docs/FRAMEWORK.md；qt/lab/store.py）。

用法（先跑过 scripts/build_pool.py；单进程，几分钟）：
    python scripts/lab_build_flags.py

    deduct_loss.parquet   扣非净利润滚动四季为负（按公告日对齐）
    neg_equity.parquet    股东权益不为正

日期和股票取缓存里 close.parquet 的；训练段、验证段各算各的；不碰封存段（取数经过 check_range）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt.lab.store import distress_flags  # noqa: E402


def build(close: pd.DataFrame, fund: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """结果的日期、股票和 close 一样；财务表里没有的股票或日子按"没有标记"。"""
    al = lambda x: x.reindex(index=close.index, columns=close.columns)
    return distress_flags({k: al(fund[k]) for k in ("deduct_net_profit_ttm", "total_equity")})


def main() -> None:
    from build_plan3d_flags import yearly_share
    from qt.panel import load_fundamentals
    from run_plan3c_train import POOL_CACHE, latest_version

    version = latest_version(POOL_CACHE)
    for seg in ("train", "valid"):
        seg_dir = version / seg
        close = pd.read_parquet(seg_dir / "close.parquet")
        can_buy = pd.read_parquet(seg_dir / "can_buy.parquet")
        print(f"== {seg}：{close.index[0].date()} ~ {close.index[-1].date()}，{close.shape[1]} 只", flush=True)
        flags = build(close, load_fundamentals(close.index, list(close.columns)))
        for name, frame in flags.items():
            frame.to_parquet(seg_dir / f"{name}.parquet")
        table = pd.DataFrame({name: yearly_share(frame, can_buy) for name, frame in flags.items()})
        print("  能买的股票里被标记的比例：")
        print(table.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
