"""计划 3d 任务 3：给 data/cache/pool/<版本>/<段>/ 补两张"日期 × 股票"的布尔表。

用法（先跑过 scripts/build_pool.py；单进程，几分钟）：
    python scripts/build_plan3d_flags.py                 # 训练段和验证段
    QT_UNLOCK_TEST=1 python scripts/build_plan3d_flags.py test   # 封存段：只在用户批准的最终测试前

    is_st.parquet        当天是不是 ST、*ST（baostock 日线 isST；缺记录的日子沿用上一个已知状态）
    loss_small.parquet   亏损且营业收入低于当时有效的退市门槛（滚动四季，按公告日对齐；门槛见 qt/plan3d.revenue_floor）

规则见 docs/DECISIONS.md 2026-11-12「计划 3d 事先登记的规则」。日期和股票取缓存里 close.parquet 的，
训练段、验证段各算各的；不碰测试段（取数经过 check_range）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt.plan3d import loss_small_revenue, revenue_floor  # noqa: E402

FIELDS = ("net_profit_ttm", "deduct_net_profit_ttm", "revenue_ttm")
REVENUE_RANGE = (1e8, 1e11)      # 全市场滚动四季营收的中位数应该落在 1 亿 ~ 1000 亿元之间


def build_flags(close: pd.DataFrame, is_st: pd.Series, fund: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """is_st：(datetime, instrument) 索引的 0/1；fund：FIELDS 三张宽表。结果的日期、股票和 close 一样。"""
    idx, cols = close.index, close.columns
    st = is_st.unstack("instrument").reindex(columns=cols)
    st = st.reindex(st.index.union(idx)).sort_index().ffill().reindex(idx)
    al = lambda k: fund[k].reindex(index=idx, columns=cols)
    return {"is_st": (st.fillna(0) > 0).astype(bool),
            "loss_small": loss_small_revenue(*(al(k) for k in FIELDS), revenue_floor(idx))}


def check_units(revenue_ttm: pd.DataFrame) -> float:
    med = float(revenue_ttm.stack().median())
    if not REVENUE_RANGE[0] <= med <= REVENUE_RANGE[1]:
        raise ValueError(f"滚动四季营收的中位数是 {med:.3g}，不像以元为单位，先核对数据")
    return med


def yearly_share(flag: pd.DataFrame, can_buy: pd.DataFrame) -> pd.Series:
    """每年：能买的股票里被标记的比例。"""
    buy = can_buy.reindex(index=flag.index, columns=flag.columns).fillna(False).astype(bool)
    by_year = lambda x: x.sum(axis=1).groupby(x.index.year).sum()
    return (by_year(flag & buy) / by_year(buy)).fillna(0.0)


def main() -> None:
    from qt.config import parse_segments
    from qt.full_a import load_daily
    from qt.panel import load_fundamentals
    from run_plan3c_train import POOL_CACHE, latest_version

    version = latest_version(POOL_CACHE)
    for seg in parse_segments(sys.argv[1] if len(sys.argv) > 1 else "both"):
        seg_dir = version / seg
        close = pd.read_parquet(seg_dir / "close.parquet")
        can_buy = pd.read_parquet(seg_dir / "can_buy.parquet")
        start, end = str(close.index[0].date()), str(close.index[-1].date())
        print(f"== {seg}：{start} ~ {end}，{close.shape[1]} 只", flush=True)
        fund = load_fundamentals(close.index, list(close.columns))
        med = check_units(fund["revenue_ttm"])
        flags = build_flags(close, load_daily(["isST"], start, end)["isST"], fund)
        for name, frame in flags.items():
            frame.to_parquet(seg_dir / f"{name}.parquet")
        table = pd.DataFrame({name: yearly_share(frame, can_buy) for name, frame in flags.items()})
        print(f"  滚动四季营收中位数 {med / 1e8:.1f} 亿元；能买的股票里被标记的比例：")
        print(table.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
