"""开发段数据（2010–2021）：把 data/cache/pool/<版本>/ 下 train、valid 两段接成一段，并备好各种标记。

    dev = load_dev(version_dir, ["ep_deduct", "bp", ...])
    dev["close"], dev["can_buy"], dev["factors"]["bp"], dev["flags"]["deduct_loss"], dev["index"]["H00300"]

因子值是已中性化、调好方向的（scripts/build_pool.py 存的）。标记都是"日期 × 股票"的布尔表，按信号日对齐：
  存下来的：is_st、loss_small（scripts/build_plan3d_flags.py）、deduct_loss、neg_equity（scripts/lab_build_flags.py）
  现算的：cheap（未复权收盘价 < 2 元）、high_vol（vol60 分数最低的 20%，即波动最高）、small30（流通市值最小的 30%）
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from qt.config import check_range

SEGMENTS = ("train", "valid")
FLOATS = ("close", "raw_close", "log_mcap")
BOOLS = ("can_buy", "can_sell")
STORED_FLAGS = ("is_st", "loss_small", "deduct_loss", "neg_equity")
VOL_FACTOR = "vol60"
PRICE_FLOOR, VOL_CUT, SMALL_CUT = 2.0, 0.20, 0.30


def concat(parts: Sequence[pd.DataFrame], kind: str = "float") -> pd.DataFrame:
    """按日期接起来，股票取并集。kind：float（缺失留空）、bool（缺失按否）、object（缺失为 None）。"""
    for a, b in zip(parts, parts[1:]):
        if not a.index.max() < b.index.min():
            raise ValueError("各段的日期有重叠或顺序不对")
    cols = sorted(set().union(*(p.columns for p in parts)))
    out = pd.concat([p.reindex(columns=cols) for p in parts])
    if kind == "bool":
        return out.fillna(False).astype(bool)
    if kind == "object":
        out = out.astype(object)
        return out.where(out.notna(), None)
    return out.astype(float)


def distress_flags(fund: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """deduct_loss：扣非净利润滚动四季为负；neg_equity：股东权益不为正。数据缺失不算。"""
    return {"deduct_loss": (fund["deduct_net_profit_ttm"] < 0).fillna(False).astype(bool),
            "neg_equity": (fund["total_equity"] <= 0).fillna(False).astype(bool)}


def derived_flags(raw_close: pd.DataFrame, log_mcap: pd.DataFrame, vol_score: pd.DataFrame) -> dict[str, pd.DataFrame]:
    like = lambda x: x.reindex(index=raw_close.index, columns=raw_close.columns)
    pct = lambda x: like(x).rank(axis=1, pct=True)
    return {"cheap": (raw_close < PRICE_FLOOR).fillna(False).astype(bool),
            "high_vol": (pct(vol_score) <= VOL_CUT).fillna(False).astype(bool),
            "small30": (pct(log_mcap) <= SMALL_CUT).fillna(False).astype(bool)}


def load_dev(version_dir: Path, factors: Sequence[str], segments: Sequence[str] = SEGMENTS) -> dict:
    version_dir = Path(version_dir)
    read = lambda seg, name: pd.read_parquet(version_dir / seg / f"{name}.parquet")

    def joined(name: str, kind: str = "float") -> pd.DataFrame:
        for seg in segments:
            if not (version_dir / seg / f"{name}.parquet").exists():
                raise FileNotFoundError(f"没有 {version_dir / seg / name}.parquet，先跑 scripts/build_pool.py、"
                                        "scripts/build_plan3d_flags.py、scripts/lab_build_flags.py")
        return concat([read(seg, name) for seg in segments], kind)

    dev: dict = {name: joined(name) for name in FLOATS}
    check_range(str(dev["close"].index[0].date()), str(dev["close"].index[-1].date()))
    cols = dev["close"].columns
    align = lambda x: x.reindex(index=dev["close"].index, columns=cols)
    dev.update({name: align(joined(name, "bool")).fillna(False).astype(bool) for name in BOOLS})
    ind = align(joined("industry", "object")).astype(object)
    dev["industry"] = ind.where(ind.notna(), None)
    index = pd.concat([read(seg, "index") for seg in segments])
    dev["index"] = index[~index.index.duplicated(keep="first")].sort_index()
    names = list(dict.fromkeys([*factors, VOL_FACTOR]))
    dev["factors"] = {f: align(joined(f"factors/{f}")).astype("float32") for f in names}
    dev["flags"] = {name: align(joined(name, "bool")).fillna(False).astype(bool) for name in STORED_FLAGS}
    dev["flags"].update(derived_flags(dev["raw_close"], dev["log_mcap"], dev["factors"][VOL_FACTOR]))
    if all((version_dir / seg / "div_yield.parquet").exists() for seg in segments):     # 红利税用；没有就不放
        dev["div_yield"] = align(joined("div_yield")).fillna(0.0)
    return dev
