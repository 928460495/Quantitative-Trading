"""实验室测试共用的小假数据。"""

import numpy as np
import pandas as pd

from qt import plan3d as p3
from qt.lab import store

from test_plan3d import seg3d


def dev3d(seed=0, n_stocks=60, start="2018-01-02", end="2019-12-31"):
    """把 seg3d 的假数据包成 load_dev 的样子（跨 2018、2019 两个阶段）。"便宜"组带真信号。"""
    seg = seg3d(seed, n_stocks, start, end)
    false = pd.DataFrame(False, index=seg["close"].index, columns=seg["close"].columns)
    flags = {"is_st": seg["is_st"], "loss_small": seg["loss_small"], "deduct_loss": false.copy(), "neg_equity": false.copy()}
    flags.update(store.derived_flags(seg["raw_close"], seg["log_mcap"], seg["factors"][p3.VOL_FACTOR]))
    return {**{k: seg[k] for k in ("close", "raw_close", "log_mcap", "can_buy", "can_sell", "industry", "index")},
            "factors": seg["factors"], "flags": flags}, seg


def write_segment(root, name, idx, cols, rng, factors=("a", "vol60"), flag_value=False):
    d = root / name
    (d / "factors").mkdir(parents=True)
    f = lambda v: pd.DataFrame(v, index=idx, columns=cols)
    f(rng.uniform(5, 20, (len(idx), len(cols)))).to_parquet(d / "close.parquet")
    f(rng.uniform(1, 20, (len(idx), len(cols)))).to_parquet(d / "raw_close.parquet")
    f(rng.normal(22, 1, (len(idx), len(cols)))).astype("float32").to_parquet(d / "log_mcap.parquet")
    for b in ("can_buy", "can_sell"):
        f(True).to_parquet(d / f"{b}.parquet")
    f("I1").astype("string").to_parquet(d / "industry.parquet")
    pd.DataFrame({"000985": 1.0, "H00300": np.arange(len(idx)) + 1.0}, index=idx).to_parquet(d / "index.parquet")
    for name_ in factors:
        f(rng.normal(size=(len(idx), len(cols)))).astype("float32").to_parquet(d / "factors" / f"{name_}.parquet")
    for flag in store.STORED_FLAGS:
        f(flag_value).to_parquet(d / f"{flag}.parquet")
    return d
