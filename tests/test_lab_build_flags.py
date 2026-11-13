"""实验室：财务困境标记表（scripts/lab_build_flags.py）。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lab_build_flags", ROOT / "scripts" / "lab_build_flags.py")
bf = importlib.util.module_from_spec(spec)
sys.modules["lab_build_flags"] = bf
spec.loader.exec_module(bf)


def test_build_aligns_to_cache_by_name_not_position():
    idx = pd.bdate_range("2015-01-05", periods=4)
    close = pd.DataFrame(10.0, index=idx, columns=["SH600000", "SH600001", "SZ000002"])
    wide = pd.bdate_range("2014-12-29", periods=10)
    fund = {"deduct_net_profit_ttm": pd.DataFrame({"SZ000002": -1.0, "SH600000": 5.0, "SH999999": -9.0}, index=wide),
            "total_equity": pd.DataFrame({"SH600001": 0.0, "SZ000002": 3.0}, index=wide)}
    fund["deduct_net_profit_ttm"].loc[idx[2]:, "SH600000"] = -2.0
    got = bf.build(close, fund)
    for frame in got.values():
        assert frame.index.equals(idx) and list(frame.columns) == list(close.columns) and frame.dtypes.eq(bool).all()
    assert got["deduct_loss"]["SZ000002"].all() and got["deduct_loss"]["SH600000"].tolist() == [False, False, True, True]
    assert not got["deduct_loss"]["SH600001"].any()                            # 没有财务数据不算
    assert got["neg_equity"]["SH600001"].all() and not got["neg_equity"]["SH600000"].any()
