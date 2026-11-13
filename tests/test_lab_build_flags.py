"""实验室：财务困境标记表（scripts/lab_build_flags.py）。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_dividend_yield_from_reference_preclose():
    idx = pd.bdate_range("2019-06-03", periods=5)
    rows = []
    for k, d in enumerate(idx):
        rows.append((d, "SH600000", 10.0 if k != 2 else 9.5, 10.0))           # 第 3 天除息 0.5 元：5%
        rows.append((d, "SH600001", 20.0 if k != 3 else 10.0, 20.0 if k < 3 else 10.0))   # 10 送 10：不算
    daily = pd.DataFrame(rows, columns=["datetime", "instrument", "preclose", "close"]).set_index(["datetime", "instrument"])
    close = pd.DataFrame(1.0, index=idx, columns=["SH600000", "SH600001", "SZ000002"])
    got = bf.dividend_yield(daily, close)
    assert got.index.equals(idx) and list(got.columns) == list(close.columns)
    assert got.loc[idx[2], "SH600000"] == pytest.approx(0.05)
    assert (got.drop(index=idx[2])["SH600000"] == 0).all()
    assert (got["SH600001"] == 0).all() and (got["SZ000002"] == 0).all()      # 送转不算；没有数据的为 0


def test_dividend_yield_compares_with_the_stocks_own_previous_trading_day():
    idx = pd.bdate_range("2019-06-03", periods=4)
    daily = pd.DataFrame([(idx[0], "A", 10.0, 10.0), (idx[3], "A", 9.8, 9.8)],      # 中间停牌两天
                         columns=["datetime", "instrument", "preclose", "close"]).set_index(["datetime", "instrument"])
    got = bf.dividend_yield(daily, pd.DataFrame(1.0, index=idx, columns=["A"]))
    assert got.loc[idx[3], "A"] == pytest.approx(0.02)
