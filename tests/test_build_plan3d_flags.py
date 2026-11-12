"""计划 3d 任务 3：ST、亏损且营收不足 3 亿元两张标记表（DECISIONS 2026-11-12）。小假数据。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_plan3d_flags", ROOT / "scripts" / "build_plan3d_flags.py")
bf = importlib.util.module_from_spec(spec)
sys.modules["build_plan3d_flags"] = bf
spec.loader.exec_module(bf)

IDX = pd.bdate_range("2015-01-05", periods=6)
COLS = ["SH600000", "SH600001", "SZ000002"]


def _close():
    return pd.DataFrame(10.0, index=IDX, columns=COLS)


def _st(rows):
    idx = pd.MultiIndex.from_tuples([(IDX[d], c) for d, c, _ in rows], names=["datetime", "instrument"])
    return pd.Series([v for _, _, v in rows], index=idx, name="isST")


def _fund(net, ded, rev, cols=COLS, idx=IDX):
    f = lambda v: pd.DataFrame([v] * len(idx), index=idx, columns=cols, dtype=float)
    return {"net_profit_ttm": f(net), "deduct_net_profit_ttm": f(ded), "revenue_ttm": f(rev)}


def test_flags_align_to_cached_dates_and_stocks():
    st = _st([(0, "SH600000", 0), (1, "SH600000", 1), (2, "SH600000", 1), (0, "SZ000002", 0), (0, "SH999999", 1)])
    fund = _fund([-1, 5, -1, 9], [-1, 5, 1, 9], [1e8, 1e8, 9e8, 1e8], cols=["SZ000002", "SH600001", "SH600000", "SH999999"])
    got = bf.build_flags(_close(), st, fund)
    assert set(got) == {"is_st", "loss_small"}
    for frame in got.values():
        assert frame.index.equals(IDX) and list(frame.columns) == COLS and frame.dtypes.eq(bool).all()
    assert got["loss_small"].iloc[0].tolist() == [False, False, True]        # 按股票名对齐，不按位置
    assert got["is_st"]["SH600001"].sum() == 0                               # 没有记录的按否


def test_st_status_carries_over_missing_days_but_not_backwards():
    st = _st([(1, "SH600000", 1), (3, "SH600000", 0), (4, "SH600001", 1)])
    got = bf.build_flags(_close(), st, _fund([1] * 3, [1] * 3, [1e9] * 3))["is_st"]
    assert got["SH600000"].tolist() == [False, True, True, False, False, False]   # 第 2 天没记录：沿用；之前缺的按否
    assert got["SH600001"].tolist() == [False, False, False, False, True, True]


def test_fundamentals_outside_cached_dates_are_ignored():
    wide = pd.bdate_range("2014-12-29", periods=12)
    fund = _fund([-1] * 3, [-1] * 3, [1e8] * 3, idx=wide)
    fund["revenue_ttm"].loc[IDX[3]:, "SH600001"] = 5e8
    got = bf.build_flags(_close(), _st([(0, "SH600000", 0)]), fund)["loss_small"]
    assert got["SH600001"].tolist() == [True, True, True, False, False, False]
    assert got["SH600000"].all()


def test_revenue_unit_check():
    ok = pd.DataFrame(np.full((3, 4), 2e9))
    assert bf.check_units(ok) == pytest.approx(2e9)
    with pytest.raises(ValueError, match="单位"):
        bf.check_units(ok / 1e4)                                             # 万元当成元
    with pytest.raises(ValueError, match="单位"):
        bf.check_units(ok * 1e4)
    with pytest.raises(ValueError, match="单位"):
        bf.check_units(ok * np.nan)


def test_yearly_share_is_among_buyable_stocks():
    flag = pd.DataFrame(False, index=IDX, columns=COLS)
    flag["SH600000"] = True
    buy = pd.DataFrame(True, index=IDX, columns=COLS)
    buy["SZ000002"] = False
    assert bf.yearly_share(flag, buy).to_dict() == {2015: pytest.approx(0.5)}
    assert bf.yearly_share(flag, buy & False).to_dict() == {2015: 0.0}
