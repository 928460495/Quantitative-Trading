"""实验室：开发段数据的拼接和标记（docs/FRAMEWORK.md 第 2 节）。"""

import numpy as np
import pandas as pd
import pytest

from qt.config import TestSegmentLockedError
from qt.lab import store

from lab_helpers import write_segment

A = pd.bdate_range("2018-12-24", "2018-12-31")
B = pd.bdate_range("2019-01-02", "2019-01-08")


def _two(tmp_path, cols_a=("SH600000", "SH600001"), cols_b=("SH600001", "SH600002")):
    rng = np.random.default_rng(0)
    write_segment(tmp_path, "train", A, list(cols_a), rng)
    write_segment(tmp_path, "valid", B, list(cols_b), rng, flag_value=True)
    return tmp_path


def test_concat_orders_dates_and_unions_stocks():
    a = pd.DataFrame([[1.0, 2.0]], index=A[:1], columns=["x", "y"])
    b = pd.DataFrame([[3.0, 4.0]], index=B[:1], columns=["y", "z"])
    got = store.concat([a, b])
    assert list(got.columns) == ["x", "y", "z"] and list(got.index) == [A[0], B[0]]
    assert np.isnan(got.loc[A[0], "z"]) and got.loc[B[0], "y"] == 3.0
    flags = store.concat([a > 1, b > 3], "bool")
    assert flags.dtypes.eq(bool).all() and flags.loc[B[0]].tolist() == [False, False, True]   # 缺失按否
    obj = store.concat([a.astype(str), b.astype(str)], "object")
    assert obj.loc[A[0], "z"] is None
    with pytest.raises(ValueError, match="重叠"):
        store.concat([b, a])
    with pytest.raises(ValueError, match="重叠"):
        store.concat([a, a])


def test_load_dev_joins_segments(tmp_path):
    dev = store.load_dev(_two(tmp_path), ["a"])
    idx = A.append(B)
    cols = ["SH600000", "SH600001", "SH600002"]
    for key in ("close", "raw_close", "log_mcap", "can_buy", "can_sell", "industry"):
        assert dev[key].index.equals(idx) and list(dev[key].columns) == cols, key
    assert set(dev["factors"]) == {"a", "vol60"}                             # vol60 总是带上（高波动标记要用）
    assert dev["factors"]["a"].index.equals(idx) and list(dev["factors"]["a"].columns) == cols
    assert not dev["can_buy"].loc[B, "SH600000"].any() and dev["can_buy"].loc[A, "SH600000"].all()   # 没数据 = 不能买
    assert dev["industry"].loc[B[0], "SH600000"] is None and dev["industry"].loc[B[0], "SH600002"] == "I1"
    assert set(dev["flags"]) == {"is_st", "loss_small", "deduct_loss", "neg_equity", "cheap", "high_vol", "small30"}
    for name, flag in dev["flags"].items():
        assert flag.dtypes.eq(bool).all() and flag.index.equals(idx) and list(flag.columns) == cols, name
    assert not dev["flags"]["is_st"].loc[A].any().any() and dev["flags"]["is_st"].loc[B, "SH600002"].all()
    assert dev["index"]["H00300"].index.is_monotonic_increasing and dev["index"].index.is_unique


def test_index_overlap_between_segments_keeps_one_row_per_day(tmp_path):
    # 真实缓存里验证段的 index.parquet 带一年预热期，和训练段重叠
    root = _two(tmp_path)
    both = A.append(B)
    pd.DataFrame({"000985": 9.0, "H00300": 99.0}, index=both).to_parquet(root / "valid" / "index.parquet")
    idx = store.load_dev(root, ["a"])["index"]
    assert idx.index.is_unique and idx.index.equals(both)
    assert idx.loc[A[0], "H00300"] == 1.0 and idx.loc[B[0], "H00300"] == 99.0   # 重叠的日子用前一段的


def test_load_dev_reports_missing_files(tmp_path):
    root = _two(tmp_path)
    (root / "valid" / "deduct_loss.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="lab_build_flags"):
        store.load_dev(root, ["a"])
    with pytest.raises(FileNotFoundError):
        store.load_dev(_two(tmp_path / "x"), ["nope"])


def test_load_dev_refuses_sealed_segment(tmp_path, monkeypatch):
    monkeypatch.delenv("QT_UNLOCK_TEST", raising=False)
    rng = np.random.default_rng(0)
    write_segment(tmp_path, "train", A, ["SH600000"], rng)
    write_segment(tmp_path, "valid", pd.bdate_range("2021-12-30", "2022-01-05"), ["SH600000"], rng)
    with pytest.raises(TestSegmentLockedError):
        store.load_dev(tmp_path, ["a"])


def test_derived_flags():
    idx = A[:1]
    cols = [f"S{i}" for i in range(10)]
    raw = pd.DataFrame([[1.99, 2.0, 2.01, np.nan, 5, 5, 5, 5, 5, 5]], index=idx, columns=cols)
    mcap = pd.DataFrame([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], index=idx, columns=cols, dtype=float)
    vol = pd.DataFrame([[10, 9, 8, 7, 6, 5, 4, 3, 2, np.nan]], index=idx, columns=cols, dtype=float)
    got = store.derived_flags(raw, mcap, vol)
    assert got["cheap"].iloc[0].tolist() == [True] + [False] * 9             # 正好 2 元不算；缺价不算
    assert got["small30"].iloc[0].tolist() == [True, True, True] + [False] * 7
    assert got["high_vol"].iloc[0].tolist() == [False] * 8 + [True, False]   # 9 个有值的里最低的 20%（1 个）；缺值不算
    assert all(f.dtypes.eq(bool).all() for f in got.values())


def test_distress_flags_ignore_missing_data():
    f = lambda *v: pd.DataFrame([list(v)], dtype=float)
    got = store.distress_flags({"deduct_net_profit_ttm": f(-1, 0, 1, np.nan), "total_equity": f(5, -1, 0, np.nan)})
    assert got["deduct_loss"].iloc[0].tolist() == [True, False, False, False]
    assert got["neg_equity"].iloc[0].tolist() == [False, True, True, False]  # 权益为零也算


def test_div_yield_is_loaded_only_when_both_segments_have_it(tmp_path):
    root = _two(tmp_path)
    assert "div_yield" not in store.load_dev(root, ["a"])
    pd.DataFrame(0.01, index=A, columns=["SH600000", "SH600001"]).to_parquet(root / "train" / "div_yield.parquet")
    assert "div_yield" not in store.load_dev(root, ["a"])                      # 只有一段：不放
    pd.DataFrame(0.02, index=B, columns=["SH600001", "SH600002"]).to_parquet(root / "valid" / "div_yield.parquet")
    div = store.load_dev(root, ["a"])["div_yield"]
    assert div.index.equals(A.append(B)) and list(div.columns) == ["SH600000", "SH600001", "SH600002"]
    assert div.loc[B[0], "SH600000"] == 0.0 and div.loc[A[0], "SH600000"] == 0.01 and div.loc[B[0], "SH600002"] == 0.02
