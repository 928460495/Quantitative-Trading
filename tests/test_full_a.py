import numpy as np
import pandas as pd
import pytest

from qt.config import TestSegmentLockedError
from qt.full_a import float_shares, listing_dates_full_a, load_daily, load_index
from qt.universe import full_a_masks


@pytest.fixture
def full_a_dir(tmp_path):
    daily = pd.DataFrame({
        "instrument": pd.Categorical(["SH600000", "SH600000", "SZ000001"]),
        "date": pd.to_datetime(["2019-01-02", "2019-01-03", "2019-01-03"]),
        "close": np.array([10.0, 10.5, 12.0], dtype="float32"),
        "isST": np.array([0, 0, 1], dtype="int8"),
    })
    daily.to_parquet(tmp_path / "daily.parquet", index=False)
    pd.DataFrame({"instrument": ["SH600000", "SZ000001"], "name": ["A", "B"], "ipo_date": ["1999-11-10", "1991-04-03"],
                  "out_date": ["", ""]}).to_parquet(tmp_path / "stocks.parquet", index=False)
    pd.DataFrame({"symbol": ["H00300", "H00300", "000985"], "date": ["2019-01-02", "2019-01-03", "2019-01-02"],
                  "close": [5000.0, 5050.0, 4000.0]}).to_parquet(tmp_path / "index.parquet", index=False)
    return tmp_path


def test_load_daily_respects_test_lock(full_a_dir):
    with pytest.raises(TestSegmentLockedError):
        load_daily(["close"], "2021-01-01", "2022-06-30", full_a_dir)


def test_load_daily_index_and_range(full_a_dir):
    df = load_daily(["close", "isST"], "2019-01-03", "2019-01-31", full_a_dir)
    assert df.index.names == ["datetime", "instrument"]
    assert list(df.columns) == ["close", "isST"]
    assert list(df.index.get_level_values("instrument")) == ["SH600000", "SZ000001"]
    assert df.loc[(pd.Timestamp("2019-01-03"), "SZ000001"), "isST"] == 1


def test_listing_dates_and_index(full_a_dir):
    listed = listing_dates_full_a(full_a_dir)
    assert listed["SZ000001"] == pd.Timestamp("1991-04-03")
    idx = load_index("H00300", "2019-01-01", "2019-01-31", full_a_dir)
    assert list(idx) == [5000.0, 5050.0] and idx.index[0] == pd.Timestamp("2019-01-02")


def test_float_shares_from_turnover():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2019-01-02", periods=3), ["SH600000"]], names=["datetime", "instrument"])
    daily = pd.DataFrame({"volume": [1_000_000.0, 0.0, 2_000_000.0], "turn": [2.0, 0.0, 4.0]}, index=idx)
    assert list(float_shares(daily)) == [50_000_000.0, 50_000_000.0, 50_000_000.0]


def _setup(entries, amount20=None, listed=None):
    """entries：(股票, T+1 的 preclose, close, tradestatus, isST)。信号日固定 2021-03-01，入场日 2021-03-02。"""
    t, t1 = pd.Timestamp("2021-03-01"), pd.Timestamp("2021-03-02")
    codes = [e[0] for e in entries]
    df = pd.DataFrame({"_amount20": amount20 or [1e5] * len(codes)},
                      index=pd.MultiIndex.from_product([[t], codes], names=["datetime", "instrument"]))
    rows = [(t1, c, p, cl, ts, st) for c, p, cl, ts, st in entries if p is not None]
    rows += [(t, c, 10.0, 10.0, 1, 0) for c in codes]
    daily = pd.DataFrame(rows, columns=["datetime", "instrument", "preclose", "close", "tradestatus", "isST"])
    daily = daily.set_index(["datetime", "instrument"]).sort_index()
    listed = listed if listed is not None else pd.Series(pd.Timestamp("2000-01-04"), index=codes)
    return full_a_masks(df, daily, listed)


def test_normal_stock_can_buy_and_sell():
    can_buy, can_sell = _setup([("SH600000", 10.0, 10.2, 1, 0)])
    assert can_buy.iloc[0] and can_sell.iloc[0]


def test_st_on_entry_day_blocks_buy():
    can_buy, can_sell = _setup([("SH600000", 10.0, 10.2, 1, 1)])
    assert not can_buy.iloc[0] and can_sell.iloc[0]


def test_suspended_or_missing_entry_day_blocks_both():
    can_buy, can_sell = _setup([("SH600000", 10.0, 10.0, 0, 0), ("SZ000001", None, None, None, None)])
    assert not can_buy.any() and not can_sell.any()


def test_limit_up_uses_real_limit_for_chinext():
    can_buy, _ = _setup([("SZ300750", 100.0, 115.0, 1, 0), ("SZ300751", 100.0, 120.0, 1, 0), ("SH600000", 10.0, 11.0, 1, 0)])
    assert list(can_buy) == [True, False, False]


def test_limit_down_blocks_sell():
    _, can_sell = _setup([("SH600000", 10.0, 9.0, 1, 0), ("SH600001", 10.0, 9.5, 1, 0)])
    assert list(can_sell) == [False, True]


def test_bottom_liquidity_excluded():
    codes = [f"SH6000{i:02d}" for i in range(10)]
    can_buy, _ = _setup([(c, 10.0, 10.1, 1, 0) for c in codes], amount20=[float(i + 1) for i in range(10)])
    assert list(can_buy) == [False, False] + [True] * 8


def test_new_listing_and_bj_excluded():
    listed = pd.Series({"SH600000": pd.Timestamp("2020-06-01"), "BJ430047": pd.Timestamp("2010-01-04")})
    can_buy, _ = _setup([("SH600000", 10.0, 10.1, 1, 0), ("BJ430047", 10.0, 10.1, 1, 0)], listed=listed)
    assert not can_buy.any()
