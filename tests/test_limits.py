import numpy as np
import pandas as pd
import pytest

from qt.limits import limit_flags, limit_pct, limit_prices


def _pct(instruments, dates, st):
    idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(dates), instruments], names=["datetime", "instrument"])
    is_st = pd.Series(st, index=idx)
    return limit_pct(idx.get_level_values("instrument"), idx.get_level_values("datetime"), is_st)


def test_limit_pct_by_board_and_date():
    out = _pct(
        ["SH600000", "SH600000", "SZ002001", "SZ300750", "SZ300750", "SZ300001", "SZ300001", "SZ301001", "SH688001", "SH689009"],
        ["2019-01-02", "2019-01-02", "2019-01-02", "2020-08-21", "2020-08-24", "2020-08-21", "2020-08-24", "2021-09-01", "2019-07-22", "2020-01-02"],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    )
    assert list(out) == [0.10, 0.05, 0.05, 0.10, 0.20, 0.05, 0.20, 0.20, 0.20, 0.20]


def test_limit_price_rounding_half_up():
    up, down = limit_prices(pd.Series([10.05, 3.33, np.nan]), pd.Series([0.10, 0.05, 0.10]))
    assert up.iloc[0] == pytest.approx(11.06) and down.iloc[0] == pytest.approx(9.05)  # 11.055 → 11.06，9.045 → 9.05
    assert up.iloc[1] == pytest.approx(3.50) and down.iloc[1] == pytest.approx(3.16)   # 3.4965 → 3.50，3.1635 → 3.16
    assert np.isnan(up.iloc[2])


def test_limit_flags_at_limit_and_not():
    at_up, at_down = limit_flags(pd.Series([11.06, 11.05, 9.05, 10.0]), pd.Series([10.05] * 4), pd.Series([0.10] * 4))
    assert list(at_up) == [True, False, False, False]
    assert list(at_down) == [False, False, True, False]


def test_ex_dividend_day_uses_reference_preclose():
    # 前一天收盘 10.00，除息后参考价 9.50，当天收盘 10.45：按 9.50 算正好涨停
    at_up, _ = limit_flags(pd.Series([10.45]), pd.Series([9.50]), pd.Series([0.10]))
    assert at_up.iloc[0]
    at_up_wrong, _ = limit_flags(pd.Series([10.45]), pd.Series([10.00]), pd.Series([0.10]))
    assert not at_up_wrong.iloc[0]


def test_missing_preclose_is_not_a_limit():
    at_up, at_down = limit_flags(pd.Series([10.0]), pd.Series([np.nan]), pd.Series([0.10]))
    assert not at_up.iloc[0] and not at_down.iloc[0]
