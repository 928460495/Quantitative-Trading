import numpy as np
import pandas as pd

from qt.universe import tradable_mask


def _frame(rows):
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp(d), c) for d, c, *_ in rows], names=["datetime", "instrument"])
    data = [r[2:] for r in rows]
    return pd.DataFrame(data, index=idx, columns=["_entry_close", "_entry_volume", "_entry_change"])


LISTED = pd.to_datetime(pd.Series({"SH600000": "2000-01-04", "SZ300999": "2015-01-05", "BJ430001": "2000-01-04"}))


def test_keeps_normal_sample():
    df = _frame([("2016-03-01", "SH600000", 10.0, 1e5, 0.01)])
    assert tradable_mask(df, LISTED).all()


def test_drops_entry_day_suspended_or_limit_up():
    df = _frame(
        [
            ("2016-03-01", "SH600000", np.nan, np.nan, np.nan),  # 入场日停牌
            ("2016-03-02", "SH600000", 10.0, 0.0, 0.0),  # 成交量为 0
            ("2016-03-03", "SH600000", 11.0, 1e5, 0.0998),  # 收盘涨停
            ("2016-03-04", "SH600000", 9.0, 1e5, -0.0999),  # 跌停可以买
        ]
    )
    assert tradable_mask(df, LISTED).tolist() == [False, False, False, True]


def test_drops_new_listings_and_bj():
    df = _frame(
        [
            ("2015-06-01", "SZ300999", 10.0, 1e5, 0.0),  # 上市不满一年
            ("2016-02-01", "SZ300999", 10.0, 1e5, 0.0),
            ("2016-02-01", "BJ430001", 10.0, 1e5, 0.0),
        ]
    )
    assert tradable_mask(df, LISTED).tolist() == [False, True, False]
