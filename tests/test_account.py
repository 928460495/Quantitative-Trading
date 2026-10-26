"""账户回测引擎：手工构造的小场景，每个测一条规则。"""

import numpy as np
import pandas as pd
import pytest

from qt.account import AccountRules, lot_shares, market_data

NAN = float("nan")


def _long(table: dict[str, list], dates) -> pd.Series:
    """{股票: 每天的值} → (datetime, instrument) 索引的长格式 Series。"""
    dates = pd.DatetimeIndex(dates)
    rows = {(d, code): v for code, values in table.items() for d, v in zip(dates, values)}
    s = pd.Series(rows, dtype=object)
    s.index.names = ["datetime", "instrument"]
    return s


def test_rules_defaults_match_spec():
    r = AccountRules()
    assert (r.cash, r.n_hold, r.sell_rank) == (100_000, 15, 45)
    assert (r.max_stock_weight, r.max_industry_weight, r.stop_loss, r.cooldown_days, r.drawdown_stop) == (0.20, 0.30, 0.20, 20, 0.20)


def test_lot_shares():
    assert lot_shares(6700, 12.3, "SH600000") == 500
    assert lot_shares(6700, 70.0, "SH600000") == 0
    assert lot_shares(6700, 30.0, "SH688001") == 223
    assert lot_shares(5000, 30.0, "SH688001") == 0
    assert lot_shares(6700, NAN, "SH600000") == 0


def test_market_data_shapes_and_missing_masks():
    dates = ["2019-01-02", "2019-01-03"]
    score = _long({"A": [1.0, 2.0], "B": [3.0, NAN], "C": [0.5, 0.1]}, dates).astype(float)
    price = _long({"A": [10.0, 10.5], "B": [20.0, NAN], "C": [5.0, 5.1]}, dates).astype(float)
    can_buy = _long({"A": [True, True]}, dates).astype(bool)
    data = market_data(score, price, price, can_buy, can_buy)
    assert list(data.instruments) == ["A", "B", "C"]
    assert data.score.shape == (2, 3) and data.adj_close.shape == (2, 3)
    assert data.can_buy.dtype == bool and data.can_buy[:, 1:].sum() == 0
    assert np.isnan(data.adj_close[1, 1])
    assert data.industry is None
