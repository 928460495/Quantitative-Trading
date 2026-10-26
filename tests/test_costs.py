import pandas as pd
import pytest

from qt.costs import MIN_COMMISSION, trade_cost

DAY = pd.Timestamp("2019-01-02")


def test_min_commission_applies_to_small_trades():
    assert MIN_COMMISSION == 5.0
    assert trade_cost(6700, "buy", DAY) == pytest.approx(5 + 6.7)


def test_large_trade_uses_rate():
    assert trade_cost(1_000_000, "buy", DAY) == pytest.approx(250 + 1000)


def test_sell_adds_stamp_duty_by_date():
    assert trade_cost(10_000, "sell", pd.Timestamp("2023-08-25")) == pytest.approx(5 + 10 + 10)
    assert trade_cost(10_000, "sell", pd.Timestamp("2023-08-28")) == pytest.approx(5 + 10 + 5)


def test_multiplier_scales_everything():
    assert trade_cost(6700, "buy", DAY, multiplier=2) == pytest.approx(2 * (5 + 6.7))


def test_unknown_side_rejected():
    with pytest.raises(ValueError):
        trade_cost(100, "short", DAY)
