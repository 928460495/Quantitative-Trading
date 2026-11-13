"""scripts/live_orders.py：在开发段数据上出单，和回测下一天的成交对得上（计划 5 任务 3）。"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from qt.lab import procedure as pc
from qt.live.daily import State, state_from_backtest

from lab_helpers import dev3d

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live_orders", ROOT / "scripts" / "live_orders.py")
lo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lo)

PROC = pc.PLAN3D.replace(exclude=["is_st"], no_buy=[], rebalance_every=10, n_hold=8, sell_mult=2)


@pytest.fixture(scope="module")
def dev():
    return dev3d(seed=1, n_stocks=60, start="2018-01-02", end="2018-12-31")[0]


def test_orders_agree_with_backtest_sells(dev, tmp_path):
    a = 5                                                       # 跟单从第 5 个交易日开始
    res = pc.run(dev, PROC, a)
    data = pc.market(dev, PROC, a)
    s = 60                                                      # 从开始日起第 60 天：调仓日
    state = state_from_backtest(res, data, s, anchor=data.dates[0])
    path, problems, orders = lo.run(dev, PROC, state, data.dates[s], tmp_path)
    day = res.trades[res.trades["date"] == data.dates[s + 1]]
    assert problems == [] and len(day) > 0 and len(orders.sells) > 0
    assert sorted(orders.sells) == sorted(zip(day.loc[day.side == "sell", "instrument"], day.loc[day.side == "sell", "reason"]))
    assert sorted(b.code for b in orders.buys) == sorted(day.loc[day.side == "buy", "instrument"])
    assert path.exists() and (tmp_path / f"orders-{data.dates[s].date()}.csv").exists()


def test_unknown_signal_day_rejected(dev, tmp_path):
    state = State(anchor=dev["close"].index[0], cash=1e5, positions={})
    with pytest.raises(ValueError, match="没有这一天"):
        lo.run(dev, PROC, state, pd.Timestamp("2018-01-06"), tmp_path)     # 星期六


def test_anchor_not_in_data_rejected(dev, tmp_path):
    state = State(anchor=pd.Timestamp("2017-06-01"), cash=1e5, positions={})
    with pytest.raises(ValueError, match="开始日"):
        lo.run(dev, PROC, state, dev["close"].index[30], tmp_path)


def _day_trades(res, date):
    day = res.trades[res.trades["date"] == date]
    sells = sorted(zip(day.loc[day.side == "sell", "instrument"], day.loc[day.side == "sell", "reason"]))
    return sells, sorted(day.loc[day.side == "buy", "instrument"])


def test_overlay_slot_limits_counted_from_anchor(dev, tmp_path):
    import numpy as np
    months = pd.date_range("2017-01-31", "2019-01-31", freq="ME")
    rng = np.random.default_rng(0)
    dev = {**dev, "overlay_signals": pd.DataFrame({"valuation": rng.random(len(months)),
                                                   "trend": rng.integers(0, 2, len(months)).astype(float)}, index=months)}
    proc = PROC.replace(overlay="value_trend")
    a = 7
    res = pc.run(dev, proc, a)
    data = pc.market(dev, proc, a)
    for s in range(20, len(data.dates) - 1, 10):
        state = state_from_backtest(res, data, s, anchor=data.dates[0])
        _, problems, orders = lo.run(dev, proc, state, data.dates[s], tmp_path)
        sells, buys = _day_trades(res, data.dates[s + 1])
        assert problems == []
        assert sorted(orders.sells) == sells, data.dates[s]
        assert sorted(b.code for b in orders.buys) == buys, data.dates[s]
    assert res.daily["slot_limit"].nunique() >= 3                # 持股只数上限真的在变


def test_failed_safety_check_is_reported(dev, tmp_path):
    state = State(anchor=dev["close"].index[0], cash=-5000.0, positions={})
    path, problems, _ = lo.run(dev, PROC, state, dev["close"].index[20], tmp_path)
    assert any("现金" in p for p in problems) and "现金" in path.read_text()
