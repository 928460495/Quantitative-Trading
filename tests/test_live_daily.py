"""出单流程：账户状态、调仓日、安全检查、清单（计划 5 任务 2、3）。"""

import json

import numpy as np
import pandas as pd
import pytest

from qt.account import AccountRules, run_account
from qt.live.daily import (State, check_orders, load_state, make_orders, save_state, state_from_backtest,
                           write_sheet)
from qt.live.orders import Buy, Orders
from test_live_orders import RULES, _market


def _flat_next_day(data, s):
    """让第 s+1 天的价格和第 s 天一样：这样按参考价算的股数应该和回测完全相同。"""
    for a in (data.adj_close, data.raw_close):
        a[s + 1] = a[s]
    return data


@pytest.mark.parametrize("seed", range(4))
def test_orders_match_backtest_next_day_when_prices_unchanged(seed):
    data = _market(seed)
    s = 40                                                     # 第 40 天是调仓日（每 5 天）
    data = _flat_next_day(data, s)
    data.can_buy[s] &= ~np.isnan(data.raw_close[s])
    res = run_account(data, RULES)
    state = state_from_backtest(res, data, s, anchor=data.dates[0])
    orders, meta = make_orders(data, RULES, None, state, data.dates[s])
    day = res.trades[res.trades["date"] == data.dates[s + 1]]
    assert meta["rebalance"] and meta["slot_limit"] == RULES.n_hold
    assert sorted(orders.sells) == sorted(zip(day.loc[day.side == "sell", "instrument"], day.loc[day.side == "sell", "reason"]))
    assert sorted((b.code, b.shares) for b in orders.buys) == sorted(
        zip(day.loc[day.side == "buy", "instrument"], day.loc[day.side == "buy", "shares"]))


def test_rebalance_counted_from_anchor_and_slot_limit_from_exposure():
    data = _market(7)
    res = run_account(data, RULES)
    state = state_from_backtest(res, data, 42, anchor=data.dates[2])
    exposure = pd.Series(0.5 + 1e-9, index=data.dates)
    _, meta = make_orders(data, RULES, exposure, state, data.dates[42])
    assert meta["rebalance"] and meta["slot_limit"] == 4
    _, meta = make_orders(data, RULES, exposure, state, data.dates[43])
    assert not meta["rebalance"]


def test_signal_day_before_anchor_rejected():
    data = _market(7)
    state = State(anchor=data.dates[10], cash=1e5, positions={})
    with pytest.raises(ValueError):
        make_orders(data, RULES, None, state, data.dates[5])


def test_state_round_trip(tmp_path):
    st = State(anchor=pd.Timestamp("2021-01-04"), cash=12345.6,
               positions={"SH600000": {"shares": 300, "buy_day": "2021-02-01"}})
    save_state(st, tmp_path / "s.json")
    assert load_state(tmp_path / "s.json") == st
    assert json.loads((tmp_path / "s.json").read_text())["cash"] == 12345.6


def _orders(**kw):
    base = dict(sells=[("SH600003", "rank")], buys=[Buy("SH600001", 200, 10.0, 2000.0, 5.0)], cash_after=100.0)
    return Orders(**{**base, **kw})


def test_check_orders_clean():
    assert check_orders(_orders(), {"SH600003": 100, "SH600004": 100}, slot_limit=2) == []


@pytest.mark.parametrize("orders,held,slot,needle", [
    (_orders(cash_after=-1.0), {"SH600003": 100}, 2, "现金"),
    (_orders(buys=[Buy("SH600001", 150, 10.0, 1500.0, 5.0)]), {"SH600003": 100}, 2, "整手"),
    (_orders(buys=[Buy("SH688001", 100, 10.0, 1000.0, 5.0)]), {"SH600003": 100}, 2, "整手"),
    (_orders(sells=[("SH600009", "rank")]), {"SH600003": 100}, 2, "没有持有"),
    (_orders(buys=[Buy("SH600003", 200, 10.0, 2000.0, 5.0)]), {"SH600003": 100}, 2, "又卖又买"),
    (_orders(), {"SH600003": 100, "SH600004": 100, "SH600005": 100}, 2, "只数"),
])
def test_check_orders_flags_problems(orders, held, slot, needle):
    problems = check_orders(orders, held, slot_limit=slot)
    assert any(needle in p for p in problems), problems


def test_write_sheet(tmp_path):
    meta = {"signal_date": "2021-06-01", "rebalance": True, "slot_limit": 2, "cash": 5000.0}
    write_sheet(_orders(), meta, ["示例问题"], tmp_path, held={"SH600003": 700})
    csv = pd.read_csv(tmp_path / "orders-2021-06-01.csv", dtype={"code": str})
    assert list(csv["side"]) == ["sell", "buy"] and list(csv["code"]) == ["SH600003", "SH600001"]
    assert list(csv["shares"]) == [700, 200]
    md = (tmp_path / "orders-2021-06-01.md").read_text()
    assert "示例问题" in md and "SH600001" in md and "跌出排名" in md and "nan" not in md


def test_state_from_backtest_is_end_of_day():
    data = _market(3)
    res = run_account(data, RULES)
    st = state_from_backtest(res, data, 41, anchor=data.dates[0])   # 第 41 天有成交（第 40 天出的信号）
    assert (res.trades["date"] == data.dates[41]).any()
    assert st.cash == float(res.daily["cash"].iloc[41])
    assert len(st.positions) == int(res.daily["n_holdings"].iloc[41])
