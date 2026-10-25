import numpy as np
import pandas as pd
import pytest

from qt.costs import buy_cost, sell_cost
from qt.simulate import BufferRule, simulate_buffer, turnover


def _panel(scores: dict[str, list[float]], rets: dict[str, list[float]] | None = None):
    dates = pd.bdate_range("2016-01-04", periods=len(next(iter(scores.values()))))
    idx = pd.MultiIndex.from_product([dates, list(scores)], names=["datetime", "instrument"])
    score = pd.Series([scores[c][i] for i in range(len(dates)) for c in scores], index=idx)
    rets = rets or {c: [0.0] * len(dates) for c in scores}
    ret = pd.Series([rets[c][i] for i in range(len(dates)) for c in scores], index=idx)
    ones = pd.Series(True, index=idx)
    return score, ret, ones


def test_buffer_keeps_holding_until_rank_falls_past_sell_line():
    # A 第一天排第 1，第二天排第 2（仍在缓冲线 2 以内），第三天排第 3（掉出缓冲线）
    score, ret, ones = _panel({"A": [3, 2, 1], "B": [2, 3, 3], "C": [1, 1, 2]})
    sim = simulate_buffer(score, ret, ones, ones, BufferRule(n_hold=1, sell_rank=2))
    assert sim["buys"].tolist() == [1, 0, 1]
    assert sim["sells"].tolist() == [0, 0, 1]


def test_no_buffer_trades_every_rank_change():
    score, ret, ones = _panel({"A": [3, 2, 1], "B": [2, 3, 3], "C": [1, 1, 2]})
    sim = simulate_buffer(score, ret, ones, ones, BufferRule(n_hold=1, sell_rank=1))
    assert sim["sells"].tolist() == [0, 1, 0]


def test_unbuyable_top_stock_is_skipped():
    score, ret, ones = _panel({"A": [3], "B": [2], "C": [1]})
    can_buy = ones.copy()
    can_buy.loc[(slice(None), "A")] = False
    sim = simulate_buffer(score, pd.Series([0.05, 0.01, 0.0], index=score.index), can_buy, ones, BufferRule(1, 1))
    # 买到的是 B，组合收益 1%
    assert sim["portfolio"].iloc[0] == pytest.approx(0.01)


def test_unsellable_stock_is_held_and_suspended_return_is_zero():
    score, _, ones = _panel({"A": [3, 1], "B": [2, 3], "C": [1, 2]})
    ret = pd.Series([0.0, 0.0, 0.0, np.nan, 0.02, 0.0], index=score.index)  # 第二天 A 停牌，没有收益
    can_sell = ones.copy()
    can_sell.loc[(score.index.get_level_values(0)[3], "A")] = False
    sim = simulate_buffer(score, ret, ones, can_sell, BufferRule(1, 1))
    assert sim["sells"].tolist() == [0, 0]
    assert sim["buys"].tolist() == [1, 0]
    assert sim["portfolio"].iloc[1] == 0.0


def test_costs_and_excess():
    score, _, ones = _panel({"A": [3, 1], "B": [2, 3], "C": [1, 2]})
    ret = pd.Series([0.03, 0.0, 0.0, 0.0, 0.03, 0.0], index=score.index)
    sim = simulate_buffer(score, ret, ones, ones, BufferRule(1, 1))
    d0, d1 = sim.index
    assert sim.loc[d0, "cost"] == pytest.approx(buy_cost(d0))
    assert sim.loc[d1, "cost"] == pytest.approx(buy_cost(d1) + sell_cost(d1))
    assert sim.loc[d0, "excess"] == pytest.approx(0.03 - 0.01)
    assert sim.loc[d1, "net"] == pytest.approx(0.03 - 0.01 - sim.loc[d1, "cost"])
    assert turnover(sim, 1).tolist() == [1.0]
