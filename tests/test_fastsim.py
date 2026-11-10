"""计划 3c 任务 4：快速模拟器。规则和 qt.account.run_account 一样，只是不按手取整、行业上限按只数算。"""

import numpy as np
import pandas as pd
import pytest

from qt.account import AccountRules, run_account
from qt.costs import trade_cost
from qt.drawdown import DEFAULT_TIERS
from qt.fastsim import SimParams, sharpe, simulate
from qt.wide_eval import _market_data


def _wide(n_days, n_stocks, seed=0, price=None, vol=0.02):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n_days)
    cols = [f"SH6{i:05d}" for i in range(n_stocks)]
    if price is None:
        adj = pd.DataFrame(10 * np.exp(np.cumsum(rng.normal(0, vol, (n_days, n_stocks)), axis=0)), index=idx, columns=cols)
    else:
        adj = pd.DataFrame(price, index=idx, columns=cols, dtype=float)
    score = pd.DataFrame(rng.normal(size=(n_days, n_stocks)), index=idx, columns=cols)
    true = pd.DataFrame(True, index=idx, columns=cols)
    return adj, score, true


def test_sim_params_defaults_match_reference():
    p = SimParams()
    assert (p.n_hold, p.rebalance_every, p.sell_mult, p.industry_cap, p.tiers, p.min_trade, p.cost_multiplier) == \
        (20, 5, 3, 0.30, True, 2000.0, 1.0)


def test_fastsim_matches_account_when_lots_negligible():
    adj, score, true = _wide(200, 30, seed=1)
    raw = adj * 0 + 0.01                                 # 0.01 元一股：按手取整的误差可以忽略
    data = _market_data(score, adj, raw, true, true, None)
    p = SimParams(n_hold=5, rebalance_every=5, sell_mult=2, industry_cap=1.0, tiers=True, min_trade=0)
    r_fast = simulate(data, p, cash=1e6)
    acct = run_account(data, AccountRules(cash=1e6, n_hold=5, sell_rank=10, max_industry_weight=1.0, rebalance_every=5,
                                          drawdown_stop=None, drawdown_tiers=DEFAULT_TIERS))
    nav_fast = 1e6 * (1 + r_fast).cumprod()
    assert r_fast.index.equals(acct.daily.index) and r_fast.iloc[0] == 0
    assert (nav_fast / acct.daily["nav"] - 1).abs().max() < 0.005


def test_fastsim_respects_can_buy_and_can_sell():
    adj, score, true = _wide(6, 3, price=10.0)
    score.loc[:, :] = [[3.0, 2.0, 1.0]] * 6
    score.iloc[1:, 0] = 0.0                              # A 从信号日 1 起排到最后
    can_buy = true.copy()
    can_buy.iloc[0, 0] = False                           # 信号日 0：A 第二天涨停买不进 → 买 B
    can_sell = true.copy()
    p = SimParams(n_hold=1, rebalance_every=1, sell_mult=1, industry_cap=1.0, tiers=False, min_trade=0)
    up = adj.copy()
    up.iloc[2:, 0] = 11.0                                # A 涨 10%：如果误买了 A，第 2 天会有正收益
    r = simulate(_market_data(score, up, up, can_buy, can_sell, None), p)
    assert r.iloc[2] == pytest.approx(0.0, abs=1e-12)
    can_sell2 = true.copy()
    can_sell2.iloc[1:4, 1] = False                       # B 跌停卖不掉：第 2–4 天还持有 B
    score2 = score.copy()
    score2.iloc[1:, 1] = -1.0                            # B 从信号日 1 起排最后
    down = adj.copy()
    down.iloc[3, 1] = 9.0
    r2 = simulate(_market_data(score2, down, down, can_buy, can_sell2, None), p)
    assert r2.iloc[3] == pytest.approx(-0.02, abs=0.001)    # 还拿着 B（仓位 20%），吃到了 10% 的下跌


def test_fastsim_min_commission_applies():
    adj, score, true = _wide(3, 5, price=10.0)
    score.iloc[:, 1:] = np.nan                           # 只有一只股票有分数
    p = SimParams(n_hold=5, rebalance_every=1, sell_mult=3, industry_cap=1.0, tiers=False, min_trade=0)
    r = simulate(_market_data(score, adj, adj, true, true, None), p, cash=25_000)
    assert r.iloc[1] == pytest.approx(-trade_cost(5000, "buy", adj.index[1]) / 25_000)


def test_fastsim_industry_cap_by_count():
    adj, score, true = _wide(4, 15, price=10.0)
    score.loc[:, :] = list(range(15, 0, -1))             # 前 10 名都在行业 X
    industry = pd.DataFrame([["X"] * 10 + ["Y"] * 5] * 4, index=adj.index, columns=adj.columns, dtype=object)
    up = adj.copy()
    up.iloc[2:, :10] = 11.0                              # X 行业第 2 天涨 10%
    data = _market_data(score, up, up, true, true, industry)
    capped = simulate(data, SimParams(n_hold=10, rebalance_every=1, sell_mult=3, industry_cap=0.2, tiers=False, min_trade=0))
    free = simulate(data, SimParams(n_hold=10, rebalance_every=1, sell_mult=3, industry_cap=1.0, tiers=False, min_trade=0))
    assert capped.iloc[2] == pytest.approx(0.02, abs=0.002)    # 只买了 2 只 X
    assert free.iloc[2] == pytest.approx(0.10, abs=0.005)


def test_sharpe_basic():
    r = pd.Series([0.01, -0.005, 0.01, -0.005])
    assert sharpe(r) == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    assert np.isnan(sharpe(pd.Series([0.0, 0.0])))


@pytest.mark.slow
def test_fastsim_runs_fast():
    import time

    adj, score, true = _wide(2200, 3000, seed=2)
    data = _market_data(score, adj, adj, true, true, None)
    t = time.time()
    simulate(data, SimParams())
    assert time.time() - t < 3
