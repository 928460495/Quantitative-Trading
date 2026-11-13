"""出单函数和回测账户引擎必须一模一样（计划 5 任务 1）。"""

import numpy as np
import pandas as pd
import pytest

from qt.account import AccountRules, MarketData, run_account
from qt.live.orders import Orders, plan_orders

RULES = AccountRules(n_hold=8, sell_rank=24, max_stock_weight=0.20, max_industry_weight=0.30, stop_loss=None,
                     cooldown_days=20, drawdown_stop=None, rebalance_every=5, min_trade=2000.0)


def _market(seed: int, n_days: int = 90, n_stocks: int = 40) -> MarketData:
    rng = np.random.default_rng(seed)
    codes = [f"SH60{i:04d}" for i in range(n_stocks - 3)] + ["SH688001", "SH688002", "SZ000001"]
    raw = 5 + 60 * rng.random(n_stocks) * np.exp(np.cumsum(rng.normal(0, 0.03, (n_days, n_stocks)), axis=0))
    adj = raw * (1 + rng.random(n_stocks))                       # 复权价和未复权价不同
    halt = rng.random((n_days, n_stocks)) < 0.03                 # 停牌：没有价格
    halt[-1] = False                                             # 最后一天停牌会被回测当成退市，出单管不了退市
    raw[halt], adj[halt] = np.nan, np.nan
    score = rng.normal(size=(n_days, n_stocks))
    score[rng.random((n_days, n_stocks)) < 0.1] = np.nan
    industry = rng.choice(np.array(["银行", "医药", "电子", "化工", None], dtype=object), size=n_stocks)
    return MarketData(dates=pd.bdate_range("2020-01-01", periods=n_days), instruments=pd.Index(sorted(codes)),
                      score=score, adj_close=adj, raw_close=raw,
                      can_buy=(rng.random((n_days, n_stocks)) > 0.1) & ~halt,
                      can_sell=(rng.random((n_days, n_stocks)) > 0.1) & ~halt,
                      industry=np.tile(industry, (n_days, 1)))


def _replay(data: MarketData, rules: AccountRules, exposure: pd.Series | None = None) -> int:
    """跑回测，再每天把回测账户的状态喂给出单函数，对比当天的成交。返回比过的有成交的天数。"""
    res = run_account(data, rules, exposure)
    marks = pd.DataFrame(data.adj_close).ffill().to_numpy()
    col = {c: j for j, c in enumerate(data.instruments)}
    units: dict[str, float] = {}
    checked = 0
    for i in range(1, len(data.dates)):
        s = i - 1
        day = res.trades[res.trades["date"] == data.dates[i]]
        cash = float(res.daily["cash"].iloc[s])
        values = {c: u * marks[i, col[c]] for c, u in units.items()}
        row = lambda a: pd.Series(a[s], index=data.instruments)
        tradable = pd.Series(~np.isnan(data.adj_close[i]) & ~np.isnan(data.raw_close[i]), index=data.instruments)
        got = plan_orders(score=row(data.score), can_buy=row(data.can_buy), can_sell=row(data.can_sell),
                          industry=row(data.industry), tradable=tradable, holdings=values, cash=cash,
                          price=pd.Series(data.raw_close[i], index=data.instruments),
                          slot_limit=int(res.daily["slot_limit"].iloc[i]), rebalance=s % rules.rebalance_every == 0,
                          rules=rules, date=data.dates[i])
        sells = day[day["side"] == "sell"]
        buys = day[day["side"] == "buy"]
        assert sorted(got.sells) == sorted(zip(sells["instrument"], sells["reason"])), data.dates[i]
        assert sorted((b.code, b.shares) for b in got.buys) == sorted(zip(buys["instrument"], buys["shares"])), data.dates[i]
        for t in day.itertuples():                                # 状态按回测的成交推进
            if t.side == "sell":
                del units[t.instrument]
            else:
                units[t.instrument] = t.value / data.adj_close[i, col[t.instrument]]
        checked += len(day) > 0
    return checked


@pytest.mark.parametrize("seed", range(6))
def test_matches_account_engine(seed):
    assert _replay(_market(seed), RULES) > 5


@pytest.mark.parametrize("seed", range(4))
def test_matches_account_engine_with_slot_limit_and_budget(seed):
    data = _market(100 + seed)
    rng = np.random.default_rng(seed)
    k = np.repeat(rng.integers(2, 9, len(data.dates) // 10 + 1), 10)[:len(data.dates)]
    exposure = pd.Series(k / RULES.n_hold + 1e-9, index=data.dates)
    rules = AccountRules(**{**RULES.__dict__, "slot_budget": 0.75 / RULES.n_hold, "cash_yield": 0.02})
    assert _replay(data, rules, exposure) > 5


def _one_day(**kw) -> Orders:
    codes = pd.Index(["SH600001", "SH600002", "SH600003"])
    base = dict(score=pd.Series([3.0, 2.0, 1.0], index=codes), can_buy=pd.Series(True, index=codes),
                can_sell=pd.Series(True, index=codes), industry=pd.Series([None] * 3, index=codes, dtype=object),
                tradable=pd.Series(True, index=codes), holdings={}, cash=100_000.0,
                price=pd.Series(10.0, index=codes), slot_limit=2, rebalance=True,
                rules=AccountRules(n_hold=2, sell_rank=2, stop_loss=None, drawdown_stop=None, max_stock_weight=0.6),
                date=pd.Timestamp("2021-06-01"))
    return plan_orders(**{**base, **kw})


def test_buys_top_ranked_in_whole_lots():
    got = _one_day()
    assert [b.code for b in got.buys] == ["SH600001", "SH600002"]
    assert all(b.shares % 100 == 0 and b.shares > 0 for b in got.buys)
    assert got.buys[0].price == 10.0 and got.buys[0].value == b_value(got.buys[0])


def b_value(b) -> float:
    return b.shares * b.price


def test_not_rebalance_day_only_trims_to_slot_limit():
    held = {"SH600003": 30_000.0, "SH600002": 30_000.0}
    got = _one_day(holdings=held, rebalance=False, slot_limit=1)
    assert got.sells == [("SH600003", "exposure")]
    assert got.buys == []


def test_rank_sell_only_on_rebalance_day():
    held = {"SH600003": 30_000.0}
    assert _one_day(holdings=held, rebalance=False, cash=70_000.0).sells == []
    assert _one_day(holdings=held, rebalance=True, cash=70_000.0).sells == [("SH600003", "rank")]


def test_unsupported_rules_rejected():
    with pytest.raises(NotImplementedError):
        _one_day(rules=AccountRules(n_hold=2, stop_loss=0.2, drawdown_stop=None))
    with pytest.raises(NotImplementedError):
        _one_day(rules=AccountRules(n_hold=2, stop_loss=None, drawdown_stop=0.2))


def test_no_buys_on_non_rebalance_day_even_with_empty_slots():
    assert _one_day(rebalance=False).buys == []


def test_budget_below_min_trade_buys_nothing():
    rules = AccountRules(n_hold=2, sell_rank=2, stop_loss=None, drawdown_stop=None, max_stock_weight=0.6, min_trade=2000.0)
    assert _one_day(rules=rules, cash=3000.0).buys == []
    assert len(_one_day(rules=rules, cash=5000.0).buys) == 2


def test_untradable_next_day_skipped_even_with_reference_price():
    codes = pd.Index(["SH600001", "SH600002", "SH600003"])
    got = _one_day(tradable=pd.Series([False, True, True], index=codes))
    assert [b.code for b in got.buys] == ["SH600002", "SH600003"]
