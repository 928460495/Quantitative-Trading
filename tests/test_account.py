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


# ---------- 任务 3：核心循环 ----------

from qt.account import run_account  # noqa: E402
from qt.costs import trade_cost  # noqa: E402


def _scenario(scores: dict[str, list], n_days: int, price=10.0, prices=None, can_buy=None, can_sell=None, industry=None,
              raw=None):
    """scores：{股票: 每天的分数}；价格默认全部 10 元，复权价和未复权价相同。"""
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    codes = list(scores)
    prices = prices or {c: [price] * n_days for c in codes}
    raw = raw or prices
    ones = {c: [True] * n_days for c in codes}
    return market_data(
        _long(scores, dates).astype(float), _long(prices, dates).astype(float), _long(raw, dates).astype(float),
        _long(can_buy or ones, dates).astype(bool), _long(can_sell or ones, dates).astype(bool),
        None if industry is None else _long({c: [industry[c]] * n_days for c in codes}, dates),
    )


def _held(result, day=-1):
    trades = result.trades
    side = trades.groupby("instrument")["side"].apply(list)
    return sorted(c for c, sides in side.items() if sides.count("buy") > sides.count("sell"))


RULES = AccountRules(n_hold=4, sell_rank=6, stop_loss=None, drawdown_stop=None)


def test_buys_top_ranked_with_equal_cash_and_lots():
    data = _scenario({c: [s, s] for c, s in zip("ABCDE", [5, 4, 3, 2, 1])}, 2)
    res = run_account(data, RULES)
    buys = res.trades[res.trades["side"] == "buy"].set_index("instrument")
    assert list(buys.index) == ["A", "B", "C", "D"]          # 前 4 名
    assert (buys["shares"] == 2000).all()                    # 分配额 min(100000/4, 20%×100000)=20000，10 元买 2000 股
    cost = trade_cost(20000, "buy", data.dates[1])
    assert res.daily["cash"].iloc[-1] == pytest.approx(100_000 - 4 * (20000 + cost))
    assert res.daily["nav"].iloc[0] == 100_000 and (res.daily["cash"] >= 0).all()


def test_sells_only_when_rank_drops_below_sell_rank():
    rules = AccountRules(n_hold=2, sell_rank=3, stop_loss=None, drawdown_stop=None, max_stock_weight=0.5)
    scores = {"A": [5, 3, 3], "B": [4, 2, 2], "C": [3, 5, 5], "D": [2, 4, 4], "E": [1, 1, 1]}
    res = run_account(_scenario(scores, 3), rules)
    sells = res.trades[res.trades["side"] == "sell"]
    assert list(sells["instrument"]) == ["B"] and list(sells["reason"]) == ["rank"]   # A 排第 3 不卖，B 排第 4 卖
    assert _held(res) == ["A", "C"]


def test_missing_score_means_out_of_pool_and_sold():
    rules = AccountRules(n_hold=1, sell_rank=3, stop_loss=None, drawdown_stop=None, max_stock_weight=1.0)
    res = run_account(_scenario({"A": [5, NAN, NAN], "B": [1, 1, 1]}, 3), rules)
    sells = res.trades[res.trades["side"] == "sell"]
    assert list(sells["instrument"]) == ["A"] and list(sells["reason"]) == ["rank"]


def test_cannot_sell_keeps_position():
    rules = AccountRules(n_hold=2, sell_rank=3, stop_loss=None, drawdown_stop=None, max_stock_weight=0.5)
    scores = {"A": [5, 3, 3], "B": [4, 2, 2], "C": [3, 5, 5], "D": [2, 4, 4], "E": [1, 1, 1]}
    can_sell = {c: [True] * 3 for c in scores} | {"B": [True, False, False]}
    res = run_account(_scenario(scores, 3, can_sell=can_sell), rules)
    assert (res.trades["side"] == "sell").sum() == 0
    assert _held(res) == ["A", "B"]


def test_stock_cap_limits_allocation():
    rules = AccountRules(n_hold=1, stop_loss=None, drawdown_stop=None)
    res = run_account(_scenario({"A": [1, 1]}, 2), rules)
    assert res.trades["shares"].tolist() == [2000]          # 只有 1 个空位，但封顶 20% 净值


def test_industry_cap_skips_to_next():
    scores = {c: [s, s] for c, s in zip("ABCDEF", [6, 5, 4, 3, 2, 1])}
    industry = {"A": "银行", "B": "银行", "C": "电子", "D": "银行", "E": "医药", "F": "传媒"}
    res = run_account(_scenario(scores, 2, industry=industry), RULES)
    assert sorted(res.trades["instrument"]) == ["A", "C", "E", "F"]   # B、D 会让银行超过 30%


def test_all_candidates_too_expensive_keeps_cash():
    res = run_account(_scenario({"A": [2, 2], "B": [1, 1]}, 2, price=300.0), RULES)
    assert res.trades.empty and res.daily["cash"].iloc[-1] == 100_000


def test_cash_never_negative_with_min_commission():
    rules = AccountRules(cash=1003, n_hold=1, max_stock_weight=1.0, stop_loss=None, drawdown_stop=None)
    res = run_account(_scenario({"A": [1, 1]}, 2), rules)        # 1 手 1000 元 + 成本 6 元 > 1003
    assert res.trades.empty and res.daily["cash"].iloc[-1] == 1003


def test_last_buy_shrinks_to_fit_cash():
    rules = AccountRules(n_hold=5, stop_loss=None, drawdown_stop=None)
    res = run_account(_scenario({c: [s, s] for c, s in zip("ABCDE", [5, 4, 3, 2, 1])}, 2), rules)
    assert res.trades["shares"].tolist() == [2000, 2000, 2000, 2000, 1900]   # 第 5 只付完成本不够 2000 股
    assert res.daily["cash"].iloc[-1] >= 0


def test_no_lookahead():
    rng = np.random.default_rng(0)
    codes = list("ABCDEFGH")
    scores = {c: list(rng.normal(size=6)) for c in codes}
    base = run_account(_scenario(scores, 6), RULES)
    # 最后一天：所有股票都没有分数（按"不在股票池"处理）、都不能买卖。偷看当天数据的代码会在最后一天全部卖出
    changed = {c: v[:-1] + [NAN] for c, v in scores.items()}
    blocked = {c: [True] * 5 + [False] for c in codes}
    other = run_account(_scenario(changed, 6, can_buy=blocked, can_sell=blocked), RULES)
    pd.testing.assert_frame_equal(base.trades, other.trades)
    assert len(base.trades) > 0


def test_nav_uses_adjusted_price_for_dividends():
    rules = AccountRules(n_hold=1, max_stock_weight=1.0, stop_loss=None, drawdown_stop=None)
    adj = {"A": [10.0] * 4}
    raw = {"A": [10.0, 10.0, 9.5, 9.5]}                      # 第 3 天除息，未复权价跌、复权价不变
    res = run_account(_scenario({"A": [1] * 4}, 4, prices=adj, raw=raw), rules)
    assert res.daily["nav"].iloc[1] == pytest.approx(res.daily["nav"].iloc[3])


# ---------- 任务 4：个股止损和冷却期 ----------

STOP = AccountRules(n_hold=1, sell_rank=3, max_stock_weight=1.0, stop_loss=0.20, cooldown_days=3, drawdown_stop=None)


def _stop_trades(path, rules=STOP, can_sell=None):
    n = len(path)
    res = run_account(_scenario({"A": [2.0] * n, "B": [1.0] * n}, n, prices={"A": path, "B": [10.0] * n},
                                can_sell=can_sell), rules)
    return res.trades


def test_stop_loss_sells_next_day():
    t = _stop_trades([10.0, 10.0, 7.99, 7.99, 7.99])       # 第 1 天按 10 元买入；第 2 天收盘跌 20.1%
    sells = t[t["side"] == "sell"]
    assert list(sells["reason"]) == ["stop"] and sells["date"].iloc[0] == pd.Timestamp("2019-01-07")  # 第 3 个交易日卖
    t2 = _stop_trades([10.0, 10.0, 8.01, 8.01, 8.01])      # 跌 19.9%：不卖
    assert (t2["side"] == "sell").sum() == 0


def test_stopped_stock_not_rebought_within_cooldown():
    path = [10.0, 10.0, 7.99] + [7.99] * 6
    res = run_account(_scenario({"A": [1.0] * 9}, 9, prices={"A": path}), STOP)
    a_buys = res.trades[res.trades["side"] == "buy"]["date"]
    dates = pd.bdate_range("2019-01-02", periods=9)
    assert list(a_buys) == [dates[1], dates[7]]             # 第 3 天卖出，冷却 3 天（第 4–6 天），第 7 天才能买回


def test_stop_loss_blocked_by_limit_down_retries():
    can_sell = {"A": [True, True, False, True, True], "B": [True] * 5}
    t = _stop_trades([10.0, 10.0, 7.99, 7.5, 7.5], can_sell=can_sell)
    sells = t[t["side"] == "sell"]
    assert list(sells["reason"]) == ["stop"] and sells["date"].iloc[0] == pd.Timestamp("2019-01-08")  # 晚一天


def test_stop_loss_disabled():
    rules = AccountRules(n_hold=1, sell_rank=3, max_stock_weight=1.0, stop_loss=None, drawdown_stop=None)
    t = _stop_trades([10.0, 10.0, 5.0, 5.0], rules=rules)
    assert (t["side"] == "sell").sum() == 0


# ---------- 任务 5：仓位控制 ----------

from qt.account import trend_exposure  # noqa: E402


def test_trend_exposure():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 5.0], index=pd.bdate_range("2019-01-02", periods=8))
    out = trend_exposure(close, window=3, low=0.5)
    # 3 日均线：NaN, NaN, 2, 3, 3.33, 3, 2, 2.67
    assert list(out) == [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0]


FOUR = {c: [s] * 6 for c, s in zip("ABCDE", [4, 3, 2, 1, 0])}
POS = AccountRules(n_hold=4, sell_rank=10, max_stock_weight=0.25, stop_loss=None, drawdown_stop=None)


def _exposure(values):
    return pd.Series(values, index=pd.bdate_range("2019-01-02", periods=len(values)))


def test_exposure_half_sells_lowest_ranked():
    res = run_account(_scenario(FOUR, 6), POS, exposure=_exposure([1, 1, 0.5, 0.5, 0.5, 0.5]))
    sells = res.trades[res.trades["side"] == "sell"]
    assert sorted(sells["instrument"]) == ["C", "D"] and set(sells["reason"]) == {"exposure"}
    assert sells["date"].iloc[0] == pd.Timestamp("2019-01-07")   # 第 2 天的信号，第 3 天卖
    assert res.daily["slot_limit"].iloc[3] == 2


def test_exposure_zero_with_limit_down_holding():
    can_sell = {c: [True] * 6 for c in FOUR} | {"A": [True, True, False, True, True, True]}
    res = run_account(_scenario(FOUR, 6, can_sell=can_sell), POS, exposure=_exposure([1, 1, 0, 0, 0, 0]))
    sells = res.trades[res.trades["side"] == "sell"].set_index("instrument")
    assert sells.loc["A", "date"] == pd.Timestamp("2019-01-08")  # 跌停晚一天卖
    assert sorted(sells.index) == ["A", "B", "C", "D"]
    assert (res.daily["slot_limit"].iloc[3:] == 0).all() and (res.daily["n_holdings"].iloc[-1] == 0)


def test_drawdown_stop_liquidates_and_waits_for_fresh_signal():
    rules = AccountRules(n_hold=1, sell_rank=3, max_stock_weight=1.0, stop_loss=None, drawdown_stop=0.20)
    path = [10.0, 10.0, 7.9, 7.9, 7.9, 7.9, 7.9, 7.9, 7.9, 6.5]
    exposure = _exposure([1, 1, 1, 1, 1, 1, 0, 1, 1, 1])
    res = run_account(_scenario({"A": [1.0] * 10}, 10, prices={"A": path}), rules, exposure=exposure)
    dates = pd.bdate_range("2019-01-02", periods=10)
    t = res.trades
    assert list(t["side"]) == ["buy", "sell", "buy"]
    assert t["reason"].iloc[1] == "drawdown" and t["date"].iloc[1] == dates[3]   # 第 2 天净值跌超 20%，第 3 天全卖
    assert t["date"].iloc[2] == dates[8]          # 均线信号先变空（第 6 天）再变满（第 7 天），第 8 天才回来
    assert (res.daily["slot_limit"].iloc[3:8] == 0).all()
    assert res.daily["n_holdings"].iloc[-1] == 1  # 回来后最高点已重置，再跌 18% 不会触发


# ---------- 任务 6：退市处理和指标 ----------

from qt.account import AccountResult, account_metrics  # noqa: E402


def test_delisted_holding_sold_at_last_price():
    rules = AccountRules(n_hold=1, sell_rank=3, max_stock_weight=1.0, stop_loss=None, drawdown_stop=None)
    path = [10.0, 10.0, 9.0, 8.0, NAN, NAN]                  # 第 3 天之后再也没有价格
    can_sell = {"A": [True, True, True, True, False, False]}
    res = run_account(_scenario({"A": [1.0, 1.0, 1.0, 1.0, NAN, NAN]}, 6, prices={"A": path}, can_sell=can_sell), rules)
    sells = res.trades[res.trades["side"] == "sell"]
    assert list(sells["reason"]) == ["delisted"] and sells["date"].iloc[0] == pd.Timestamp("2019-01-08")
    assert sells["price"].iloc[0] == 8.0
    assert res.daily["n_holdings"].iloc[-1] == 0


def test_metrics_on_known_path():
    dates = pd.bdate_range("2019-01-02", periods=4)
    daily = pd.DataFrame({"nav": [100.0, 110.0, 99.0, 121.0], "cash": [100.0, 10.0, 9.0, 21.0],
                          "stock_value": [0.0, 100.0, 90.0, 100.0], "cost": [0.0, 0.1, 0.0, 0.2]}, index=dates)
    trades = pd.DataFrame([(dates[2], "A", "sell", 100, 1.0, 50.0, 0.1, "stop"),
                           (dates[3], "B", "sell", 100, 1.0, 20.0, 0.1, "delisted")], columns=[
        "date", "instrument", "side", "shares", "price", "value", "cost", "reason"])
    bench = pd.Series([1000.0, 1010.0, 1020.0, 1030.0], index=dates)
    m = account_metrics(AccountResult(daily=daily, trades=trades), bench)
    assert m["max_drawdown"] == pytest.approx(-0.10)
    assert m["annual_return"] == pytest.approx(1.21 ** (252 / 3) - 1)
    assert m["bench_annual"] == pytest.approx(1.03 ** (252 / 3) - 1)
    assert m["excess_annual"] == pytest.approx(m["annual_return"] - m["bench_annual"])
    assert m["turnover"] == pytest.approx((50 / 99 + 20 / 121) / 3)
    assert (m["n_trades"], m["n_stops"], m["n_delisted"]) == (2, 1, 1)
    assert m["avg_exposure"] == pytest.approx(np.mean([100 / 110, 90 / 99, 100 / 121]))


# ---------- 任务 7：和 Qlib 回测核对（需要数据包和第 5 个 PR 的缓存）----------


@pytest.mark.data
def test_account_engine_matches_qlib_backtest():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("check_account", root / "scripts" / "check_account.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    out = check.compare()
    assert abs(out["ours"]["annual_return"] - out["qlib"]["annual_return"]) < 0.01
    assert out["daily_corr"] > 0.95


# ---------- 最终审查发现的问题 ----------


def test_missing_industry_is_none_not_nan():
    d = pd.bdate_range("2019-01-02", periods=2)
    idx = pd.MultiIndex.from_product([d, ["A", "B"]], names=["datetime", "instrument"])
    s = pd.Series(1.0, index=idx)
    data = market_data(s, s, s, s.astype(bool), s.astype(bool), pd.Series(["银行"], index=idx[:1]))
    assert data.industry[0, 0] == "银行"
    assert all(x is None for x in [data.industry[0, 1], data.industry[1, 0], data.industry[1, 1]])


def test_exposure_without_matching_dates_is_rejected():
    data = _scenario(FOUR, 6)
    wrong = pd.Series(0.0, index=pd.bdate_range("2030-01-01", periods=6))
    with pytest.raises(ValueError, match="exposure"):
        run_account(data, POS, exposure=wrong)
