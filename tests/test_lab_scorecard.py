"""实验室：成绩单、配对比较、采纳规则、顶端证据（docs/FRAMEWORK.md 第 5、6 节）。小假数据。"""

import numpy as np
import pandas as pd
import pytest

from qt.account import AccountResult
from qt.diagnostics import annualize, holdable_benchmark
from qt.lab import evidence as ev
from qt.lab import gates
from qt.lab import procedure as pc
from qt.lab import scorecard as sc
from qt.lab.protocol import BIG_LOSS

from lab_helpers import dev3d

IDX = pd.bdate_range("2018-12-03", "2019-02-28")


def _result(nav, trades=(), cost=0.0, idx=IDX):
    daily = pd.DataFrame({"nav": nav, "cash": 0.0, "stock_value": nav, "n_holdings": 1, "slot_limit": 1,
                          "cost": cost, "n_buys": 0, "n_sells": 0}, index=pd.Index(idx, name="date"))
    tr = pd.DataFrame(list(trades), columns=["date", "instrument", "side", "shares", "price", "value", "cost", "reason"])
    return AccountResult(daily=daily, trades=tr)


def _grow(rate, n=len(IDX), start=100_000.0):
    return start * (1 + rate) ** np.arange(n)


def test_closed_positions_pair_each_buy_with_the_next_sell_net_of_costs():
    t = [(IDX[1], "A", "buy", 100, 10.0, 1000.0, 5.0, "buy"), (IDX[2], "B", "buy", 100, 10.0, 1000.0, 5.0, "buy"),
         (IDX[5], "A", "sell", 100, 5.0, 500.0, 5.0, "rank"), (IDX[6], "A", "buy", 100, 5.0, 500.0, 5.0, "buy"),
         (IDX[9], "A", "sell", 100, 8.0, 800.0, 5.0, "event")]
    got = sc.closed_positions(_result(_grow(0.0), t))
    assert got["instrument"].tolist() == ["A", "A"]                          # B 还没卖，不算
    assert got["ret"].tolist() == pytest.approx([495 / 1005 - 1, 795 / 505 - 1])
    assert got["pnl"].tolist() == pytest.approx([-510.0, 290.0])
    assert got["buy_date"].tolist() == [IDX[1], IDX[6]] and got["sell_date"].tolist() == [IDX[5], IDX[9]]


def test_daily_returns_start_when_all_phases_have_started():
    res = {0: _result(_grow(0.001)), 5: _result(_grow(0.002, len(IDX) - 5), idx=IDX[5:])}
    r = sc.daily_returns(res)
    assert r.index[0] == IDX[5] and list(r.columns) == [0, 5]
    assert r[0].iloc[0] == pytest.approx(0.001) and r[5].iloc[0] == 0.0 and r[5].iloc[1] == pytest.approx(0.002)


def _dev_for(idx=IDX):
    close = pd.DataFrame({"A": 10.0 * 1.001 ** np.arange(len(idx)), "B": 10.0 * 0.999 ** np.arange(len(idx))}, index=idx)
    buy = pd.DataFrame(True, index=idx, columns=close.columns)
    index = pd.DataFrame({"H00300": 100.0 * 1.0005 ** np.arange(len(idx))}, index=idx)
    return {"close": close, "can_buy": buy, "index": index}


def test_scorecard_splits_by_era_and_counts_big_losses_by_sell_date():
    lose = lambda day, pct: [(IDX[1], f"X{day}", "buy", 100, 10.0, 10_000.0, 0.0, "buy"),
                             (IDX[day], f"X{day}", "sell", 100, 1.0, 10_000.0 * (1 + pct), 0.0, "rank")]
    trades0 = lose(10, -0.50) + lose(30, -0.40) + lose(31, -0.399) + lose(32, -0.80)   # 第 10 天在 2018（阶段二）
    res = {0: _result(_grow(0.001), trades0, cost=10.0), 5: _result(_grow(0.003, len(IDX) - 5), lose(40, -0.45), idx=IDX[5:])}
    card = sc.scorecard(_dev_for(), res)
    assert list(card.index) == ["二", "三", "全段"] and list(card.columns) == sc.COLUMNS
    assert card["大亏只数"].to_dict() == {"二": 1.0, "三": 3.0, "全段": 4.0}        # −39.9% 不算；两个起点合计
    assert BIG_LOSS == -0.40
    r = sc.daily_returns(res)
    y3 = r.index.year == 2019
    want = np.mean([(1 + r.loc[y3, k]).prod() ** (252 / y3.sum()) - 1 for k in (0, 5)])
    assert card.loc["三", "年化"] == pytest.approx(want)
    assert card.loc["三", "年化"] == pytest.approx((1.001 ** 252 - 1 + 1.003 ** 252 - 1) / 2, rel=1e-6)
    assert card.loc["三", "沪深300全收益"] == pytest.approx(1.0005 ** 252 - 1, rel=1e-6)
    assert (card["超额"] == card["年化"] - card["能持有等权"]).all()
    assert card.loc["全段", "最差起点超额"] < card.loc["全段", "超额"]              # 起点 0 赚得少
    assert card.loc["全段", "最大回撤"] == 0.0 and card.loc["全段", "年成本"] > 0
    # 最差五笔：起点 0 在阶段三亏 4000+3990+8000，起点 5 亏 4500；各自除以 10 万再平均
    assert card.loc["三", "最差五笔占比"] == pytest.approx(-(15_990 + 4_500) / 2 / 100_000)


def test_scorecard_benchmark_is_phase_matched():
    dev = _dev_for()
    res = {0: _result(_grow(0.001)), 5: _result(_grow(0.001, len(IDX) - 5), idx=IDX[5:])}
    card = sc.scorecard(dev, res)
    hold, _ = sc.bench_returns(dev, res)
    want5 = holdable_benchmark(dev["close"].iloc[5:], dev["can_buy"], 20).reindex(hold.index).fillna(0.0)
    pd.testing.assert_series_equal(hold[5], want5, check_names=False)
    assert not hold[0].equals(hold[5])                                         # 两个起点的基准不是同一条
    assert card.loc["全段", "能持有等权"] == pytest.approx(np.mean([sc._annual(hold[k]) for k in (0, 5)]))


def test_max_drawdown_counts_a_loss_on_the_first_day():
    assert sc._max_drawdown(pd.Series([-0.10, 0.05])) == pytest.approx(-0.10)
    assert sc._max_drawdown(pd.Series([0.10, -0.20, 0.30])) == pytest.approx(-0.20)
    assert sc._max_drawdown(pd.Series([0.01, 0.01])) == 0.0


def test_paired_difference_by_era_and_t_value():
    rng = np.random.default_rng(0)
    base_r = rng.normal(0.0005, 0.01, len(IDX))
    edge = np.where(IDX.year == 2019, 0.002, -0.001)                           # 阶段三每天多 0.2%，阶段二少 0.1%
    nav = lambda r: 100_000 * np.cumprod(1 + r)
    base = {0: _result(nav(base_r)), 5: _result(nav(base_r))}
    cand = {0: _result(nav(base_r + edge)), 5: _result(nav(base_r + edge))}
    got = sc.paired(cand, base)
    assert got["差值"]["三"] == pytest.approx(0.002 * 252, rel=0.02) and got["差值"]["二"] == pytest.approx(-0.001 * 252, rel=0.05)
    assert got["差值"]["全段"] == pytest.approx(got["daily"].mean() * 252)
    noisy = base_r + rng.normal(0.002, 0.004, len(IDX))                        # 每天平均多 0.2%，有噪声
    strong = sc.paired({0: _result(nav(noisy)), 5: _result(nav(noisy))}, base)
    d = strong["daily"]
    assert strong["t"] > 2
    from qt.evaluate import newey_west_t
    assert strong["t"] == pytest.approx(newey_west_t(d, 20))
    assert sc.paired(base, cand)["t"] == pytest.approx(-got["t"])
    assert np.isnan(sc.paired(base, base)["t"]) or sc.paired(base, base)["t"] == 0
    with pytest.raises(ValueError, match="起点"):
        sc.paired({0: base[0]}, base)


def _cards(big_base=(10, 10, 10), big_cand=(10, 10, 10)):
    mk = lambda big: pd.DataFrame({"大亏只数": [big[0], big[1], big[2], sum(big)], "超额": 0.05}, index=["一", "二", "三", "全段"])
    return mk(big_cand), mk(big_base)


def _pair(e1=0.01, e2=0.01, e3=0.01, t=2.5):
    return {"差值": pd.Series({"一": e1, "二": e2, "三": e3, "全段": (e1 + e2 + e3) / 3}), "t": t}


UP = {"metric": "配对.全段", "direction": "升"}


def test_return_gate_needs_every_condition():
    cand, base = _cards()
    ok = lambda **k: gates.decide("收益型", k.get("cand", cand), base, k.get("pair", _pair()), k.get("cost2", _pair()),
                                  k.get("pred", UP))
    table, adopt = ok()
    assert adopt and table["满足"].all() and len(table) == 6
    assert not ok(pair=_pair(t=1.99))[1] and ok(pair=_pair(t=2.0))[1]
    assert not ok(pair=_pair(e3=-0.001))[1]                                    # 最近的阶段必须为正
    assert not ok(pair=_pair(e1=-0.01, e2=-0.01, e3=0.05))[1]                  # 更早的阶段至少一个为正
    assert ok(pair=_pair(e1=-0.01, e2=0.01, e3=0.05))[1]
    assert not ok(cost2=_pair(e1=-0.05, e2=-0.05, e3=0.01))[1]                 # 成本翻倍后全段要为正
    assert not ok(cand=_cards(big_cand=(10, 10, 11))[0])[1] and ok(cand=_cards(big_cand=(9, 10, 10))[0])[1]
    assert not ok(pred={"metric": "配对.全段", "direction": "降"})[1]            # 预期方向反了
    failed = ok(pair=_pair(t=1.0))[0]
    assert failed.index[~failed["满足"]].tolist() == ["配对差值的 t 值"]


def test_safety_gate_needs_fewer_big_losses_in_latest_era_without_getting_worse():
    down = {"metric": "三.大亏只数", "direction": "降"}
    run = lambda big_cand=(10, 10, 7), pair=_pair(e1=-0.03, e2=-0.03, e3=0.01, t=-0.5), pred=down, big_base=(10, 10, 10): \
        gates.decide("防雷型", *_cards(big_base, big_cand), pair, _pair(), pred)
    table, adopt = run()
    assert adopt and len(table) == 4                                           # 更早的阶段少赚没关系
    assert not run(big_cand=(10, 10, 8))[1]                                    # 只少 20%
    assert not run(pair=_pair(e3=-0.001, t=0.5))[1]
    assert not run(pair=_pair(e1=-0.2, e2=-0.2, e3=0.01, t=-1.0))[1] and run(pair=_pair(e3=0.01, t=-0.99))[1]
    assert not run(big_base=(10, 10, 0), big_cand=(10, 10, 0))[1]              # 基线本来就没有大亏：谈不上减少
    assert not run(pred={"metric": "三.大亏只数", "direction": "升"})[1]


def test_prediction_metrics_and_bad_inputs():
    cand, base = _cards(big_cand=(10, 10, 7))
    assert gates.prediction_holds({"metric": "三.大亏只数", "direction": "降"}, cand, base, _pair())
    assert not gates.prediction_holds({"metric": "一.大亏只数", "direction": "降"}, cand, base, _pair())   # 没变不算降
    assert gates.prediction_holds({"metric": "配对.三", "direction": "降"}, cand, base, _pair(e3=-0.01))
    with pytest.raises(ValueError):
        gates.prediction_holds({"metric": "配对.三", "direction": "变好"}, cand, base, _pair())
    with pytest.raises(ValueError):
        gates.decide("随便", cand, base, _pair(), _pair(), UP)


def test_scorecard_end_to_end_on_fake_market_finds_the_signal():
    dev, _ = dev3d()
    cache = {}
    base = sc.run_phases(dev, pc.PLAN3D, cache=cache)
    assert list(base) == [0, 5, 10, 15] and len(cache) == 1                    # 合成分只算一次
    card = sc.scorecard(dev, base)
    assert list(card.index) == ["二", "三", "全段"] and card.loc["全段", "超额"] > 0.10
    worse = sc.run_phases(dev, pc.PLAN3D.replace(families={k: v for k, v in pc.PLAN3D.to_dict()["families"].items() if k != "便宜"}))
    pair = sc.paired(worse, base)
    assert pair["差值"]["全段"] < -0.10 and pair["t"] < -2                       # 去掉带信号的那组，明显变差


def test_top_tail_orders_buckets_and_respects_can_buy():
    idx = pd.bdate_range("2019-01-02", periods=260)
    cols = [f"S{i:03d}" for i in range(200)]
    rng = np.random.default_rng(1)
    quality = np.linspace(0.002, -0.002, 200)                                   # 排前面的每天多涨
    close = pd.DataFrame(10 * np.cumprod(1 + quality + rng.normal(0, 0.01, (260, 200)), axis=0), index=idx, columns=cols)
    score = pd.DataFrame(np.tile(-np.arange(200.0), (260, 1)), index=idx, columns=cols)
    buy = pd.DataFrame(True, index=idx, columns=cols)
    got = ev.top_tail(score, close, buy)
    assert list(got.index) == ["三", "全段"]
    row = got.loc["三"]
    assert row["前15名"] > row["前50名"] > row["前150名"] > row["全体"] and row["最好五分之一"] > row["全体"]
    assert row["随机挑15只的期望"] >= 0 and row["前15名跌超20%的只数"] >= 0
    buy.iloc[:, :15] = False                                                    # 最好的 15 只不能买 → 前 15 名换成后面的
    assert ev.top_tail(score, close, buy).loc["三", "前15名"] < row["前15名"]
    crash = close.copy()
    crash.iloc[25:, :15] *= 0.5                                                 # 前 15 名在第 25 天集体腰斩
    assert ev.top_tail(score, crash, pd.DataFrame(True, index=idx, columns=cols)).loc["三", "前15名跌超20%的只数"] > 0.5


def test_worst_five_takes_exactly_five_and_ignores_winners():
    t = []
    for k, pnl in enumerate([-6000, -5000, -4000, -3000, -2000, -1000, 500]):   # 六笔亏、一笔赚，都在 2019 年平仓
        t += [(IDX[1], f"X{k}", "buy", 100, 10.0, 10_000.0, 0.0, "buy"),
              (IDX[30 + k], f"X{k}", "sell", 100, 1.0, 10_000.0 + pnl, 0.0, "rank")]
    card = sc.scorecard(_dev_for(), {0: _result(_grow(0.0), t)})
    assert card.loc["三", "最差五笔占比"] == pytest.approx(-20_000 / 100_000)
    win = [(IDX[1], "W", "buy", 100, 10.0, 10_000.0, 0.0, "buy"), (IDX[30], "W", "sell", 100, 11.0, 11_000.0, 0.0, "rank")]
    assert sc.scorecard(_dev_for(), {0: _result(_grow(0.0), win)}).loc["三", "最差五笔占比"] == 0.0
    assert sc.scorecard(_dev_for(), {0: _result(_grow(0.0))}).loc["三", "最差五笔占比"] == 0.0   # 一笔都没有
