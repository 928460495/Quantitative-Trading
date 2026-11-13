"""计划 3d：四组因子等权、参数事先定死（DECISIONS 2026-11-12）。小假数据，不碰真实数据。"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from qt import plan3d as p3
from qt.diagnostics import annualize, holdable_benchmark

from test_walkforward import fake_seg


def seg3d(seed=0, n_stocks=60, start="2010-01-04", end="2011-12-30"):
    """"便宜"组带真信号，其余三组和 vol60 是噪声；价格 10 元，没有 ST、没有亏损小营收。"""
    seg = fake_seg(seed, n_stocks, start, end)
    rng = np.random.default_rng(seed + 100)
    shape, idx, cols = seg["close"].shape, seg["close"].index, seg["close"].columns
    noise = lambda: pd.DataFrame(rng.normal(size=shape), index=idx, columns=cols)
    sig = seg["factors"]["sig"]
    sig = sig.sub(sig.mean(axis=1), axis=0).div(sig.std(axis=1), axis=0)
    factors = {f: (sig + 0.3 * noise() if g == "便宜" else noise()) for g, fs in p3.GROUPS.items() for f in fs}
    factors[p3.VOL_FACTOR] = noise()
    seg["factors"] = factors
    seg["raw_close"] = seg["close"] * 0 + 10.0
    false = pd.DataFrame(False, index=idx, columns=cols)
    seg["is_st"], seg["loss_small"] = false.copy(), false.copy()
    return seg


def test_groups_cover_16_registered_factors_without_price_level():
    assert p3.GROUPS == {
        "便宜": ("ep_deduct", "bp", "sp_ttm"),
        "跌得多": ("ret60", "ret120", "high52w", "lit_high52w", "a158_MA60", "a158_QTLU60", "a158_ROC60", "a158_MAX60"),
        "冷门": ("turn_12m", "gtja100", "gtja081", "gtja132"),
        "业绩超预期": ("sue",),
    }
    names = [f for fs in p3.GROUPS.values() for f in fs]
    assert len(names) == len(set(names)) == 16 and "price_level" not in names
    assert p3.VOL_FACTOR == "vol60" and p3.VOL_FACTOR not in names


def test_main_spec_matches_registered_values():
    m = p3.MAIN
    assert (m.groups, m.n_hold, m.rebalance_every, m.sell_mult) == (tuple(p3.GROUPS), 15, 20, 3)
    assert m.stop_loss is None and m.vol_cut == 0.20 and m.price_floor == 2.0 and m.drop_loss_small
    assert m.cost_multiplier == 1.0 and m.offset == 0
    assert p3.BENCH_EVERY == 20 and p3.REVENUE_FLOOR == 3e8


def test_loss_small_needs_both_loss_and_small_revenue():
    f = lambda *v: pd.DataFrame([list(v)])
    nan = np.nan
    #            亏+小   盈+小   亏+大   只扣非亏+小  缺营收  缺利润   正好 3 亿
    net = f(-1.0, 5.0, -1.0, 2.0, -1.0, nan, -1.0)
    ded = f(-2.0, 4.0, -2.0, -0.5, -1.0, nan, -1.0)
    rev = f(1e8, 1e8, 5e8, 2.9e8, nan, 1e8, 3e8)
    got = p3.loss_small_revenue(net, ded, rev)
    assert got.iloc[0].tolist() == [True, False, False, True, False, False, False]
    assert got.dtypes.eq(bool).all()


def test_high_vol_flags_lowest_fifth_of_vol_score():
    # vol_score 越大 = 波动越低（已调好方向）；最低的 20% 是波动最高的
    row = pd.DataFrame([[5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 10.0, np.nan]])
    got = p3.high_vol(row, 0.20)
    assert got.iloc[0].tolist() == [False, True, False, False, False, True, False, False, False, False, False]
    assert p3.high_vol(row, 0.10).iloc[0].sum() == 1


def test_composite_ignores_dropped_group():
    seg = seg3d()
    kept = tuple(g for g in p3.GROUPS if g != "冷门")
    base = p3.composite_score(seg, kept)
    seg["factors"]["turn_12m"] = seg["factors"]["turn_12m"] * -3 + 1          # 被去掉那组的因子随便改
    pd.testing.assert_frame_equal(p3.composite_score(seg, kept), base)
    seg["factors"]["bp"] = seg["factors"]["bp"] * -3 + 1                      # 留着的组一改就变
    assert not p3.composite_score(seg, kept).equals(base)
    assert not p3.composite_score(seg, tuple(p3.GROUPS)).equals(p3.composite_score(seg, kept))


def test_groups_are_equally_weighted_not_factors():
    # "跌得多"有 8 个因子、"业绩超预期"只有 1 个：两组对总分的影响应该一样大
    seg = seg3d()
    idx, cols = seg["close"].index, seg["close"].columns
    rng = np.random.default_rng(5)
    a = pd.DataFrame(rng.normal(size=seg["close"].shape), index=idx, columns=cols)
    b = pd.DataFrame(rng.normal(size=seg["close"].shape), index=idx, columns=cols)
    for f in p3.GROUPS["跌得多"]:
        seg["factors"][f] = a
    seg["factors"]["sue"] = b
    seg["industry"] = seg["industry"].map(lambda _: "I0")                    # 只留一个行业，中性化影响小
    score = p3.composite_score(seg, ("跌得多", "业绩超预期"))
    ca = score.corrwith(a, axis=1).mean()
    cb = score.corrwith(b, axis=1).mean()
    assert ca == pytest.approx(cb, abs=0.05) and ca > 0.5


def _flagged(seg, key, col, start):
    seg[key] = seg[key].copy()
    seg[key].loc[start:, col] = True
    return seg


def test_high_vol_blocks_buy_but_keeps_score():
    seg = seg3d()
    col = seg["close"].columns[3]
    seg["factors"][p3.VOL_FACTOR][col] = -99.0                               # 永远是波动最高的
    data = p3.market(seg, p3.MAIN)
    j = list(data.instruments).index(col)
    assert not data.can_buy[:, j].any() and not np.isnan(data.score[:, j]).any()
    free = p3.market(seg, dataclasses.replace(p3.MAIN, vol_cut=None))
    assert free.can_buy[:, j].all()
    assert data.can_buy.mean() == pytest.approx(0.8, abs=0.02)               # 每天挡掉约 20%


@pytest.mark.parametrize("key", ["is_st", "loss_small", "cheap"])
def test_st_and_loss_small_and_cheap_lose_score_and_buy(key):
    seg = seg3d()
    col, start = seg["close"].columns[7], "2011-01-03"
    if key == "cheap":
        seg["raw_close"].loc[start:, col] = 1.99
    else:
        _flagged(seg, key, col, start)
    data = p3.market(seg, dataclasses.replace(p3.MAIN, vol_cut=None))
    j = list(data.instruments).index(col)
    after = data.dates >= pd.Timestamp(start)
    assert np.isnan(data.score[after, j]).all() and not data.can_buy[after, j].any()
    assert not np.isnan(data.score[~after, j]).any() and data.can_buy[~after, j].all()
    assert not np.isnan(np.delete(data.score, j, axis=1)).any()               # 只动这一只


def test_price_floor_is_strict_and_optional_filters_can_be_switched_off():
    seg = seg3d()
    c2, c3 = seg["close"].columns[2], seg["close"].columns[5]
    seg["raw_close"][c2] = 2.0                                                # 正好 2 元不算低价
    seg["raw_close"][c3] = 1.5
    _flagged(seg, "loss_small", seg["close"].columns[9], "2010-01-04")
    _flagged(seg, "is_st", seg["close"].columns[11], "2010-01-04")
    on = p3.market(seg, dataclasses.replace(p3.MAIN, vol_cut=None))
    off = p3.market(seg, dataclasses.replace(p3.MAIN, vol_cut=None, price_floor=None, drop_loss_small=False))
    nan_cols = lambda d: set(np.flatnonzero(np.isnan(d.score).all(axis=0)))
    assert nan_cols(on) == {5, 9, 11}
    assert nan_cols(off) == {11}                                              # ST 始终排除


def test_rules_have_no_stop_loss_no_drawdown_control():
    r = p3.rules(p3.MAIN)
    assert (r.cash, r.n_hold, r.sell_rank, r.rebalance_every) == (100_000, 15, 45, 20)
    assert r.stop_loss is None and r.drawdown_stop is None and r.drawdown_tiers is None
    assert (r.max_stock_weight, r.max_industry_weight, r.min_trade, r.cost_multiplier) == (0.20, 0.30, 2000.0, 1.0)
    r2 = p3.rules(dataclasses.replace(p3.MAIN, n_hold=10, sell_mult=2, stop_loss=0.2, cost_multiplier=2.0, rebalance_every=40))
    assert (r2.n_hold, r2.sell_rank, r2.stop_loss, r2.cooldown_days, r2.cost_multiplier, r2.rebalance_every) == (10, 20, 0.2, 20, 2.0, 40)


def test_offset_starts_later_and_benchmark_uses_same_window():
    seg = seg3d()
    data = p3.market(seg, dataclasses.replace(p3.MAIN, offset=5))
    assert data.dates[0] == seg["close"].index[5] and len(data.dates) == len(seg["close"]) - 5
    nav = p3.run(seg, dataclasses.replace(p3.MAIN, offset=5)).daily["nav"]
    assert nav.index[0] == seg["close"].index[5]
    got = p3.benchmarks(seg, nav)
    want = annualize((1 + holdable_benchmark(seg["close"].iloc[5:], seg["can_buy"], 20)).cumprod())
    assert got["能持有等权"] == pytest.approx(want)
    full = annualize((1 + holdable_benchmark(seg["close"], seg["can_buy"], 20)).cumprod())
    assert got["能持有等权"] != pytest.approx(full)
    hs = seg["index"]["H00300"].iloc[5:]
    assert got["沪深300全收益"] == pytest.approx(annualize(hs))


def test_benchmark_ignores_strategy_filters():
    seg = seg3d()
    nav = p3.run(seg, p3.MAIN).daily["nav"]
    before = p3.benchmarks(seg, nav)["能持有等权"]
    _flagged(seg, "is_st", seg["close"].columns[0], "2010-01-04")             # 策略的排除名单不改基准
    seg["factors"][p3.VOL_FACTOR] = seg["factors"][p3.VOL_FACTOR] * -1
    assert p3.benchmarks(seg, nav)["能持有等权"] == pytest.approx(before)


def test_variants_differ_from_main_in_exactly_one_field():
    assert list(p3.VARIANTS) == [
        "主方案", "错开 5 天", "错开 10 天", "错开 15 天", "成本翻倍",
        "去掉便宜", "去掉跌得多", "去掉冷门", "去掉业绩超预期",
        "持股 10 只", "持股 20 只", "每 10 天调仓", "每 40 天调仓", "跌出前 30 名卖", "跌出前 60 名卖",
        "加回止损", "不剔除高波动", "不排除低价和亏损小营收"]
    assert p3.VARIANTS["主方案"] == p3.MAIN
    main = dataclasses.asdict(p3.MAIN)
    changed = {}
    for name, spec in p3.VARIANTS.items():
        diff = {k: v for k, v in dataclasses.asdict(spec).items() if v != main[k]}
        changed[name] = diff
        if name == "主方案":
            assert diff == {}
        elif name == "不排除低价和亏损小营收":
            assert diff == {"price_floor": None, "drop_loss_small": False}
        else:
            assert len(diff) == 1, (name, diff)
    assert [changed[f"错开 {k} 天"] for k in (5, 10, 15)] == [{"offset": 5}, {"offset": 10}, {"offset": 15}]
    assert changed["成本翻倍"] == {"cost_multiplier": 2.0}
    for g in p3.GROUPS:
        assert changed[f"去掉{g}"] == {"groups": tuple(x for x in p3.GROUPS if x != g)}
    assert changed["持股 10 只"] == {"n_hold": 10} and changed["持股 20 只"] == {"n_hold": 20}
    assert changed["每 10 天调仓"] == {"rebalance_every": 10} and changed["每 40 天调仓"] == {"rebalance_every": 40}
    assert changed["跌出前 30 名卖"] == {"sell_mult": 2} and changed["跌出前 60 名卖"] == {"sell_mult": 4}
    assert changed["加回止损"] == {"stop_loss": 0.20} and changed["不剔除高波动"] == {"vol_cut": None}


def test_checks_reference_existing_variants():
    assert p3.CHECKS == {
        "1 主方案高于两个基准": ("主方案",),
        "2 错开调仓起点都高于能持有等权": ("主方案", "错开 5 天", "错开 10 天", "错开 15 天"),
        "3 成本翻倍仍高于能持有等权": ("成本翻倍",),
        "4 去掉任何一组仍高于能持有等权": ("去掉便宜", "去掉跌得多", "去掉冷门", "去掉业绩超预期"),
    }
    assert all(m in p3.VARIANTS for ms in p3.CHECKS.values() for m in ms)


def _table(excess: dict, hs300=0.0):
    rows = {n: {"年化": 0.05 + e, "能持有等权": 0.05, "沪深300全收益": hs300, "超额": e} for n, e in excess.items()}
    return pd.DataFrame(rows).T


def test_check_table_fails_when_any_member_trails_benchmark():
    good = {n: 0.03 for n in p3.VARIANTS}
    assert p3.check_table(_table(good))["满足"].all()
    for name, check in [("错开 10 天", "2 错开调仓起点都高于能持有等权"), ("成本翻倍", "3 成本翻倍仍高于能持有等权"),
                        ("去掉冷门", "4 去掉任何一组仍高于能持有等权")]:
        got = p3.check_table(_table({**good, name: -0.001}))
        assert got["满足"].to_dict() == {c: c != check for c in p3.CHECKS}, name
        assert got.loc[check, "最差的版本"] == name and got.loc[check, "最小超额"] == pytest.approx(-0.001)
    # 平均为正也不行：每个都要高于基准；等于基准不算高于
    assert not p3.check_table(_table({**good, "错开 5 天": 0.0})).loc["2 错开调仓起点都高于能持有等权", "满足"]
    # 只报告的版本不影响检查
    assert p3.check_table(_table({**good, "持股 10 只": -0.5, "加回止损": -0.5}))["满足"].all()


def test_check_one_also_requires_beating_hs300():
    good = {n: 0.03 for n in p3.VARIANTS}
    got = p3.check_table(_table(good, hs300=0.10))                            # 年化 8% < 沪深300 的 10%
    assert not got.loc["1 主方案高于两个基准", "满足"]
    assert got.drop(index="1 主方案高于两个基准")["满足"].all()


def test_st_holding_is_sold_at_next_rebalance_but_high_vol_holding_stays():
    seg = seg3d()
    cols = seg["close"].columns
    for f in p3.GROUPS["便宜"]:                                                 # 两只股票永远排最前，一定被买入
        seg["factors"][f][cols[0]] = 50.0
        seg["factors"][f][cols[1]] = 50.0
    flag_day = seg["close"].index[150]
    _flagged(seg, "is_st", cols[0], flag_day)
    seg["factors"][p3.VOL_FACTOR].loc[flag_day:, cols[1]] = -99.0             # 买入之后才变成高波动
    res = p3.run(seg, p3.MAIN)
    tr = res.trades
    buys = tr[tr["side"] == "buy"].groupby("instrument")["date"].min()
    assert buys[cols[0]] < flag_day and buys[cols[1]] < flag_day
    sold0 = tr[(tr["side"] == "sell") & (tr["instrument"] == cols[0])]
    assert len(sold0) == 1
    first_reb = seg["close"].index[160]                                       # 第 160 个信号日是调仓日，第 161 天成交
    assert sold0["date"].iloc[0] == seg["close"].index[161] and flag_day < first_reb
    assert tr[(tr["instrument"] == cols[0]) & (tr["date"] > flag_day) & (tr["side"] == "buy")].empty
    assert tr[(tr["side"] == "sell") & (tr["instrument"] == cols[1])].empty   # 高波动的不逼卖


def test_no_trades_between_rebalance_days():
    res = p3.run(seg3d(), p3.MAIN)
    idx = seg3d()["close"].index
    pos = pd.Index(idx).get_indexer(pd.DatetimeIndex(res.trades["date"]))
    assert len(pos) > 30 and ((pos - 1) % 20 == 0).all()                      # 信号日是 0、20、40 …，第二天成交


def test_evaluate_table_logs_each_variant_and_finds_the_signal():
    seg = seg3d()
    variants = {k: p3.VARIANTS[k] for k in ("主方案", "错开 5 天", "去掉便宜", "成本翻倍")}
    logged = []
    table, results = p3.evaluate(seg, variants, log=lambda kind, params, metrics: logged.append((kind, params, metrics)),
                                 tag="train")
    assert list(table.index) == list(variants) and set(results) == set(variants)
    assert list(table.columns) == ["年化", "能持有等权", "沪深300全收益", "超额", "最大回撤", "日均换手", "年成本", "交易笔数"]
    assert (table["超额"] == table["年化"] - table["能持有等权"]).all()
    assert table.loc["主方案", "超额"] > 0.10                                  # "便宜"组带真信号
    assert table.loc["去掉便宜", "超额"] < table.loc["主方案", "超额"] - 0.10     # 去掉它就没了
    assert table.loc["成本翻倍", "年化"] < table.loc["主方案", "年化"]
    assert table.loc["成本翻倍", "年成本"] == pytest.approx(2 * table.loc["主方案", "年成本"], rel=0.15)
    assert table.loc["主方案", "年化"] == pytest.approx(annualize(results["主方案"].daily["nav"]))
    assert [k for k, _, _ in logged] == ["plan3d"] * 4
    assert [p["variant"] for _, p, _ in logged] == list(variants) and {p["segment"] for _, p, _ in logged} == {"train"}
    assert logged[0][1]["n_hold"] == 15 and set(logged[0][2]) == {"annual_return", "sharpe", "max_drawdown"}


def test_revenue_floor_follows_the_rule_in_force_on_each_day():
    # 复查 R001 的 M1：营收门槛按当时有效的退市规则，不能把 2024 年的 3 亿元用到更早
    idx = pd.to_datetime(["2020-12-11", "2020-12-14", "2021-06-01", "2024-04-11", "2024-04-12", "2026-01-05"])
    assert p3.revenue_floor(idx).tolist() == [0.0, 1e8, 1e8, 1e8, 3e8, 3e8]


def test_loss_small_uses_the_dated_floor():
    idx = pd.to_datetime(["2019-06-03", "2021-06-01", "2024-06-03"])
    cols = ["A", "B"]
    f = lambda *rows: pd.DataFrame(list(rows), index=idx, columns=cols, dtype=float)
    net = f([-1, -1], [-1, -1], [-1, -1])
    ded = f([-1, -1], [-1, -1], [-1, -1])
    rev = f([5e7, 2e8], [5e7, 2e8], [5e7, 2e8])
    got = p3.loss_small_revenue(net, ded, rev, p3.revenue_floor(idx))
    assert got.to_numpy().tolist() == [[False, False], [True, False], [True, True]]   # 2019 年还没有这条规则
