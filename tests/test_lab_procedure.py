"""实验室：流程（docs/FRAMEWORK.md 第 3 节）。和计划 3d 的实现逐笔对上，再测新加的开关。"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from qt import plan3d as p3
from qt.lab import procedure as pc

from lab_helpers import dev3d


def test_plan3d_baseline_matches_registered_spec():
    b = pc.PLAN3D
    assert dict(b.families) == p3.GROUPS and len(b.factors()) == 16
    assert (b.n_hold, b.rebalance_every, b.sell_mult, b.stop_loss, b.cost_multiplier) == (15, 20, 3, None, 1.0)
    assert b.exclude == ("cheap", "is_st", "loss_small") and b.no_buy == ("high_vol",) and not b.exit_next_day
    assert [n for n, _ in b.families] == sorted(p3.GROUPS)


@pytest.mark.parametrize("offset", [0, 5])
def test_baseline_reproduces_plan3d_trade_for_trade(offset):
    dev, seg = dev3d()
    seg["is_st"].loc["2019-03-01":, seg["close"].columns[3]] = True
    seg["loss_small"].loc["2018-09-03":, seg["close"].columns[8]] = True
    seg["raw_close"].loc["2019-06-03":, seg["close"].columns[12]] = 1.5
    dev, _ = dev3d()
    dev["flags"]["is_st"], dev["flags"]["loss_small"] = seg["is_st"], seg["loss_small"]
    dev["raw_close"] = seg["raw_close"]
    dev["flags"]["cheap"] = seg["raw_close"] < 2.0
    want = p3.run(seg, dataclasses.replace(p3.MAIN, offset=offset))
    got = pc.run(dev, pc.PLAN3D, offset)
    pd.testing.assert_frame_equal(got.trades, want.trades)
    pd.testing.assert_series_equal(got.daily["nav"], want.daily["nav"])


def test_round_trip_and_replace():
    p = pc.PLAN3D.replace(n_hold=30, exclude=["is_st", "deduct_loss"], families={"甲": ["bp"], "乙": ["sue", "ret60"]})
    assert p.n_hold == 30 and p.exclude == ("deduct_loss", "is_st") and p.families == (("乙", ("sue", "ret60")), ("甲", ("bp",)))
    q = pc.PLAN3D.replace(exclude=["is_st", "deduct_loss", "is_st"], families={"甲": ["bp"], "乙": ["sue", "ret60"]})
    assert q.exclude == p.exclude and q.families == p.families                 # 写法不同、去重后是同一个
    import yaml
    for sort in (True, False):
        text = yaml.safe_dump(p.to_dict(), allow_unicode=True, sort_keys=sort)
        assert pc.Procedure.from_dict(yaml.safe_load(text)) == p              # 写进文件再读回来不变
    assert pc.Procedure.from_dict(p.to_dict()) == p and hash(p) == hash(pc.Procedure.from_dict(p.to_dict()))
    assert pc.PLAN3D.replace() == pc.PLAN3D
    with pytest.raises(ValueError, match="没有这些字段"):
        pc.PLAN3D.replace(n_holdings=30)


def test_unknown_flag_is_an_error():
    dev, _ = dev3d()
    with pytest.raises(KeyError, match="没有这个标记"):
        pc.market(dev, pc.PLAN3D.replace(exclude=["is_st", "typo"]))


def test_new_exclusion_removes_score_and_buy_only_where_flagged():
    dev, _ = dev3d()
    col = dev["close"].columns[4]
    dev["flags"]["deduct_loss"].loc["2019-01-02":, col] = True
    base = pc.market(dev, pc.PLAN3D.replace(no_buy=[]))
    new = pc.market(dev, pc.PLAN3D.replace(no_buy=[], exclude=["is_st", "loss_small", "cheap", "deduct_loss"]))
    j = list(new.instruments).index(col)
    after = new.dates >= pd.Timestamp("2019-01-02")
    assert np.isnan(new.score[after, j]).all() and not new.can_buy[after, j].any()
    assert np.array_equal(new.score[~after], base.score[~after], equal_nan=True)
    assert np.array_equal(np.delete(new.score, j, axis=1), np.delete(base.score, j, axis=1), equal_nan=True)
    assert new.force_exit is None


def test_no_buy_flag_keeps_score():
    dev, _ = dev3d()
    col = dev["close"].columns[4]
    dev["flags"]["small30"] = dev["flags"]["small30"] & False
    dev["flags"]["small30"][col] = True
    data = pc.market(dev, pc.PLAN3D.replace(no_buy=["small30"]))
    j = list(data.instruments).index(col)
    assert not data.can_buy[:, j].any() and not np.isnan(data.score[:, j]).any()
    assert data.can_buy[:, np.arange(len(data.instruments)) != j].all()


def test_exit_next_day_sells_before_the_rebalance_day():
    dev, _ = dev3d()
    cols = dev["close"].columns
    for f in p3.GROUPS["便宜"]:
        dev["factors"][f][cols[0]] = 50.0                                     # 一定被买入
    flag_day = dev["close"].index[150]
    dev["flags"]["is_st"].loc[flag_day:, cols[0]] = True
    slow = pc.run(dev, pc.PLAN3D).trades
    fast = pc.run(dev, pc.PLAN3D.replace(exit_next_day=True)).trades
    sold = lambda tr: tr[(tr["side"] == "sell") & (tr["instrument"] == cols[0])]
    assert sold(slow)["date"].tolist() == [dev["close"].index[161]] and sold(slow)["reason"].tolist() == ["rank"]
    assert sold(fast)["date"].tolist() == [dev["close"].index[151]] and sold(fast)["reason"].tolist() == ["event"]
    data = pc.market(dev, pc.PLAN3D.replace(exit_next_day=True), offset=5)
    assert data.force_exit.shape == data.score.shape and data.force_exit[145:, 0].all() and not data.force_exit[:145].any()


def test_rules_follow_the_procedure():
    r = pc.rules(pc.PLAN3D.replace(n_hold=30, sell_mult=2, rebalance_every=10, cost_multiplier=2.0, stop_loss=0.2))
    assert (r.n_hold, r.sell_rank, r.rebalance_every, r.cost_multiplier, r.stop_loss) == (30, 60, 10, 2.0, 0.2)
    assert r.drawdown_stop is None and r.drawdown_tiers is None and r.min_trade == 2000.0
    assert (r.max_stock_weight, r.max_industry_weight) == (0.20, 0.30)


# ---------- 某一组分数缺失时怎么处理（H001） ----------

def test_missing_family_skip_is_the_old_behaviour_and_zero_counts_it_as_average():
    dev, _ = dev3d()
    cols = dev["close"].columns
    assert pc.PLAN3D.missing == "skip"
    dev["factors"]["sue"] = dev["factors"]["sue"].copy()
    dev["factors"]["sue"][cols[:10]] = np.nan                                 # 前 10 只没有"业绩超预期"
    skip = pc.composite(dev, pc.PLAN3D)
    zero = pc.composite(dev, pc.PLAN3D.replace(missing="zero"))
    from qt.composite import neutral_composite
    group_of = {f: g for g, fs in pc.PLAN3D.families for f in fs}
    want = neutral_composite({f: dev["factors"][f] for f in group_of}, group_of, dev["industry"], dev["log_mcap"])
    pd.testing.assert_frame_equal(skip, want)                                 # skip 和以前完全一样
    assert zero.notna().equals(skip.notna())                                  # 有没有总分不变
    # 缺一组的股票：skip 时按其余三组平均，zero 时这一组记 0 → 四组平均，分数向 0 收
    from qt.composite import category_scores
    cats = category_scores({f: dev["factors"][f] for f in group_of}, group_of)
    d = dev["close"].index[100]
    others = np.mean([cats[g].loc[d, cols[0]] for g in cats if g != "业绩超预期"])
    raw_zero = sum(cats[g].loc[d, cols[0]] for g in cats if g != "业绩超预期") / 4
    assert raw_zero == pytest.approx(others * 3 / 4)
    full = [c for c in cols[10:]]
    assert np.corrcoef(skip.loc[:, full].stack(), zero.loc[:, full].stack())[0, 1] > 0.98   # 数据齐的股票几乎不受影响
    miss = list(cols[:10])
    shrink = zero[miss].abs().mean().mean() / skip[miss].abs().mean().mean()
    assert 0.6 < shrink < 0.9                                                 # 缺一组的股票分数向 0 收（约 3/4）


def test_missing_zero_still_needs_at_least_one_family():
    dev, _ = dev3d()
    col = dev["close"].columns[3]
    for f in pc.PLAN3D.factors():
        dev["factors"][f] = dev["factors"][f].copy()
        dev["factors"][f][col] = np.nan                                       # 一组都没有
    zero = pc.composite(dev, pc.PLAN3D.replace(missing="zero"))
    assert zero[col].isna().all()


def test_missing_option_is_validated_and_round_trips():
    p = pc.PLAN3D.replace(missing="zero")
    assert pc.Procedure.from_dict(p.to_dict()) == p and p != pc.PLAN3D
    with pytest.raises(ValueError, match="missing"):
        pc.PLAN3D.replace(missing="median")


def test_cache_tells_apart_procedures_that_differ_only_in_how_scores_are_built():
    # H001 第 1 次运行作废的原因：缓存只按因子族认，"缺失怎么处理"不同的两个流程拿到了同一个合成分
    dev, _ = dev3d()
    dev["factors"]["sue"] = dev["factors"]["sue"].copy()
    dev["factors"]["sue"][dev["close"].columns[:20]] = np.nan
    cache = {}
    a = pc.market(dev, pc.PLAN3D, cache=cache)
    b = pc.market(dev, pc.PLAN3D.replace(missing="zero"), cache=cache)
    assert not np.array_equal(a.score, b.score, equal_nan=True)
    fresh = pc.market(dev, pc.PLAN3D.replace(missing="zero"))
    assert np.array_equal(b.score, fresh.score, equal_nan=True)
    c = pc.market(dev, pc.PLAN3D.replace(n_hold=30), cache=cache)              # 只改持股数：可以共用
    assert np.array_equal(a.score, c.score, equal_nan=True) and len(cache) == 2


def test_every_field_that_changes_the_composite_is_in_the_cache_key():
    # 防同类错误：以后给流程加字段，只要它改变合成分而没登记进 SCORE_FIELDS，这里就会失败
    import dataclasses
    dev, _ = dev3d()
    dev["factors"]["sue"] = dev["factors"]["sue"].copy()
    dev["factors"]["sue"][dev["close"].columns[:20]] = np.nan
    other = {"families": {"甲": ["bp", "sue"], "乙": ["ret60"]}, "n_hold": 30, "rebalance_every": 10, "sell_mult": 2,
             "exclude": ["is_st"], "no_buy": [], "exit_next_day": True, "stop_loss": 0.2, "cost_multiplier": 2.0,
             "missing": "zero", "blend": "rank", "overlay": "value_trend", "overlay_low": 0.2, "overlay_high": 0.85,
             "cash_yield": 0.02, "dividend_tax": True}
    fields = [f.name for f in dataclasses.fields(pc.Procedure)]
    assert set(other) == set(fields), "给新字段补一个不同的取值"
    base = pc.composite(dev, pc.PLAN3D)
    for name in fields:
        changed = pc.composite(dev, pc.PLAN3D.replace(**{name: other[name]}))
        moves = not changed.equals(base)
        assert moves == (name in pc.SCORE_FIELDS), name


# ---------- 族间合成：平均标准分还是平均百分位排名（H005） ----------

def test_rank_blend_limits_how_far_one_extreme_family_can_lift_a_stock():
    dev, _ = dev3d()
    cols = dev["close"].columns
    star = cols[7]
    for f in ("ep_deduct", "bp", "sp_ttm"):                                   # "便宜"这一组极端
        dev["factors"][f] = dev["factors"][f].copy()
        dev["factors"][f][star] = 40.0
    for f in pc.PLAN3D.factors():
        if f not in ("ep_deduct", "bp", "sp_ttm"):
            dev["factors"][f] = dev["factors"][f].copy()
            dev["factors"][f][star] = -0.5                                    # 其他组都略低于平均
    assert pc.PLAN3D.blend == "z"
    z = pc.composite(dev, pc.PLAN3D)
    r = pc.composite(dev, pc.PLAN3D.replace(blend="rank"))
    rank_z = z.rank(axis=1, ascending=False)[star].median()
    rank_r = r.rank(axis=1, ascending=False)[star].median()
    assert rank_z <= 3 and rank_r > 10                                         # 标准分相加时冲到最前面；排名平均时不会
    assert z.notna().equals(r.notna())
    others = [c for c in cols if c != star]
    assert np.corrcoef(z[others].stack(), r[others].stack())[0, 1] > 0.85      # 正常的股票排序大体一致（实测约 0.89）


def test_rank_blend_treats_missing_family_like_the_missing_option():
    dev, _ = dev3d()
    cols = dev["close"].columns
    dev["factors"]["sue"] = dev["factors"]["sue"].copy()
    dev["factors"]["sue"][cols[:10]] = np.nan
    skip = pc.composite(dev, pc.PLAN3D.replace(blend="rank", missing="skip"))
    zero = pc.composite(dev, pc.PLAN3D.replace(blend="rank", missing="zero"))
    assert not skip.equals(zero) and skip.notna().equals(zero.notna())
    miss = list(cols[:10])
    assert zero[miss].abs().mean().mean() < skip[miss].abs().mean().mean()     # 缺的那组按中间排名计入，向 0 收


def test_blend_option_is_validated():
    with pytest.raises(ValueError, match="blend"):
        pc.PLAN3D.replace(blend="max")
    assert pc.Procedure.from_dict(pc.PLAN3D.replace(blend="rank").to_dict()).blend == "rank"


# ---------- 估值 + 趋势仓位接进流程（DECISIONS 2026-11-13） ----------

def _signals(dev, valuation=0.5, trend=1.0):
    months = pd.date_range("2017-06-30", "2020-01-31", freq="ME")
    return pd.DataFrame({"valuation": valuation, "trend": trend}, index=months)


def test_overlay_fields_and_rules():
    p = pc.PLAN3D.replace(overlay="value_trend", cash_yield=0.02)
    assert (p.overlay_low, p.overlay_high) == (0.25, 0.75)
    r = pc.rules(p)
    assert r.slot_budget == pytest.approx(0.75 / 15) and r.cash_yield == 0.02
    assert pc.rules(pc.PLAN3D).slot_budget is None and pc.rules(pc.PLAN3D).cash_yield == 0.0
    with pytest.raises(ValueError, match="overlay"):
        pc.PLAN3D.replace(overlay="vol")
    assert pc.Procedure.from_dict(p.to_dict()) == p


def test_overlay_sets_the_number_of_holdings():
    dev, _ = dev3d()
    for valuation, trend, slots in ((1.0, 1.0, 15), (0.4, 1.0, 12), (0.0, 0.0, 5)):
        dev["overlay_signals"] = _signals(dev, valuation, trend)
        res = pc.run(dev, pc.PLAN3D.replace(overlay="value_trend"))
        held = res.daily["n_holdings"].iloc[25:]
        assert held.max() == slots and held.median() == slots, (valuation, trend)
        assert res.trades.query("side == 'buy'")["value"].max() <= res.daily["nav"].max() * 0.05 + 1
    no = pc.run(dev, pc.PLAN3D)
    assert no.daily["n_holdings"].iloc[25:].median() == 15


def test_overlay_runs_without_signals_injected_use_the_file(monkeypatch):
    dev, _ = dev3d()
    called = []
    monkeypatch.setattr(pc.overlay, "load_monthly", lambda *a, **k: called.append(1) or pd.DataFrame(
        {"ep": 0.05, "index": 100.0}, index=pd.date_range("2010-01-31", "2020-12-31", freq="ME")))
    pc.run(dev, pc.PLAN3D.replace(overlay="value_trend"))
    assert called


def test_overlay_uses_the_procedures_own_range():
    dev, _ = dev3d()
    dev["overlay_signals"] = _signals(dev, 0.0, 0.0)                           # 最贵、趋势向下 → 下限
    res = pc.run(dev, pc.PLAN3D.replace(overlay="value_trend", overlay_low=0.10, overlay_high=0.90))
    assert res.daily["n_holdings"].iloc[25:].median() == 2                     # round(15 × 0.10 ÷ 0.90) = 2


def test_dividend_tax_switch_passes_dividends_to_the_engine():
    dev, _ = dev3d()
    div = dev["close"] * 0.0
    div.iloc[100:, :] = 0.0
    div.iloc[100, :] = 0.03                                                    # 第 100 天每只都分红 3%
    dev["div_yield"] = div
    on = pc.run(dev, pc.PLAN3D.replace(dividend_tax=True))
    off = pc.run(dev, pc.PLAN3D)
    assert on.daily["tax"].iloc[100] > 0 and on.daily["tax"].drop(index=dev["close"].index[100]).sum() == 0
    assert off.daily["tax"].sum() == 0
    assert pc.rules(pc.PLAN3D.replace(dividend_tax=True)).dividend_tax
    shifted = pc.run(dev, pc.PLAN3D.replace(dividend_tax=True), offset=5)      # 错开起点后仍在同一天扣
    assert shifted.daily["tax"].loc[dev["close"].index[100]] > 0
    with pytest.raises(KeyError, match="div_yield"):
        pc.run({k: v for k, v in dev.items() if k != "div_yield"}, pc.PLAN3D.replace(dividend_tax=True))
