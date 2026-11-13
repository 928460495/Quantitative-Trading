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
             "missing": "zero"}
    fields = [f.name for f in dataclasses.fields(pc.Procedure)]
    assert set(other) == set(fields), "给新字段补一个不同的取值"
    base = pc.composite(dev, pc.PLAN3D)
    for name in fields:
        changed = pc.composite(dev, pc.PLAN3D.replace(**{name: other[name]}))
        moves = not changed.equals(base)
        assert moves == (name in pc.SCORE_FIELDS), name
