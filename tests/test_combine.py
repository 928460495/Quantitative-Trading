import numpy as np
import pandas as pd
import pytest

from qt import combine
from qt.preprocess import zscore


def _panel(n_days=40, n_stocks=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2016-01-04", periods=n_days), [f"s{i:02d}" for i in range(n_stocks)]],
        names=["datetime", "instrument"],
    )
    return idx, rng


def test_composite_fills_missing_factor_with_mean_and_keeps_all_missing_nan():
    idx, rng = _panel(3, 40)
    scores = pd.DataFrame({"a": rng.normal(size=len(idx)), "b": rng.normal(size=len(idx))}, index=idx)
    scores.iloc[0, 0] = np.nan  # 只缺一个因子：按 0 处理
    scores.iloc[1, :] = np.nan  # 全缺：结果为 NaN
    out = combine.composite(scores, pd.Series({"a": 0.5, "b": 0.5}))
    assert out.notna().iloc[0] and np.isnan(out.iloc[1])
    day = out.xs(idx[0][0], level="datetime")
    assert day.mean() == pytest.approx(0, abs=1e-12) and day.std() == pytest.approx(1)


def test_composite_ignores_weights_for_absent_columns():
    idx, rng = _panel(2, 40)
    scores = pd.DataFrame({"a": rng.normal(size=len(idx))}, index=idx)
    out = combine.composite(scores, pd.Series({"a": 1.0, "z": 3.0}))
    assert np.allclose(out, zscore(scores["a"]))


def test_icir_weights_proportional_to_icir():
    ic = pd.DataFrame({"a": [0.02, 0.04, 0.03, 0.05], "b": [0.01, -0.01, 0.02, 0.0]})
    w = combine.icir_weights(ic)
    icir = ic.mean() / ic.std()
    assert w["a"] / w["b"] == pytest.approx(icir["a"] / icir["b"])
    assert w.abs().sum() == pytest.approx(1)


def test_max_icir_prefers_independent_factor_over_duplicate():
    rng = np.random.default_rng(1)
    base = rng.normal(0.03, 0.05, 500)
    ic = pd.DataFrame({
        "a": base,
        "a_copy": base + rng.normal(0, 0.005, 500),  # 和 a 几乎一样
        "c": rng.normal(0.03, 0.05, 500),  # 和 a 独立
    })
    w = combine.max_icir_weights(ic)
    # 两个几乎一样的因子合起来的权重，和独立因子的权重相当，而不是 2 倍
    assert w["a"] + w["a_copy"] == pytest.approx(w["c"], rel=0.35)


def test_linear_weights_recover_true_coefficients():
    idx, rng = _panel(30, 80)
    x = pd.DataFrame({"a": rng.normal(size=len(idx)), "b": rng.normal(size=len(idx))}, index=idx)
    y = 2 * x["a"] + 1 * x["b"] + rng.normal(0, 0.1, len(idx))
    w = combine.linear_weights(x, y)
    assert w["a"] / w["b"] == pytest.approx(2, rel=0.05)


def test_rank_label_is_centered_with_unit_variance():
    idx, rng = _panel(5, 200)
    lab = combine.rank_label(pd.Series(rng.normal(size=len(idx)), index=idx))
    day = lab.xs(idx[0][0], level="datetime")
    # 和 Qlib 的 CSRankNorm 一样，排名百分比从 1/n 到 1，均值偏 1/(2n)
    assert day.mean() == pytest.approx(np.sqrt(12) / (2 * len(day)))
    assert day.std() == pytest.approx(1, rel=0.02)


def test_veto_block_marks_bottom_share_each_day():
    idx, rng = _panel(4, 50)
    s = pd.Series(rng.normal(size=len(idx)), index=idx)
    block = combine.veto_block(s, 0.2)
    assert block.groupby(level="datetime").sum().eq(10).all()
    day = s.xs(idx[0][0], level="datetime")
    blocked = block.xs(idx[0][0], level="datetime")
    assert day[blocked].max() < day[~blocked].min()
    assert not combine.veto_block(s, 0.0).any()


def test_veto_block_leaves_missing_score_unblocked():
    idx, rng = _panel(1, 50)
    s = pd.Series(rng.normal(size=len(idx)), index=idx)
    s.iloc[0] = np.nan
    assert not combine.veto_block(s, 0.2).iloc[0]


def test_smooth_uses_only_past_values():
    idx, rng = _panel(10, 40)
    s = pd.Series(rng.normal(size=len(idx)), index=idx)
    out = combine.smooth(s, 3)
    future = s.copy()
    last_day = idx.get_level_values("datetime").max()
    future.loc[last_day] = 100.0  # 改最后一天，之前的平滑值不应该变
    out2 = combine.smooth(future, 3)
    before = out.index.get_level_values("datetime") < last_day
    assert np.allclose(out[before], out2[before])


def test_smooth_is_rolling_mean_per_stock():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2016-01-04", periods=4), ["a", "b"]], names=["datetime", "instrument"])
    s = pd.Series([1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], index=idx)
    rolled = s.groupby(level="instrument").rolling(3, min_periods=1).mean().droplevel(0).reindex(idx)
    out = combine.smooth(s, 3)
    assert np.allclose(out, zscore(rolled))
    assert combine.smooth(s, 1) is s


def test_lgbm_learns_simple_signal():
    pytest.importorskip("lightgbm")
    idx, rng = _panel(60, 50)
    x = pd.DataFrame({"a": rng.normal(size=len(idx)), "b": rng.normal(size=len(idx))}, index=idx)
    y = combine.rank_label(x["a"] + rng.normal(0, 0.5, len(idx)))
    dates = idx.get_level_values("datetime")
    fit, stop = dates < dates.unique()[40], dates >= dates.unique()[40]
    model, best = combine.fit_lgbm(x[fit], y[fit], x[stop], y[stop])
    pred = combine.predict_lgbm(model, x[stop])
    assert best >= 1
    assert pred.corr(x.loc[stop, "a"], method="spearman") > 0.5
