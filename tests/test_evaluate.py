import numpy as np
import pandas as pd
import pytest

from qt.costs import round_trip_cost, sell_cost
from qt.evaluate import (
    REJECTED,
    SCORING,
    VETO,
    evaluate_factor,
    factor_return_series,
    ic_series,
    newey_west_t,
    quantile_excess_returns,
    rebalance_dates,
)
from qt.simulate import BufferRule
from qt.factors import forward_return_expr
from qt.preprocess import winsorize_mad, zscore


def _panel(n_days=60, n_stocks=100, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-05", periods=n_days)
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, codes], names=["datetime", "instrument"])
    return idx, rng


def test_forward_return_expr_enters_next_day():
    assert forward_return_expr(5) == "Ref($close,-6)/Ref($close,-1)-1"
    with pytest.raises(ValueError):
        forward_return_expr(0)


def test_rank_ic_is_one_for_monotone_signal():
    idx, rng = _panel()
    ret = pd.Series(rng.normal(size=len(idx)), index=idx)
    ic = ic_series(np.exp(ret), ret, "spearman")
    assert np.allclose(ic, 1.0)


def test_ic_nan_when_too_few_stocks():
    idx, rng = _panel(n_stocks=10)
    x = pd.Series(rng.normal(size=len(idx)), index=idx)
    assert ic_series(x, x).isna().all()


def test_newey_west_matches_plain_t_without_lags():
    x = pd.Series(np.random.default_rng(1).normal(0.1, 1, 2000))
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_t(x, lag=0) == pytest.approx(plain)


def test_newey_west_shrinks_t_for_overlapping_series():
    rng = np.random.default_rng(2)
    shocks = rng.normal(0.02, 1, 3000)
    overlapping = pd.Series(np.convolve(shocks, np.ones(5) / 5, mode="valid"))
    assert abs(newey_west_t(overlapping, lag=4)) < abs(newey_west_t(overlapping, lag=0))


def test_quantiles_are_monotone_when_score_equals_return():
    idx, rng = _panel()
    ret = pd.Series(rng.normal(size=len(idx)), index=idx)
    q = quantile_excess_returns(ret, ret, rebalance_dates(idx, 5), n_groups=5)
    means = q[[1, 2, 3, 4, 5]].mean()
    assert means.is_monotonic_increasing
    # 等分 5 组时各组超额之和为 0
    assert np.allclose(q[[1, 2, 3, 4, 5]].sum(axis=1), 0, atol=1e-12)


def test_factor_return_recovers_slope():
    idx, rng = _panel()
    z = pd.Series(rng.normal(size=len(idx)), index=idx)
    ret = 0.02 * z + 0.001
    assert np.allclose(factor_return_series(z, ret), 0.02)


def test_winsorize_clips_outlier_only():
    idx = pd.MultiIndex.from_product([[pd.Timestamp("2020-01-02")], list("ABCDEFGHIJ")], names=["datetime", "instrument"])
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000.0], index=idx)
    w = winsorize_mad(s)
    assert w.iloc[:9].tolist() == s.iloc[:9].tolist()
    assert w.iloc[-1] < 20
    z = zscore(w)
    assert z.mean() == pytest.approx(0, abs=1e-12)


def test_stamp_duty_halved_after_cut():
    assert sell_cost(pd.Timestamp("2023-08-25")) == pytest.approx(0.00225)
    assert sell_cost(pd.Timestamp("2023-08-28")) == pytest.approx(0.00175)
    assert round_trip_cost(pd.Timestamp("2018-01-02")) == pytest.approx(0.0035)


HORIZONS = (1, 2, 3, 5, 10, 20)
RULES = (BufferRule(30, 90),)


def _returns_with_signal(strength, seed, n_days=300):
    idx, rng = _panel(n_days=n_days, n_stocks=200, seed=seed)
    signal = pd.Series(rng.normal(size=len(idx)), index=idx)
    returns = {h: strength * signal * 0.01 + pd.Series(rng.normal(scale=0.01, size=len(idx)), index=idx) for h in HORIZONS}
    ones = pd.Series(True, index=idx)
    return signal, returns, ones


def _evaluate(name, factor, returns, ones, sign):
    return evaluate_factor(name, factor, zscore(factor), returns, ones, ones, sign, rules=RULES)


def test_planted_signal_is_a_scoring_factor():
    signal, returns, ones = _returns_with_signal(0.5, seed=3)
    res = _evaluate("planted", -signal, returns, ones, -1)
    assert res.rank_ic[5]["mean"] > 0.2
    assert res.category == SCORING
    assert res.monotonicity == pytest.approx(1.0)
    assert res.factor_return_main["t_nw"] > 3


def test_noise_is_rejected():
    signal, returns, ones = _returns_with_signal(0.0, seed=4)
    res = _evaluate("noise", signal, returns, ones, None)
    assert not res.checks["t 值 > 3"]
    assert res.category == REJECTED


def test_wrong_direction_is_rejected():
    signal, returns, ones = _returns_with_signal(0.5, seed=5)
    res = _evaluate("wrong_way", signal, returns, ones, -1)
    assert res.rank_ic[5]["mean"] < 0
    assert res.category == REJECTED


def test_signal_that_only_flags_losers_is_a_veto_factor():
    # 只有最差的 20% 会跑输，其余股票之间没有差别：做多赚不到钱，但能用来排雷
    idx, rng = _panel(n_days=300, n_stocks=200, seed=6)
    signal = pd.Series(rng.normal(size=len(idx)), index=idx)
    worst = signal.groupby(level="datetime").rank(pct=True) <= 0.2
    returns = {h: pd.Series(rng.normal(scale=0.01, size=len(idx)), index=idx) - 0.01 * worst for h in HORIZONS}
    ones = pd.Series(True, index=idx)
    res = _evaluate("veto", signal, returns, ones, 1)
    assert res.category == VETO
    assert res.bottom_t < -3


def test_vectorized_ic_matches_pandas_corr():
    idx, rng = _panel(n_days=20, n_stocks=80, seed=9)
    f = pd.Series(rng.normal(size=len(idx)), index=idx)
    r = 0.3 * f + pd.Series(rng.normal(size=len(idx)), index=idx)
    f.iloc[::7] = np.nan
    for method in ("pearson", "spearman"):
        expected = pd.DataFrame({"f": f, "r": r}).dropna().groupby(level="datetime").apply(lambda g: g["f"].corr(g["r"], method=method))
        assert np.allclose(ic_series(f, r, method), expected)
    slope = pd.DataFrame({"f": f, "r": r}).dropna().groupby(level="datetime").apply(lambda g: np.polyfit(g["f"], g["r"], 1)[0])
    assert np.allclose(factor_return_series(f, r), slope)
