import numpy as np
import pandas as pd
import pytest

from qt.costs import round_trip_cost, sell_cost
from qt.evaluate import (
    evaluate_factor,
    ic_series,
    newey_west_t,
    quantile_excess_returns,
    rebalance_dates,
    top_group_turnover,
)
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


def test_turnover_zero_when_ranking_never_changes():
    idx, _ = _panel()
    score = pd.Series(np.tile(np.arange(100.0), 60), index=idx)
    t = top_group_turnover(score, rebalance_dates(idx, 5))
    assert t.iloc[0] == 1.0
    assert (t.iloc[1:] == 0).all()


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


def _returns_with_signal(strength, seed):
    idx, rng = _panel(n_days=500, n_stocks=200, seed=seed)
    signal = pd.Series(rng.normal(size=len(idx)), index=idx)
    returns = {h: strength * signal + pd.Series(rng.normal(size=len(idx)), index=idx) for h in (1, 5, 10, 20)}
    return signal, returns


def test_evaluate_factor_detects_planted_signal_with_expected_sign():
    signal, returns = _returns_with_signal(0.3, seed=3)
    res = evaluate_factor("planted", -signal, zscore(-signal), returns, expected_sign=-1)
    assert res.rank_ic[5]["mean"] > 0.2
    assert res.passed
    assert res.monotonicity == pytest.approx(1.0)


def test_evaluate_factor_rejects_noise():
    signal, returns = _returns_with_signal(0.0, seed=4)
    res = evaluate_factor("noise", signal, zscore(signal), returns, expected_sign=None)
    assert not res.checks["t 值 > 3"]


def test_evaluate_factor_flags_wrong_direction():
    signal, returns = _returns_with_signal(0.3, seed=5)
    res = evaluate_factor("wrong_way", signal, zscore(signal), returns, expected_sign=-1)
    assert res.rank_ic[5]["mean"] < 0
    assert not res.passed
