"""宽表版的单因子检验指标：和长表版（qt/evaluate.py，沪深300 上已经核对过）在同一份数据上逐值对比。"""

import numpy as np
import pandas as pd
import pytest

from qt import evaluate as long_eval
from qt import wide_eval as we

DATES = pd.bdate_range("2015-01-05", periods=80)
CODES = [f"S{i:03d}" for i in range(120)]


def _wide(seed, nan_share=0.1):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(len(DATES), len(CODES))), index=DATES, columns=CODES)
    return x.mask(rng.random(x.shape) < nan_share)


def _long(wide: pd.DataFrame) -> pd.Series:
    s = wide.stack(future_stack=True)
    s.index.names = ["datetime", "instrument"]
    return s


def _close(seed=1):
    rng = np.random.default_rng(seed)
    c = pd.DataFrame(10 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(DATES), len(CODES))), axis=0)),
                     index=DATES, columns=CODES)
    return c.mask(rng.random(c.shape) < 0.05)   # 停牌


# ---------- 远期收益 ----------

def test_forward_returns_enter_next_day_and_hold_h_days():
    c = _close()
    r = we.forward_returns(c, [1, 5])
    t = 10
    assert r[5].iloc[t, 0] == pytest.approx(c.iloc[t + 6, 0] / c.iloc[t + 1, 0] - 1, nan_ok=True)
    assert r[1].iloc[t, 3] == pytest.approx(c.iloc[t + 2, 3] / c.iloc[t + 1, 3] - 1, nan_ok=True)
    assert r[5].iloc[-6:].isna().all().all()          # 段尾没有完整的持有期


def test_forward_returns_do_not_depend_on_prices_after_the_holding_period():
    c = _close()
    changed = c.copy()
    changed.iloc[30:] *= 3
    a, b = we.forward_returns(c, [5])[5], we.forward_returns(changed, [5])[5]
    pd.testing.assert_frame_equal(a.iloc[:23], b.iloc[:23])   # T+6 ≤ 29


def test_drop_tail():
    x = _wide(0)
    assert we.drop_tail(x, 21).index[-1] == DATES[-22]
    with pytest.raises(ValueError):
        we.drop_tail(x.iloc[:5], 21)


# ---------- RankIC ----------

def test_rank_ic_matches_long_version():
    f, r = _wide(2), _wide(3)
    mask = pd.DataFrame(np.random.default_rng(4).random(f.shape) > 0.2, index=DATES, columns=CODES)
    wide = we.rank_ic(f, r, mask)
    long = long_eval.ic_series(_long(f.where(mask)), _long(r), "spearman")
    pd.testing.assert_series_equal(wide, long, check_names=False, check_freq=False, atol=1e-12)


def test_rank_ic_handles_ties_like_pandas():
    f = _wide(5).round(0)          # 大量并列
    r = _wide(6)
    long = long_eval.ic_series(_long(f), _long(r), "spearman")
    pd.testing.assert_series_equal(we.rank_ic(f, r), long, check_names=False, check_freq=False, atol=1e-12)


def test_rank_ic_missing_when_fewer_than_30_stocks():
    f, r = _wide(7), _wide(8)
    f.iloc[3, 25:] = np.nan
    ic = we.rank_ic(f, r)
    assert np.isnan(ic.iloc[3]) and ic.notna().iloc[4]


def test_rank_ic_constant_day_is_missing_not_error():
    f, r = _wide(9, 0), _wide(10)
    f.iloc[2] = 1.0
    assert np.isnan(we.rank_ic(f, r).iloc[2])


# ---------- 汇总 ----------

def test_ic_summary_is_long_version():
    ic = we.rank_ic(_wide(11), _wide(12))
    assert we.ic_summary(ic, 5) == long_eval.ic_stats(ic, 5)


def test_yearly_reversed_ratio():
    idx = pd.to_datetime(["2015-03-01", "2015-06-01", "2016-03-01", "2017-03-01", "2017-04-01"])
    ic = pd.Series([0.05, 0.03, -0.02, 0.01, -0.04], index=idx)
    # 年均：2015 +0.04，2016 −0.02，2017 −0.015 → 方向为正时反了 2/3 年
    assert we.yearly_reversed_ratio(ic, 1) == pytest.approx(2 / 3)
    assert we.yearly_reversed_ratio(ic, -1) == pytest.approx(1 / 3)


# ---------- 分层 ----------

def test_quantile_excess_matches_long_version():
    f, r = _wide(13), _wide(14)
    mask = pd.DataFrame(np.random.default_rng(15).random(f.shape) > 0.3, index=DATES, columns=CODES)
    score = f.where(mask)
    wide = we.quantile_excess(score, r, every=5, n_groups=5)
    long_score = _long(score)
    dates = long_eval.rebalance_dates(long_score.dropna().index, every=5)
    long = long_eval.quantile_excess_returns(long_score, _long(r), dates, 5)
    pd.testing.assert_frame_equal(wide, long, check_freq=False, check_names=False, atol=1e-12)


def test_quantile_excess_monotone_when_score_is_return():
    r = _wide(16, 0)
    q = we.quantile_excess(r, r, every=5, n_groups=5)
    assert (q[[1, 2, 3, 4, 5]].mean().diff().dropna() > 0).all()


# ---------- 速度 ----------

@pytest.mark.slow
def test_full_size_rank_ic_is_fast():
    import time

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2009-01-05", periods=2400)
    f = pd.DataFrame(rng.normal(size=(2400, 2800)), index=idx)
    r = pd.DataFrame(rng.normal(size=(2400, 2800)), index=idx)
    t = time.time()
    we.rank_ic(f, r)
    we.quantile_excess(f, r)
    assert time.time() - t < 15
