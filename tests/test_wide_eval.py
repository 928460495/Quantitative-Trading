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


# ---------- 预处理和中性化（任务 3） ----------

from qt.neutralize import neutralize as long_neutralize  # noqa: E402
from qt.preprocess import standardize as long_standardize  # noqa: E402


def _exposures(seed=20):
    rng = np.random.default_rng(seed)
    inds = np.array(["11", "21", "22", "23", "24", "27", "28"])
    p = np.array([0.3, 0.25, 0.2, 0.12, 0.1, 0.02, 0.01])   # 后两个行业常常不足 5 只，要并入"其他"
    industry = pd.DataFrame(rng.choice(inds, size=(len(DATES), len(CODES)), p=p), index=DATES, columns=CODES,
                            dtype=object)
    industry = industry.mask(rng.random(industry.shape) < 0.05)
    log_mcap = pd.DataFrame(rng.normal(22, 1, (len(DATES), len(CODES))), index=DATES, columns=CODES)
    log_mcap = log_mcap.mask(rng.random(log_mcap.shape) < 0.05)
    return industry, log_mcap


def _with_outliers(seed):
    x = _wide(seed)
    x.iloc[::7, ::11] *= 50                       # 极端值
    x.iloc[5] = x.iloc[5].round(0) * 0 + 1.0      # 一天全相同：MAD 为 0，不截
    return x


def test_standardize_matches_long_version():
    x = _with_outliers(21)
    got = we.standardize(x)
    pd.testing.assert_series_equal(_long(got), long_standardize(_long(x)), check_names=False, atol=1e-12)


def test_neutralize_matches_long_version():
    x = we.standardize(_with_outliers(22))
    industry, log_mcap = _exposures()
    x.iloc[7, 20:] = np.nan                       # 这天不足 30 只：整行缺失
    got = we.neutralize(x, industry, log_mcap)
    want = long_neutralize(_long(x), _long(industry), _long(log_mcap))
    pd.testing.assert_series_equal(_long(got), want, check_names=False, atol=1e-10)
    assert got.iloc[7].isna().all()


def test_neutral_score_removes_size_but_industry_only_keeps_it():
    industry, log_mcap = _exposures(23)
    rng = np.random.default_rng(24)
    factor = log_mcap + rng.normal(0, 0.5, log_mcap.shape)       # 一个"规模"因子
    full = we.neutral_score(factor, industry, log_mcap)
    ind_only = we.neutral_score(factor, industry, log_mcap, size=False)
    corr = lambda a: a.corrwith(log_mcap, axis=1).mean()
    assert abs(corr(full)) < 1e-8
    assert corr(ind_only) > 0.5


def test_industry_only_is_within_industry_demeaning():
    industry, log_mcap = _exposures(25)
    x = we.standardize(_wide(26, 0))
    got = we.neutralize(x, industry, log_mcap, size=False)
    d = DATES[10]
    ind = industry.loc[d]
    big = ind.map(ind.value_counts()) >= 5
    row = x.loc[d].where(ind.notna())
    # 大行业：减去行业均值；并入"其他"的小行业不设哑变量，残差就是原值；最后在全部有效股票上标准化
    resid = (row - row.groupby(ind.where(big)).transform("mean")).where(big, row)
    resid = (resid - resid.mean()) / resid.std()
    assert (~big & ind.notna()).any()             # 这天确实有并入"其他"的股票
    pd.testing.assert_series_equal(got.loc[d], resid, check_names=False, atol=1e-10)


def test_neutralize_uses_only_same_day_data():
    x = we.standardize(_wide(27))
    industry, log_mcap = _exposures(28)
    changed = x.copy()
    changed.iloc[40:] = changed.iloc[40:] * 5 + 1
    a, b = we.neutralize(x, industry, log_mcap), we.neutralize(changed, industry, log_mcap)
    pd.testing.assert_frame_equal(a.iloc[:40], b.iloc[:40])


@pytest.mark.slow
def test_full_size_neutral_score_is_fast():
    import time

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2009-01-05", periods=2400)
    x = pd.DataFrame(rng.normal(size=(2400, 2800)), index=idx)
    industry = pd.DataFrame(rng.integers(0, 31, (2400, 2800)).astype(str), index=idx, dtype=object)
    log_mcap = pd.DataFrame(rng.normal(22, 1, (2400, 2800)), index=idx)
    t = time.time()
    we.neutral_score(x, industry, log_mcap)
    assert time.time() - t < 20


# ---------- 按实际账户规则扣成本（任务 4） ----------

from qt.account import AccountRules  # noqa: E402


def _market(n_days=300, n_stocks=60, seed=30, vol=0.02):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-04", periods=n_days)
    cols = [f"SH600{i:03d}" for i in range(n_stocks)]
    adj = pd.DataFrame(10 * np.exp(np.cumsum(rng.normal(0, vol, (n_days, n_stocks)), axis=0)), index=idx, columns=cols)
    true = pd.DataFrame(True, index=idx, columns=cols)
    industry = pd.DataFrame(np.array([f"{10 + i % 8}" for i in range(n_stocks)])[None, :].repeat(n_days, 0),
                            index=idx, columns=cols, dtype=object)
    return adj, true, industry


def test_screen_rules_are_account_defaults_without_drawdown_stop():
    assert we.SCREEN_RULES == AccountRules(drawdown_stop=None)


def test_equal_weight_benchmark_by_hand():
    idx = pd.bdate_range("2012-01-04", periods=4)
    adj = pd.DataFrame({"A": [10, 10, 11, 11.0], "B": [10, 10, 10, 12.0]}, index=idx)
    can_buy = pd.DataFrame({"A": [True, True, True, True], "B": [False, True, True, True]}, index=idx)
    b = we.buyable_benchmark(adj, can_buy)
    # 第 t 天的收益：信号日 t−2 能买、t−1 收盘买进的股票，t−1 → t 的平均收益
    # 第 2 天：信号日 0 只有 A 能买 → +10%；第 3 天：A、B 都能买 → (0 + 20%) / 2
    assert b.iloc[2] == pytest.approx(0.10) and b.iloc[3] == pytest.approx(0.10)
    assert b.iloc[:2].tolist() == [0.0, 0.0]


def test_perfect_foresight_beats_benchmark():
    adj, true, industry = _market()
    score = we.forward_returns(adj, [1])[1]          # 信号日就知道 T+1 → T+2 的收益
    res = we.account_excess(score, adj, adj, true, true, industry)
    assert res["excess_annual"] > 1.0
    assert res["n_trades"] > 100


def test_flat_market_loses_only_costs():
    adj, true, industry = _market(vol=0.0)
    score = pd.DataFrame(np.random.default_rng(31).normal(size=adj.shape), index=adj.index, columns=adj.columns)
    res = we.account_excess(score, adj, adj, true, true, industry)
    assert res["bench_annual"] == pytest.approx(0.0)
    assert res["annual_return"] < 0 and res["excess_annual"] < 0
    assert res["cost_annual"] > 0


def test_account_excess_ignores_drawdown_stop():
    adj, true, industry = _market(vol=0.0)
    adj.iloc[50:] *= 0.5                              # 全市场腰斩：开了回撤兜底就会清仓
    score = pd.DataFrame(np.random.default_rng(32).normal(size=adj.shape), index=adj.index, columns=adj.columns)
    res = we.account_excess(score, adj, adj, true, true, industry)
    assert res["n_drawdown_sells"] == 0


@pytest.mark.slow
def test_full_size_account_backtest_is_fast():
    import time

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-04", periods=2200)
    cols = [f"SH6{i:05d}" for i in range(2800)]
    adj = pd.DataFrame(10 * np.exp(np.cumsum(rng.normal(0, 0.02, (2200, 2800)), axis=0)), index=idx, columns=cols)
    true = pd.DataFrame(True, index=idx, columns=cols)
    industry = pd.DataFrame(rng.integers(10, 41, (2200, 2800)).astype(str), index=idx, columns=cols, dtype=object)
    score = pd.DataFrame(rng.normal(size=(2200, 2800)), index=idx, columns=cols)
    t = time.time()
    we.account_excess(score, adj, adj, true, true, industry)
    assert time.time() - t < 60
