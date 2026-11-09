"""类内筛选：按 ICIR 排序、相关性 > 0.7 跳过、每类最多 3 个（docs/DECISIONS.md 2026-11-01）。"""

import numpy as np
import pandas as pd
import pytest

from qt import screen_wide as sw

DATES = pd.bdate_range("2015-01-05", periods=60)
CODES = [f"S{i:03d}" for i in range(200)]


def _base(seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(len(DATES), len(CODES))), index=DATES, columns=CODES)


def _mix(a, b, w, seed):
    """和 a 的相关性约为 w 的因子：w·a + √(1−w²)·噪声。"""
    return w * a + np.sqrt(1 - w * w) * _base(seed) if b is None else w * a + np.sqrt(1 - w * w) * b


def test_pair_corr_is_mean_daily_spearman():
    a, b = _base(1), _base(2)
    b.iloc[:, :20] = np.nan
    want = np.mean([a.loc[d].corr(b.loc[d], method="spearman") for d in DATES])
    assert sw.pair_corr(a, b) == pytest.approx(want, abs=1e-12)


def test_pick_skips_near_duplicates_and_keeps_at_most_three():
    a = _base(10)
    scores = {
        "a": a,
        "a_copy": _mix(a, None, 0.95, 11),     # 和 a 很像 → 跳过
        "b": _base(12),
        "c": _base(13),
        "d": _base(14),                        # 前面已经留够 3 个 → 不再看
    }
    table = pd.DataFrame({"icir": [0.9, 0.8, 0.7, 0.6, 0.5]}, index=list(scores))
    res = sw.pick(table, scores)
    assert res.index[res["kept"]].tolist() == ["a", "b", "c"]
    assert res.loc["a_copy", "dropped_by"] == "a" and res.loc["a_copy", "corr"] > 0.9
    assert res.loc["d", "status"] == "已满 3 个，未比较"


def test_pick_uses_absolute_correlation():
    a = _base(20)
    scores = {"a": a, "neg": -_mix(a, None, 0.9, 21)}
    res = sw.pick(pd.DataFrame({"icir": [0.5, 0.4]}, index=list(scores)), scores)
    assert res.index[res["kept"]].tolist() == ["a"]


def test_pick_moderate_correlation_is_kept():
    a = _base(30)
    scores = {"a": a, "b": _mix(a, None, 0.5, 31)}
    res = sw.pick(pd.DataFrame({"icir": [0.5, 0.4]}, index=list(scores)), scores)
    assert res["kept"].all()


def test_pick_ties_broken_by_name_and_empty_category():
    scores = {"z": _base(40), "m": _base(41)}
    res = sw.pick(pd.DataFrame({"icir": [0.3, 0.3]}, index=["z", "m"]), scores)
    assert res.index.tolist() == ["m", "z"]
    empty = sw.pick(pd.DataFrame({"icir": []}, dtype=float), {})
    assert empty.empty and "kept" in empty.columns


def test_select_by_category_runs_each_category_separately():
    a = _base(50)
    scores = {"m1": a, "v1": _mix(a, None, 0.95, 51)}      # 跨类很像也都留：跨类不去重
    table = pd.DataFrame({"icir": [0.5, 0.4], "category": ["动量/反转", "价值"]}, index=["m1", "v1"])
    res = sw.select_by_category(table, scores)
    assert res["kept"].all()
    assert set(res["category"]) == {"动量/反转", "价值"}


def test_corr_matrix_symmetric_with_unit_diagonal():
    scores = {"a": _base(60), "b": _base(61)}
    m = sw.corr_matrix(scores)
    assert m.loc["a", "a"] == 1.0 and m.loc["a", "b"] == m.loc["b", "a"]


# ---------- 单个因子的完整检验（任务 6） ----------

from qt import wide_eval as we  # noqa: E402


def _ctx(n_days=400, n_stocks=200, seed=80):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n_days)
    cols = [f"SH600{i:03d}" for i in range(n_stocks)]
    close = pd.DataFrame(10 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n_stocks)), axis=0)), index=idx, columns=cols)
    true = pd.DataFrame(True, index=idx, columns=cols)
    industry = pd.DataFrame(np.array([f"{10 + i % 10}" for i in range(n_stocks)])[None, :].repeat(n_days, 0),
                            index=idx, columns=cols, dtype=object)
    log_mcap = pd.DataFrame(rng.normal(22, 1, (n_days, n_stocks)), index=idx, columns=cols)
    return sw.ScreenContext(close=close, raw_close=close, can_buy=true, can_sell=true, industry=industry,
                            log_mcap=log_mcap, horizons=(1, 5, 10, 20), main_horizon=5)


def test_planted_signal_passes_all_three_checks():
    ctx = _ctx()
    fwd = we.forward_returns(ctx.close, [5])[5]
    noise = pd.DataFrame(np.random.default_rng(81).normal(size=fwd.shape), index=fwd.index, columns=fwd.columns)
    row, sub = sw.evaluate_factor_wide("planted", fwd.rank(axis=1) + 40 * noise.rank(axis=1) / 200, "动量/反转", ctx)
    assert row["direction"] == 1 and row["t_nw_5d"] > 3
    assert row["pass_t"] and row["pass_stable"] and row["pass_cost"] and row["passed"]
    assert sub is not None and len(sub) == pytest.approx(len(ctx.dates) / 5, abs=1)   # 去掉段尾之后每 5 天一次


def test_reversed_signal_gets_negative_direction():
    ctx = _ctx()
    fwd = we.forward_returns(ctx.close, [5])[5]
    row, _ = sw.evaluate_factor_wide("rev", -fwd, "动量/反转", ctx)
    assert row["direction"] == -1 and row["t_nw_5d"] > 3


def test_noise_fails_and_skips_account_backtest():
    ctx = _ctx()
    noise = pd.DataFrame(np.random.default_rng(82).normal(size=ctx.close.shape), index=ctx.close.index,
                         columns=ctx.close.columns)
    row, sub = sw.evaluate_factor_wide("noise", noise, "价值", ctx)
    assert not row["pass_t"] and not row["passed"] and sub is None
    assert np.isnan(row["excess_annual"])                 # 前两条没过，不跑账户回测
    assert row["fail_reason"].startswith("t 值")


def test_size_category_is_only_industry_neutralized():
    ctx = _ctx()
    rng = np.random.default_rng(83)
    f = ctx.log_mcap + rng.normal(0, 0.3, ctx.log_mcap.shape)
    row_size, _ = sw.evaluate_factor_wide("lnsize", f, "规模", ctx)
    row_other, _ = sw.evaluate_factor_wide("lnsize", f, "价值", ctx)
    assert row_size["size_neutralized"] is False and row_other["size_neutralized"] is True
    assert row_size["rank_ic_5d"] != pytest.approx(row_other["rank_ic_5d"])


def test_evaluate_drops_warmup_and_tail():
    ctx = _ctx()
    fwd = we.forward_returns(ctx.close, [5])[5]
    row, _ = sw.evaluate_factor_wide("planted", fwd, "动量/反转", ctx)
    # 段尾 21 天被去掉（ScreenContext 构造时去掉），IC 天数 ≤ 400 − 21
    assert row["n_days_5d"] <= 400 - 21


def test_wide_masks_match_long_full_a_masks():
    from qt.universe import full_a_masks

    rng = np.random.default_rng(90)
    idx = pd.bdate_range("2012-03-01", periods=30)
    cols = [f"SH600{i:03d}" for i in range(20)] + [f"SZ300{i:03d}" for i in range(5)]   # 创业板不能买
    amount = pd.DataFrame(rng.uniform(1e6, 1e8, (30, 25)), index=idx, columns=cols)
    long_idx = pd.MultiIndex.from_product([idx, cols], names=["datetime", "instrument"])
    pre = rng.uniform(9, 11, len(long_idx))
    daily = pd.DataFrame({"preclose": pre, "close": pre * rng.choice([1.0, 1.1, 0.9, 1.02], len(long_idx)),
                          "tradestatus": rng.choice([1, 1, 1, 0], len(long_idx)),
                          "isST": rng.choice([0, 0, 0, 1], len(long_idx))}, index=long_idx)
    daily = daily.drop(daily.sample(frac=0.05, random_state=1).index)          # 部分日子没有记录 = 停牌
    listed = pd.Series(pd.Timestamp("2005-01-04"), index=cols)
    listed["SH600003"] = pd.Timestamp("2011-12-01")                             # 上市不满一年
    buy, sell = sw.wide_masks(amount, daily, listed)
    df = pd.DataFrame({"_amount20": amount.rolling(20, min_periods=1).mean().stack()})
    df.index.names = ["datetime", "instrument"]
    want_buy, want_sell = full_a_masks(df, daily, listed)
    pd.testing.assert_series_equal(_long_bool(buy), want_buy, check_names=False)
    pd.testing.assert_series_equal(_long_bool(sell), want_sell, check_names=False)
    assert not buy["SZ300000"].any() and not buy["SH600003"].any()


def _long_bool(w):
    s = w.stack()
    s.index.names = ["datetime", "instrument"]
    return s.astype(bool)


@pytest.mark.slow
def test_full_size_one_factor_without_account_is_fast():
    import time

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2009-01-05", periods=2400)
    cols = [f"SH6{i:05d}" for i in range(2800)]
    close = pd.DataFrame(10 * np.exp(np.cumsum(rng.normal(0, 0.02, (2400, 2800)), axis=0)), index=idx, columns=cols)
    true = pd.DataFrame(True, index=idx, columns=cols)
    industry = pd.DataFrame(rng.integers(10, 41, (2400, 2800)).astype(str), index=idx, columns=cols, dtype=object)
    log_mcap = pd.DataFrame(rng.normal(22, 1, (2400, 2800)), index=idx, columns=cols)
    ctx = sw.ScreenContext(close=close, raw_close=close, can_buy=true, can_sell=true, industry=industry,
                           log_mcap=log_mcap, start="2010-01-01")
    noise = pd.DataFrame(rng.normal(size=(2400, 2800)), index=idx, columns=cols)
    t = time.time()
    row, _ = sw.evaluate_factor_wide("noise", noise, "价值", ctx)
    assert not row["pass_t"]
    assert time.time() - t < 30


def test_subs_share_one_date_grid_even_when_factors_start_late():
    """成长类要等 8 期财报才有值，开始得晚；抽样日必须和其他因子对齐，否则相关性算不出来（2026-11-09 修）。"""
    ctx = _ctx()
    fwd = we.forward_returns(ctx.close, [5])[5]
    noise = pd.DataFrame(np.random.default_rng(84).normal(size=fwd.shape), index=fwd.index, columns=fwd.columns)
    early = fwd.rank(axis=1) + 40 * noise.rank(axis=1) / 200
    late = early.copy()
    late.iloc[:37] = np.nan                                # 晚 37 天才有值，37 不是 5 的倍数
    _, sub_a = sw.evaluate_factor_wide("early", early, "成长", ctx)
    _, sub_b = sw.evaluate_factor_wide("late", late, "成长", ctx)
    assert sub_a.index.equals(pd.DatetimeIndex(ctx.dates[::5]))
    assert sub_b.index.equals(sub_a.index)
    assert sw.pair_corr(sub_a, sub_b) > 0.9


def test_screen_score_is_what_evaluate_keeps():
    ctx = _ctx()
    fwd = we.forward_returns(ctx.close, [5])[5]
    row, sub = sw.evaluate_factor_wide("x", -fwd, "动量/反转", ctx)
    again = sw.screen_score(-fwd, "动量/反转", row["direction"], ctx)
    pd.testing.assert_frame_equal(again, sub)
