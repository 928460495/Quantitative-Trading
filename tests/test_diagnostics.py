"""计划 3c 任务 8：诊断报告和五条成功标准（DECISIONS 2026-11-10）。"""

import numpy as np
import pandas as pd
import pytest

from qt.account import AccountResult
from qt.diagnostics import criteria, diagnose, holdable_benchmark, rolling_shortfall, size_corr, write_report
from qt.wide_eval import buyable_benchmark

VALID = pd.bdate_range("2019-01-02", "2021-12-31")


def _nav(rate, idx=VALID):
    return pd.Series(100_000 * (1 + rate) ** (np.arange(len(idx)) / 252), index=idx)


def test_36_months_is_exactly_one_window():
    out = rolling_shortfall(_nav(0.1), _nav(0.1))
    assert len(out) == 1
    assert out["start"].iloc[0] == VALID[0] and out["end"].iloc[0] == VALID[-1]
    assert out["shortfall"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_rolling_windows_slide_monthly():
    idx = pd.bdate_range("2013-01-02", "2018-12-28")
    out = rolling_shortfall(_nav(0.1, idx), _nav(0.05, idx))
    assert len(out) == 72 - 36 + 1
    assert (out["shortfall"] < 0).all()                 # 策略每年多赚 5%，没有一个窗口落后


def test_criteria_rows_and_rolling_failure():
    nav, bench = _nav(0.10), _nav(0.17)                 # 基准每年多 7 个点：④ 不满足
    got = criteria(nav, bench, nav_cost2x=_nav(0.08), baseline_annual=0.05)
    assert list(got.index) == ["①", "②", "③", "④", "⑤"]
    assert list(got.columns) == ["数值", "门槛", "满足"]
    assert not got.loc["①", "满足"] and not got.loc["④", "满足"] and got.loc["⑤", "满足"]
    ok = criteria(_nav(0.20), _nav(0.17), nav_cost2x=_nav(0.18), baseline_annual=0.05)
    assert ok["满足"].loc[["①", "③", "④", "⑤"]].all()


def test_criteria_drawdown_compares_to_bench():
    idx = VALID
    nav = _nav(0.10)
    bench = _nav(0.10).copy()
    bench.iloc[300:] *= 0.8                             # 基准回撤 20%，策略没有回撤
    got = criteria(nav, bench, nav_cost2x=nav, baseline_annual=0.0)
    assert got.loc["②", "满足"]
    nav2 = nav.copy()
    nav2.iloc[300:] *= 0.7
    assert not criteria(nav2, bench, nav_cost2x=nav2, baseline_annual=0.0).loc["②", "满足"]


def test_size_corr_of_size_itself_is_one():
    rng = np.random.default_rng(1)
    m = pd.DataFrame(rng.normal(size=(20, 50)), index=VALID[:20])
    assert size_corr(m, m) == pytest.approx(1.0)
    assert size_corr(pd.DataFrame(rng.normal(size=(20, 50)), index=VALID[:20]), m) < 0.2


def _seg(n_days=60, n_stocks=40):
    rng = np.random.default_rng(3)
    idx = VALID[:n_days]
    cols = [f"S{i}" for i in range(n_stocks)]
    close = pd.DataFrame(10 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, n_stocks)), axis=0)), index=idx, columns=cols)
    return {"close": close, "can_buy": pd.DataFrame(True, index=idx, columns=cols),
            "index": pd.DataFrame({"H00300": close.mean(axis=1)}),
            "log_mcap": pd.DataFrame(rng.normal(22, 1, (n_days, n_stocks)), index=idx, columns=cols)}


def test_diagnose_sections_from_account_result_and_returns():
    seg = _seg()
    idx = seg["close"].index
    nav = _nav(0.1, idx)
    cash = pd.Series([0.0] * 50 + list(nav.iloc[50:]), index=idx)          # 最后 10 天空仓
    daily = pd.DataFrame({"nav": nav, "cash": cash, "cost": 5.0, "slot_limit": [20] * 40 + [13] * 10 + [0] * 10}, index=idx)
    trades = pd.DataFrame({"date": [idx[1]], "side": ["buy"], "value": [5000.0], "reason": ["buy"]})
    score = seg["log_mcap"] * 0 + np.random.default_rng(4).normal(size=seg["close"].shape)
    out = diagnose(AccountResult(daily=daily, trades=trades), seg, score)
    assert set(out) == {"summary", "yearly", "rolling"}
    s = out["summary"]["值"]
    assert s["扣成本前年化"] > s["扣成本后年化"]
    assert s["分档减仓天数"] == 20 and s["交易笔数"] == 1
    assert s["平均持仓比例"] == pytest.approx(50 / 60)
    assert "能持有等权年化" in s.index
    y = out["yearly"]
    assert {"能持有等权", "相对能持有等权", "平均持仓比例", "空仓天数"} <= set(y.columns)
    assert y["空仓天数"].sum() == 10
    r = diagnose(_nav(0.1, idx).pct_change().fillna(0.0), seg, score)
    assert np.isnan(r["summary"]["值"]["交易笔数"]) and np.isnan(r["summary"]["值"]["平均持仓比例"])


def test_write_report(tmp_path):
    path = tmp_path / "r.md"
    write_report({"表": pd.DataFrame({"a": [1]})}, path, "标题", ["说明一"])
    text = path.read_text()
    assert text.startswith("# 标题") and "## 表" in text and "说明一" in text


def _two(rets, idx=VALID[:6]):
    close = pd.DataFrame((1 + pd.DataFrame(rets, index=idx)).cumprod() * 10)
    return close


def test_holdable_every_day_equals_daily_buyable():
    seg = _seg()
    cb = seg["can_buy"].copy()
    cb.iloc[::3, ::4] = False
    got = holdable_benchmark(seg["close"], cb, every=1)
    pd.testing.assert_series_equal(got, buyable_benchmark(seg["close"], cb), check_names=False)


def test_holdable_keeps_stock_that_becomes_unbuyable():
    # A 第 2 天起涨停买不进（can_buy False），但第 0 天信号买入后一直持有；旧基准第 3 天起把它剔除
    close = _two({"A": [0, 0, 0.1, 0.1, 0.1, 0.1], "B": [0] * 6})
    cb = pd.DataFrame({"A": [True, True, False, False, False, False], "B": True}, index=close.index)
    got = holdable_benchmark(close, cb, every=5)
    assert got.iloc[:2].tolist() == [0.0, 0.0]                      # 第 1 天收盘才买进
    assert got.iloc[2] == pytest.approx(0.05)
    assert got.iloc[3] == pytest.approx((1.21 + 1) / (1.1 + 1) - 1)  # 期间持有，权重随涨跌漂移
    assert buyable_benchmark(close, cb).iloc[4] == pytest.approx(0.0)


def test_holdable_rebalances_on_schedule_and_survives_suspension():
    close = _two({"A": [0, 0, 0.1, 0.0, 0.0, 0.0], "B": [0, 0, 0.0, 0.0, -0.1, 0.0]})
    close.iloc[3:5, 0] = np.nan                                    # A 停牌两天：当天收益记 0
    cb = pd.DataFrame(True, index=close.index, columns=["A", "B"])
    cb.iloc[2, 1] = False                                          # 第 2 天信号时 B 不能买 → 第 3 天收盘只买 A
    got = holdable_benchmark(close, cb, every=2)
    assert got.iloc[2] == pytest.approx(0.05)
    assert got.iloc[4] == pytest.approx(0.0)                       # 只拿着 A，A 停牌；B 跌 10% 不影响
    assert not got.isna().any()


def test_holdable_counts_return_over_suspension_on_resume_day():
    close = _two({"A": [0] * 6, "B": [0] * 6})
    close.iloc[3:5, 0] = np.nan
    close.iloc[5, 0] = 12.0                                        # 停牌前 10 元，复牌 12 元：复牌日 +20%
    cb = pd.DataFrame(True, index=close.index, columns=["A", "B"])
    got = holdable_benchmark(close, cb, every=10)
    assert got.iloc[5] == pytest.approx(0.1)
