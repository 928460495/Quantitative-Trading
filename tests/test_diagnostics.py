"""计划 3c 任务 8：诊断报告和五条成功标准（DECISIONS 2026-11-10）。"""

import numpy as np
import pandas as pd
import pytest

from qt.account import AccountResult
from qt.diagnostics import criteria, diagnose, rolling_shortfall, size_corr, write_report

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
    daily = pd.DataFrame({"nav": _nav(0.1, idx), "cost": 5.0, "slot_limit": [20] * 50 + [13] * 10}, index=idx)
    trades = pd.DataFrame({"date": [idx[1]], "side": ["buy"], "value": [5000.0], "reason": ["buy"]})
    score = seg["log_mcap"] * 0 + np.random.default_rng(4).normal(size=seg["close"].shape)
    out = diagnose(AccountResult(daily=daily, trades=trades), seg, score)
    assert set(out) == {"summary", "yearly", "rolling"}
    s = out["summary"]["值"]
    assert s["扣成本前年化"] > s["扣成本后年化"]
    assert s["分档减仓天数"] == 10 and s["交易笔数"] == 1
    r = diagnose(_nav(0.1, idx).pct_change().fillna(0.0), seg, score)
    assert np.isnan(r["summary"]["值"]["交易笔数"])


def test_write_report(tmp_path):
    path = tmp_path / "r.md"
    write_report({"表": pd.DataFrame({"a": [1]})}, path, "标题", ["说明一"])
    text = path.read_text()
    assert text.startswith("# 标题") and "## 表" in text and "说明一" in text
