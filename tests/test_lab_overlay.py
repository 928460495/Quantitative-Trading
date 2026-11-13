"""估值 + 趋势仓位的信号（qt/lab/overlay.py；DECISIONS 2026-11-13）。小假数据，不碰真实数据。"""

import numpy as np
import pandas as pd
import pytest

from qt.lab import overlay as ov


def _monthly(path, start="1999-01-29", n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="ME")
    pe = 20 * np.exp(np.cumsum(rng.normal(0, 0.05, n)))
    idx = 2000 * np.exp(np.cumsum(rng.normal(0, 0.06, n)))
    pd.DataFrame({"日期": dates.strftime("%Y-%m-%d"), "指数": idx, "平均市盈率": pe}).to_csv(path, index=False)
    return dates, pe, idx


def test_sealed_rows_are_dropped_unless_unlocked(tmp_path, monkeypatch):
    path = tmp_path / "pe.csv"
    _monthly(path)                                                             # 1999-01 起 300 个月，到 2023 年底
    monkeypatch.delenv("QT_UNLOCK_TEST", raising=False)
    m = ov.load_monthly(path)
    assert m.index.max() < pd.Timestamp("2022-01-01") and m.index.min() == pd.Timestamp("1999-01-31")
    monkeypatch.setenv("QT_UNLOCK_TEST", "1")
    assert ov.load_monthly(path).index.max() > pd.Timestamp("2022-01-01")


def test_monthly_signals_match_the_registered_rules(tmp_path, monkeypatch):
    monkeypatch.delenv("QT_UNLOCK_TEST", raising=False)
    path = tmp_path / "pe.csv"
    dates, pe, idx = _monthly(path)
    m = ov.load_monthly(path)
    sig = ov.monthly_signals(m)
    ep = 1 / pe[:len(m)]
    k = 80
    assert sig["valuation"].iloc[k] == pytest.approx((ep[k - 59:k + 1] <= ep[k]).mean())
    assert np.isnan(sig["valuation"].iloc[58]) and not np.isnan(sig["valuation"].iloc[59])
    assert sig["trend"].iloc[k] == float(idx[k] >= idx[k - 9:k + 1].mean())
    assert np.isnan(sig["trend"].iloc[8]) and not np.isnan(sig["trend"].iloc[9])


def test_daily_target_uses_only_months_strictly_before_the_day(tmp_path, monkeypatch):
    monkeypatch.delenv("QT_UNLOCK_TEST", raising=False)
    path = tmp_path / "pe.csv"
    _monthly(path)
    sig = ov.monthly_signals(ov.load_monthly(path))
    days = pd.bdate_range("2015-03-25", "2015-04-03")
    f = ov.daily_target(sig, days)
    feb, mar = pd.Timestamp("2015-02-28"), pd.Timestamp("2015-03-31")
    want = lambda month: 0.25 + 0.50 * (0.5 * sig.loc[month, "valuation"] + 0.5 * sig.loc[month, "trend"])
    assert f.loc["2015-03-31"] == pytest.approx(want(feb))                    # 月末当天还不能用当月的值
    assert f.loc["2015-04-01"] == pytest.approx(want(mar))
    assert ov.daily_target(sig, days, low=0.2, high=0.85).loc["2015-04-01"] == pytest.approx(
        0.2 + 0.65 * (0.5 * sig.loc[mar, "valuation"] + 0.5 * sig.loc[mar, "trend"]))


def test_changing_future_months_does_not_change_earlier_targets(tmp_path, monkeypatch):
    monkeypatch.delenv("QT_UNLOCK_TEST", raising=False)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _monthly(a)
    df = pd.read_csv(a)
    df.loc[df["日期"] >= "2016-06-30", ["指数", "平均市盈率"]] *= 3.0
    df.to_csv(b, index=False)
    days = pd.bdate_range("2014-01-02", "2016-12-30")
    fa = ov.daily_target(ov.monthly_signals(ov.load_monthly(a)), days)
    fb = ov.daily_target(ov.monthly_signals(ov.load_monthly(b)), days)
    cut = pd.Timestamp("2016-06-30")
    pd.testing.assert_series_equal(fa[fa.index <= cut], fb[fb.index <= cut])
    assert not fa[fa.index > cut].equals(fb[fb.index > cut])


def test_slots_update_every_20_days_with_a_two_slot_band():
    days = pd.bdate_range("2020-01-01", periods=100)
    f = pd.Series(0.75, index=days)
    f.iloc[5:] = 0.70                                                          # 第 5 天：14 只，差 1 只
    f.iloc[25:] = 0.55                                                         # 第 25 天：11 只，差 4 只，但要等到第 40 天
    f.iloc[45:] = 0.50                                                         # 10 只，和 11 只差 1 只：不动
    f.iloc[65:] = 0.25                                                         # 5 只
    k = ov.slot_counts(f, n_hold=15, high=0.75, every=20, band=2)
    assert k.iloc[0] == 15 and k.iloc[39] == 15 and k.iloc[40] == 11
    assert k.iloc[60] == 11 and k.iloc[79] == 11 and k.iloc[80] == 5
    assert set(np.flatnonzero(np.diff(k.to_numpy()))) <= {39, 79}             # 只在更新日变
    assert ov.slot_counts(f.iloc[3:], 15, 0.75, 20, 2).iloc[0] == 15          # 从别的起点开始也按自己的第 0 天算


def test_slots_band_is_strictly_more_than_two():
    days = pd.bdate_range("2020-01-01", periods=45)
    f = pd.Series(0.75, index=days)
    f.iloc[1:] = 0.75 - 0.10                                                   # 正好差 2 只
    assert (ov.slot_counts(f, 15, 0.75, 20, 2) == 15).all()
    f.iloc[1:] = 0.75 - 0.11
                                                                               # 差 2.2 只，四舍五入成 2 只 → 仍不动
    assert (ov.slot_counts(f, 15, 0.75, 20, 2) == 15).all()
    f.iloc[1:] = 0.55
    assert ov.slot_counts(f, 15, 0.75, 20, 2).iloc[20] == 11


def test_trend_counts_equal_to_average_as_up():
    dates = pd.date_range("2000-01-31", periods=12, freq="ME")
    m = pd.DataFrame({"ep": 0.05, "index": 100.0}, index=dates)               # 指数一直不变：等于均线
    assert (ov.monthly_signals(m)["trend"].dropna() == 1.0).all()


def test_missing_target_is_an_error_not_silently_zero():
    f = pd.Series([0.5, np.nan, 0.5], index=pd.bdate_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match="缺失"):
        ov.slot_counts(f, 15)
