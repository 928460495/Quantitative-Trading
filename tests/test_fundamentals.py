import numpy as np
import pandas as pd
import pytest

from qt.fundamentals import align_to_days, single_quarter, tidy_reports, ttm, yoy


def _yjbb(rows):
    """rows：(代码, 报告期, 公告日, 净利润累计, 证券类型)。"""
    return pd.DataFrame({
        "instrument": [("SH" if c.startswith("6") else "SZ" if c[0] in "023" else "BJ") + c for c, *_ in rows],
        "report_date": [r for _, r, *_ in rows],
        "SECURITY_CODE": [c for c, *_ in rows],
        "NOTICE_DATE": [n + " 00:00:00" for _, _, n, *_ in rows],
        "PARENT_NETPROFIT": [v for *_, v, _ in rows],
        "TOTAL_OPERATE_INCOME": [v * 10 for *_, v, _ in rows],
        "SECURITY_TYPE": [t for *_, t in rows],
        "BASIC_EPS": 1.0, "BPS": 5.0, "WEIGHTAVG_ROE": 10.0, "XSMLL": 30.0, "MGJYXJJE": 0.5, "YSTZ": 1.0, "SJLTZ": 2.0,
    })


def test_tidy_keeps_a_shares_and_merges_balance():
    yj = _yjbb([("600000", "2020-12-31", "2021-03-30", 100.0, "A股"), ("900901", "2020-12-31", "2021-03-30", 1.0, "B股"),
                ("830799", "2020-12-31", "2021-03-30", 1.0, "A股"), ("430047", "2020-12-31", "2021-03-30", 1.0, "三板股")])
    bal = pd.DataFrame({"instrument": ["SH600000"], "report_date": ["2020-12-31"], "TOTAL_ASSETS": [1000.0],
                        "TOTAL_LIABILITIES": [600.0], "TOTAL_EQUITY": [400.0], "NOTICE_DATE": ["2022-03-30 00:00:00"]})
    out = tidy_reports({"yjbb": yj, "balance": bal})
    assert list(out["instrument"]) == ["SH600000"]
    row = out.iloc[0]
    assert row["net_profit"] == 100.0 and row["total_equity"] == 400.0
    assert row["notice_date"] == pd.Timestamp("2021-03-30")      # 用业绩报表的首次公告日，不用资产负债表的
    assert row["report_date"] == pd.Timestamp("2020-12-31")


def test_restatement_keeps_first_notice():
    yj = _yjbb([("600000", "2020-12-31", "2021-03-30", 100.0, "A股"), ("600000", "2020-12-31", "2021-05-10", 90.0, "A股")])
    out = tidy_reports({"yjbb": yj})
    assert len(out) == 1 and out["notice_date"].iloc[0] == pd.Timestamp("2021-03-30") and out["net_profit"].iloc[0] == 100.0


def _records(values, start="2019-03-31", code="SH600000"):
    dates = pd.date_range(start, periods=len(values), freq="QE")
    return pd.DataFrame({"instrument": code, "report_date": dates, "net_profit": values,
                         "notice_date": dates + pd.Timedelta(days=30)})


def test_single_quarter_and_ttm_from_cumulative():
    df = _records([10.0, 25.0, 45.0, 70.0, 12.0])
    assert list(single_quarter(df, "net_profit")) == [10.0, 15.0, 20.0, 25.0, 12.0]
    t = ttm(df, "net_profit")
    assert np.isnan(t.iloc[2]) and t.iloc[3] == 70.0 and t.iloc[4] == 72.0   # 15+20+25+12


def test_single_quarter_needs_previous_quarter():
    df = _records([10.0, 25.0, 45.0, 70.0]).drop(index=1)       # 缺半年报：三季度单季算不出
    sq = single_quarter(df, "net_profit")
    assert sq.iloc[0] == 10.0 and np.isnan(sq.iloc[1]) and sq.iloc[2] == 25.0


def test_yoy_nan_when_base_not_positive():
    df = _records([10.0, 25.0, 45.0, 70.0, 12.0, 20.0])       # 2020Q1 单季 12 vs 10；2020Q2 单季 8 vs 15
    y = yoy(df, "net_profit")
    assert y.iloc[4] == pytest.approx(0.2) and y.iloc[5] == pytest.approx(8 / 15 - 1)
    neg = _records([-10.0, -5.0, 0.0, 5.0, 3.0])
    assert np.isnan(yoy(neg, "net_profit").iloc[4])            # 基数 -10 ≤ 0


CAL = pd.bdate_range("2021-04-26", "2021-05-07")


def _index(codes=("SH600000",)):
    return pd.MultiIndex.from_product([CAL, list(codes)], names=["datetime", "instrument"])


def test_annual_report_not_visible_before_notice():
    rec = pd.DataFrame({"instrument": ["SH600000"], "report_date": [pd.Timestamp("2020-12-31")],
                        "notice_date": [pd.Timestamp("2021-04-28")], "net_profit": [100.0]})
    out = align_to_days(rec, ["net_profit"], _index(), CAL)
    s = out["net_profit"].droplevel("instrument")
    assert np.isnan(s[pd.Timestamp("2021-04-28")]) and s[pd.Timestamp("2021-04-29")] == 100.0
    assert out["report_date"].droplevel("instrument")[pd.Timestamp("2021-05-07")] == pd.Timestamp("2020-12-31")


def test_older_period_announced_later_does_not_override_newer():
    rec = pd.DataFrame({"instrument": ["SH600000", "SH600000"],
                        "report_date": [pd.Timestamp("2021-03-31"), pd.Timestamp("2020-12-31")],
                        "notice_date": [pd.Timestamp("2021-04-27"), pd.Timestamp("2021-04-30")],
                        "net_profit": [30.0, 100.0]})
    s = align_to_days(rec, ["net_profit"], _index(), CAL)["net_profit"].droplevel("instrument")
    assert s[pd.Timestamp("2021-05-06")] == 30.0                # 一季报更新，晚到的年报不覆盖它


def test_single_quarter_with_several_stocks():
    df = pd.concat([_records([10.0, 25.0, 45.0, 70.0]), _records([1.0, 3.0, 6.0, 10.0], code="SZ000001")], ignore_index=True)
    assert list(single_quarter(df, "net_profit")) == [10.0, 15.0, 20.0, 25.0, 1.0, 2.0, 3.0, 4.0]


def test_align_accepts_categorical_instrument_index():
    rec = pd.DataFrame({"instrument": ["SH600000"], "report_date": [pd.Timestamp("2020-12-31")],
                        "notice_date": [pd.Timestamp("2021-04-28")], "net_profit": [100.0]})
    idx = pd.MultiIndex.from_arrays([CAL, pd.Categorical(["SH600000"] * len(CAL))], names=["datetime", "instrument"])
    out = align_to_days(rec, ["net_profit"], idx, CAL)
    assert out["net_profit"].iloc[-1] == 100.0
