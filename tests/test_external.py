import numpy as np
import pandas as pd
import pytest

from qt.external import float_shares_on, industry_level1, industry_on, tidy_industry, tidy_share_changes, to_symbol


def _index(dates, codes):
    return pd.MultiIndex.from_product([pd.to_datetime(dates), codes], names=["datetime", "instrument"])


def test_code_helpers():
    assert to_symbol("SH600000") == "600000"
    assert industry_level1("480301") == "48"


def test_industry_is_point_in_time():
    raw = pd.DataFrame(
        {"symbol": ["600000", "600000", "000001"], "start_date": ["2011-01-01", "2014-06-30", "2011-01-01"],
         "industry_code": ["480301", "490101", "480101"], "update_time": [None, None, None]}
    )
    ind = tidy_industry(raw)
    idx = _index(["2010-12-31", "2014-06-27", "2014-06-30"], ["SH600000", "SZ000001"])
    out = industry_on(idx, ind)
    assert out.isna().tolist()[:2] == [True, True]  # 2010 年还没有记录
    assert out.loc[(pd.Timestamp("2014-06-27"), "SH600000")] == "48"
    assert out.loc[(pd.Timestamp("2014-06-30"), "SH600000")] == "49"
    assert out.loc[(pd.Timestamp("2014-06-30"), "SZ000001")] == "48"


def _cninfo(rows):
    return pd.DataFrame(rows, columns=["公告日期", "变动日期", "总股本", "人民币普通股", "已流通股份"])


def test_share_change_effective_date_is_later_of_declare_and_change():
    raw = _cninfo(
        [
            ["2015-01-10", "2015-01-05", 100.0, 60.0, 60.0],  # 事后公告：1 月 10 日才生效
            ["2015-03-01", "2015-03-20", 100.0, 80.0, 80.0],  # 提前公告：3 月 20 日才生效
            ["2015-05-01", "2015-05-01", 100.0, np.nan, 90.0],  # 人民币普通股缺失，用已流通股份
        ]
    )
    sh = tidy_share_changes(raw, "600000")
    assert sh["effective_date"].dt.strftime("%Y-%m-%d").tolist() == ["2015-01-10", "2015-03-20", "2015-05-01"]
    assert sh["float_a_shares"].tolist() == [60.0, 80.0, 90.0]

    idx = _index(["2015-01-09", "2015-01-12", "2015-03-19", "2015-03-20", "2015-05-04"], ["SH600000"])
    out = float_shares_on(idx, sh)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1:].tolist() == [60.0, 60.0, 80.0, 90.0]


def test_empty_share_response():
    assert tidy_share_changes(pd.DataFrame(), "600000").empty


def test_tidy_drops_bad_dates_and_pads_codes():
    raw = pd.DataFrame({"symbol": [1, "600000"], "start_date": ["2015-01-01", "not a date"],
                        "industry_code": [110101, "480301"], "update_time": [None, None]})
    ind = tidy_industry(raw)
    assert ind["symbol"].tolist() == ["000001"]
    assert ind["industry_code"].tolist() == ["110101"]
