import numpy as np
import pandas as pd
import pytest

from qt.panel import industry_levels, main_board_instruments


def test_industry_levels_from_six_digit_code():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-02", periods=2), ["SH600000", "SZ000001"]],
                                     names=["datetime", "instrument"])
    table = pd.DataFrame({"symbol": ["600000", "000001"], "start_date": pd.to_datetime(["2014-02-21", "2014-02-21"]),
                          "industry_code": ["480301", "480101"]})
    lv = industry_levels(idx, table)
    assert lv["industry_l1"].iloc[0] == "48" and lv["industry_l2"].iloc[0] == "4803" and lv["industry_l3"].iloc[0] == "480301"
    assert lv["industry_l2"].iloc[1] == "4801"


def test_main_board_instruments(tmp_path):
    (tmp_path / "instruments").mkdir()
    (tmp_path / "instruments" / "all.txt").write_text(
        "SH600000\t2000-01-04\t2026-10-23\nSZ300750\t2018-06-11\t2026-10-23\nSZ000003\t2000-01-04\t2002-06-14\n"
        "SH688001\t2019-07-22\t2026-10-23\nSZ001979\t2015-12-10\t2026-10-23\n")
    assert main_board_instruments("2016-01-01", "2016-12-31", tmp_path) == ["SH600000", "SZ001979"]


def test_load_fundamentals_refuses_test_segment(tmp_path, monkeypatch):
    from qt import panel as pn
    from qt.config import TestSegmentLockedError

    monkeypatch.delenv("QT_UNLOCK_TEST", raising=False)
    called = []
    monkeypatch.setattr(pn, "_fundamental_wide", lambda dates, inst, d: called.append((dates, inst, d)) or {"x": 1})
    ok = pd.bdate_range("2021-12-01", "2021-12-31")
    assert pn.load_fundamentals(ok, ["SH600000"], tmp_path) == {"x": 1}
    assert called[0][1] == ["SH600000"] and called[0][2] == tmp_path and called[0][0].equals(ok)
    with pytest.raises(TestSegmentLockedError):
        pn.load_fundamentals(pd.bdate_range("2021-12-01", "2022-01-04"), ["SH600000"], tmp_path)
    assert len(called) == 1                                # 碰到测试段：读数据之前就报错

# ---------- 需要数据包和全 A 补充数据 ----------


@pytest.fixture(scope="module")
def panel():
    from qt.panel import load_panel

    return load_panel("2019-03-01", "2019-06-28")


@pytest.mark.data
def test_panel_shapes_and_main_board(panel):
    close = panel["close"]
    for key in ["open", "high", "low", "vwap", "volume", "amount", "returns", "raw_close", "turn", "float_mcap",
                "industry_l1", "industry_l2", "industry_l3", "benchmark_close", "pe_ttm", "net_profit_ttm"]:
        assert panel[key].shape == close.shape, key
    assert all(c.startswith(("SH60", "SZ00")) for c in close.columns)
    assert (panel["benchmark_close"].nunique(axis=1) == 1).all()       # 每天各列相同


@pytest.mark.data
def test_fundamentals_not_visible_on_notice_day(panel):
    from qt.full_a import FULL_A_DIR
    from qt.fundamentals import tidy_reports

    raw = {n: pd.read_parquet(FULL_A_DIR / "reports" / f"{n}.parquet") for n in ["yjbb"]}
    rec = tidy_reports(raw)
    days = panel["close"].index
    # 2019 年一季报：找一只在样本期内公告、公告日后还有交易日的主板股票
    q1 = rec[(rec["report_date"] == "2019-03-31") & rec["instrument"].isin(panel["close"].columns)
             & (rec["notice_date"] > days[0]) & (rec["notice_date"] < days[-5])]
    row = q1.iloc[0]
    notice = row["notice_date"]
    on_or_before = days[days <= notice][-1]
    after = days[days > notice][0]
    rep = panel["report_date"][row["instrument"]]
    assert rep[on_or_before] < pd.Timestamp("2019-03-31")
    assert rep[after] == pd.Timestamp("2019-03-31")
