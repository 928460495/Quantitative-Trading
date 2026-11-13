import numpy as np
import pandas as pd
import pytest

from qt.config import TestSegmentLockedError, check_range, segment
from qt.data import drop_embargo


@pytest.fixture(autouse=True)
def _lock_test(monkeypatch):
    monkeypatch.delenv("QT_UNLOCK_TEST", raising=False)


def test_train_and_valid_segments_are_open():
    assert segment("train") == ("2010-01-01", "2018-12-31")
    assert segment("valid") == ("2019-01-01", "2021-12-31")


def test_test_segment_locked_by_default():
    with pytest.raises(TestSegmentLockedError):
        segment("test")


def test_test_segment_unlocks_with_env(monkeypatch):
    monkeypatch.setenv("QT_UNLOCK_TEST", "1")
    assert segment("test")[0] == "2022-01-01"


@pytest.mark.parametrize(
    "start,end",
    [("2020-01-01", "2022-01-01"), ("2022-03-01", "2022-06-01"), ("2015-01-01", None)],
)
def test_check_range_rejects_test_segment(start, end):
    with pytest.raises(TestSegmentLockedError):
        check_range(start, end)


def test_check_range_allows_valid_end():
    check_range("2010-01-01", "2021-12-31")


def test_drop_embargo_removes_last_n_dates():
    dates = pd.bdate_range("2020-01-01", periods=10)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
    df = pd.DataFrame({"x": np.arange(20)}, index=idx)
    out = drop_embargo(df, 3)
    kept = out.index.get_level_values("datetime").unique()
    assert len(kept) == 7
    assert kept.max() == dates[6]


def test_drop_embargo_rejects_short_sample():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=2), ["A"]], names=["datetime", "instrument"])
    with pytest.raises(ValueError):
        drop_embargo(pd.DataFrame({"x": [1, 2]}, index=idx), 3)


def test_segment_range_gives_concrete_end_for_test(monkeypatch):
    from qt.config import FINAL_TEST_END, segment_range
    assert segment_range("valid") == ("2019-01-01", "2021-12-31")
    with pytest.raises(TestSegmentLockedError):
        segment_range("test")
    monkeypatch.setenv("QT_UNLOCK_TEST", "1")
    assert segment_range("test") == ("2022-01-01", FINAL_TEST_END) and FINAL_TEST_END == "2026-10-23"


def test_parse_segments(monkeypatch):
    from qt.config import parse_segments
    assert parse_segments("both") == ["train", "valid"] and parse_segments("valid") == ["valid"]
    with pytest.raises(TestSegmentLockedError):
        parse_segments("test")
    with pytest.raises(KeyError):
        parse_segments("all")
    monkeypatch.setenv("QT_UNLOCK_TEST", "1")
    assert parse_segments("test") == ["test"]


def test_spent_test_segment_allows_reading_after_2022(monkeypatch):
    import qt.config as cfg
    monkeypatch.setattr(cfg, "TEST_SEGMENT_SPENT", True)
    cfg.check_range("2022-01-01", "2026-11-01")
    assert cfg.segment("live")[0] == "2026-10-26"
    assert cfg.parse_segments("live") == ["live"]


def test_live_segment_locked_until_spent():
    from qt.config import segment
    with pytest.raises(TestSegmentLockedError):
        segment("live")


def test_latest_data_date_is_earlier_of_two_sources(tmp_path):
    from qt.config import latest_data_date
    (tmp_path / "q" / "calendars").mkdir(parents=True)
    (tmp_path / "q" / "calendars" / "day.txt").write_text("2026-11-10\n2026-11-11\n2026-11-12\n")
    (tmp_path / "f").mkdir()
    pd.DataFrame({"instrument": ["SH600000"] * 2, "date": pd.to_datetime(["2026-11-10", "2026-11-11"]),
                  "close": [1.0, 2.0]}).to_parquet(tmp_path / "f" / "daily.parquet")
    assert latest_data_date(tmp_path / "q", tmp_path / "f") == "2026-11-11"
