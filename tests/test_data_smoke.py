"""需要本地数据包的冒烟测试：没有数据时跳过。"""

import pytest

from qt.config import DATA_DIR

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not (DATA_DIR / "calendars" / "day.txt").exists(), reason="没有本地 Qlib 数据"),
]


def test_csi300_membership_is_point_in_time():
    from qt.data import load_features

    df = load_features({"close": "$close"}, "csi300", "2015-01-05", "2015-01-30")
    per_day = df.groupby(level="datetime").size()
    assert per_day.between(295, 300).all()


def test_close_returns_are_adjusted_and_change_is_raw():
    from qt.data import load_features

    df = load_features(
        {
            "r_close": "$close/Ref($close,1)-1",
            "r_adj": "$adjclose/Ref($adjclose,1)-1",
            "r_raw": "($close/$factor)/(Ref($close,1)/Ref($factor,1))-1",
            "change": "$change",
        },
        "csi300",
        "2019-01-02",
        "2019-12-31",
    ).dropna()
    assert (df["r_close"] - df["r_adj"]).abs().max() < 1e-5
    assert (df["r_raw"] - df["change"]).abs().max() < 1e-3
