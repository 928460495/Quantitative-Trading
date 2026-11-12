"""研究框架 2.0 的冻结常数和噪声夏普（docs/FRAMEWORK.md）。"""

import pytest

from qt.lab import protocol as pr


def test_frozen_values_match_framework_document():
    assert pr.FRAMEWORK_VERSION == "2.0"
    assert pr.DEV == ("2010-01-01", "2021-12-31")
    assert pr.ERAS == {"一": ("2010-01-01", "2014-12-31"), "二": ("2015-01-01", "2018-12-31"),
                       "三": ("2019-01-01", "2021-12-31")}
    assert pr.LATEST_ERA == "三" and pr.FULL == "全段"
    assert pr.OFFSETS == (0, 5, 10, 15) and pr.BENCH_EVERY == 20 and pr.NW_LAG == 20
    assert pr.BIG_LOSS == -0.40
    assert (pr.MAX_RUNS_PER_CARD, pr.MAX_CARDS, pr.MAX_CARDS_PER_FAMILY, pr.COOLDOWN_AFTER) == (6, 12, 3, 2)
    assert (pr.T_ADOPT, pr.T_NOT_WORSE, pr.BIG_LOSS_CUT) == (2.0, -1.0, 0.30)


def test_eras_tile_the_development_segment_without_gaps_or_overlap():
    import pandas as pd
    spans = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in pr.ERAS.values()]
    assert spans[0][0] == pd.Timestamp(pr.DEV[0]) and spans[-1][1] == pd.Timestamp(pr.DEV[1])
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start == end + pd.Timedelta(days=1)
    assert pd.Timestamp(pr.DEV[1]) < pd.Timestamp("2022-01-01")              # 不碰封存段


def test_noise_sharpe_matches_hand_calculation():
    # 12 年、独立试验：综述里重算过的三个数
    assert pr.noise_sharpe(20, 12) == pytest.approx(0.549, abs=0.005)
    assert pr.noise_sharpe(100, 12) == pytest.approx(0.73, abs=0.01)
    assert pr.noise_sharpe(3722, 12) == pytest.approx(1.04, abs=0.01)
    assert pr.noise_sharpe(100, 48) == pytest.approx(pr.noise_sharpe(100, 12) / 2)   # 按 1/√年数 缩小
    assert pr.noise_sharpe(1000, 12) > pr.noise_sharpe(100, 12)
    assert pr.noise_sharpe(1, 12) == 0.0
    with pytest.raises(ValueError):
        pr.noise_sharpe(0, 12)


def test_era_of_dates():
    import pandas as pd
    idx = pd.to_datetime(["2010-01-04", "2014-12-31", "2015-01-05", "2018-12-28", "2019-01-02", "2021-12-31"])
    assert list(pr.era_of(idx)) == ["一", "一", "二", "二", "三", "三"]
    with pytest.raises(ValueError):
        pr.era_of(pd.to_datetime(["2022-01-04"]))
