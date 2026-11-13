"""最终测试的成绩单和判定（DECISIONS 2026-11-13「最终测试的程序」）。"""

import numpy as np
import pandas as pd
import pytest

from qt.lab import final
from qt.lab import procedure as pc
from qt.lab import scorecard as sc

from lab_helpers import dev3d

PROC = pc.PLAN3D.replace(exclude=["is_st"], no_buy=[], rebalance_every=10, n_hold=8, sell_mult=2)


@pytest.fixture(scope="module")
def runs():
    dev = dev3d(seed=2, n_stocks=60, start="2018-01-02", end="2019-12-31")[0]
    return dev, sc.run_phases(dev, PROC)


def test_scorecard_rows_are_full_period_and_years(runs):
    dev, res = runs
    card = final.final_scorecard(dev, res)
    assert list(card.index) == ["全段", "2018", "2019"]
    for col in ("夏普", "等权夏普", "沪深300夏普", "最差起点夏普", "年化", "能持有等权", "沪深300全收益", "最大回撤"):
        assert col in card.columns


def test_full_period_sharpe_is_average_over_starts(runs):
    dev, res = runs
    card = final.final_scorecard(dev, res)
    rets = sc.daily_returns(res)
    hold, hs = sc.bench_returns(dev, res)
    each = [sc.risk_metrics(rets[k], hold[k], hs) for k in rets]
    assert card.loc["全段", "夏普"] == pytest.approx(np.mean([m["夏普"] for m in each]))
    assert card.loc["全段", "最差起点夏普"] == pytest.approx(min(m["夏普"] for m in each))
    assert card.loc["全段", "沪深300夏普"] == pytest.approx(each[0]["沪深300夏普"])
    y = rets.index.year == 2019
    assert card.loc["2019", "夏普"] == pytest.approx(np.mean([sc.risk_metrics(rets[k][y], hold[k][y], hs[y])["夏普"] for k in rets]))


def _card(s, ew, hs):
    return pd.DataFrame({"夏普": [s], "等权夏普": [ew], "沪深300夏普": [hs]}, index=["全段"])


@pytest.mark.parametrize("s,ew,hs,ok", [(0.5, 0.3, 0.4, True), (0.5, 0.6, 0.4, False), (0.5, 0.3, 0.5, False),
                                        (0.5, 0.5, 0.1, False), (-0.1, -0.3, -0.2, True)])
def test_verdict_needs_sharpe_strictly_above_both(s, ew, hs, ok):
    passed, table = final.verdict(_card(s, ew, hs))
    assert passed is ok
    assert list(table.index) == ["夏普高于沪深300全收益", "夏普高于能持有等权"]


def test_dates_outside_window_rejected(runs):
    dev, res = runs
    with pytest.raises(ValueError, match="窗口"):
        final.check_window(dev["close"].index, "2018-01-01", "2019-06-30")
    final.check_window(dev["close"].index, "2018-01-01", "2019-12-31")
