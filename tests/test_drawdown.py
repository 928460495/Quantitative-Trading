"""计划 3c 任务 3：按回撤分档减仓的状态机（DECISIONS 2026-11-10「计划 3c 事先登记的规则」）。"""

import pytest

from qt.drawdown import DEFAULT_TIERS, DrawdownTiers


def test_default_tiers_match_registered_rule():
    assert DEFAULT_TIERS == ((0.10, 2 / 3), (0.15, 1 / 3), (0.20, 0.0))


def test_tiers_deepen_immediately_and_recover_with_buffer():
    t = DrawdownTiers()
    assert t.update(100) == 1.0
    assert t.update(89) == pytest.approx(2 / 3)      # 回撤 11%
    assert t.update(84) == pytest.approx(1 / 3)      # 16%
    assert t.update(79) == 0.0                       # 21%
    assert t.update(81.5) == 0.0                     # 18.5%：没低于 20% − 2%，仍空仓
    assert t.update(82.1) == pytest.approx(1 / 3)    # 17.9% < 18%：回到 1/3 档
    assert t.update(92.1) == 1.0                     # 7.9% < 8%：直接回到满仓（可一次跨多档）
    t2 = DrawdownTiers()
    t2.update(100)
    assert t2.update(90) == pytest.approx(2 / 3)     # 正好 10%：算进 10–15% 档


def test_tiers_stay_when_inside_buffer():
    t = DrawdownTiers()
    t.update(100)
    assert t.update(88) == pytest.approx(2 / 3)      # 12%
    assert t.update(91) == pytest.approx(2 / 3)      # 9%：没低于 8%，不加仓


def test_tiers_peak_window_forgets_old_high():
    t = DrawdownTiers(window=3)
    for v in (100, 90, 90, 90):
        f = t.update(v)
    assert f == 1.0                                  # 100 已出窗口，最高点 90；从 2/3 档回到满仓


def test_tiers_short_history_uses_available_days():
    assert DrawdownTiers().update(50) == 1.0


def test_slots_floor():
    assert DrawdownTiers.slots(20, 2 / 3) == 13
    assert DrawdownTiers.slots(15, 1 / 3) == 5
    assert DrawdownTiers.slots(20, 1.0) == 20
    assert DrawdownTiers.slots(20, 0.0) == 0
