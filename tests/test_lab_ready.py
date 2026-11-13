"""实验室：能否进最终测试的检查（scripts/lab_ready.py；FRAMEWORK 第 8 节第 2、3 条，DECISIONS 2026-11-13）。小假数据。"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from qt import plan3d as p3
from qt.lab import procedure as pc

from lab_helpers import dev3d

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lab_ready", ROOT / "scripts" / "lab_ready.py")
lr = importlib.util.module_from_spec(spec)
sys.modules["lab_ready"] = lr
spec.loader.exec_module(lr)


def test_neighbours_are_the_registered_six():
    assert lr.NEIGHBOURS == {"持股 10 只": {"n_hold": 10}, "持股 20 只": {"n_hold": 20},
                             "每 10 天调仓": {"rebalance_every": 10}, "每 40 天调仓": {"rebalance_every": 40},
                             "跌出前 2N 名才卖": {"sell_mult": 2}, "跌出前 4N 名才卖": {"sell_mult": 4}}


def test_strong_signal_passes_every_check(tmp_path):
    dev, _ = dev3d(n_stocks=150)                                              # 股票够多，持 20 只仍是精选
    log = tmp_path / "trials.jsonl"
    checks, versions = lr.readiness(dev, pc.PLAN3D, log=log)
    assert list(checks.index) == ["2a 三个阶段的夏普都高于两个基准", "2b 四个起点各自的夏普高于能持有等权",
                                  "2c 成本翻倍后夏普高于两个基准", "3 邻近参数的夏普都高于能持有等权"]
    assert checks["满足"].all(), checks
    assert set(versions) == {"基线", "成本翻倍", *lr.NEIGHBOURS}
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert {r["kind"] for r in rows} == {"lab_ready"} and len(rows) == 2 + len(lr.NEIGHBOURS)


def _bad_dev():
    dev, _ = dev3d()
    for f in p3.GROUPS["便宜"]:
        dev["factors"][f] = -dev["factors"][f]                                # 选出来的是之后跌得最多的
    return dev


def test_reversed_signal_fails_and_names_the_failed_checks():
    checks, _ = lr.readiness(_bad_dev(), pc.PLAN3D)
    assert not checks["满足"].any(), checks


def test_each_check_uses_the_right_numbers(monkeypatch):
    dev, _ = dev3d()
    checks, versions = lr.readiness(dev, pc.PLAN3D)
    base = versions["基线"]
    eras = [e for e in base.index if e != "全段"]
    gap = lambda card, row: min(card.loc[row, "夏普"] - card.loc[row, "等权夏普"], card.loc[row, "夏普"] - card.loc[row, "沪深300夏普"])
    assert checks.loc["2a 三个阶段的夏普都高于两个基准", "最小余量"] == pytest.approx(min(gap(base, e) for e in eras))
    assert checks.loc["2b 四个起点各自的夏普高于能持有等权", "最小余量"] == pytest.approx(base.loc["全段", "最差起点夏普差"])
    c2 = versions["成本翻倍"]
    assert checks.loc["2c 成本翻倍后夏普高于两个基准", "最小余量"] == pytest.approx(gap(c2, "全段"))
    assert c2.loc["全段", "年成本"] > base.loc["全段", "年成本"] * 1.8
    assert checks.loc["3 邻近参数的夏普都高于能持有等权", "最小余量"] == pytest.approx(
        min(versions[n].loc["全段", "夏普"] - versions[n].loc["全段", "等权夏普"] for n in lr.NEIGHBOURS))


def test_report_lists_checks_and_versions(tmp_path):
    dev, _ = dev3d()
    checks, versions = lr.readiness(dev, pc.PLAN3D)
    text = lr.report(checks, versions, n_trials=10)
    for word in ("能否进最终测试", "全部满足", "邻近参数", "持股 20 只", "纯噪声"):
        assert word in text, word


def test_hs300_counts_in_checks_2a_and_2c():
    import numpy as np
    dev, _ = dev3d(n_stocks=150)
    n = len(dev["index"])
    dev["index"] = dev["index"].copy()
    dev["index"]["H00300"] = 100 * np.cumprod(1 + np.random.default_rng(2).normal(0.004, 0.001, n))   # 沪深300 稳涨、波动很小：夏普极高
    checks, _ = lr.readiness(dev, pc.PLAN3D)
    assert checks["满足"].to_dict() == {"2a 三个阶段的夏普都高于两个基准": False, "2b 四个起点各自的夏普高于能持有等权": True,
                                        "2c 成本翻倍后夏普高于两个基准": False, "3 邻近参数的夏普都高于能持有等权": True}
