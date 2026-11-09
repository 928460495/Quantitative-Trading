"""计划 4 任务 5 的运行脚本：假数据从头跑到尾（不碰真实数据）。"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_position_control", ROOT / "scripts" / "run_position_control.py")
rpc = importlib.util.module_from_spec(spec)
sys.modules["run_position_control"] = rpc
spec.loader.exec_module(rpc)

from test_position_control import _segment  # noqa: E402


def _segs():
    return {"train": _segment(n_days=300, seed=1), "valid": _segment(n_days=200, seed=2)}


def test_runs_every_variant_and_applies_the_selection_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(rpc.pc, "choose_a", lambda *a, **k: rpc.pc.a_name(120, 0.5))
    trials = tmp_path / "trials.jsonl"
    res = rpc.run_all(_segs(), trials_path=trials, data_version="test")
    m = res["metrics"]
    base = 1 + 6 + 1                                                    # 不控仓、A 的 6 个、B
    assert len(m) == 2 * base + 2 * 3                                   # 选中后每段 3 个只报告变体
    assert set(m["segment"]) == {"train", "valid"}
    assert m.loc[m["variant"] == rpc.NO_CONTROL, "c_triggers"].eq(0).all()   # 不控仓也不用 C
    assert len(trials.read_text().splitlines()) == len(m)
    assert {json.loads(l)["kind"] for l in trials.read_text().splitlines()} == {"position_control"}

    rpc.write_report(res, tmp_path / "out")
    readme = (tmp_path / "out" / "README.md").read_text()
    assert "选中：A_120日_跌破后50%" in readme and "成本翻倍" in readme
    assert json.loads((tmp_path / "out" / "chosen.json").read_text())["chosen"] == "A_120日_跌破后50%"
    assert (tmp_path / "out" / "nav_valid.png").exists()


def test_stops_when_no_combination_meets_drawdown_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(rpc.pc, "choose_a", lambda *a, **k: None)
    res = rpc.run_all(_segs(), trials_path=None)
    assert res["chosen"] is None
    assert not res["metrics"]["variant"].str.contains("｜").any()        # 不跑只报告的变体
    rpc.write_report(res, tmp_path / "out")
    assert "按事先写定的规则停下来讨论" in (tmp_path / "out" / "README.md").read_text()


def test_a_table_feeds_the_real_rule():
    res = rpc.run_all(_segs(), trials_path=None)
    limits = res["limits"]
    assert res["chosen"] == rpc.pc.choose_a(res["a_table"], train_limit=limits["train"], valid_limit=limits["valid"])
    assert limits["train"] == pytest.approx(-rpc.pc.bench_max_drawdown(_segs()["train"]))
    assert len(res["a_table"]) == 6
