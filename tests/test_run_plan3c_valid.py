"""计划 3c 任务 10：验证段只跑一次。小假数据。"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_plan3c_valid", ROOT / "scripts" / "run_plan3c_valid.py")
rv = importlib.util.module_from_spec(spec)
sys.modules["run_plan3c_valid"] = rv
spec.loader.exec_module(rv)

from test_walkforward import POOL, fake_seg  # noqa: E402

FINAL = {"factors": ["sig"], "params": {"n_hold": 5, "rebalance_every": 5, "sell_mult": 3, "industry_cap": 1.0,
                                        "tiers": True, "min_trade": 0.0, "cost_multiplier": 1.0},
         "smooth": 5, "lowvol_weight": 1.0}


def _valid():
    seg = fake_seg(start="2019-01-02", end="2021-12-31")
    base = pd.DataFrame(np.random.default_rng(3).normal(size=seg["close"].shape), index=seg["close"].index,
                        columns=seg["close"].columns)
    return seg, base


def _trials(tmp_path):
    path = tmp_path / "trials.jsonl"
    path.write_text("\n".join(json.dumps({"kind": "plan3c_grid", "config_hash": f"h{i}", "params": {},
                                          "metrics": {"sharpe": s}}) for i, s in enumerate([0.5, 0.8, 1.1])) + "\n")
    return path


def test_runs_once_and_reports_five_criteria(tmp_path):
    seg, base = _valid()
    trials = _trials(tmp_path)
    code = rv.run(seg, POOL, FINAL, base, tmp_path / "out", trials)
    assert code == 0
    text = (tmp_path / "out" / "README.md").read_text()
    for mark in ("①", "②", "③", "④", "⑤", "DSR", "验证段在计划 4 已看过两次", "能持有等权", "平均持仓比例", "空仓天数"):
        assert mark in text, mark
    assert (tmp_path / "out" / "backtest.csv").exists()
    kinds = [json.loads(line)["kind"] for line in trials.read_text().splitlines()]
    assert kinds.count("plan3c_valid") >= 1


def test_refuses_second_run(tmp_path):
    seg, base = _valid()
    trials = _trials(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "README.md").write_text("已经跑过")
    before = trials.read_text()
    code = rv.run(seg, POOL, FINAL, base, out, trials)
    assert code != 0
    assert (out / "README.md").read_text() == "已经跑过" and sorted(p.name for p in out.iterdir()) == ["README.md"]
    assert trials.read_text() == before


def test_missing_final_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rv.load_final(tmp_path / "final.json")


def test_refuses_after_partial_run(tmp_path):
    # 上次跑到一半崩了（只写了 backtest.csv、没写 README）：也不能再跑
    seg, base = _valid()
    trials = _trials(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "backtest.csv").write_text("x")
    assert rv.run(seg, POOL, FINAL, base, out, trials) != 0
    assert sorted(p.name for p in out.iterdir()) == ["backtest.csv"]
