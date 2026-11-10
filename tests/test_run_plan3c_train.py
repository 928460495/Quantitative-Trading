"""计划 3c 任务 9：训练段脚本的流程和"停下讨论"的出口。小假数据。"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_plan3c_train", ROOT / "scripts" / "run_plan3c_train.py")
rt = importlib.util.module_from_spec(spec)
sys.modules["run_plan3c_train"] = rt
spec.loader.exec_module(rt)

from test_walkforward import POOL, REF, SMALL_GRID, fake_seg  # noqa: E402

KW = dict(grid=SMALL_GRID, years=(2012, 2013), reference=REF, train_end="2013-12-31")


def test_stop_at_consistency_check(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "MAX_GAP", -1.0)
    code = rt.run(fake_seg(), POOL, tmp_path / "out", tmp_path / "cache", tmp_path / "trials.jsonl", **KW)
    assert code != 0
    assert "停下讨论" in (tmp_path / "out" / "README.md").read_text()
    assert not (tmp_path / "out" / "final.json").exists()


def test_stop_at_pbo(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "MAX_PBO", -1.0)
    code = rt.run(fake_seg(), POOL, tmp_path / "out", tmp_path / "cache", None, **KW)
    assert code != 0 and not (tmp_path / "out" / "final.json").exists()
    assert "PBO" in (tmp_path / "out" / "README.md").read_text()


def test_happy_path_writes_final_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "MAX_PBO", 1.0)
    log = tmp_path / "trials.jsonl"
    code = rt.run(fake_seg(), POOL, tmp_path / "out", tmp_path / "cache", log, **KW)
    assert code == 0
    final = json.loads((tmp_path / "out" / "final.json").read_text())
    assert set(final) == {"factors", "params", "smooth", "lowvol_weight"} and final["factors"][0] == "sig"
    for name in ("walk_forward.csv", "grid.csv", "ic_decay.csv", "README.md"):
        assert (tmp_path / "out" / name).exists(), name
    kinds = {json.loads(line)["kind"] for line in log.read_text().splitlines()}
    assert kinds == {"plan3c_step", "plan3c_grid"}
    wf = pd.read_csv(tmp_path / "out" / "walk_forward.csv", index_col=0, parse_dates=True)
    assert wf.index.min().year == 2012
