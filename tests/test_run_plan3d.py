"""计划 3d 任务 4、5：训练段四条检查和 2019–2021 中途检查的流程（DECISIONS 2026-11-12）。小假数据。"""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from qt import plan3d as p3

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_plan3d", ROOT / "scripts" / "run_plan3d.py")
rp = importlib.util.module_from_spec(spec)
sys.modules["run_plan3d"] = rp
spec.loader.exec_module(rp)

from test_plan3d import seg3d  # noqa: E402


def bad_seg():
    """"便宜"组的信号反过来：选出来的是之后跌得最多的。"""
    seg = seg3d()
    for f in p3.GROUPS["便宜"]:
        seg["factors"][f] = -seg["factors"][f]
    return seg


def _kinds(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_train_happy_path_writes_final_and_logs_every_variant(tmp_path, monkeypatch):
    # 假数据里只有"便宜"组有信号，去掉它不可能过第 4 条；这里只留前三条，第 4 条另测
    checks = {k: v for k, v in p3.CHECKS.items() if not k.startswith("4")}
    log = tmp_path / "trials.jsonl"
    code = rp.run_train(seg3d(), tmp_path / "out", log, checks=checks)
    assert code == 0
    final = json.loads((tmp_path / "out" / "final.json").read_text())
    assert final["spec"] == json.loads(json.dumps(dataclasses.asdict(p3.MAIN)))
    assert final["groups"] == {g: list(fs) for g, fs in p3.GROUPS.items()}
    table = pd.read_csv(tmp_path / "out" / "variants.csv", index_col=0)
    assert list(table.index) == list(p3.VARIANTS)
    nav = pd.read_csv(tmp_path / "out" / "backtest.csv", index_col=0, parse_dates=True)["nav"]
    assert nav.iloc[0] == 100_000 and len(nav) == len(seg3d()["close"])
    readme = (tmp_path / "out" / "README.md").read_text()
    for word in ("检查", "各版本", "错开起点", "能持有等权", "最大回撤", "所有检查都满足", "只报告"):
        assert word in readme, word
    assert "停下讨论" not in readme
    rows = _kinds(log)
    assert [r["kind"] for r in rows] == ["plan3d"] * len(p3.VARIANTS)
    assert [r["params"]["variant"] for r in rows] == list(p3.VARIANTS)
    assert {r["params"]["segment"] for r in rows} == {"train"}


def test_train_stops_when_a_check_fails_and_removes_old_final(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "final.json").write_text("{}")                       # 上一次通过留下的
    code = rp.run_train(bad_seg(), out, None)
    assert code == 1 and not (out / "final.json").exists()
    readme = (out / "README.md").read_text()
    assert "停下讨论" in readme and "1 主方案高于两个基准" in readme and "所有检查都满足" not in readme
    assert (out / "variants.csv").exists()                      # 结果照样留下，方便讨论


def test_train_uses_all_four_checks_by_default(tmp_path, monkeypatch):
    seg = seg3d()
    _, results = p3.evaluate(seg, {"主方案": p3.MAIN})

    def fake(excess):
        rows = {n: {"年化": 0.10 + excess.get(n, 0.03), "能持有等权": 0.10, "沪深300全收益": 0.0,
                    "超额": excess.get(n, 0.03), "最大回撤": -0.3, "日均换手": 0.01, "年成本": 0.01, "交易笔数": 1.0}
                for n in p3.VARIANTS}
        return lambda *a, **k: (pd.DataFrame(rows).T, results)

    monkeypatch.setattr(p3, "evaluate", fake({}))
    assert rp.run_train(seg, tmp_path / "a", None) == 0
    for name in ("错开 15 天", "成本翻倍", "去掉冷门"):                 # 第 2、3、4 条各坏一个成员
        monkeypatch.setattr(p3, "evaluate", fake({name: -0.01}))
        out = tmp_path / name
        assert rp.run_train(seg, out, None) == 1, name
        assert not (out / "final.json").exists() and "停下讨论" in (out / "README.md").read_text()
    monkeypatch.setattr(p3, "evaluate", fake({"持股 10 只": -0.5, "加回止损": -0.5, "不剔除高波动": -0.5}))
    assert rp.run_train(seg, tmp_path / "b", None) == 0               # 只报告的版本再差也不影响


def _final(tmp_path, **changes):
    path = tmp_path / "final.json"
    path.write_text(json.dumps({"spec": dataclasses.asdict(dataclasses.replace(p3.MAIN, **changes))}))
    return rp.load_final(path)


def test_load_final_requires_file_and_restores_tuple(tmp_path):
    with pytest.raises(FileNotFoundError, match="训练段"):
        rp.load_final(tmp_path / "nope.json")
    assert _final(tmp_path) == p3.MAIN
    assert _final(tmp_path, n_hold=10).n_hold == 10


def test_valid_runs_once_with_offsets_and_reports(tmp_path):
    log = tmp_path / "trials.jsonl"
    out = tmp_path / "valid"
    code = rp.run_valid(seg3d(), _final(tmp_path, n_hold=10), out, log)
    assert code == 0
    table = pd.read_csv(out / "variants.csv", index_col=0)
    assert list(table.index) == ["主方案", "错开 5 天", "错开 10 天", "错开 15 天", "成本翻倍"]
    rows = _kinds(log)
    assert {r["params"]["segment"] for r in rows} == {"valid"} and {r["params"]["n_hold"] for r in rows} == {10}
    assert [r["params"]["offset"] for r in rows] == [0, 5, 10, 15, 0]
    readme = (out / "README.md").read_text()
    assert "通过不算证据" in readme and "看过三次" in readme and "没有明显坏掉" in readme
    # 第二次：拒绝运行，什么都不改
    before = readme
    assert rp.run_valid(seg3d(), _final(tmp_path), out, log) == 1
    assert (out / "README.md").read_text() == before and len(_kinds(log)) == 5


def test_valid_refuses_after_a_crash_left_only_the_marker(tmp_path):
    out = tmp_path / "valid"
    out.mkdir()
    (out / "STARTED").write_text("x")
    assert rp.run_valid(seg3d(), _final(tmp_path), out, None) == 1
    assert sorted(p.name for p in out.iterdir()) == ["STARTED"]


def test_valid_leaves_marker_when_it_crashes_midway(tmp_path, monkeypatch):
    out = tmp_path / "valid"

    def boom(*a, **k):
        raise RuntimeError("跑到一半出错")

    monkeypatch.setattr(p3, "evaluate", boom)
    with pytest.raises(RuntimeError):
        rp.run_valid(seg3d(), _final(tmp_path), out, None)
    assert (out / "STARTED").exists()
    monkeypatch.undo()
    assert rp.run_valid(seg3d(), _final(tmp_path), out, None) == 1          # 崩过一次也不能重跑
    assert not (out / "README.md").exists()


def test_valid_flags_obviously_broken_plan(tmp_path):
    out = tmp_path / "valid"
    code = rp.run_valid(bad_seg(), _final(tmp_path), out, None)
    assert code == 1
    readme = (out / "README.md").read_text()
    assert "明显坏掉" in readme and "停下讨论" in readme and "没有明显坏掉" not in readme


def test_valid_threshold_is_mean_excess_of_four_offsets_minus_5_points():
    t = lambda *e: pd.DataFrame({"超额": list(e) + [-0.9]}, index=list(p3.OFFSETS) + ["成本翻倍"])
    assert rp.broken(t(-0.049, -0.049, -0.049, -0.049)) is False
    assert rp.broken(t(-0.051, -0.051, -0.051, -0.051)) is True
    assert rp.broken(t(-0.20, 0.0, 0.0, 0.0)) is False            # 看平均：−5 个点整不算坏掉
    assert rp.broken(t(-0.21, 0.0, 0.0, 0.0)) is True
    assert rp.MAX_SHORTFALL == 0.05
