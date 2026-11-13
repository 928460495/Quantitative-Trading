"""实验室：一张卡从头跑到尾（scripts/lab_run.py）。小假数据，不碰真实数据。"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from qt.lab import cards as cd
from qt.lab import procedure as pc

from lab_helpers import dev3d
from test_lab_cards import GOOD

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lab_run", ROOT / "scripts" / "lab_run.py")
lr = importlib.util.module_from_spec(spec)
sys.modules["lab_run"] = lr
spec.loader.exec_module(lr)

DROP_CHEAP = {**GOOD, "id": "H001", "kind": "收益型", "family": "因子族", "change": "去掉便宜组",
              "prediction": {"metric": "配对.全段", "direction": "降"},
              "variant": {"families": {k: list(v) for k, v in pc.PLAN3D.families if k != "便宜"}}}


def _lab(tmp_path, card=DROP_CHEAP):
    lab = tmp_path / "lab"
    (lab / "cards").mkdir(parents=True)
    (lab / "baseline.yaml").write_text(yaml.safe_dump({"procedure": pc.PLAN3D.to_dict()}, allow_unicode=True), encoding="utf-8")
    path = lab / "cards" / f"{card['id']}.yaml"
    path.write_text(yaml.safe_dump(card, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return lab, path


def test_run_card_writes_report_runs_and_trials(tmp_path):
    lab, path = _lab(tmp_path)
    dev, _ = dev3d()
    trials = tmp_path / "trials.jsonl"
    assert lr.run_card(dev, path, lab, trials, check_git=False) == 0
    runs = cd.load_runs(lab / "runs.jsonl")
    assert len(runs) == 1 and runs[0]["card"] == "H001" and runs[0]["run"] == 1
    r = runs[0]
    assert r["adopt"] is False                                                 # 去掉带信号的组：收益型规则不会通过
    assert r["diff"]["全段"] < 0 and r["t"] < -2 and "配对差值的 t 值" in r["failed"]
    assert r["candidate"]["families"] == DROP_CHEAP["variant"]["families"] and r["framework"] == "2.0"
    assert r["baseline"] == lr.proc_hash(pc.PLAN3D)
    text = (lab / "reports" / "H001-r1.md").read_text()
    for word in ("不采纳", "规则逐条", "配对比较", "成绩单：候选", "成绩单：基线", "最顶端各档", "纯噪声", "不算干净的证据"):
        assert word in text, word
    logged = [json.loads(x) for x in trials.read_text().splitlines()]
    assert [x["kind"] for x in logged] == ["lab"] and logged[0]["params"]["card"] == "H001"
    assert lr.run_card(dev, path, lab, trials, check_git=False) == 0
    assert [x["run"] for x in cd.load_runs(lab / "runs.jsonl")] == [1, 2] and (lab / "reports" / "H001-r2.md").exists()


def test_inverse_card_is_adopted_on_fake_data(tmp_path):
    # 基线不含"便宜"组，候选把它加回来：假数据里只有这组带信号，应该采纳
    no_cheap = pc.PLAN3D.replace(families={k: list(v) for k, v in pc.PLAN3D.families if k != "便宜"})
    card = {**DROP_CHEAP, "change": "加回便宜组", "prediction": {"metric": "配对.全段", "direction": "升"},
            "variant": {"families": {k: list(v) for k, v in pc.PLAN3D.families}}}
    lab, path = _lab(tmp_path, card)
    (lab / "baseline.yaml").write_text(yaml.safe_dump({"procedure": no_cheap.to_dict()}, allow_unicode=True), encoding="utf-8")
    dev, _ = dev3d()
    out = lr.evaluate(dev, cd.load_card(path), cd.load_baseline(lab / "baseline.yaml"))
    assert out["adopt"], out["gates"]
    full = lambda key: out[key].loc["全段", "大亏只数"]
    assert out["gates"].loc["大亏只数没有明显变多", "数值"] == full("候选") - full("基线")   # 候选、基线没弄反
    assert out["候选"].loc["全段", "年化"] > out["基线"].loc["全段", "年化"]
    # 成本翻倍那一遍真的翻倍：两个流程都多付一倍成本，配对差值和标准成本下不一样
    assert out["pair2"]["差值"]["全段"] != pytest.approx(out["pair"]["差值"]["全段"], abs=1e-6)


def test_refuses_uncommitted_card_and_identical_variant(tmp_path):
    lab, path = _lab(tmp_path)
    dev, _ = dev3d()
    with pytest.raises(cd.CardError, match="H001.yaml 还没提交"):
        lr.run_card(dev, path, lab, None)                                     # tmp_path 不是 git 仓库
    same = {**DROP_CHEAP, "variant": {"n_hold": 15}}
    lab2, path2 = _lab(tmp_path / "b", same)
    with pytest.raises(cd.CardError, match="一模一样"):
        lr.run_card(dev, path2, lab2, None, check_git=False)
    assert not (lab2 / "runs.jsonl").exists()


def test_refuses_when_budget_is_used_up(tmp_path):
    lab, path = _lab(tmp_path)
    for i in range(6):
        cd.append_run(lab / "runs.jsonl", {"card": "H001", "run": i + 1})
    with pytest.raises(cd.CardError, match="最多 6 次"):
        lr.run_card(dev3d()[0], path, lab, None, check_git=False)


def test_committed_card_in_a_real_repo_runs(tmp_path):
    lab, path = _lab(tmp_path)
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    run("add", "-A"); run("commit", "-q", "-m", "x")
    assert lr.run_card(dev3d()[0], path, lab, None) == 0
    (lab / "baseline.yaml").write_text((lab / "baseline.yaml").read_text() + "\n")
    with pytest.raises(cd.CardError, match="baseline"):
        lr.run_card(dev3d()[0], path, lab, None)
