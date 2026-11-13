"""scripts/lab_final.py：守门（解锁、只跑一次、代码冻结、登记已提交）和运行（假数据）。"""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from qt.lab import procedure as pc

from lab_helpers import dev3d

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lab_final", ROOT / "scripts" / "lab_final.py")
lf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lf)

PROC = pc.PLAN3D.replace(exclude=["is_st"], no_buy=[], rebalance_every=10, n_hold=8, sell_mult=2)


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "qt").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "lab" / "reports").mkdir(parents=True)
    (tmp_path / "qt" / "x.py").write_text("A = 1\n")
    (tmp_path / "scripts" / "y.py").write_text("B = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "code")
    frozen = _git(tmp_path, "rev-parse", "HEAD")
    reg = {"procedure": PROC.to_dict(), "code_commit": frozen, "window": {"start": "2018-01-01", "end": "2019-12-31"},
           "approval": {"date": "2026-11-13", "quote": "同意"}}
    (tmp_path / "lab" / "FINAL.yaml").write_text(yaml.safe_dump(reg, allow_unicode=True))
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "register")
    return tmp_path


def test_ready_when_everything_in_place(repo, monkeypatch):
    monkeypatch.setenv("QT_UNLOCK_TEST", "1")
    reg = lf.check_ready(repo)
    assert reg["code_commit"] and pc.Procedure.from_dict(reg["procedure"]) == PROC


def test_refuses_when_locked(repo, monkeypatch):
    monkeypatch.delenv("QT_UNLOCK_TEST", raising=False)
    with pytest.raises(lf.FinalError, match="解锁"):
        lf.check_ready(repo)


def test_refuses_second_run(repo, monkeypatch):
    monkeypatch.setenv("QT_UNLOCK_TEST", "1")
    (repo / "lab" / "reports" / "FINAL.md").write_text("done")
    with pytest.raises(lf.FinalError, match="只跑一次"):
        lf.check_ready(repo)


def test_refuses_code_changed_after_freeze(repo, monkeypatch):
    monkeypatch.setenv("QT_UNLOCK_TEST", "1")
    (repo / "qt" / "x.py").write_text("A = 2\n")
    with pytest.raises(lf.FinalError, match="没提交"):
        lf.check_ready(repo)
    _git(repo, "commit", "-qam", "change")
    with pytest.raises(lf.FinalError, match="冻结"):
        lf.check_ready(repo)


def test_refuses_uncommitted_registration(repo, monkeypatch):
    monkeypatch.setenv("QT_UNLOCK_TEST", "1")
    reg = yaml.safe_load((repo / "lab" / "FINAL.yaml").read_text())
    reg["approval"]["quote"] = "改过"
    (repo / "lab" / "FINAL.yaml").write_text(yaml.safe_dump(reg, allow_unicode=True))
    with pytest.raises(lf.FinalError, match="FINAL.yaml"):
        lf.check_ready(repo)


def test_registration_must_name_approval(repo, monkeypatch):
    monkeypatch.setenv("QT_UNLOCK_TEST", "1")
    reg = yaml.safe_load((repo / "lab" / "FINAL.yaml").read_text())
    del reg["approval"]
    (repo / "lab" / "FINAL.yaml").write_text(yaml.safe_dump(reg, allow_unicode=True))
    _git(repo, "commit", "-qam", "drop approval")
    with pytest.raises(lf.FinalError, match="approval"):
        lf.check_ready(repo)


def test_run_final_on_fake_data_writes_report_and_trial(tmp_path):
    dev = dev3d(seed=3, n_stocks=60, start="2018-01-02", end="2019-12-31")[0]
    reg = {"procedure": PROC.to_dict(), "code_commit": "abc", "window": {"start": "2018-01-01", "end": "2019-12-31"},
           "approval": {"date": "2026-11-13", "quote": "同意"}}
    trials = tmp_path / "trials.jsonl"
    out = tmp_path / "FINAL.md"
    passed = lf.run_final(dev, reg, out, trials)
    text = out.read_text()
    assert ("通过" if passed else "不通过") in text.splitlines()[0]
    assert "夏普高于沪深300全收益" in text and "成本翻倍" in text and "2019" in text
    rec = json.loads(trials.read_text().splitlines()[-1])
    assert rec["kind"] == "final" and rec["metrics"]["passed"] == passed


def test_run_final_rejects_data_outside_window(tmp_path):
    dev = dev3d(seed=3, n_stocks=60, start="2018-01-02", end="2019-12-31")[0]
    reg = {"procedure": PROC.to_dict(), "code_commit": "abc", "window": {"start": "2018-01-01", "end": "2019-06-30"},
           "approval": {"date": "2026-11-13", "quote": "同意"}}
    with pytest.raises(ValueError, match="窗口"):
        lf.run_final(dev, reg, tmp_path / "FINAL.md", tmp_path / "t.jsonl")
    assert not (tmp_path / "FINAL.md").exists()


def test_failed_verdict_is_reported_and_logged(tmp_path, monkeypatch):
    import pandas as pd
    dev = dev3d(seed=3, n_stocks=60, start="2018-01-02", end="2019-12-31")[0]
    reg = {"procedure": PROC.to_dict(), "code_commit": "abc", "window": {"start": "2018-01-01", "end": "2019-12-31"},
           "approval": {"date": "2026-11-13", "quote": "同意"}}
    monkeypatch.setattr(lf.final, "verdict", lambda card: (False, pd.DataFrame({"满足": [False]}, index=["夏普高于沪深300全收益"])))
    assert lf.run_final(dev, reg, tmp_path / "FINAL.md", tmp_path / "t.jsonl") is False
    assert (tmp_path / "FINAL.md").read_text().startswith("# 最终测试：不通过")
    assert json.loads((tmp_path / "t.jsonl").read_text().splitlines()[-1])["metrics"]["passed"] is False
