"""scripts/paper_day.py：纸面账户一键运行（计划 5 任务 5）。"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from qt.lab import procedure as pc

from lab_helpers import dev3d

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_day", ROOT / "scripts" / "paper_day.py")
pd_ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd_)

PROC = pc.PLAN3D.replace(exclude=["is_st"], no_buy=[], rebalance_every=10, n_hold=8, sell_mult=2)


@pytest.fixture(scope="module")
def full():
    return dev3d(seed=4, n_stocks=60, start="2018-01-02", end="2018-12-31")[0]


def _cut(dev, end):
    def cut(x):
        if isinstance(x, dict):
            return {k: cut(v) for k, v in x.items()}
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return x.loc[:end]
        return x
    return cut(dev)


def test_first_day_sets_anchor_and_issues_buys(full, tmp_path):
    d = full["close"].index
    out = pd_.run_day(_cut(full, d[30]), PROC, tmp_path, start=d[30])
    cfg = json.loads((tmp_path / "paper.json").read_text())
    assert cfg["anchor"] == str(d[30].date()) and pc.Procedure.from_dict(cfg["procedure"]) == PROC
    assert out["rebalance"] and out["n_buys"] > 0 and out["log_problems"] == []
    assert (tmp_path / f"report-{d[30].date()}.md").exists() and (tmp_path / "orders_log.jsonl").exists()


def test_later_days_rerun_and_match_logged_orders(full, tmp_path):
    d = full["close"].index
    pd_.run_day(_cut(full, d[30]), PROC, tmp_path, start=d[30])
    for k in (35, 40, 41, 55, 70):
        out = pd_.run_day(_cut(full, d[k]), PROC, tmp_path)
        assert out["log_problems"] == [], (k, out["log_problems"])
    nav = pd.read_csv(tmp_path / "nav.csv", index_col=0, parse_dates=True)
    assert nav.index[0] == d[30] and nav.index[-1] == d[70]
    assert len(pd_.load_log(tmp_path / "orders_log.jsonl")) == 3            # 跑过的日子里第 0、10、40 天是调仓日
    report = (tmp_path / f"report-{d[70].date()}.md").read_text()
    assert "停止规则" in report and "沪深300" in report


def test_tampered_log_is_reported(full, tmp_path):
    d = full["close"].index
    pd_.run_day(_cut(full, d[30]), PROC, tmp_path, start=d[30])
    log = tmp_path / "orders_log.jsonl"
    rec = json.loads(log.read_text())
    rec["buys"] = [["SH999999", 100, 1.0]]
    log.write_text(json.dumps(rec) + "\n")
    out = pd_.run_day(_cut(full, d[35]), PROC, tmp_path)
    assert out["log_problems"] and "对不上" in (tmp_path / f"report-{d[35].date()}.md").read_text()


def test_start_needed_first_time_and_not_after(full, tmp_path):
    d = full["close"].index
    with pytest.raises(ValueError, match="开始日"):
        pd_.run_day(_cut(full, d[30]), PROC, tmp_path)
    pd_.run_day(_cut(full, d[30]), PROC, tmp_path, start=d[30])
    with pytest.raises(ValueError, match="已经开始"):
        pd_.run_day(_cut(full, d[31]), PROC, tmp_path, start=d[31])


def test_start_must_be_latest_day(full, tmp_path):
    d = full["close"].index
    with pytest.raises(ValueError, match="最新"):
        pd_.run_day(_cut(full, d[30]), PROC, tmp_path, start=d[29])
