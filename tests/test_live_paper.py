"""纸面账户：停止规则检查、清单存档和核对（计划 5 任务 5；DECISIONS 2026-11-13 停止规则）。"""

import numpy as np
import pandas as pd
import pytest

from qt.live.orders import Buy, Orders
from qt.live.paper import check_log, load_log, log_orders, stop_checks

DAYS = pd.bdate_range("2026-11-02", periods=800)


def _series(daily, days=DAYS):
    return pd.Series(daily, index=days[: len(daily)])


def test_drawdown_line_triggers_review():
    r = np.r_[np.full(10, 0.01), np.full(60, -0.015)]
    out = stop_checks(_series(r), _series(np.zeros(len(r))), dd_line=-0.54)
    assert out["回撤"] < -0.54 and out["①回撤复查"] is True
    out = stop_checks(_series(r[:20]), _series(np.zeros(20)), dd_line=-0.54)
    assert out["①回撤复查"] is False


def test_failure_line_only_checked_after_24_months_at_6_month_marks():
    rng = np.random.default_rng(0)
    h = rng.normal(0.0004, 0.012, 800)
    bad = h - 0.002                                       # 每天比沪深300 少 0.2%：明显变差
    short = stop_checks(_series(bad[:300]), _series(h[:300]))
    assert short["②失效检查"] == "未满 24 个月"
    on_mark = stop_checks(_series(bad), _series(h), months=24)
    assert on_mark["t"] < -1 and on_mark["②失效检查"] == "停用"
    good = stop_checks(_series(h + 0.0005), _series(h), months=24)
    assert good["②失效检查"] == "继续"
    off_mark = stop_checks(_series(bad), _series(h), months=27)
    assert off_mark["②失效检查"] == "不是检查月"


def test_failure_t_matches_registered_formula():
    rng = np.random.default_rng(1)
    h = rng.normal(0.0003, 0.012, 600)
    r = 1.3 * h + rng.normal(0, 0.006, 600)
    out = stop_checks(_series(r), _series(h), months=24)
    from qt.evaluate import newey_west_t
    rf = 1.02 ** (1 / 252) - 1
    a = (r - rf) - (r.std(ddof=1) / h.std(ddof=1)) * (h - rf)
    assert out["t"] == pytest.approx(newey_west_t(pd.Series(a), 20))


def _orders(sells=(), buys=()):
    return Orders(sells=list(sells), buys=[Buy(c, 100, 10.0, 1000.0, 5.0) for c in buys], cash_after=0.0)


def test_log_and_check(tmp_path):
    path = tmp_path / "log.jsonl"
    log_orders(path, pd.Timestamp("2026-11-13"), _orders([("SH600001", "rank")], ["SH600002"]))
    log_orders(path, pd.Timestamp("2026-11-13"), _orders([("SH600001", "rank")], ["SH600002"]))   # 同一天重跑不重复记
    assert len(load_log(path)) == 1
    days = pd.DatetimeIndex(["2026-11-13", "2026-11-16"])
    same = pd.DataFrame({"date": [days[1]] * 2, "instrument": ["SH600001", "SH600002"], "side": ["sell", "buy"],
                         "reason": ["rank", "buy"]})
    assert check_log(path, same, days) == []
    other = same.assign(instrument=["SH600001", "SH600009"])
    problems = check_log(path, other, days)
    assert problems and "2026-11-13" in problems[0]
    assert check_log(path, same.iloc[:0], days[:1]) == []         # 下一天的数据还没有：不核对


@pytest.mark.parametrize("t,verdict", [(-1.5, "停用"), (-0.99, "继续"), (-1.01, "停用")])
def test_failure_threshold_is_minus_one(monkeypatch, t, verdict):
    import qt.live.paper as paper
    monkeypatch.setattr(paper, "newey_west_t", lambda a, lag: t)
    h = np.random.default_rng(2).normal(0, 0.01, 100)
    assert stop_checks(_series(h + 0.001), _series(h), months=30)["②失效检查"] == verdict
