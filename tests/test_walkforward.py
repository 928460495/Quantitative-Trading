"""计划 3c 任务 9：训练段滚动前推（逐步加因子 + 组合网格）。小假数据。"""

import json

import numpy as np
import pandas as pd
import pytest

from qt import walkforward as wf
from qt.fastsim import SimParams
from qt.wide_eval import forward_returns


def fake_seg(seed=0, n_stocks=40, start="2010-01-04", end="2013-12-31"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    cols = [f"SH6{i:05d}" for i in range(n_stocks)]
    close = pd.DataFrame(10 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(idx), n_stocks)), axis=0)), index=idx, columns=cols)
    sig = forward_returns(close, [5])[5].rank(axis=1)                       # 真信号（假数据里可以看未来）
    noise = lambda: pd.DataFrame(rng.normal(size=close.shape), index=idx, columns=cols)
    true = pd.DataFrame(True, index=idx, columns=cols)
    return {
        "factors": {"sig": sig.fillna(0.0), "n1": noise(), "n2": noise()},
        "close": close, "raw_close": close * 0 + 1.0, "can_buy": true, "can_sell": true,
        "industry": pd.DataFrame([[f"I{i % 4}" for i in range(n_stocks)]] * len(idx), index=idx, columns=cols, dtype=object),
        "log_mcap": pd.DataFrame(rng.normal(22, 1, close.shape), index=idx, columns=cols),
        "index": pd.DataFrame({"000985": close.mean(axis=1), "H00300": close.mean(axis=1)}),
    }


POOL = pd.DataFrame({"category": ["价值", "动量/反转", "低波动"], "direction": [1, 1, 1]}, index=["sig", "n1", "n2"])
SMALL_GRID = [(SimParams(n_hold=n, rebalance_every=5, sell_mult=3, industry_cap=1.0, min_trade=0), k, 1.0)
              for n in (5, 8) for k in (1, 5)]
REF = SimParams(n_hold=5, rebalance_every=5, sell_mult=3, industry_cap=1.0, min_trade=0)


def test_grid_has_216_unique_configs_with_registered_values():
    assert len(wf.GRID) == 216
    labels = {wf.label(*g) for g in wf.GRID}
    assert len(labels) == 216
    p, k, w = zip(*wf.GRID)
    assert {x.n_hold for x in p} == {15, 20} and {x.rebalance_every for x in p} == {1, 5, 10}
    assert {x.sell_mult for x in p} == {3, 4, 6} and {x.industry_cap for x in p} == {0.30, 0.20}
    assert set(k) == {1, 5, 10} and set(w) == {1.0, 2.0}
    assert wf.REFERENCE == SimParams(n_hold=20, rebalance_every=5, sell_mult=3, industry_cap=0.30)


def test_forward_select_picks_signal_first_and_stops():
    seg = fake_seg()
    logged = []
    got = wf.forward_select(seg, POOL, ("2010-01-04", "2011-12-31"), reference=REF,
                            log=lambda kind, params, metrics: logged.append(kind))
    assert got[0] == "sig"
    assert len(got) < 3                                # 噪声加进来提高不到 0.05
    assert logged and set(logged) == {"plan3c_step"}


def test_grid_select_returns_all_columns_and_best():
    seg = fake_seg()
    rets, best = wf.grid_select(seg, POOL, ["sig"], ("2010-01-04", "2011-12-31"), grid=SMALL_GRID)
    assert list(rets.columns) == [wf.label(*g) for g in SMALL_GRID]
    assert rets.index.min() >= pd.Timestamp("2010-01-04") and rets.index.max() <= pd.Timestamp("2011-12-31")
    assert best in SMALL_GRID


def test_walk_forward_uses_only_window_data(tmp_path):
    seg = fake_seg()
    a = wf.walk_forward(seg, POOL, years=(2012,), cache_dir=tmp_path / "a", grid=SMALL_GRID, reference=REF)
    bumped = fake_seg()
    bumped["close"].loc["2012-01-01":] *= np.random.default_rng(9).uniform(0.5, 1.5, bumped["close"].loc["2012-01-01":].shape)
    b = wf.walk_forward(bumped, POOL, years=(2012,), cache_dir=tmp_path / "b", grid=SMALL_GRID, reference=REF)
    assert a["years"][2012] == b["years"][2012]
    r = a["returns"]
    assert r.index.min().year == 2012 and r.index.max().year == 2012


def test_walk_forward_resumes_from_cache(tmp_path, monkeypatch):
    seg = fake_seg()
    first = wf.walk_forward(seg, POOL, years=(2012,), cache_dir=tmp_path, grid=SMALL_GRID, reference=REF)
    assert json.loads((tmp_path / "wf_2012.json").read_text())["factors"] == first["years"][2012]["factors"]
    monkeypatch.setattr(wf, "forward_select", lambda *a, **k: (_ for _ in ()).throw(AssertionError("不该重算")))
    again = wf.walk_forward(seg, POOL, years=(2012,), cache_dir=tmp_path, grid=SMALL_GRID, reference=REF)
    assert again["years"] == first["years"]


def test_account_rules_mirror_sim_params():
    r = wf.account_rules(SimParams(n_hold=15, rebalance_every=10, sell_mult=4, industry_cap=0.2))
    assert (r.n_hold, r.sell_rank, r.rebalance_every, r.max_industry_weight, r.min_trade) == (15, 60, 10, 0.2, 2000.0)
    assert r.drawdown_stop is None and r.drawdown_tiers is not None
