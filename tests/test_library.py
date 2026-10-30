import numpy as np
import pandas as pd
import pytest

from qt.formula import evaluate
from qt.library import ALPHA101, ALPHA191, HELD, FactorDef


def _env(n_days=300, n_stocks=12, seed=0):
    idx = pd.bdate_range("2018-01-02", periods=n_days)
    cols = [f"S{i}" for i in range(n_stocks)]
    rng = np.random.default_rng(seed)
    walk = lambda: pd.DataFrame(10 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n_stocks)), axis=0)), index=idx, columns=cols)
    close = walk()
    env = {"close": close, "open": close * rng.uniform(0.98, 1.02, close.shape), "vwap": close * rng.uniform(0.99, 1.01, close.shape)}
    env["high"] = np.maximum(env["open"], close) * 1.01
    env["low"] = np.minimum(env["open"], close) * 0.99
    env["volume"] = pd.DataFrame(rng.uniform(1e5, 1e6, close.shape), index=idx, columns=cols)
    env["amount"] = env["volume"] * env["vwap"]
    env["float_mcap"] = close * 1e8
    env["returns"] = close.pct_change()
    for lv, k in (("industry_l1", 3), ("industry_l2", 5), ("industry_l3", 7)):
        env[lv] = pd.DataFrame(np.tile((np.arange(n_stocks) % k).astype(str), (n_days, 1)), index=idx, columns=cols)
    bench = 3000 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
    env["benchmark_close"] = pd.DataFrame(np.tile(bench[:, None], (1, n_stocks)), index=idx, columns=cols)
    env["benchmark_open"] = env["benchmark_close"].shift(1).fillna(3000) * rng.uniform(0.995, 1.005, (n_days, 1))
    return env


def test_counts():
    assert len(ALPHA101) == 101
    assert len(ALPHA191) == 182 and len(HELD) == 9
    assert all(isinstance(f, FactorDef) and f.formula for f in ALPHA101 + ALPHA191)
    assert {f.family for f in ALPHA101} == {"alpha101"} and {f.family for f in ALPHA191} == {"alpha191"}


@pytest.mark.parametrize("factor", ALPHA101 + ALPHA191, ids=lambda f: f.name)
def test_every_formula_evaluates(factor):
    env = _env()
    out = evaluate(factor.formula, env, factor.dialect)
    assert out.shape == env["close"].shape
    assert out.notna().to_numpy().sum() > 0, "整列缺失"


def test_held_have_reasons():
    assert all(h.note for h in HELD)
    assert {h.name for h in HELD} == {"gtja030", "gtja054", "gtja143", "gtja149", "gtja165", "gtja166",
                                      "gtja181", "gtja183", "gtja190"}


def test_corrections_are_documented():
    fixed = {f.name: f for f in ALPHA191 if f.formula != f.original}
    assert set(fixed) == {"gtja002", "gtja010", "gtja052", "gtja083", "gtja099", "gtja100", "gtja111", "gtja116",
                          "gtja127", "gtja131", "gtja159", "gtja180", "gtja188"}
    assert all(f.note for f in fixed.values())
