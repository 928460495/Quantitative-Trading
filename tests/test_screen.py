import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("screen_factors", ROOT / "scripts" / "screen_factors.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def _cands(rows):
    return pd.DataFrame(rows, columns=["factor", "category", "icir_5d"]).set_index("factor")


def test_keeps_higher_icir_of_correlated_pair():
    cands = _cands([("a", "打分因子", 0.3), ("b", "打分因子", 0.4), ("c", "打分因子", 0.2)])
    corr = pd.DataFrame(np.eye(3), index=list("abc"), columns=list("abc"))
    corr.loc["a", "b"] = corr.loc["b", "a"] = 0.8
    out = screen.greedy_select(cands, corr)
    assert out["kept"].to_dict() == {"b": True, "a": False, "c": True}
    assert out.loc["a", "dropped_by"] == "b"


def test_negative_correlation_also_counts():
    cands = _cands([("a", "打分因子", 0.3), ("b", "打分因子", 0.4)])
    corr = pd.DataFrame([[1, -0.75], [-0.75, 1]], index=list("ab"), columns=list("ab"))
    assert screen.greedy_select(cands, corr)["kept"].to_dict() == {"b": True, "a": False}


def test_scoring_factor_is_not_displaced_by_veto_factor():
    cands = _cands([("s", "打分因子", 0.2), ("v", "排雷因子", 0.5)])
    corr = pd.DataFrame([[1, 0.9], [0.9, 1]], index=list("sv"), columns=list("sv"))
    out = screen.greedy_select(cands, corr)
    assert out["kept"].to_dict() == {"s": True, "v": False}


def test_mean_rank_corr():
    dates = pd.bdate_range("2016-01-04", periods=3)
    idx = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(50)]], names=["datetime", "instrument"])
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=len(idx)), index=idx)
    scores = pd.DataFrame({"x": x, "y": np.exp(x), "z": -x})
    corr = screen.mean_rank_corr(scores, dates)
    assert np.isclose(corr.loc["x", "y"], 1) and np.isclose(corr.loc["x", "z"], -1)
