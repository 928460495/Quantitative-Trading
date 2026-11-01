"""类内筛选：按 ICIR 排序、相关性 > 0.7 跳过、每类最多 3 个（docs/DECISIONS.md 2026-11-01）。"""

import numpy as np
import pandas as pd
import pytest

from qt import screen_wide as sw

DATES = pd.bdate_range("2015-01-05", periods=60)
CODES = [f"S{i:03d}" for i in range(200)]


def _base(seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(len(DATES), len(CODES))), index=DATES, columns=CODES)


def _mix(a, b, w, seed):
    """和 a 的相关性约为 w 的因子：w·a + √(1−w²)·噪声。"""
    return w * a + np.sqrt(1 - w * w) * _base(seed) if b is None else w * a + np.sqrt(1 - w * w) * b


def test_pair_corr_is_mean_daily_spearman():
    a, b = _base(1), _base(2)
    b.iloc[:, :20] = np.nan
    want = np.mean([a.loc[d].corr(b.loc[d], method="spearman") for d in DATES])
    assert sw.pair_corr(a, b) == pytest.approx(want, abs=1e-12)


def test_pick_skips_near_duplicates_and_keeps_at_most_three():
    a = _base(10)
    scores = {
        "a": a,
        "a_copy": _mix(a, None, 0.95, 11),     # 和 a 很像 → 跳过
        "b": _base(12),
        "c": _base(13),
        "d": _base(14),                        # 前面已经留够 3 个 → 不再看
    }
    table = pd.DataFrame({"icir": [0.9, 0.8, 0.7, 0.6, 0.5]}, index=list(scores))
    res = sw.pick(table, scores)
    assert res.index[res["kept"]].tolist() == ["a", "b", "c"]
    assert res.loc["a_copy", "dropped_by"] == "a" and res.loc["a_copy", "corr"] > 0.9
    assert res.loc["d", "status"] == "已满 3 个，未比较"


def test_pick_uses_absolute_correlation():
    a = _base(20)
    scores = {"a": a, "neg": -_mix(a, None, 0.9, 21)}
    res = sw.pick(pd.DataFrame({"icir": [0.5, 0.4]}, index=list(scores)), scores)
    assert res.index[res["kept"]].tolist() == ["a"]


def test_pick_moderate_correlation_is_kept():
    a = _base(30)
    scores = {"a": a, "b": _mix(a, None, 0.5, 31)}
    res = sw.pick(pd.DataFrame({"icir": [0.5, 0.4]}, index=list(scores)), scores)
    assert res["kept"].all()


def test_pick_ties_broken_by_name_and_empty_category():
    scores = {"z": _base(40), "m": _base(41)}
    res = sw.pick(pd.DataFrame({"icir": [0.3, 0.3]}, index=["z", "m"]), scores)
    assert res.index.tolist() == ["m", "z"]
    empty = sw.pick(pd.DataFrame({"icir": []}, dtype=float), {})
    assert empty.empty and "kept" in empty.columns


def test_select_by_category_runs_each_category_separately():
    a = _base(50)
    scores = {"m1": a, "v1": _mix(a, None, 0.95, 51)}      # 跨类很像也都留：跨类不去重
    table = pd.DataFrame({"icir": [0.5, 0.4], "category": ["动量/反转", "价值"]}, index=["m1", "v1"])
    res = sw.select_by_category(table, scores)
    assert res["kept"].all()
    assert set(res["category"]) == {"动量/反转", "价值"}


def test_corr_matrix_symmetric_with_unit_diagonal():
    scores = {"a": _base(60), "b": _base(61)}
    m = sw.corr_matrix(scores)
    assert m.loc["a", "a"] == 1.0 and m.loc["a", "b"] == m.loc["b", "a"]


def test_every_nth_day_starts_at_first_valid_day():
    x = _base(70)
    x.iloc[:3] = np.nan
    sub = sw.every_nth_day(x, 5)
    assert sub.index[0] == DATES[3] and sub.index[1] == DATES[8]
