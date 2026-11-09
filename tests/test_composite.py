"""计划 4 的合成：类内等权 → 每类标准化 → 类间等权 → 标准化（docs/DECISIONS.md 2026-11-09）。"""

import numpy as np
import pandas as pd
import pytest

from qt.composite import category_scores, composite
from qt.wide_eval import zscore

DATES = pd.bdate_range("2019-01-02", periods=30)
CODES = [f"S{i:03d}" for i in range(80)]
NAN = np.nan


def _w(seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(len(DATES), len(CODES))), index=DATES, columns=CODES)


def test_category_mean_ignores_missing_factor_and_restandardizes():
    idx = DATES[:1]
    a = pd.DataFrame([[1.0, 2.0, 3.0, NAN]], index=idx, columns=list("WXYZ"))
    b = pd.DataFrame([[3.0, NAN, 1.0, 4.0]], index=idx, columns=list("WXYZ"))
    got = category_scores({"a": a, "b": b}, {"a": "价值", "b": "价值"})["价值"]
    raw = pd.DataFrame([[2.0, 2.0, 2.0, 4.0]], index=idx, columns=list("WXYZ"))   # 缺值的因子不参与平均
    pd.testing.assert_frame_equal(got, zscore(raw))


def test_single_factor_composite_is_the_factor_standardized():
    f = _w(1)
    got = composite(category_scores({"f": f}, {"f": "价值"}))
    pd.testing.assert_frame_equal(got, zscore(zscore(f)))
    pd.testing.assert_frame_equal(got, zscore(f), atol=1e-12)


def test_categories_weigh_equally_however_many_factors_they_have():
    # 价值类 3 个互不相关的因子（平均后波动变小），质量类 1 个：两类在总分里的分量应该一样
    fs = {"v1": _w(2), "v2": _w(3), "v3": _w(4), "q1": _w(5)}
    cats = {"v1": "价值", "v2": "价值", "v3": "价值", "q1": "质量"}
    cs = category_scores(fs, cats)
    total = composite(cs)
    corr = lambda x: total.corrwith(x, axis=1).mean()
    assert corr(cs["价值"]) == pytest.approx(corr(cs["质量"]), abs=0.02)


def test_category_missing_for_a_stock_uses_the_rest():
    v, q = _w(6), _w(7)
    q.iloc[:, :5] = NAN                     # 这 5 只没有质量类（比如新股没财报）
    total = composite(category_scores({"v": v, "q": q}, {"v": "价值", "q": "质量"}))
    assert total.iloc[:, :5].notna().all().all()
    q.iloc[:, :] = NAN
    total2 = composite(category_scores({"v": v, "q": q}, {"v": "价值", "q": "质量"}))
    pd.testing.assert_frame_equal(total2, zscore(zscore(v)), atol=1e-12)


def test_no_look_ahead():
    fs = {"a": _w(8), "b": _w(9)}
    cats = {"a": "价值", "b": "成长"}
    changed = {k: v.copy() for k, v in fs.items()}
    for v in changed.values():
        v.iloc[20:] = v.iloc[20:] * 3 + 1
    a = composite(category_scores(fs, cats))
    b = composite(category_scores(changed, cats))
    pd.testing.assert_frame_equal(a.iloc[:20], b.iloc[:20])


def test_category_names_must_cover_every_factor():
    with pytest.raises(KeyError):
        category_scores({"a": _w(10)}, {})
