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


# ---------- 计划 3c 任务 5：低波动加权、池内市值行业中性、平滑 ----------

from qt.composite import neutral_composite  # noqa: E402

IND = pd.DataFrame([[f"I{i % 4}" for i in range(len(CODES))]] * len(DATES), index=DATES, columns=CODES, dtype=object)


def test_composite_weights_default_unchanged():
    cs = category_scores({"a": _w(11), "b": _w(12)}, {"a": "价值", "b": "低波动"})
    pd.testing.assert_frame_equal(composite(cs, None), composite(cs))
    pd.testing.assert_frame_equal(composite(cs, {"低波动": 1.0}), composite(cs))


def test_lowvol_weight_two_counts_double():
    cs = category_scores({"a": _w(13), "b": _w(14)}, {"a": "低波动", "b": "价值"})
    got = composite(cs, {"低波动": 2.0})
    want = zscore((2 * cs["低波动"] + cs["价值"]) / 3)
    pd.testing.assert_frame_equal(got, want, atol=1e-12)


def test_neutral_composite_uncorrelated_with_size():
    log_mcap = _w(15) * 2 + 20
    f = log_mcap + 0.3 * _w(16)                       # 因子几乎就是市值
    got = neutral_composite({"f": f}, {"f": "价值"}, IND, log_mcap)
    corr = got.T.corrwith(log_mcap.T)
    assert corr.abs().max() < 1e-8


def test_neutral_composite_lowvol_weight_changes_result():
    log_mcap = _w(17) + 20
    fs = {"a": _w(18), "b": _w(19)}
    cats = {"a": "低波动", "b": "价值"}
    one = neutral_composite(fs, cats, IND, log_mcap, lowvol_weight=1.0)
    two = neutral_composite(fs, cats, IND, log_mcap, lowvol_weight=2.0)
    assert one.T.corrwith(fs["a"].T).mean() < two.T.corrwith(fs["a"].T).mean()


def test_smooth_uses_only_past():
    log_mcap = _w(20) + 20
    f = _w(21)
    base = neutral_composite({"f": f}, {"f": "价值"}, IND, log_mcap, smooth=5)
    g = f.copy()
    g.iloc[11:] += 5 * _w(22).iloc[11:]
    bumped = neutral_composite({"f": g}, {"f": "价值"}, IND, log_mcap, smooth=5)
    pd.testing.assert_frame_equal(base.iloc[:11], bumped.iloc[:11])
    assert not base.iloc[11:].equals(bumped.iloc[11:])


def test_smooth_one_equals_unsmoothed_and_first_days_use_available():
    log_mcap = _w(23) + 20
    f = _w(24)
    s1 = neutral_composite({"f": f}, {"f": "价值"}, IND, log_mcap, smooth=1)
    s10 = neutral_composite({"f": f}, {"f": "价值"}, IND, log_mcap, smooth=10)
    assert s10.iloc[0].notna().all()                  # 不足 10 天用已有天数
    pd.testing.assert_series_equal(s1.iloc[0], s10.iloc[0])


def test_factor_missing_early_years_does_not_drop_stock():
    log_mcap = _w(25) + 20
    late = _w(26)
    late.iloc[:10] = np.nan                           # 成长类：前面几年没有财报
    got = neutral_composite({"late": late, "early": _w(27)}, {"late": "成长", "early": "价值"}, IND, log_mcap)
    assert got.iloc[0].notna().all()


def test_smooth_keeps_missing_today_missing():
    log_mcap = _w(28) + 20
    f = _w(29)
    f.iloc[15:, 3] = np.nan                           # 第 15 天起这只股票不在股票池（比如变成 ST）
    got = neutral_composite({"f": f}, {"f": "价值"}, IND, log_mcap, smooth=5)
    assert got.iloc[15:, 3].isna().all()
