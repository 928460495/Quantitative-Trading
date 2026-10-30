"""宽表算子：行是日期，列是股票。每个算子一个手算的小例子。"""

import time

import numpy as np
import pandas as pd
import pytest

from qt import ops

NAN = np.nan


def _df(rows, cols=("A", "B")):
    return pd.DataFrame(rows, index=pd.bdate_range("2019-01-02", periods=len(rows)), columns=list(cols), dtype=float)


X = _df([[1, 5], [2, 4], [3, 3], [2, 2], [5, 1]])


def test_delay_delta():
    assert ops.delay(X, 1)["A"].tolist()[1:] == [1, 2, 3, 2]
    assert ops.delta(X, 2)["A"].tolist()[2:] == [2, 0, 2]


def test_ts_basic_windows():
    assert ops.ts_sum(X, 3)["A"].tolist()[2:] == [6, 7, 10]
    assert ops.ts_mean(X, 3)["B"].tolist()[2:] == [4, 3, 2]
    assert ops.ts_min(X, 3)["A"].tolist()[2:] == [1, 2, 2]
    assert ops.ts_max(X, 3)["A"].tolist()[2:] == [3, 3, 5]
    assert ops.ts_std(X, 3)["A"].iloc[2] == pytest.approx(1.0)       # 1,2,3 的样本标准差
    assert ops.ts_product(X, 3)["A"].tolist()[2:] == [6, 12, 30]
    assert np.isnan(ops.ts_sum(X, 3)["A"].iloc[1])


def test_window_missing_up_to_a_quarter_uses_available_days():
    # 窗口 4 天缺 1 天（25%）：用剩下 3 天算；缺 2 天（50%）：缺失
    y = _df([[1, 1], [NAN, NAN], [3, NAN], [4, 4], [5, 5]])
    assert ops.ts_mean(y, 4)["A"].iloc[3] == pytest.approx((1 + 3 + 4) / 3)
    assert np.isnan(ops.ts_mean(y, 4)["B"].iloc[3])
    assert ops.ts_rank(y, 4)["A"].iloc[3] == 3            # 4 在 1、3、4 里排第 3
    assert ops.decay_linear(y, 4)["A"].iloc[3] == pytest.approx((4 * 4 + 3 * 3 + 1 * 1) / (4 + 3 + 1))   # 前 2 天缺失（权重 2）
    assert ops.ts_argmax(y, 4)["A"].iloc[3] == 4
    assert ops.ts_product(y, 4)["A"].iloc[3] == 12
    assert ops.ts_corr(y, y * 2, 4)["A"].iloc[3] == pytest.approx(1.0)


def test_suspended_today_rank_is_missing():
    y = _df([[1, 1], [2, 2], [3, 3], [NAN, 4]])
    assert np.isnan(ops.ts_rank(y, 4)["A"].iloc[3])       # 当天没有值：排名缺失


def test_window_not_full_yet_is_missing():
    assert np.isnan(ops.ts_mean(X, 3)["A"].iloc[1]) and np.isnan(ops.ts_rank(X, 3)["A"].iloc[1])


def test_ts_rank_and_argmax():
    # A 最近三天 3,2,5：5 排第 3；B 3,2,1：1 排第 1
    assert ops.ts_rank(X, 3).iloc[-1].tolist() == [3, 1]
    # 并列取平均：2,2 → 1.5
    assert ops.ts_rank(_df([[1, 2], [2, 2]]), 2).iloc[-1].tolist() == [2, 1.5]
    # 最大值在窗口里的位置：1 = 最早，3 = 当天
    assert ops.ts_argmax(X, 3).iloc[-1].tolist() == [3, 1]
    assert ops.ts_argmin(X, 3).iloc[-1].tolist() == [2, 3]
    # 最高点距今天数（国泰君安 HIGHDAY）：当天是 1
    assert ops.highday(X, 3).iloc[-1].tolist() == [1, 3]


def test_decay_linear_and_wma():
    # 权重 1,2,3（越新越大），A 最近三天 3,2,5 → (3+4+15)/6
    assert ops.decay_linear(X, 3)["A"].iloc[-1] == pytest.approx(22 / 6)
    w = 0.9 ** np.array([2, 1, 0])
    assert ops.wma(X, 3)["A"].iloc[-1] == pytest.approx((w * [3, 2, 5]).sum() / w.sum())


def test_sma_is_gtja_recursive_mean():
    # Y_t = (m·A_t + (n−m)·Y_{t−1}) / n，n=3、m=1
    y = [1.0]
    for a in [2, 3, 2, 5]:
        y.append((a + 2 * y[-1]) / 3)
    assert ops.sma(X, 3, 1)["A"].tolist() == pytest.approx(y)


def test_corr_cov():
    c = ops.ts_corr(X, ops.delay(X, 1), 3)
    a, b = np.array([2, 3, 2.0]), np.array([1, 2, 3.0])
    assert c["A"].iloc[3] == pytest.approx(np.corrcoef(a, b)[0, 1])
    assert ops.ts_cov(X, X, 3)["A"].iloc[2] == pytest.approx(1.0)


def test_count_sumif():
    cond = X > 2
    assert ops.count(cond, 3)["A"].tolist()[2:] == [1, 1, 2]
    assert ops.sumif(X, 3, cond)["A"].tolist()[2:] == [3, 3, 8]


def test_regbeta_regresi_against_sequence():
    y = _df([[1, 2], [3, 4], [5, 6], [7, 9]])
    beta = ops.regbeta(y, ops.sequence(3), 3)
    assert beta.iloc[-1].tolist() == pytest.approx([2.0, 2.5])
    resid = ops.regresi(y, ops.sequence(3), 3)
    assert resid["A"].iloc[-1] == pytest.approx(0.0)
    beta_x = ops.regbeta(y, y * 2, 3)                 # 对另一张表回归
    assert beta_x.iloc[-1].tolist() == pytest.approx([0.5, 0.5])


def test_cross_sectional():
    x = _df([[1, 2, 3, 4], [4, 3, 2, NAN]], cols="ABCD")
    r = ops.cs_rank(x)
    assert r.iloc[0].tolist() == [0.25, 0.5, 0.75, 1.0]
    assert np.isnan(r.iloc[1]["D"]) and r.iloc[1]["A"] == 1.0
    s = ops.cs_scale(x)
    assert s.abs().sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    groups = _df([[1, 1, 2, 2], [1, 1, 2, 2]], cols="ABCD")
    n = ops.ind_neutralize(x, groups)
    assert n.iloc[0].tolist() == [-0.5, 0.5, -0.5, 0.5]


def test_cross_sectional_does_not_look_ahead():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(5, 6)), index=pd.bdate_range("2019-01-02", periods=5))
    y = x.copy()
    y.iloc[-1] = rng.normal(size=6) * 100
    pd.testing.assert_frame_equal(ops.cs_rank(x).iloc[:-1], ops.cs_rank(y).iloc[:-1])
    pd.testing.assert_frame_equal(ops.ts_rank(x, 3).iloc[:-1], ops.ts_rank(y, 3).iloc[:-1])


def test_elementwise():
    assert ops.signed_power(_df([[-2, 3]]), 2).iloc[0].tolist() == [-4, 9]


@pytest.mark.slow
def test_full_market_scale_is_fast_enough():
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(3000, 5000)))
    for fn, arg in [(ops.ts_rank, 20), (ops.decay_linear, 20)]:
        t = time.time()
        fn(x, arg)
        assert time.time() - t < 10, fn.__name__


def test_corr_constant_window_is_zero_but_missing_stays_missing():
    x = _df([[1, 1], [2, 1], [3, 1], [4, NAN]])
    y = _df([[3, 2], [1, 5], [2, 4], [6, 1]])
    c = ops.ts_corr(x, y, 3)
    assert c["B"].iloc[2] == 0.0           # B 的 x 连续 3 天不变：相关系数记 0
    assert np.isnan(c["B"].iloc[3])        # 窗口里有缺失：仍然缺失
    assert c["A"].iloc[2] == pytest.approx(np.corrcoef([1, 2, 3], [3, 1, 2])[0, 1])
