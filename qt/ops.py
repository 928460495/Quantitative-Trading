"""因子算子：在"日期 × 股票"的宽表上运算（行是交易日，列是股票）。

Qlib 的表达式只能逐只股票算时间序列，Alpha101、Alpha191 大量用到同一天股票之间的排名，所以另写一套。
约定（和 Qlib、pandas rolling 一致）：
- 时间序列算子的窗口里有缺失，结果就是缺失
- 横截面算子只在同一天的股票之间算，不用到别的日子
- 时序排名、线性衰减加权这类算子用"错位相加"实现：循环窗口长度次，每次整表运算，不逐窗口调 Python 函数
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Frame = pd.DataFrame


# ---------- 时间序列 ----------

def delay(x: Frame, d: int) -> Frame:
    return x.shift(int(d))


def delta(x: Frame, d: int) -> Frame:
    return x - x.shift(int(d))


def _roll(x: Frame, d: int):
    d = int(d)
    return x.rolling(d, min_periods=d)


def ts_sum(x: Frame, d: int) -> Frame:
    return _roll(x, d).sum()


def ts_mean(x: Frame, d: int) -> Frame:
    return _roll(x, d).mean()


def ts_std(x: Frame, d: int) -> Frame:
    return _roll(x, d).std()


def ts_min(x: Frame, d: int) -> Frame:
    return _roll(x, d).min()


def ts_max(x: Frame, d: int) -> Frame:
    return _roll(x, d).max()


def ts_corr(x: Frame, y: Frame, d: int) -> Frame:
    """滚动相关系数。窗口里数据完整、但有一边是常数（方差为 0）时记 0，不记缺失：
    否则缺失会顺着后面的时间窗口一路扩散（Alpha101 里对短窗口排名求相关很常见）。"""
    out = _roll(x, d).corr(y).replace([np.inf, -np.inf], np.nan)
    complete = (x.notna() & y.notna()).astype(float).rolling(int(d), min_periods=int(d)).sum() == int(d)
    return out.mask(out.isna() & complete, 0.0)


def ts_cov(x: Frame, y: Frame, d: int) -> Frame:
    return _roll(x, d).cov(y)


def _nan_window(arr: np.ndarray, d: int) -> np.ndarray:
    """窗口（当天往前 d 天）里有没有缺失、窗口满不满。"""
    isnan = np.isnan(arr)
    out = isnan.copy()
    for k in range(1, d):
        out[k:] |= isnan[:-k]
    out[: d - 1] = True
    return out


def _finish(x: Frame, values: np.ndarray, nan_any: np.ndarray) -> Frame:
    values = values.astype(float)
    values[nan_any] = np.nan
    return pd.DataFrame(values, index=x.index, columns=x.columns)


def ts_product(x: Frame, d: int) -> Frame:
    d = int(d)
    arr = x.to_numpy(dtype=float)
    acc = arr.copy()
    for k in range(1, d):
        acc[k:] *= arr[:-k]
    return _finish(x, acc, _nan_window(arr, d))


def ts_rank(x: Frame, d: int) -> Frame:
    """当天的值在过去 d 天里排第几（1 = 最小，并列取平均）。"""
    d = int(d)
    cur = x.to_numpy(dtype=float)
    less = np.zeros(cur.shape, dtype=np.float32)
    equal = np.ones(cur.shape, dtype=np.float32)  # 当天和自己相等
    for k in range(1, d):
        less[k:] += cur[:-k] < cur[k:]
        equal[k:] += cur[:-k] == cur[k:]
    return _finish(x, less + (equal + 1) / 2, _nan_window(cur, d))


def _arg_extreme(x: Frame, d: int, take_max: bool) -> tuple[np.ndarray, np.ndarray]:
    """最大（小）值在窗口里的位置：1 = 最早一天，d = 当天；并列取最近的一天。"""
    arr = x.to_numpy(dtype=float)
    best = arr.copy()
    pos = np.full(arr.shape, float(d))
    for k in range(1, d):
        lag = arr[:-k]
        better = (lag > best[k:]) if take_max else (lag < best[k:])
        best[k:] = np.where(better, lag, best[k:])
        pos[k:] = np.where(better, d - k, pos[k:])
    return pos, _nan_window(arr, d)


def ts_argmax(x: Frame, d: int) -> Frame:
    pos, nan_any = _arg_extreme(x, int(d), True)
    return _finish(x, pos, nan_any)


def ts_argmin(x: Frame, d: int) -> Frame:
    pos, nan_any = _arg_extreme(x, int(d), False)
    return _finish(x, pos, nan_any)


def highday(x: Frame, d: int) -> Frame:
    """国泰君安 HIGHDAY：窗口内最大值离今天几天，今天是 1。"""
    return int(d) + 1 - ts_argmax(x, d)


def lowday(x: Frame, d: int) -> Frame:
    return int(d) + 1 - ts_argmin(x, d)


def _weighted(x: Frame, weights_newest_first: np.ndarray) -> Frame:
    d = len(weights_newest_first)
    arr = x.to_numpy(dtype=float)
    acc = arr * weights_newest_first[0]
    for k in range(1, d):
        acc[k:] += weights_newest_first[k] * arr[:-k]
    return _finish(x, acc / weights_newest_first.sum(), _nan_window(arr, d))


def decay_linear(x: Frame, d: int) -> Frame:
    """线性衰减加权平均：当天权重 d，最早一天权重 1。"""
    d = int(d)
    return _weighted(x, np.arange(d, 0, -1, dtype=float))


def wma(x: Frame, d: int) -> Frame:
    """国泰君安 WMA：权重 0.9^i，i 是离今天的天数。"""
    d = int(d)
    return _weighted(x, 0.9 ** np.arange(d, dtype=float))


def sma(x: Frame, n: float, m: float) -> Frame:
    """国泰君安 SMA：Y_t = (m·A_t + (n−m)·Y_{t−1}) / n。"""
    return x.ewm(alpha=m / n, adjust=False).mean()


def count(cond: Frame, d: int) -> Frame:
    return ts_sum(cond.astype(float).where(cond.notna()), d)


def sumif(x: Frame, d: int, cond: Frame) -> Frame:
    return ts_sum(x.where(cond.astype(bool), 0.0), d)


class Sequence:
    """国泰君安 SEQUENCE(n)：1, 2, ……, n，只在回归里当自变量用。"""

    def __init__(self, n: int):
        self.n = int(n)


def sequence(n: int) -> Sequence:
    return Sequence(n)


def _regress(y: Frame, x, d: int) -> tuple[Frame, Frame, Frame]:
    """滚动一元回归 y = a + b·x，返回 (b, a, x 的当天值)。x 是宽表或 Sequence。"""
    d = int(d)
    if isinstance(x, Sequence):
        t = pd.DataFrame(np.broadcast_to(np.arange(len(y), dtype=float)[:, None], y.shape),
                         index=y.index, columns=y.columns)
        x = t
    mx, my = ts_mean(x, d), ts_mean(y, d)
    b = (ts_mean(x * y, d) - mx * my) / (ts_mean(x * x, d) - mx * mx)
    b = b.replace([np.inf, -np.inf], np.nan)
    return b, my - b * mx, x


def regbeta(y: Frame, x, d: int) -> Frame:
    return _regress(y, x, d)[0]


def regresi(y: Frame, x, d: int) -> Frame:
    """当天的回归残差。"""
    b, a, xv = _regress(y, x, d)
    return y - (a + b * xv)


# ---------- 横截面 ----------

def cs_rank(x: Frame) -> Frame:
    """同一天股票之间的百分比排名（并列取平均），缺失不参与。"""
    return x.rank(axis=1, pct=True)


def cs_scale(x: Frame, a: float = 1.0) -> Frame:
    """同一天按绝对值之和缩放到 a。"""
    return x.mul(a / x.abs().sum(axis=1), axis=0)


def ind_neutralize(x: Frame, groups: Frame) -> Frame:
    """同一天同一组（行业）内减去组均值。"""
    long = pd.DataFrame({"x": x.stack(future_stack=True), "g": groups.reindex_like(x).stack(future_stack=True)})
    mean = long.groupby([long.index.get_level_values(0), long["g"]])["x"].transform("mean")
    out = (long["x"] - mean).where(long["g"].notna())
    return out.unstack().reindex_like(x)


# ---------- 逐元素 ----------

def signed_power(x: Frame, e: float) -> Frame:
    return np.sign(x) * np.abs(x) ** e
