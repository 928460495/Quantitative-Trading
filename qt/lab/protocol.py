"""框架 2.0 的冻结常数（docs/FRAMEWORK.md）。一个世代内不改；要改就升 FRAMEWORK_VERSION，旧结论不追溯。"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

FRAMEWORK_VERSION = "2.0"

# 数据：开发段和三个制度阶段（第 2 节）。封存段 2022-01-01 起，不在这里出现
DEV = ("2010-01-01", "2021-12-31")
ERAS = {"一": ("2010-01-01", "2014-12-31"), "二": ("2015-01-01", "2018-12-31"), "三": ("2019-01-01", "2021-12-31")}
LATEST_ERA = "三"
FULL = "全段"

# 评价口径（第 5 节）
OFFSETS = (0, 5, 10, 15)      # 四个错开的调仓起点（交易日）
BENCH_EVERY = 20              # 能持有等权每 20 个交易日调一次
NW_LAG = 20                   # 配对差值的 Newey-West 滞后
BIG_LOSS = -0.40              # 平仓时亏损超过它算"大亏"

# 采纳规则（第 6 节）
T_ADOPT = 2.0                 # 收益型：配对差值 t 值下限
T_NOT_WORSE = -1.0            # 防雷型：全段 t 值不能低于它
BIG_LOSS_CUT = 0.30           # 防雷型：阶段三大亏只数至少少这么多
BIG_LOSS_SLACK = 0.10         # 收益型：大亏只数最多比基线多这么多（至少容许多 1 笔）

# 预算（第 7 节）
MAX_RUNS_PER_CARD = 6
MAX_CARDS = 12
MAX_CARDS_PER_FAMILY = 3
COOLDOWN_AFTER = 2            # 同一方向连续这么多张卡不采纳就冷却

_EULER = 0.5772156649015329


def noise_sharpe(n_trials: int, years: float) -> float:
    """试 n_trials 次（独立）、样本 years 年：纯噪声里最好的年化夏普的期望（Bailey 等 2014）。"""
    if n_trials < 1 or years <= 0:
        raise ValueError("试验次数至少 1，年数要为正")
    if n_trials == 1:
        return 0.0
    inv = NormalDist().inv_cdf
    z = (1 - _EULER) * inv(1 - 1 / n_trials) + _EULER * inv(1 - 1 / (n_trials * np.e))
    return float(z / np.sqrt(years))


def era_of(dates: pd.DatetimeIndex) -> np.ndarray:
    """每个日期属于哪个阶段；开发段以外的日期报错。"""
    dates = pd.DatetimeIndex(dates)
    out = np.full(len(dates), None, dtype=object)
    for name, (start, end) in ERAS.items():
        out[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))] = name
    if any(x is None for x in out):
        raise ValueError("有日期不在开发段内")
    return out
