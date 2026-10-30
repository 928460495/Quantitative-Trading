"""抽查 20 个 Alpha101 / Alpha191 因子：用不经过 qt.ops 的手写 pandas 计算，在真实数据上逐值对比。

目的是抓两类错误：公式文本抄错（任务 4 的修正是否对），算子实现和原文定义不符。
只比最后 40 个交易日、数据完整的 5 只股票；横截面排名在全部 60 只股票上算。
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def env():
    from qt.panel import load_panel, main_board_instruments

    codes = main_board_instruments("2018-10-01", "2019-06-28")[:60]
    return load_panel("2018-10-08", "2019-06-28", instruments=codes)


def rank(x):
    return x.rank(axis=1, pct=True)


def corr(x, y, d):
    return x.rolling(d).corr(y)


def apply(x, d, fn):
    return x.rolling(d).apply(fn, raw=True)


def sma(x, n, m):
    out = x.copy() * np.nan
    for c in x.columns:
        prev = np.nan
        for i, v in enumerate(x[c].to_numpy()):
            if np.isnan(v):
                out.iloc[i, out.columns.get_loc(c)] = prev
                continue
            prev = v if np.isnan(prev) else (m * v + (n - m) * prev) / n
            out.iloc[i, out.columns.get_loc(c)] = prev
    return out


def hand(name, e):
    o, h, lo, c, v, vw, r = (e[k] for k in ("open", "high", "low", "close", "volume", "vwap", "returns"))
    pc = c.shift(1)
    if name == "wq001":
        x = pd.DataFrame(np.where(r < 0, r.rolling(20).std(), c), index=c.index, columns=c.columns)
        sp = np.sign(x) * np.abs(x) ** 2
        return rank(apply(sp, 5, lambda a: len(a) - np.argmax(a[::-1]))) - 0.5      # 并列取最近一天
    if name == "wq002":
        return -corr(rank(np.log(v).diff(2)), rank((c - o) / o), 6)
    if name == "wq003":
        return -corr(rank(o), rank(v), 10)
    if name == "wq006":
        return -corr(o, v, 10)
    if name == "wq012":
        return np.sign(v.diff()) * (-c.diff())
    if name == "wq013":
        return -rank(rank(c).rolling(5).cov(rank(v)))
    if name == "wq033":
        return rank(-(1 - o / c))
    if name == "wq041":
        return np.sqrt(h * lo) - vw
    if name == "wq053":
        return -(((c - lo) - (h - c)) / (c - lo)).diff(9)
    if name == "wq101":
        return (c - o) / ((h - lo) + 0.001)
    if name == "gtja001":
        return -corr(rank(np.log(v).diff(1)), rank((c - o) / o), 6)
    if name == "gtja003":
        inner = np.where(c == pc, 0.0, c - np.where(c > pc, np.minimum(lo, pc), np.maximum(h, pc)))
        return pd.DataFrame(inner, index=c.index, columns=c.columns).rolling(6).sum()
    if name == "gtja014":
        return c - c.shift(5)
    if name == "gtja021":
        t = np.arange(1, 7)
        return apply(c.rolling(6).mean(), 6, lambda a: np.polyfit(t, a, 1)[0])
    if name == "gtja031":
        m = c.rolling(12).mean()
        return (c - m) / m * 100
    if name == "gtja063":
        d = c.diff()
        return sma(d.clip(lower=0), 6, 1) / sma(d.abs(), 6, 1) * 100
    if name == "gtja076":
        a = (c / pc - 1).abs() / v
        return a.rolling(20).std() / a.rolling(20).mean()
    if name == "gtja109":
        s = sma(h - lo, 10, 2)
        return s / sma(s, 10, 2)
    if name == "gtja128":
        tp = (h + lo + c) / 3
        pos = pd.DataFrame(np.where(tp > tp.shift(1), tp * v, 0.0), index=c.index, columns=c.columns)
        neg = pd.DataFrame(np.where(tp < tp.shift(1), tp * v, 0.0), index=c.index, columns=c.columns)
        return 100 - 100 / (1 + pos.rolling(14).sum() / neg.rolling(14).sum())
    if name == "gtja150":
        return (c + h + lo) / 3 * v
    raise KeyError(name)


NAMES = ["wq001", "wq002", "wq003", "wq006", "wq012", "wq013", "wq033", "wq041", "wq053", "wq101",
         "gtja001", "gtja003", "gtja014", "gtja021", "gtja031", "gtja063", "gtja076", "gtja109", "gtja128", "gtja150"]


@pytest.mark.parametrize("name", NAMES)
def test_formula_matches_hand_calculation(env, name):
    from qt.formula import evaluate
    from qt.library import ALPHA101, ALPHA191

    f = {x.name: x for x in ALPHA101 + ALPHA191}[name]
    ours = evaluate(f.formula, env, f.dialect)
    ref = hand(name, env)
    full = env["close"].notna().all() & env["volume"].notna().all()
    cols = list(env["close"].columns[full][:5])
    a, b = ours[cols].iloc[-40:], ref[cols].iloc[-40:]
    mask = b.notna() & np.isfinite(b)
    assert mask.to_numpy().sum() > 100, "可比的数值太少"
    assert (a.notna() | ~mask).all().all(), "手算有值的地方我们算成了缺失"
    diff = (a[mask] - b[mask]).abs() / np.maximum(1.0, b[mask].abs())
    assert np.nanmax(diff.to_numpy()) < 1e-8
