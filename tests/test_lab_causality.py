"""时间因果测试（docs/FRAMEWORK.md 第 9 节，失效模式 F7）：把某一天之后的数据全改掉，这一天及以前的东西一个数都不能变。

统计修正（DSR、PBO）抓不到偷看未来，只能在代码层面查。
"""

import numpy as np
import pandas as pd
import pytest

from qt.formula import evaluate
from qt.lab import procedure as pc
from qt.lab import store
from qt.library import all_factors

from lab_helpers import dev3d
from test_factor_funcs import _panel

DEFS = {f.name: f for f in all_factors()}
LOCAL = [n for n in (*pc.PLAN3D.factors(), store.VOL_FACTOR) if DEFS[n].dialect != "qlib"]
QLIB = [n for n in pc.PLAN3D.factors() if DEFS[n].dialect == "qlib"]
CUT = 220


def _scramble_after(p: dict, cut: int, seed: int = 9) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in p.items():
        v = v.copy()
        if isinstance(v, pd.DataFrame) and np.issubdtype(v.to_numpy().dtype, np.number):
            v.iloc[cut:] = v.iloc[cut:].to_numpy() * rng.uniform(0.3, 3.0, v.iloc[cut:].shape)
        out[k] = v
    return out


def _compute(name, panel):
    f = DEFS[name]
    return evaluate(f.formula, panel, f.dialect) if f.formula else f.func(panel)


def test_baseline_splits_into_locally_testable_and_qlib_factors():
    assert len(LOCAL) + len(QLIB) == 17 and set(QLIB) == {"lit_high52w", "a158_MA60", "a158_QTLU60", "a158_ROC60", "a158_MAX60"}


@pytest.mark.parametrize("name", LOCAL)
def test_factor_values_do_not_depend_on_the_future(name):
    p = _panel(n_days=320, n_stocks=8)
    a = _compute(name, p).iloc[:CUT]
    b = _compute(name, _scramble_after(p, CUT)).iloc[:CUT]
    assert a.notna().to_numpy().sum() > 0, name
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)


def test_scramble_helper_really_changes_the_future():
    p = _panel(n_days=320, n_stocks=8)
    q = _scramble_after(p, CUT)
    assert p["close"].iloc[:CUT].equals(q["close"].iloc[:CUT]) and not p["close"].iloc[CUT:].equals(q["close"].iloc[CUT:])
    future = _compute("ret60", q).iloc[CUT:]
    assert not future.equals(_compute("ret60", p).iloc[CUT:])               # 改的是真会影响结果的数


def _scramble_dev(dev: dict, cut: int) -> dict:
    rng = np.random.default_rng(3)
    out = {**dev, "factors": dict(dev["factors"]), "flags": dict(dev["flags"])}
    for key in ("close", "raw_close", "log_mcap"):
        v = dev[key].copy()
        v.iloc[cut:] = v.iloc[cut:].to_numpy() * rng.uniform(0.1, 3.0, v.iloc[cut:].shape)   # 能跌到 2 元以下
        out[key] = v
    for name, v in dev["factors"].items():
        v = v.copy()
        v.iloc[cut:] = rng.normal(size=v.iloc[cut:].shape)
        out["factors"][name] = v
    for name in store.STORED_FLAGS:                                           # 存下来的标记：后面随便改
        v = dev["flags"][name].copy()
        v.iloc[cut:] = rng.uniform(size=v.iloc[cut:].shape) < 0.3
        out["flags"][name] = v
    # 现算的标记：用改过的价格、市值、波动率重新算（藏在计算里的偷看才抓得到）
    out["flags"].update(store.derived_flags(out["raw_close"], out["log_mcap"], out["factors"][store.VOL_FACTOR]))
    for key in ("can_buy", "can_sell"):
        v = dev[key].copy()
        v.iloc[cut:] = rng.uniform(size=v.iloc[cut:].shape) < 0.8
        out[key] = v
    ind = dev["industry"].copy()
    ind.iloc[cut:] = "I9"
    out["industry"] = ind
    return out


@pytest.mark.parametrize("proc", [pc.PLAN3D, pc.PLAN3D.replace(exit_next_day=True, exclude=["is_st", "deduct_loss", "cheap"]),
                                  pc.PLAN3D.replace(no_buy=["high_vol", "small30"]), pc.PLAN3D.replace(missing="zero")],
                         ids=["基线", "次日卖出", "不买小市值", "缺失记零"])
def test_signals_and_account_up_to_a_day_ignore_everything_after_it(proc):
    dev, _ = dev3d()
    cut = 300
    fut = _scramble_dev(dev, cut)
    a, b = pc.market(dev, proc), pc.market(fut, proc)
    for field in ("can_buy", "can_sell", "adj_close", "raw_close"):
        assert np.array_equal(getattr(a, field)[:cut], getattr(b, field)[:cut], equal_nan=True), field
    # 合成分只容许浮点加法顺序带来的误差（实测约 1e-15）；缺失的位置必须完全一样
    assert np.array_equal(np.isnan(a.score[:cut]), np.isnan(b.score[:cut]))
    np.testing.assert_allclose(a.score[:cut], b.score[:cut], rtol=0, atol=1e-12)
    if proc.exit_next_day:
        assert np.array_equal(a.force_exit[:cut], b.force_exit[:cut])
    assert not np.array_equal(a.score[cut:], b.score[cut:], equal_nan=True)  # 后面确实变了
    for name in ("cheap", "high_vol", "small30"):
        assert not dev["flags"][name].iloc[cut:].equals(fut["flags"][name].iloc[cut:]), name   # 重新算过，确实变了
    ra, rb = pc.run(dev, proc), pc.run(fut, proc)
    # 第 cut 天成交用的是第 cut−1 天的信号和第 cut 天的价格：净值到第 cut−1 天、交易到第 cut−1 天都必须一样
    pd.testing.assert_series_equal(ra.daily["nav"].iloc[:cut], rb.daily["nav"].iloc[:cut])
    day = dev["close"].index[cut]
    pd.testing.assert_frame_equal(ra.trades[ra.trades["date"] < day].reset_index(drop=True),
                                  rb.trades[rb.trades["date"] < day].reset_index(drop=True))


@pytest.mark.data
@pytest.mark.parametrize("name", QLIB)
def test_qlib_factor_values_do_not_depend_on_later_data(name):
    """Qlib 算的因子：同样的股票，取数截止日不同，重叠部分必须一样（需要本地数据包）。"""
    from qt.panel import _qlib_wide

    codes = ["SH600000", "SH600519", "SZ000001", "SZ000002", "SH601318"]
    short = _qlib_wide(codes, {name: DEFS[name].formula}, "2016-01-01", "2017-06-30")[name]
    long = _qlib_wide(codes, {name: DEFS[name].formula}, "2016-01-01", "2018-12-31")[name]
    assert short.notna().to_numpy().sum() > 100
    pd.testing.assert_frame_equal(short, long.loc[short.index, short.columns], check_freq=False)


CURRENT = pc.PLAN3D.replace(missing="zero", blend="rank",
                            families={"业绩超预期": ["sue"], "便宜": ["ep_deduct", "bp", "sp_ttm"],
                                      "冷门": ["turn_12m", "gtja100", "gtja081", "gtja132"]})


@pytest.mark.parametrize("offset", [0, 5, 15])
def test_current_baseline_ignores_the_future_at_every_start(offset):
    # 复查 R001 的 S4：现行基线（排名平均、缺失按 0 计入）和错开的调仓起点都要查
    dev, _ = dev3d()
    cut = 300
    fut = _scramble_dev(dev, cut)
    a, b = pc.market(dev, CURRENT, offset), pc.market(fut, CURRENT, offset)
    k = cut - offset
    for field in ("can_buy", "can_sell", "adj_close", "raw_close"):
        assert np.array_equal(getattr(a, field)[:k], getattr(b, field)[:k], equal_nan=True), field
    assert np.array_equal(np.isnan(a.score[:k]), np.isnan(b.score[:k]))
    np.testing.assert_allclose(a.score[:k], b.score[:k], rtol=0, atol=1e-12)
    ra, rb = pc.run(dev, CURRENT, offset), pc.run(fut, CURRENT, offset)
    pd.testing.assert_series_equal(ra.daily["nav"].iloc[:k], rb.daily["nav"].iloc[:k])


@pytest.mark.parametrize("offset", [0, 5])
def test_overlay_ignores_future_months_and_future_market_data(offset):
    # 估值 + 趋势仓位：把某天之后的月度信号和行情全改掉，之前的持股只数和净值都不能变
    dev, _ = dev3d()
    months = pd.date_range("2016-01-31", "2020-01-31", freq="ME")
    rng = np.random.default_rng(1)
    sig = pd.DataFrame({"valuation": rng.uniform(size=len(months)), "trend": (rng.uniform(size=len(months)) > 0.5) * 1.0},
                       index=months)
    dev["overlay_signals"] = sig
    proc = CURRENT.replace(overlay="value_trend", cash_yield=0.02)
    cut = 300
    fut = _scramble_dev(dev, cut)
    day = dev["close"].index[cut]
    later = sig.copy()
    later.loc[later.index >= day] = rng.uniform(size=(int((later.index >= day).sum()), 2)).round()
    fut["overlay_signals"] = later
    ra, rb = pc.run(dev, proc, offset), pc.run(fut, proc, offset)
    k = cut - offset
    pd.testing.assert_series_equal(ra.daily["nav"].iloc[:k], rb.daily["nav"].iloc[:k])
    pd.testing.assert_series_equal(ra.daily["slot_limit"].iloc[:k], rb.daily["slot_limit"].iloc[:k])
