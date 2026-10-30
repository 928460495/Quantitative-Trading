"""函数形式的因子：价量补充，以及估值、财务、规模因子。输入是 qt/panel.py 的宽表字典。

口径（docs/DATA_NOTES.md）：
- baostock 估值（市盈率、市净率、市销率、市现率）用前一个交易日反推的每股数据，除以当天的价格：
  公告当天收盘后的估值已经用了新财报，直接用会偷看一天
- 财务数据在宽表里已经按首次公告日的下一个交易日起可用
- 总股本 = 净利润 / 每股收益（同一期累计值），总市值 = 未复权收盘价 × 总股本；拿不到时为缺失
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from qt import ops

Panel = dict[str, pd.DataFrame]


def _pos(x: pd.DataFrame) -> pd.DataFrame:
    return x.where(x > 0)


def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return (a / b).replace([np.inf, -np.inf], np.nan)


def _market_returns(p: Panel) -> pd.DataFrame:
    return p["benchmark_close"] / p["benchmark_close"].shift(1) - 1


# ---------- 动量 / 反转 ----------

def _ret(d: int):
    def f(p: Panel) -> pd.DataFrame:
        return p["close"] / p["close"].shift(d) - 1
    f.__name__ = f"ret{d}"
    return f


def mom12_1(p: Panel) -> pd.DataFrame:
    """12 个月动量，去掉最近 1 个月。"""
    return p["close"].shift(21) / p["close"].shift(252) - 1


def overnight20(p: Panel) -> pd.DataFrame:
    """20 日累计隔夜收益（前收到今开）。"""
    return ops.ts_sum(np.log(p["open"] / p["close"].shift(1)), 20)


def intraday20(p: Panel) -> pd.DataFrame:
    """20 日累计日内收益（今开到今收）。"""
    return ops.ts_sum(np.log(p["close"] / p["open"]), 20)


def high52w(p: Panel) -> pd.DataFrame:
    return p["close"] / ops.ts_max(p["high"], 250)


def low52w(p: Panel) -> pd.DataFrame:
    return p["close"] / ops.ts_min(p["low"], 250)


def ideal_amp20(p: Panel, n: int = 20, share: float = 0.25) -> pd.DataFrame:
    """开源证券"理想振幅"：过去 n 天里收盘价最高的 25% 天的平均振幅，减去收盘价最低的 25% 天的平均振幅。"""
    close = p["close"].to_numpy(dtype=float)
    amp = (p["high"] / p["low"] - 1).to_numpy(dtype=float)
    k = max(int(round(n * share)), 1)
    out = np.full(close.shape, np.nan)
    for c0 in range(0, close.shape[1], 256):
        c1 = min(c0 + 256, close.shape[1])
        wc = sliding_window_view(close[:, c0:c1], n, axis=0)      # (T-n+1, C, n)
        wa = sliding_window_view(amp[:, c0:c1], n, axis=0)
        order = np.argsort(wc, axis=2)
        sorted_amp = np.take_along_axis(wa, order, axis=2)
        val = sorted_amp[..., -k:].mean(axis=2) - sorted_amp[..., :k].mean(axis=2)
        bad = np.isnan(wc).any(axis=2) | np.isnan(wa).any(axis=2)
        out[n - 1:, c0:c1] = np.where(bad, np.nan, val)
    return pd.DataFrame(out, index=p["close"].index, columns=p["close"].columns)


# ---------- 低波动 ----------

def _vol(d: int):
    def f(p: Panel) -> pd.DataFrame:
        return ops.ts_std(p["returns"], d)
    f.__name__ = f"vol{d}"
    return f


def _ivol(d: int):
    def f(p: Panel) -> pd.DataFrame:
        """相对沪深300 的 CAPM 残差波动：var(残差) = var(r) − cov(r, m)² / var(m)。"""
        r, m = p["returns"], _market_returns(p)
        var_r = ops.ts_std(r, d) ** 2
        cov = ops.ts_cov(r, m, d)
        var_m = ops.ts_std(m, d) ** 2
        return np.sqrt((var_r - cov ** 2 / var_m).clip(lower=0))
    f.__name__ = f"ivol{d}"
    return f


def beta252(p: Panel) -> pd.DataFrame:
    m = _market_returns(p)
    return _safe_div(ops.ts_cov(p["returns"], m, 252), ops.ts_std(m, 252) ** 2)


def _max(d: int):
    def f(p: Panel) -> pd.DataFrame:
        return ops.ts_max(p["returns"], d)
    f.__name__ = f"max{d}"
    return f


def _skew(d: int):
    def f(p: Panel) -> pd.DataFrame:
        return p["returns"].rolling(d, min_periods=d).skew()
    f.__name__ = f"skew{d}"
    return f


def range20(p: Panel) -> pd.DataFrame:
    """20 日平均振幅（最高减最低，除以前收）。"""
    return ops.ts_mean((p["high"] - p["low"]) / p["close"].shift(1), 20)


# ---------- 流动性 ----------

def turn_1m(p: Panel) -> pd.DataFrame:
    """Barra STOM：近 21 天换手率之和的对数。"""
    return np.log(_pos(ops.ts_sum(p["turn"] / 100, 21)))


def turn_3m(p: Panel) -> pd.DataFrame:
    """Barra STOQ：近 3 个月每月换手率平均的对数。"""
    return np.log(_pos(ops.ts_sum(p["turn"] / 100, 63) / 3))


def turn_12m(p: Panel) -> pd.DataFrame:
    """Barra STOA：近 12 个月每月换手率平均的对数。"""
    return np.log(_pos(ops.ts_sum(p["turn"] / 100, 252) / 12))


def abn_turn(p: Panel) -> pd.DataFrame:
    """Liu, Stambaugh & Yuan 异常换手：近 20 日平均换手 / 近 250 日平均换手。"""
    return _safe_div(ops.ts_mean(p["turn"], 20), ops.ts_mean(p["turn"], 250))


def turn_std20(p: Panel) -> pd.DataFrame:
    return ops.ts_std(p["turn"], 20)


def turn_cv20(p: Panel) -> pd.DataFrame:
    """换手率稳定性：20 日换手率标准差 / 均值。"""
    return _safe_div(ops.ts_std(p["turn"], 20), ops.ts_mean(p["turn"], 20))


def _amihud(d: int):
    def f(p: Panel) -> pd.DataFrame:
        return ops.ts_mean(_safe_div(p["returns"].abs(), _pos(p["amount"])), d)
    f.__name__ = f"amihud{d}"
    return f


def ret_turn_corr20(p: Panel) -> pd.DataFrame:
    return ops.ts_corr(p["returns"], p["turn"], 20)


# ---------- 规模 ----------

def log_float_mcap(p: Panel) -> pd.DataFrame:
    return np.log(_pos(p["float_mcap"]))


def _total_mcap(p: Panel) -> pd.DataFrame:
    shares = _pos(_safe_div(p["net_profit"], p["eps"]))
    return p["raw_close"] * shares


def log_total_mcap(p: Panel) -> pd.DataFrame:
    return np.log(_pos(_total_mcap(p)))


def nlsize(p: Panel) -> pd.DataFrame:
    """Barra NLSIZE：标准化后的对数市值取立方，再对对数市值做横截面回归取残差。"""
    x = log_float_mcap(p)
    z = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
    y = z ** 3
    zc = z.sub(z.mean(axis=1), axis=0)
    yc = y.sub(y.mean(axis=1), axis=0)
    beta = (yc * zc).sum(axis=1) / (zc * zc).sum(axis=1)
    return yc - zc.mul(beta, axis=0)


def price_level(p: Panel) -> pd.DataFrame:
    """低价效应：未复权价格的对数。"""
    return np.log(_pos(p["raw_close"]))


# ---------- 价值 ----------

def _lagged_yield(key: str):
    def f(p: Panel) -> pd.DataFrame:
        per_share = _safe_div(p["raw_close"], p[key])      # 反推的每股收益（净资产、销售额、现金流）
        return _safe_div(per_share.shift(1), p["raw_close"])
    return f


ep_ttm = _lagged_yield("pe_ttm")
bp = _lagged_yield("pb_mrq")
sp_ttm = _lagged_yield("ps_ttm")
cfp_ttm = _lagged_yield("pcf_ttm")


def ep_deduct(p: Panel) -> pd.DataFrame:
    return _safe_div(p["deduct_net_profit_ttm"], _total_mcap(p))


def ep_self(p: Panel) -> pd.DataFrame:
    return _safe_div(p["net_profit_ttm"], _total_mcap(p))


# ---------- 成长 ----------

def revenue_yoy(p: Panel) -> pd.DataFrame:
    return p["revenue_yoy"]


def profit_yoy(p: Panel) -> pd.DataFrame:
    return p["net_profit_yoy"]


def deduct_yoy(p: Panel) -> pd.DataFrame:
    return p["deduct_net_profit_yoy"]


def sue(p: Panel) -> pd.DataFrame:
    return p["net_profit_sue"]


# ---------- 质量 ----------

def roe_ttm(p: Panel) -> pd.DataFrame:
    return _safe_div(p["net_profit_ttm"], _pos(p["total_equity"]))


def roa_ttm(p: Panel) -> pd.DataFrame:
    return _safe_div(p["net_profit_ttm"], _pos(p["total_assets"]))


def gross_margin_ttm(p: Panel) -> pd.DataFrame:
    return _safe_div(p["revenue_ttm"] - p["operate_cost_ttm"], _pos(p["revenue_ttm"]))


def net_margin_ttm(p: Panel) -> pd.DataFrame:
    return _safe_div(p["net_profit_ttm"], _pos(p["revenue_ttm"]))


def cfo_to_np(p: Panel) -> pd.DataFrame:
    return _safe_div(p["op_cash_flow_ttm"], _pos(p["net_profit_ttm"]))


def accruals(p: Panel) -> pd.DataFrame:
    return _safe_div(p["net_profit_ttm"] - p["op_cash_flow_ttm"], _pos(p["total_assets"]))


def debt_ratio(p: Panel) -> pd.DataFrame:
    return _safe_div(p["total_liab"], _pos(p["total_assets"]))


def asset_growth(p: Panel) -> pd.DataFrame:
    return p["total_assets_yoy"]


# 带窗口参数的因子，给出固定名字方便直接调用
ret5, ret20, ret60, ret120 = _ret(5), _ret(20), _ret(60), _ret(120)
vol20, vol60 = _vol(20), _vol(60)
ivol20, ivol60 = _ivol(20), _ivol(60)
max5, max20 = _max(5), _max(20)
skew20, skew60 = _skew(20), _skew(60)
amihud5, amihud20, amihud60 = _amihud(5), _amihud(20), _amihud(60)
