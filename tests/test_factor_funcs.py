import numpy as np
import pandas as pd
import pytest

from qt import factor_funcs as ff
from qt.library import FUNDAMENTAL, PRICE_EXTRA


def _panel(n_days=300, n_stocks=8, seed=0):
    idx = pd.bdate_range("2017-01-02", periods=n_days)
    cols = [f"SH60{i:04d}" for i in range(n_stocks)]
    rng = np.random.default_rng(seed)
    frame = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    close = frame(10 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n_stocks)), axis=0)))
    p = {"close": close, "raw_close": close, "open": close * rng.uniform(0.98, 1.02, close.shape)}
    p["high"] = np.maximum(p["open"], close) * 1.01
    p["low"] = np.minimum(p["open"], close) * 0.99
    p["returns"] = close.pct_change()
    p["amount"] = frame(rng.uniform(1e4, 1e5, close.shape))
    p["volume"] = p["amount"] / close
    p["turn"] = frame(rng.uniform(0.5, 3.0, close.shape))
    p["float_mcap"] = close * frame(np.tile(rng.uniform(1e8, 1e10, n_stocks), (n_days, 1)))
    bench = 3000 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
    p["benchmark_close"] = frame(np.tile(bench[:, None], (1, n_stocks)))
    for k, lo, hi in [("pe_ttm", 5, 50), ("pb_mrq", 0.5, 5), ("ps_ttm", 0.5, 10), ("pcf_ttm", 3, 30)]:
        p[k] = frame(np.tile(rng.uniform(lo, hi, n_stocks), (n_days, 1))) * close / close.iloc[0]
    for k in ["net_profit_ttm", "revenue_ttm", "op_cash_flow_ttm", "deduct_net_profit_ttm", "operate_cost_ttm",
              "total_assets", "total_liab", "total_equity", "net_profit", "eps"]:
        p[k] = frame(np.tile(rng.uniform(1e7, 1e9, n_stocks), (n_days, 1)))
    p["revenue_ttm"] = p["operate_cost_ttm"] * 1.3
    p["total_assets"] = p["total_liab"] + p["total_equity"]
    for k in ["net_profit_yoy", "revenue_yoy", "deduct_net_profit_yoy", "net_profit_sue", "total_assets_yoy"]:
        p[k] = frame(rng.normal(0, 0.2, close.shape))
    return p


def test_valuation_uses_previous_day_per_share_value():
    p = _panel()
    p["pe_ttm"] = p["raw_close"] / 2.0          # 每股收益 2
    p["pe_ttm"].iloc[100:] = p["raw_close"].iloc[100:] / 3.0   # 第 100 天公告后变成 3
    ep = ff.ep_ttm(p)
    assert ep.iloc[100, 0] == pytest.approx(2.0 / p["raw_close"].iloc[100, 0])   # 公告当天还用旧值
    assert ep.iloc[101, 0] == pytest.approx(3.0 / p["raw_close"].iloc[101, 0])


def test_abnormal_turnover():
    p = _panel()
    p["turn"] = p["turn"] * 0 + 1.0
    p["turn"].iloc[-20:] = 2.0
    v = ff.abn_turn(p).iloc[-1]
    assert v.iloc[0] == pytest.approx(2.0 / ((230 * 1 + 20 * 2) / 250))


def test_beta_and_idiosyncratic_vol():
    p = _panel(n_days=400)
    m = p["benchmark_close"].pct_change()
    noise = pd.DataFrame(np.random.default_rng(3).normal(0, 0.01, m.shape), index=m.index, columns=m.columns)
    p["returns"] = 2 * m + noise
    assert ff.beta252(p).iloc[-1].mean() == pytest.approx(2.0, abs=0.1)
    assert ff.ivol60(p).iloc[-1].mean() == pytest.approx(0.01, rel=0.2)


def test_ideal_amplitude_high_minus_low_price_days():
    p = _panel(n_days=40, n_stocks=2)
    close = pd.DataFrame(np.tile(np.arange(40, dtype=float)[:, None] % 20 + 10, (1, 2)), index=p["close"].index,
                         columns=p["close"].columns)
    amp = np.where(close >= 25, 0.10, 0.02)      # 20 天里价格最高的 5 天（25%）振幅 10%，其余 2%
    p["close"] = close
    p["low"] = close
    p["high"] = close * (1 + pd.DataFrame(amp, index=close.index, columns=close.columns))
    v = ff.ideal_amp20(p).iloc[-1]
    assert v.iloc[0] == pytest.approx(0.10 - 0.02)


def test_nlsize_is_orthogonal_to_size():
    p = _panel()
    nl = ff.nlsize(p)
    size = np.log(p["float_mcap"])
    for d in [50, 150]:
        assert np.corrcoef(nl.iloc[d], size.iloc[d])[0, 1] == pytest.approx(0, abs=1e-8)


def test_registry_counts_and_categories():
    names = [f.name for f in PRICE_EXTRA + FUNDAMENTAL]
    assert len(names) == len(set(names)) and len(names) >= 50
    assert {f.category for f in PRICE_EXTRA + FUNDAMENTAL} == {"动量/反转", "价值", "规模", "低波动", "成长", "质量", "流动性"}


@pytest.mark.parametrize("factor", PRICE_EXTRA + FUNDAMENTAL, ids=lambda f: f.name)
def test_every_function_runs(factor):
    p = _panel()
    out = factor.func(p)
    assert out.shape == p["close"].shape
    assert out.notna().to_numpy().sum() > 0
