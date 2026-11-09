"""计划 4 任务 5：仓位控制回测和挑选（docs/DECISIONS.md 2026-11-09）。用假数据，不碰真实数据。"""

import numpy as np
import pandas as pd
import pytest

from qt import position_control as pc
from qt.account import AccountRules


def _segment(n_days=200, n_stocks=60, crash_at=120, seed=0):
    """假的一段：中证全指在 crash_at 那天跌破 20 日均线；股票价格随机游走。和 build_composite.load_segment 的结构一样。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n_days)
    warm = pd.bdate_range(end=idx[0] - pd.Timedelta(days=1), periods=30)
    cols = [f"SH600{i:03d}" for i in range(n_stocks)]
    close = pd.DataFrame(10 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n_stocks)), axis=0)), index=idx, columns=cols)
    true = pd.DataFrame(True, index=idx, columns=cols)
    industry = pd.DataFrame(np.array([f"{10 + i % 8}" for i in range(n_stocks)], dtype=object)[None, :].repeat(n_days, 0),
                            index=idx, columns=cols)
    all_days = warm.append(idx)
    level = np.r_[np.linspace(1000, 1200, len(warm) + crash_at), np.linspace(800, 500, len(all_days) - len(warm) - crash_at)]   # 跌破后一路往下
    index = pd.DataFrame({"000985": level, "H00300": np.linspace(3000, 3300, len(all_days))}, index=all_days)
    score = pd.DataFrame(rng.normal(size=(n_days, n_stocks)), index=idx, columns=cols)
    cats = {"价值": score, "短周期价量": -score}
    return {"composite": score, "categories": cats, "close": close, "raw_close": close, "can_buy": true,
            "can_sell": true, "industry": industry, "index": index}


def test_trend_filter_halves_slots_the_day_after_index_breaks_below_ma():
    seg = _segment()
    exposure = pc.trend(seg, window=20, low=0.5)
    m, result = pc.run_variant(seg, seg["composite"], AccountRules(), exposure)
    slots = result.daily["slot_limit"]
    crash = seg["close"].index[120]
    assert exposure.loc[crash] == 0.5 and exposure.iloc[119] == 1.0      # 跌破当天判断出来
    assert slots.loc[crash] == 15                                       # 当天还是按前一天的信号
    assert slots.iloc[121] == 7                                         # 第二天收盘减到 7 只
    assert result.daily["n_holdings"].iloc[121] <= 7


def test_zero_low_means_all_cash():
    seg = _segment()
    _, result = pc.run_variant(seg, seg["composite"], AccountRules(), pc.trend(seg, window=20, low=0.0))
    assert result.daily["n_holdings"].iloc[125:].eq(0).all()
    assert result.daily["stock_value"].iloc[125:].eq(0).all()


def test_metrics_include_benchmark_yearly_and_c_triggers():
    seg = _segment()
    m, result = pc.run_variant(seg, seg["composite"], AccountRules(drawdown_stop=0.01), None)
    assert m["bench_annual"] == pytest.approx((3300 / seg["index"]["H00300"].loc[seg["close"].index[0]]) ** (252 / 199) - 1, rel=0.05)
    assert m["c_triggers"] >= 1
    yearly = pc.yearly(result)
    assert list(yearly.columns) == ["return", "max_drawdown"]
    assert yearly.index.tolist() == [2019]


def test_vol_target_comes_from_training_segment_only():
    train, valid = _segment(seed=1), _segment(seed=2)
    target = pc.vol_target(train)
    from qt.account import index_volatility

    s = index_volatility(train["index"]["000985"], 20).reindex(train["close"].index)
    assert target == pytest.approx(s.median())


@pytest.mark.parametrize("rows, want", [
    ({"A1": (-0.15, -0.10, 0.05), "A2": (-0.18, -0.19, 0.12), "A3": (-0.10, -0.12, 0.08)}, "A2"),    # 都达标：验证段收益最高
    ({"A1": (-0.15, -0.10, 0.05), "A2": (-0.25, -0.10, 0.30), "A3": (-0.10, -0.22, 0.20)}, "A1"),    # 只有 A1 两段都达标
    ({"A1": (-0.21, -0.10, 0.05), "A2": (-0.25, -0.30, 0.30)}, None),                                # 都不达标：不放宽
    ({"A1": (-0.20, -0.20, 0.05)}, "A1"),                                                             # 恰好 20% 算达标
])
def test_choose_a(rows, want):
    table = pd.DataFrame(rows, index=["train_max_drawdown", "valid_max_drawdown", "valid_annual"]).T
    assert pc.choose_a(table) == want


def test_composite_without_a_category():
    seg = _segment()
    from qt.composite import composite

    got = pc.composite_without(seg, "短周期价量")
    pd.testing.assert_frame_equal(got, composite({"价值": seg["categories"]["价值"]}))


def test_moving_average_uses_warmup_before_the_segment():
    # 段开始第 5 天就跌破 20 日均线：只有用上预热期的指数才算得出均线
    seg = _segment(crash_at=5)
    exposure = pc.trend(seg, window=20, low=0.5)
    assert exposure.iloc[5] == 0.5
