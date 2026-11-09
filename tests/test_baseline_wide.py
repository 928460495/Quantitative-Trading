"""计划 4 任务 6：Alpha158 + LightGBM 对照基准（全 A 宽表版）。假数据，不碰真实数据。"""

import numpy as np
import pandas as pd
import pytest

from qt import baseline_wide as bw
from qt.combine import rank_label


def _long(n_days=300, n_stocks=40, n_feat=5, seed=0, start="2015-01-05"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    codes = [f"SH600{i:03d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, codes], names=["datetime", "instrument"])
    x = pd.DataFrame(rng.normal(size=(len(idx), n_feat)), index=idx, columns=[f"F{i}" for i in range(n_feat)])
    ret5 = 0.02 * x["F0"] + rng.normal(0, 0.02, len(idx))              # F0 有信号
    return x.astype("float32"), ret5.rename("ret5")


def test_split_drops_the_gap_before_early_stop_and_at_the_end():
    dates = pd.bdate_range("2015-12-01", "2017-01-31")
    fit, stop = bw.split_dates(dates, fit_end="2015-12-31", stop_start="2016-01-01", stop_end="2016-12-31", gap=5)
    assert fit.max() == pd.bdate_range("2015-12-01", "2015-12-31")[-6]  # 训练段最后 5 天不用
    assert stop.min() == pd.Timestamp("2016-01-01")
    assert stop.max() == pd.bdate_range("2016-01-01", "2016-12-31")[-6]  # 早停段最后 5 天的标签会看到段外
    assert fit.intersection(stop).empty


def test_labels_rank_only_buyable_rows():
    x, ret5 = _long()
    can_buy = pd.Series(np.random.default_rng(1).random(len(x)) > 0.3, index=x.index)
    xs, y = bw.training_rows(x, ret5, can_buy, x.index.get_level_values("datetime").unique()[:50])
    assert len(xs) == len(y) and y.notna().all()
    assert can_buy.reindex(xs.index).all()
    want = rank_label(ret5.where(can_buy)).reindex(y.index)
    pd.testing.assert_series_equal(y, want.astype("float32"), check_names=False)


def test_model_learns_the_planted_feature_and_predicts_every_stock():
    x, ret5 = _long()
    can_buy = pd.Series(True, index=x.index)
    dates = x.index.get_level_values("datetime").unique()
    fit, stop = dates[:200], dates[210:260]
    model, best = bw.train(x, ret5, can_buy, fit, stop)
    assert best >= 1
    imp = pd.Series(model.feature_importance("gain"), index=x.columns)
    assert imp.idxmax() == "F0"
    pred_x = x[x.index.get_level_values("datetime") >= dates[270]]
    wide = bw.predict_wide(model, pred_x)
    assert wide.shape == (30, 40)                                       # 每天、每只股票都有分（不只能买的）
    assert wide.mean(axis=1).abs().max() < 1e-6                         # 每天标准化过
    ic = wide.corrwith(ret5.loc[pred_x.index].unstack("instrument"), axis=1, method="spearman").mean()
    assert ic > 0.3


def test_standardize_by_day_matches_long_version():
    from qt.preprocess import standardize

    x, _ = _long(n_days=5)
    pd.testing.assert_frame_equal(bw.standardize_by_day(x), standardize(x.astype("float64")).astype("float32"))


def test_year_windows_have_warmup_and_stay_out_of_test_segment():
    w = bw.year_window(2021)
    assert w == ("2020-09-01", "2021-12-31")                             # 前面多取 4 个月给 60 日窗口预热
    with pytest.raises(ValueError):
        bw.year_window(2022)
