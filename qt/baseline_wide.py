"""计划 4 任务 6：对照基准 Alpha158 + LightGBM 在全 A 主板上（docs/DECISIONS.md 2026-11-09）。

- 特征：Qlib Alpha158 的 158 个因子，每天横截面去极值、标准化（不中性化），float32
- 标签：5 天远期收益（T+1 收盘买入）的横截面排名，只在第二天能买进的股票里排
- 训练：2010–2016 每天（去掉最后 21 个交易日），2017–2018 早停（同样去掉最后 21 个交易日：标签会看到段外），
  参数 qt/combine.LGBM_PARAMS
- 预测：验证段每天、每只股票都打分（已持有但第二天买不进的也要排名，才能决定卖不卖），每天标准化

特征按年算（每年前面多取 4 个月给 60 日窗口预热），算一年存一年，内存只装得下一年也能跑。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qt.combine import LGBM_EARLY_STOP, LGBM_PARAMS, LGBM_ROUNDS, rank_label
from qt.config import SEGMENTS
from qt.preprocess import standardize

WARMUP_MONTHS = 4      # Alpha158 最长窗口 60 个交易日，约 3 个月
GAP = 21               # 训练段和早停段之间、早停段末尾各去掉的交易日数


def year_window(year: int) -> tuple[str, str]:
    """这一年取数的起止（含预热）。不能碰测试段。"""
    test_start = pd.Timestamp(SEGMENTS["test"][0])
    end = pd.Timestamp(f"{year}-12-31")
    if end >= test_start:
        raise ValueError(f"{year} 年碰到了测试段（{test_start.date()} 起）")
    start = pd.Timestamp(f"{year}-01-01") - pd.DateOffset(months=WARMUP_MONTHS)
    return str(start.date()), str(end.date())


def standardize_by_day(x: pd.DataFrame) -> pd.DataFrame:
    """每天横截面去极值（中位数 ± 3 倍 MAD）、标准化；同 qt.preprocess.standardize（长表）。"""
    return standardize(x.astype("float64")).astype("float32")


def split_dates(dates: pd.DatetimeIndex, fit_end: str, stop_start: str, stop_end: str, gap: int = GAP
                ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """训练日期（到 fit_end，去掉最后 gap 天）和早停日期（stop_start ~ stop_end，去掉最后 gap 天）。"""
    dates = pd.DatetimeIndex(dates).sort_values()
    fit = dates[dates <= pd.Timestamp(fit_end)]
    stop = dates[(dates >= pd.Timestamp(stop_start)) & (dates <= pd.Timestamp(stop_end))]
    return fit[:-gap] if gap else fit, stop[:-gap] if gap else stop


def training_rows(x: pd.DataFrame, ret5: pd.Series, can_buy: pd.Series, dates: pd.DatetimeIndex
                  ) -> tuple[pd.DataFrame, pd.Series]:
    """这些日期里第二天能买进、标签有值的行；标签 = 能买的股票里 5 天收益的横截面排名。"""
    buy = can_buy.reindex(x.index, fill_value=False).astype(bool)
    label = rank_label(ret5.where(buy)).astype("float32")
    keep = buy & label.notna() & x.index.get_level_values("datetime").isin(dates)
    return x[keep], label[keep]


def train(x: pd.DataFrame, ret5: pd.Series, can_buy: pd.Series, fit: pd.DatetimeIndex, stop: pd.DatetimeIndex):
    import lightgbm as lgb

    xf, yf = training_rows(x, ret5, can_buy, fit)
    xs, ys = training_rows(x, ret5, can_buy, stop)
    return fit_arrays(xf.to_numpy(np.float32), yf.to_numpy(np.float32), xs.to_numpy(np.float32),
                      ys.to_numpy(np.float32), list(x.columns))


def fit_arrays(xf: np.ndarray, yf: np.ndarray, xs: np.ndarray, ys: np.ndarray, names: list[str]):
    """直接从 float32 数组训练（真实数据按年拼起来，不经过 DataFrame，省内存）。返回模型和早停轮数。"""
    import lightgbm as lgb

    dtrain = lgb.Dataset(xf, yf, feature_name=names, free_raw_data=True)
    dstop = lgb.Dataset(xs, ys, reference=dtrain, free_raw_data=True)
    model = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=LGBM_ROUNDS, valid_sets=[dstop],
                      callbacks=[lgb.early_stopping(LGBM_EARLY_STOP, verbose=False)])
    return model, model.best_iteration


def predict_wide(model, x: pd.DataFrame) -> pd.DataFrame:
    """每天、每只股票的打分（宽表），每天标准化；全部特征都缺的股票为缺失。"""
    pred = pd.Series(model.predict(x.to_numpy(np.float32), num_iteration=model.best_iteration), index=x.index)
    pred = pred.where(x.notna().any(axis=1))
    wide = pred.unstack("instrument")
    return wide.sub(wide.mean(axis=1), axis=0).div(wide.std(axis=1), axis=0)
