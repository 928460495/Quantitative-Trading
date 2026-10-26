"""因子合成：把留下的打分因子合成一个分数，把排雷因子合成一个排雷分。

输入的因子值都是中性化、按方向调整后的分数（每天横截面标准化过，越大越好）。
合成顺序和权重估计方法见 docs/DECISIONS.md（2026-10-26）：权重只用训练段估计，验证段只用来比较。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qt.preprocess import zscore

RANK_SCALE = np.sqrt(12)  # 0–1 均匀分布的标准差是 1/√12，乘上它方差为 1

# Qlib benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml 的参数（线程数按本机改）
LGBM_PARAMS = {
    "objective": "regression",
    "colsample_bytree": 0.8879,
    "learning_rate": 0.2,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
    "num_threads": 4,
    "seed": 0,
    "verbosity": -1,
}
LGBM_ROUNDS = 1000
LGBM_EARLY_STOP = 50


def rank_label(ret: pd.Series) -> pd.Series:
    """每天横截面排名，映射到均值约 0、方差约 1（和 Qlib 的 CSRankNorm 相同）。"""
    pct = ret.groupby(level="datetime").rank(pct=True)
    return (pct - 0.5) * RANK_SCALE


def composite(scores: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """加权求和后再做一次横截面标准化。缺个别因子的按 0（当天平均）处理，全缺的为 NaN。"""
    w = weights.reindex(scores.columns).fillna(0.0)
    total = scores.fillna(0.0) @ w
    return zscore(total.where(scores.notna().any(axis=1)))


def equal_weights(columns) -> pd.Series:
    return pd.Series(1.0 / len(columns), index=list(columns))


def icir_weights(ic: pd.DataFrame) -> pd.Series:
    """ic：逐日 IC，每列一个因子（已按方向调整）。"""
    w = ic.mean() / ic.std()
    return w / w.abs().sum()


def max_icir_weights(ic: pd.DataFrame) -> pd.Series:
    """最大化合成后 ICIR 的权重 ∝ Σ⁻¹μ。Σ 用 Ledoit-Wolf 收缩，因子相关时估计更稳。"""
    from sklearn.covariance import LedoitWolf

    x = ic.dropna()
    cov = LedoitWolf().fit(x.to_numpy()).covariance_
    w = pd.Series(np.linalg.solve(cov, x.mean().to_numpy()), index=ic.columns)
    return w / w.abs().sum()


def linear_weights(scores: pd.DataFrame, label: pd.Series) -> pd.Series:
    """混合最小二乘：标签对全部因子回归（因子和标签每天均值都约为 0，不设截距）。"""
    df = scores.fillna(0.0).join(label.rename("_y")).dropna(subset=["_y"])
    beta, *_ = np.linalg.lstsq(df[scores.columns].to_numpy(), df["_y"].to_numpy(), rcond=None)
    w = pd.Series(beta, index=scores.columns)
    return w / w.abs().sum()


def fit_lgbm(x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series):
    """训练 LightGBM，在 x_valid 上早停。返回模型和早停时的轮数。"""
    import lightgbm as lgb

    def _ds(x, y, ref=None):
        keep = y.notna()
        return lgb.Dataset(x[keep].to_numpy(np.float32), y[keep].to_numpy(np.float32), reference=ref, free_raw_data=True)

    train = _ds(x_train, y_train)
    valid = _ds(x_valid, y_valid, ref=train)
    model = lgb.train(
        LGBM_PARAMS, train, num_boost_round=LGBM_ROUNDS, valid_sets=[valid],
        callbacks=[lgb.early_stopping(LGBM_EARLY_STOP, verbose=False)],
    )
    return model, model.best_iteration


def predict_lgbm(model, x: pd.DataFrame) -> pd.Series:
    pred = pd.Series(model.predict(x.to_numpy(np.float32), num_iteration=model.best_iteration), index=x.index)
    return zscore(pred.where(x.notna().any(axis=1)))


def veto_block(veto_score: pd.Series, share: float) -> pd.Series:
    """每天排雷分最低的 share 比例标为 True（不能新买）。share 为 0 时全部 False。"""
    if share <= 0:
        return pd.Series(False, index=veto_score.index)
    pct = veto_score.groupby(level="datetime").rank(pct=True)
    return (pct <= share).fillna(False)


def smooth(score: pd.Series, days: int) -> pd.Series:
    """每只股票取最近 days 个交易日（含当天）分数的平均，再做横截面标准化。只用到过去的值。"""
    if days <= 1:
        return score
    rolled = score.groupby(level="instrument", group_keys=False).rolling(days, min_periods=1).mean()
    rolled.index = rolled.index.droplevel(0) if rolled.index.nlevels == 3 else rolled.index
    return zscore(rolled.reindex(score.index))
