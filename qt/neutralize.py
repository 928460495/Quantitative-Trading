"""行业和市值中性化：每个交易日把因子对申万一级行业哑变量和对数流通市值做横截面回归，取残差。

- 当天股票数少于 MIN_INDUSTRY_SIZE 的行业合并成一个"其他"组：只有一两只股票的行业，哑变量会把它们的
  因子值整个吸收掉（残差恒为 0），这些股票就再也排不到前面了。"其他"组本身也可能很小，所以它不设
  哑变量，不单独估计行业效应（相当于按全体平均处理），只控制市值
- 残差再做一次横截面标准化
- 缺行业或缺市值的股票当天不参与（结果为 NaN）

impute_log_mcap 用成交额估算缺失的流通市值（巨潮不保留退市公司的股本记录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qt.preprocess import zscore

MIN_STOCKS = 30
MIN_INDUSTRY_SIZE = 5
OTHER = "其他"


def merge_small_industries(industry: pd.Series, min_size: int = MIN_INDUSTRY_SIZE) -> pd.Series:
    """每天把股票数少于 min_size 的行业改成"其他"。"""
    size = industry.groupby([industry.index.get_level_values("datetime"), industry]).transform("size")
    return industry.where(size >= min_size, OTHER).where(industry.notna())


def impute_log_mcap(log_mcap: pd.Series, log_amount: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    """缺失的对数流通市值，用当天"对数市值 ~ 对数成交额"的横截面回归估算。

    返回补齐后的序列和质量指标：估算比例、每天拟合值与真实值相关系数的均值和最小值。
    """
    df = pd.DataFrame({"m": log_mcap, "a": log_amount})
    out = df["m"].copy()
    corrs = []
    for _, g in df.groupby(level="datetime"):
        known = g.dropna()
        missing = g["m"].isna() & g["a"].notna()
        if len(known) < MIN_STOCKS:
            continue
        slope, intercept = np.polyfit(known["a"], known["m"], 1)
        corrs.append(np.corrcoef(slope * known["a"] + intercept, known["m"])[0, 1])
        if missing.any():
            out.loc[missing[missing].index] = slope * g.loc[missing, "a"] + intercept
    quality = {
        "imputed_share": float((df["m"].isna() & out.notna()).mean()),
        "fit_corr_mean": float(np.mean(corrs)) if corrs else np.nan,
        "fit_corr_min": float(np.min(corrs)) if corrs else np.nan,
    }
    return out, quality


def neutralize(factor: pd.Series, industry: pd.Series, log_mcap: pd.Series, min_industry_size: int = MIN_INDUSTRY_SIZE) -> pd.Series:
    df = pd.DataFrame({"f": factor, "ind": industry, "size": log_mcap}).dropna()
    df["ind"] = merge_small_industries(df["ind"], min_industry_size)
    out = []
    for _, g in df.groupby(level="datetime"):
        if len(g) < MIN_STOCKS:
            continue
        dummies = pd.get_dummies(g["ind"], dtype=float).drop(columns=OTHER, errors="ignore")
        size = (g["size"] - g["size"].mean()) / g["size"].std() if g["size"].std() > 0 else g["size"] * 0
        x = np.column_stack([dummies.to_numpy(), size.to_numpy()])
        y = g["f"].to_numpy()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        out.append(pd.Series(y - x @ beta, index=g.index))
    resid = pd.concat(out) if out else pd.Series(dtype=float)
    return zscore(resid.reindex(factor.index))
