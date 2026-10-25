"""行业和市值中性化：每个交易日把因子对申万一级行业哑变量和对数流通市值做横截面回归，取残差。

残差再做一次横截面标准化。缺行业或缺市值的股票当天不参与（结果为 NaN）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qt.preprocess import zscore

MIN_STOCKS = 30


def neutralize(factor: pd.Series, industry: pd.Series, log_mcap: pd.Series) -> pd.Series:
    df = pd.DataFrame({"f": factor, "ind": industry, "size": log_mcap}).dropna()
    out = []
    for _, g in df.groupby(level="datetime"):
        if len(g) < MIN_STOCKS:
            continue
        dummies = pd.get_dummies(g["ind"], dtype=float)
        size = (g["size"] - g["size"].mean()) / g["size"].std() if g["size"].std() > 0 else g["size"] * 0
        x = np.column_stack([dummies.to_numpy(), size.to_numpy()])
        y = g["f"].to_numpy()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        out.append(pd.Series(y - x @ beta, index=g.index))
    resid = pd.concat(out) if out else pd.Series(dtype=float)
    return zscore(resid.reindex(factor.index))
