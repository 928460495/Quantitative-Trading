"""单因子检验和因子筛选共用的数据准备。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qt.external import float_shares_on, industry_on, load_external
from qt.neutralize import impute_log_mcap, neutralize
from qt.preprocess import standardize

# 中性化要用到的字段，和因子一起取
EXPOSURE_EXPRS: dict[str, str] = {
    "_raw_close": "$close/$factor",
    "_log_amount20": "Log(Mean($amount,20))",
}


def exposures(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, float, dict[str, float]]:
    """行业、对数流通市值（缺股本的用成交额估算）、覆盖率、估算质量。df 需要包含 EXPOSURE_EXPRS 的列。"""
    ind_table, share_table = load_external()
    industry = industry_on(df.index, ind_table)
    shares = float_shares_on(df.index, share_table)
    log_amount = df["_log_amount20"].replace([np.inf, -np.inf], np.nan)
    log_mcap, quality = impute_log_mcap(np.log(df["_raw_close"] * shares), log_amount)
    coverage = float((industry.notna() & log_mcap.notna()).mean())
    return industry, log_mcap, coverage, quality


def neutral_score(raw: pd.Series, industry: pd.Series, log_mcap: pd.Series) -> pd.Series:
    """在当天全部成分股上标准化再中性化：已持有但第二天买不进的股票也需要排名，才能决定留还是卖。"""
    return neutralize(standardize(raw), industry, log_mcap)
