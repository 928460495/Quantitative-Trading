import numpy as np
import pandas as pd
import pytest

from qt.neutralize import neutralize


def _panel(seed=0, n_days=5, n_stocks=120):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n_days)
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, codes], names=["datetime", "instrument"])
    industry = pd.Series(np.tile(np.arange(n_stocks) % 6, n_days).astype(str), index=idx)
    size = pd.Series(rng.normal(10, 1, len(idx)), index=idx)
    noise = pd.Series(rng.normal(size=len(idx)), index=idx)
    return idx, industry, size, noise


def test_residual_is_orthogonal_to_industry_and_size():
    idx, industry, size, noise = _panel()
    factor = 0.8 * (size - 10) + industry.astype(int) * 0.5 + noise
    out = neutralize(factor, industry, size)
    df = pd.DataFrame({"r": out, "size": size, "ind": industry})
    for _, g in df.groupby(level="datetime"):
        assert abs(g["r"].corr(g["size"])) < 1e-8
        assert g.groupby("ind")["r"].mean().abs().max() < 1e-8
        assert abs(g["r"].mean()) < 1e-8 and abs(g["r"].std() - 1) < 1e-8


def test_missing_exposures_give_nan():
    idx, industry, size, noise = _panel()
    industry.iloc[0] = np.nan
    size.iloc[1] = np.nan
    out = neutralize(noise, industry, size)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].notna().all()


def test_pure_industry_factor_is_removed():
    idx, industry, size, noise = _panel()
    factor = industry.astype(int).astype(float) * 3 + noise
    # 行业部分被完全吸收：结果和只对噪声做中性化一样
    assert np.allclose(neutralize(factor, industry, size), neutralize(noise, industry, size))


def test_small_industries_are_merged():
    from qt.neutralize import OTHER, merge_small_industries

    idx = pd.MultiIndex.from_product([pd.bdate_range("2016-01-04", periods=2), [f"S{i}" for i in range(12)]],
                                     names=["datetime", "instrument"])
    ind = pd.Series((["A"] * 8 + ["B"] * 3 + ["C"]) * 2, index=idx)
    out = merge_small_industries(ind, min_size=5)
    assert out.value_counts().to_dict() == {"A": 16, OTHER: 8}


def test_single_stock_industry_is_not_zeroed_after_merge():
    idx, industry, size, noise = _panel()
    industry = industry.copy()
    lonely = idx[::120]  # 每天第一只股票单独成一个行业
    industry.loc[lonely] = "lonely"
    raw = neutralize(noise, industry, size, min_industry_size=1)
    merged = neutralize(noise, industry, size, min_industry_size=5)
    assert np.allclose(raw.loc[lonely], 0, atol=1e-8)  # 不合并：残差被吸收成 0
    assert (merged.loc[lonely].abs() > 1e-6).all()


def test_impute_log_mcap_fills_missing_from_amount():
    from qt.neutralize import impute_log_mcap

    idx, _, _, noise = _panel(seed=1)
    log_amount = pd.Series(np.random.default_rng(2).normal(12, 1, len(idx)), index=idx)
    log_mcap = 1.5 + 0.9 * log_amount + 0.1 * noise
    hidden = log_mcap.copy()
    hidden.iloc[::10] = np.nan
    filled, quality = impute_log_mcap(hidden, log_amount)
    assert filled.notna().all()
    assert np.allclose(filled.iloc[1::10], log_mcap.iloc[1::10])  # 有数据的不动
    assert (filled.iloc[::10] - log_mcap.iloc[::10]).abs().max() < 0.5
    assert quality["imputed_share"] == pytest.approx(0.1)
    assert quality["fit_corr_mean"] > 0.99
