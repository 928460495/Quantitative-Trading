import numpy as np
import pandas as pd

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
