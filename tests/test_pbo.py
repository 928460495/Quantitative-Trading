"""计划 3c 任务 7：CSCV 估计回测过拟合概率（Bailey、Borwein、López de Prado、Zhu 2017）。"""

import numpy as np
import pandas as pd
import pytest

from qt.pbo import cscv_pbo


def _noise(n_days, n_cfg, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.01, (n_days, n_cfg)), index=pd.bdate_range("2010-01-04", periods=n_days))


def test_dominating_config_has_low_pbo():
    r = _noise(1600, 20, 1)
    r[0] += 0.002                                      # 方案 0 每天都多赚 0.2%
    out = cscv_pbo(r)
    assert out["pbo"] < 0.05


def test_pure_noise_pbo_near_half():
    out = cscv_pbo(_noise(1600, 200, 2))
    assert 0.3 < out["pbo"] < 0.7


def test_combination_count_and_logits():
    out = cscv_pbo(_noise(320, 5, 3), n_blocks=16)
    assert out["n_combos"] == 12870
    assert len(out["logits"]) == 12870


def test_odd_blocks_or_too_short_raises():
    with pytest.raises(ValueError):
        cscv_pbo(_noise(320, 5, 4), n_blocks=15)
    with pytest.raises(ValueError):
        cscv_pbo(_noise(10, 5, 5), n_blocks=16)
