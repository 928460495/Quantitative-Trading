import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_factor_library", ROOT / "scripts" / "build_factor_library.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_health_stats():
    idx = pd.bdate_range("2019-01-02", periods=4)
    close = pd.DataFrame(1.0, index=idx, columns=list("ABCD"))
    close.iloc[0, 3] = np.nan                      # D 第一天停牌：不算进分母
    f = pd.DataFrame([[1, 2, 3, 9], [1, 1, 1, 1], [1, 2, np.inf, 4], [np.nan, 2, 3, 4]], index=idx, columns=list("ABCD"))
    h = build.health(f, close)
    assert h["coverage"] == round(13 / 15, 4)       # 15 个有价格的格子里 13 个有值（inf 不算）
    assert h["inf_share"] == round(1 / 15, 4)
    assert h["constant_days"] == 0.25               # 第 2 天四只一样
    assert h["distinct_share"] == 1.0               # 每天不同值的个数 / 有值个数，取中位数
