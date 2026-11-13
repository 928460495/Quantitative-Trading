"""计划 3c 任务 6：59 个已通过因子的数据缓存和"买不买得起"检查。用假数据，不碰真实数据。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for name in ("build_composite", "build_pool"):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
bc, bp = sys.modules["build_composite"], sys.modules["build_pool"]

from test_screen_wide import _ctx  # noqa: E402


def test_read_pool_has_59_factors_with_direction_and_category():
    pool = bp.read_pool()
    assert len(pool) == 59 and pool.index.is_unique
    assert list(pool.columns) == ["category", "direction"]
    assert set(pool["direction"]) <= {1, -1}
    assert pool.loc["ep_ttm", "category"] == "价值"


def test_build_segment_saves_log_mcap_when_asked(tmp_path):
    from qt.screen_wide import ScreenContext

    base = _ctx()
    ctx = ScreenContext(close=base.close, raw_close=base.raw_close, can_buy=base.can_buy, can_sell=base.can_sell,
                        industry=base.industry, log_mcap=base.log_mcap, start=str(base.close.index[60].date()), tail=0)
    sel = pd.DataFrame({"category": ["价值"], "direction": [1]}, index=pd.Index(["v1"], name="factor"))
    raw = pd.DataFrame(np.random.default_rng(1).normal(size=base.close.shape), index=base.close.index,
                       columns=base.close.columns)
    index = {"000985": base.close.mean(axis=1), "H00300": base.close.mean(axis=1)}
    bc.build_segment(ctx, sel, lambda n: raw, tmp_path, index, save_log_mcap=True)
    back = bc.load_segment(tmp_path)
    pd.testing.assert_frame_equal(back["log_mcap"], ctx.trim(base.log_mcap), check_dtype=False, check_freq=False)
    bc.build_segment(ctx, sel, lambda n: raw, tmp_path / "no", index)
    assert "log_mcap" not in bc.load_segment(tmp_path / "no")


def test_lot_feasibility_share_by_year():
    idx = pd.bdate_range("2019-12-30", periods=4)              # 跨两年
    raw = pd.DataFrame({"a": [60.0] * 4, "b": [10.0] * 4}, index=idx)
    can_buy = pd.DataFrame(True, index=idx, columns=raw.columns)
    got = bp.lot_feasibility(raw, can_buy, nav=100_000, n_hold=20)   # 一手 6,000 > 5,000 买不起
    assert list(got.index) == [2019, 2020]
    assert got.loc[2019] == pytest.approx(0.5) and got.loc[2020] == pytest.approx(0.5)
    can_buy["a"] = False
    assert bp.lot_feasibility(raw, can_buy, nav=100_000, n_hold=20).max() == 0.0
