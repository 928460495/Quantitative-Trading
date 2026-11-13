"""计划 4 任务 4：算合成分并存下来。用假数据从头跑到尾，不碰真实数据。"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qt.composite import category_scores, composite
from qt.screen_wide import factor_score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_composite", ROOT / "scripts" / "build_composite.py")
bc = importlib.util.module_from_spec(spec)
sys.modules["build_composite"] = bc
spec.loader.exec_module(bc)

from test_screen_wide import _ctx  # noqa: E402


def _selected():
    return pd.DataFrame({"category": ["价值", "价值", "动量/反转"], "direction": [1, -1, -1]},
                        index=pd.Index(["v1", "v2", "m1"], name="factor"))


def _raws(ctx):
    rng = np.random.default_rng(5)
    shape, idx, cols = ctx.close.shape, ctx.close.index, ctx.close.columns
    return {n: pd.DataFrame(rng.normal(size=shape), index=idx, columns=cols) for n in ["v1", "v2", "m1"]}


def test_build_segment_writes_scores_composite_and_market(tmp_path):
    from qt.screen_wide import ScreenContext

    base = _ctx()
    can_buy = base.can_buy.astype(object)
    can_buy.iloc[100:110, :5] = None                                        # 真实数据里会有缺失
    ctx = ScreenContext(close=base.close, raw_close=base.raw_close, can_buy=can_buy, can_sell=base.can_sell,
                        industry=base.industry, log_mcap=base.log_mcap, start=str(base.close.index[60].date()), tail=0)
    raws = _raws(ctx)
    index = {"000985": base.close.mean(axis=1), "H00300": base.close.iloc[:, :50].mean(axis=1)}
    total = bc.build_segment(ctx, _selected(), lambda name: raws[name], tmp_path, index, meta={"segment": "train"})

    sel = _selected()
    scores = {n: factor_score(raws[n], sel.loc[n, "category"], int(sel.loc[n, "direction"]), ctx) for n in sel.index}
    want = composite(category_scores(scores, sel["category"].to_dict()))
    pd.testing.assert_frame_equal(total, want, check_dtype=False, atol=1e-6)
    assert total.index[0] == ctx.dates[0]                                    # 预热期不在结果里

    back = bc.load_segment(tmp_path)
    pd.testing.assert_frame_equal(back["composite"], total, check_dtype=False, check_freq=False, atol=1e-6)
    assert set(back["categories"]) == {"价值", "动量/反转"}
    assert set(back["factors"]) == {"v1", "v2", "m1"}
    for k in ("close", "raw_close", "can_buy", "can_sell", "industry"):
        assert back[k].index.equals(total.index), k
    assert back["can_buy"].dtypes.eq(bool).all()
    assert back["index"].index[0] == base.close.index[0]                     # 指数保留预热期（算均线要用）
    assert json.loads((tmp_path / "meta.json").read_text())["segment"] == "train"


def test_selected_list_is_read_from_csv(tmp_path):
    csv = tmp_path / "selected.csv"
    pd.DataFrame({"factor": ["b", "a"], "category": ["价值", "成长"], "direction": [1, -1], "icir": [0.3, 0.2],
                  "family": ["x", "y"]}).to_csv(csv, index=False)
    sel = bc.read_selected(csv)
    assert sel.index.tolist() == ["b", "a"] and sel.loc["a", "direction"] == -1
    assert list(sel.columns) == ["category", "direction"]


def test_warmup_is_one_year_before_start():
    assert bc.warmup_start("2019-01-01") == "2018-01-01"
    assert bc.warmup_start("2010-01-01") == "2009-01-05"                  # baostock 日线从 2009-01-05 开始
