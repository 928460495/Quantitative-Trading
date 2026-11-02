"""用假的小宽表把 scripts/run_full_a_screen.py 从头跑到尾（不碰真实数据）。"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qt.library import FactorDef
from qt.wide_eval import forward_returns

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_full_a_screen", ROOT / "scripts" / "run_full_a_screen.py")
script = importlib.util.module_from_spec(spec)
sys.modules["run_full_a_screen"] = script   # 子进程按名字找函数
spec.loader.exec_module(script)

from test_screen_wide import _ctx  # noqa: E402


def _factors(ctx):
    fwd = forward_returns(ctx.close, [5])[5]
    shape, idx, cols = fwd.shape, fwd.index, fwd.columns
    noise = lambda seed: pd.DataFrame(np.random.default_rng(seed).normal(size=shape), index=idx, columns=cols)
    values = {
        "planted": fwd.rank(axis=1) + 0.2 * noise(1).rank(axis=1),
        "planted_twin": fwd.rank(axis=1) + 0.25 * noise(2).rank(axis=1),     # 和 planted 很像 → 被去掉
        "noise": noise(3),
        "lnsize": ctx.log_mcap,
        "qlib_like": -fwd,                                                   # 主进程成批算，方向为负
    }
    defs = [FactorDef("planted", "price_extra", "动量/反转", "测试", func=lambda p: values["planted"]),
            FactorDef("planted_twin", "price_extra", "动量/反转", "测试", func=lambda p: values["planted_twin"]),
            FactorDef("noise", "price_extra", "价值", "测试", func=lambda p: values["noise"]),
            FactorDef("lnsize", "fundamental", "规模", "测试", func=lambda p: values["lnsize"]),
            FactorDef("broken", "price_extra", "质量", "测试", func=lambda p: 1 / 0),
            FactorDef("qlib_like", "alpha158", "短周期价量", "测试", formula="$close", dialect="qlib")]
    compute = lambda f: f.func(None)
    parent_compute = lambda batch: {f.name: values[f.name] for f in batch}
    return defs, compute, parent_compute


@pytest.mark.parametrize("jobs", [1, 2])
def test_end_to_end(tmp_path, jobs):
    ctx = _ctx()
    defs, compute, parent_compute = _factors(ctx)
    trials = tmp_path / "trials.jsonl"
    summary, subs = script.run_screen(ctx, defs, compute, jobs, parent_compute,
                                      on_result=script.trial_logger("test", trials))
    assert summary.index.tolist() == [f.name for f in defs]            # 按登记顺序
    assert summary.loc["planted", "passed"] and summary.loc["planted_twin", "passed"]
    assert summary.loc["qlib_like", "passed"] and summary.loc["qlib_like", "direction"] == -1
    assert not summary.loc["noise", "passed"]
    assert summary.loc["broken", "error"].startswith("ZeroDivisionError")
    assert not summary.loc["lnsize", "size_neutralized"]                 # 规模类只对行业中性化

    selection, corr = script.select(summary, subs)
    kept = selection.index[selection["kept"].astype(bool)].tolist()
    assert "planted" in kept and "planted_twin" not in kept and "qlib_like" in kept
    assert selection.loc["planted_twin", "dropped_by"] == "planted"
    assert list(corr.index) == kept

    script.write_reports(summary, selection, corr, "测试样本", tmp_path / "single", tmp_path / "select")
    assert len(pd.read_csv(tmp_path / "single" / "summary.csv")) == len(defs)
    sel = pd.read_csv(tmp_path / "select" / "selected.csv", index_col=0)
    assert sel.index.tolist() == kept
    readme = (tmp_path / "select" / "README.md").read_text()
    assert "空着的大类" in readme and "价值" in readme
    assert "计算出错" in (tmp_path / "single" / "README.md").read_text()

    logged = [json.loads(line) for line in trials.read_text().splitlines()]
    assert len(logged) == len(defs) - 1                                 # 出错的不记
    assert {r["params"]["market"] for r in logged} == {"full_a_main"}


def test_resume_from_cache_skips_finished_factors(tmp_path):
    ctx = _ctx()
    defs, compute, parent_compute = _factors(ctx)
    cache = tmp_path / "cache"
    trials = tmp_path / "trials.jsonl"
    first, subs1 = script.run_screen(ctx, defs, compute, 1, parent_compute,
                                     on_result=script.trial_logger("test", trials), cache_dir=cache)
    n_logged = len(trials.read_text().splitlines())

    def boom(*_):
        raise AssertionError("已经算完的不该再算")

    # 出错的 broken 没存，会再试一次；其余都从缓存读
    retry = [f for f in defs if f.name == "broken"]
    second, subs2 = script.run_screen(ctx, defs, lambda f: compute(f) if f in retry else boom(), 1, boom,
                                      on_result=script.trial_logger("test", trials), cache_dir=cache)
    cols = ["direction", "rank_ic_5d", "t_nw_5d", "passed", "excess_annual"]
    pd.testing.assert_frame_equal(second[cols], first[cols], check_dtype=False)
    assert set(subs2) == set(subs1)
    pd.testing.assert_frame_equal(subs2["planted"], subs1["planted"], check_freq=False)
    assert len(trials.read_text().splitlines()) == n_logged          # 缓存里的不重记试验日志


def test_interrupted_run_keeps_finished_factors(tmp_path):
    ctx = _ctx()
    defs, compute, parent_compute = _factors(ctx)
    cache = tmp_path / "cache"

    def stop_at_noise(f):
        if f.name == "noise":
            raise KeyboardInterrupt
        return compute(f)

    with pytest.raises(KeyboardInterrupt):
        script.run_screen(ctx, defs, stop_at_noise, 1, parent_compute, cache_dir=cache)
    assert {p.stem for p in cache.glob("*.json")} == {"planted", "planted_twin"}
