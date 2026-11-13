"""计划 4 任务 6 的运行脚本：假数据从头跑到尾（不碰真实数据、不调 Qlib）。"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qt.wide_eval import forward_returns

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_baseline", ROOT / "scripts" / "run_baseline.py")
rb = importlib.util.module_from_spec(spec)
sys.modules["run_baseline"] = rb
spec.loader.exec_module(rb)

from test_position_control import _segment  # noqa: E402


def _seg(start, n_days, seed):
    seg = _segment(n_days=n_days, n_stocks=60, seed=seed)
    new_idx = pd.bdate_range(start, periods=n_days)
    for k in ("composite", "close", "raw_close", "can_buy", "can_sell", "industry"):
        seg[k].index = new_idx
    seg["categories"] = {c: v.set_axis(new_idx) for c, v in seg["categories"].items()}
    warm = pd.bdate_range(end=new_idx[0] - pd.Timedelta(days=1), periods=len(seg["index"]) - n_days)
    seg["index"].index = warm.append(new_idx)
    return seg


def _fake_loader(segs, calls):
    rng = np.random.default_rng(9)

    def load(codes, year, with_label):
        calls.append((year, with_label))
        seg = segs["valid"] if year == 2017 else segs["train"]
        close = seg["close"]
        fwd = forward_returns(close, [5])[5]
        days = close.index[close.index.year == year]
        idx = pd.MultiIndex.from_product([days, close.columns], names=["datetime", "instrument"])
        f0 = fwd.loc[days].stack(future_stack=True).reindex(idx).fillna(0) * 50 + rng.normal(0, 1, len(idx))
        df = pd.DataFrame({"F0": f0, **{f"F{i}": rng.normal(size=len(idx)) for i in range(1, 5)}}, index=idx)
        if with_label:
            df["ret5"] = fwd.loc[days].stack(future_stack=True).reindex(idx)
        return df.astype("float32")

    return load


def test_end_to_end(tmp_path):
    segs = {"train": _seg("2015-01-01", 520, 1), "valid": _seg("2017-01-02", 250, 2)}
    calls = []
    codes = {y: list(segs["train"]["close"].columns) for y in (2015, 2016, 2017)}
    trials = tmp_path / "trials.jsonl"
    kw = dict(train_years=(2015, 2016), pred_years=(2017,), fit_end="2015-12-31", stop_start="2016-01-01",
              stop_end="2016-12-31")
    res = rb.run(segs["train"], segs["valid"], codes, tmp_path / "cache", _fake_loader(segs, calls), trials, "test", **kw)

    assert (2017, True) not in calls                                    # 验证年不算标签
    assert res["best_iteration"] >= 1 and res["importance"].idxmax() == "F0"
    assert res["valid_ic"]["对照基准"] > 0.2
    assert res["score"].index.equals(segs["valid"]["close"].index)
    m = res["metrics"]
    assert len(m) == 1 + 6 + 1 and "bench_max_drawdown" in m.columns
    logged = [json.loads(l) for l in trials.read_text().splitlines()]
    assert len(logged) == 8 and {r["kind"] for r in logged} == {"baseline"}

    ours = pd.DataFrame({"variant": m.index, "segment": "valid", "annual_return": 0.1, "max_drawdown": -0.2})
    rb.write_report(res, tmp_path / "out", ours=ours)
    text = (tmp_path / "out" / "baseline.md").read_text()
    assert "合成分年化" in text and "早停在第" in text

    calls.clear()                                                       # 第二次：特征全从缓存读
    rb.run(segs["train"], segs["valid"], codes, tmp_path / "cache", _fake_loader(segs, calls), None, "test", **kw)
    assert calls == []
