"""按更正后的基准重新判第 ③ 条、重新筛选（DECISIONS 2026-11-11）。用假的小宽表，不碰真实数据。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qt.diagnostics import holdable_benchmark
from qt.wide_eval import buyable_benchmark

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("rescreen_full_a", ROOT / "scripts" / "rescreen_full_a.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)

from test_run_full_a_screen import _factors, script  # noqa: E402
from test_screen_wide import _ctx  # noqa: E402


def _summary():
    return pd.DataFrame({
        "category": ["价值", "价值", "质量", "成长", "规模"],
        "pass_t": [True, True, True, False, True],
        "pass_stable": [True, True, True, True, False],
        "annual_return": [0.02, -0.05, -0.003, np.nan, np.nan],
        "bench_annual": [-0.0986, -0.0986, -0.0986, np.nan, np.nan],
        "excess_annual": [0.1186, 0.0486, 0.0956, np.nan, np.nan],
        "excess_t": [2.0, 0.5, 1.0, np.nan, np.nan],
        "pass_cost": [True, True, True, False, False],
        "passed": [True, True, True, False, False],
        "fail_reason": ["", "", "", "t 值 ≤ 3", "方向反了的年份过半"],
    }, index=pd.Index(["good", "thin", "edge", "weak", "flip"], name="factor"))


def test_annualized_skips_first_day_like_account_excess():
    r = pd.Series([0.5, 0.1, 0.1], index=pd.bdate_range("2010-01-04", periods=3))
    assert rs.annualized(r) == pytest.approx(1.21 ** (252 / 2) - 1)


def test_rescreen_only_changes_the_third_check():
    out = rs.rescreen(_summary(), bench_annual=-0.003)
    assert out["passed"].tolist() == [True, False, False, False, False]      # 等于基准不算跑赢
    assert out.loc["good", "excess_annual"] == pytest.approx(0.023)
    assert out.loc["thin", "excess_annual"] == pytest.approx(-0.047)
    assert out.loc["thin", "fail_reason"] == "扣成本后跑输能持有等权" and not out.loc["thin", "pass_cost"]
    assert out.loc["weak", "fail_reason"] == "t 值 ≤ 3" and out.loc["flip", "fail_reason"] == "方向反了的年份过半"
    assert (out["bench_annual"].dropna() == -0.003).all()
    assert out.loc["thin", "bench_annual_old"] == -0.0986 and out.loc["thin", "excess_annual_old"] == 0.0486
    assert out.loc["thin", "excess_t_old"] == 0.5 and out["excess_t"].isna().all()   # 没按新基准重算，留空
    assert out[["pass_t", "pass_stable"]].equals(_summary()[["pass_t", "pass_stable"]])


def test_rescreen_still_requires_the_first_two_checks():
    s = _summary()
    s.loc["weak", "annual_return"] = 0.30                 # 第 ① 条没过的因子，账户再好也不算通过
    out = rs.rescreen(s, bench_annual=-0.003)
    assert not out.loc["weak", "passed"] and out.loc["weak", "fail_reason"] == "t 值 ≤ 3"


def test_rescreen_does_not_touch_the_input():
    s = _summary()
    rs.rescreen(s, bench_annual=0.0)
    pd.testing.assert_frame_equal(s, _summary())


def _screened(tmp_path):
    ctx = _ctx()
    defs, compute, parent_compute = _factors(ctx)
    summary, subs = script.run_screen(ctx, defs, compute, 1, parent_compute)
    return ctx, summary, subs


def test_run_rewrites_reports_against_holdable_benchmark(tmp_path):
    ctx, summary, subs = _screened(tmp_path)
    close, can_buy = ctx.trim(ctx.close), ctx.trim(ctx.can_buy).fillna(False).astype(bool)
    code = rs.run(summary, subs, close, can_buy, tmp_path / "single", tmp_path / "select", "测试样本", every=20)
    assert code == 0
    new = pd.read_csv(tmp_path / "single" / "summary.csv", index_col=0)
    want = rs.annualized(holdable_benchmark(close, can_buy, 20))
    assert new["bench_annual"].dropna().unique() == pytest.approx([want], abs=1e-6)
    ran = new["annual_return"].notna()
    assert (new.loc[ran, "pass_cost"] == (new.loc[ran, "annual_return"] > want)).all()
    cands = pd.read_csv(tmp_path / "select" / "all_candidates.csv", index_col=0)
    assert set(cands.index) == set(new.index[new["passed"]])
    readme = (tmp_path / "single" / "README.md").read_text()
    assert "2026-11-11" in readme and "每 20 个交易日" in readme


def test_run_stops_when_old_benchmark_cannot_be_reproduced(tmp_path):
    ctx, summary, subs = _screened(tmp_path)
    close, can_buy = ctx.trim(ctx.close), ctx.trim(ctx.can_buy).fillna(False).astype(bool)
    assert rs.annualized(buyable_benchmark(close, can_buy)) == pytest.approx(summary["bench_annual"].dropna().iloc[0])
    wrong = close.iloc[:-30]                                       # 区间对不上：旧基准重算不出缓存里的数
    code = rs.run(summary, subs, wrong, can_buy.iloc[:-30], tmp_path / "single", tmp_path / "select", "测试样本")
    assert code == 1 and not (tmp_path / "single").exists()


def test_run_drops_factors_that_only_beat_the_old_benchmark(tmp_path):
    ctx, summary, subs = _screened(tmp_path)
    close, can_buy = ctx.trim(ctx.close), ctx.trim(ctx.can_buy).fillna(False).astype(bool)
    bench = rs.annualized(holdable_benchmark(close, can_buy, 20))
    s = summary.copy()
    s.loc["planted", "annual_return"] = bench - 0.01                # 原来通过，新基准下差 1 个点
    rs.run(s, subs, close, can_buy, tmp_path / "single", tmp_path / "select", "测试样本")
    cands = pd.read_csv(tmp_path / "select" / "all_candidates.csv", index_col=0)
    assert "planted" not in cands.index and "planted_twin" in cands.index
    sel = pd.read_csv(tmp_path / "select" / "selected.csv", index_col=0)
    assert "planted_twin" in sel.index                              # 原来被 planted 挤掉，现在补上
