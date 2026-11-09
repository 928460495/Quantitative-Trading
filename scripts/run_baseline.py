"""计划 4 任务 6：对照基准 Alpha158 + LightGBM（全 A 主板），在验证段上按同样的账户规则、同样的仓位控制回测。

用法（先跑过 scripts/build_composite.py；单进程）：
    python scripts/run_baseline.py

规则见 docs/DECISIONS.md 2026-11-09：
- 特征：Alpha158 的 158 个因子每天横截面去极值、标准化（不中性化）；按年算、算完一年存一年（可断点续跑）
- 标签：5 天远期收益（T+1 收盘买入）在第二天能买进的股票里的横截面排名；只给 2010–2018 算标签
  （2019–2021 只要预测，不算标签：2021 年底的 5 天收益会看到测试段的价格）
- 训练：2010–2016 每天（去掉最后 21 个交易日），2017–2018 早停（去掉最后 21 个交易日），参数 qt/combine.LGBM_PARAMS
- 验证段：每天每只股票打分，按和合成分完全一样的账户规则回测：不控仓、A 的 6 个组合、B
输出 reports/combination/full_a/baseline.md、baseline.csv；中间结果在 data/cache/baseline/<数据版本>/（不进仓库）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt import baseline_wide as bw  # noqa: E402
from qt import position_control as pc  # noqa: E402
from qt.account import AccountRules  # noqa: E402
from qt.trials import TRIALS_FILE, log_trial  # noqa: E402
from qt.wide_eval import forward_returns, rank_ic  # noqa: E402

OUT_DIR = ROOT / "reports" / "combination" / "full_a"
CACHE_ROOT = ROOT / "data" / "cache" / "baseline"
TRAIN_YEARS = tuple(range(2010, 2019))
PRED_YEARS = (2019, 2020, 2021)
FIT_END, STOP_START, STOP_END = "2016-12-31", "2017-01-01", "2018-12-31"
NO_CONTROL, B_NAME = "不控仓", "B_波动率控制"


def year_features(year: int, codes: list[str], with_label: bool, cache_dir: Path,
                  load: Callable[[list[str], int, bool], pd.DataFrame]) -> pd.DataFrame:
    """这一年的特征（已标准化，float32）和标签用的 ret5（with_label 时）。算过就读缓存。"""
    path = cache_dir / f"features_{year}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    t0 = time.time()
    df = load(codes, year, with_label)
    feats = [c for c in df.columns if c != "ret5"]
    out = bw.standardize_by_day(df[feats])
    if with_label:
        out["ret5"] = df["ret5"].astype("float32")
    tmp = path.with_name(f".{path.name}")
    out.to_parquet(tmp)
    tmp.replace(path)
    print(f"  特征 {year}：{len(out):,} 行，{(time.time() - t0) / 60:.1f} 分钟", flush=True)
    return out


def _stack_bool(wide: pd.DataFrame) -> pd.Series:
    s = wide.astype(bool).stack()
    s.index.names = ["datetime", "instrument"]
    return s


def build_training(train_years, codes, cache_dir, load, can_buy_wide: pd.DataFrame,
                   fit_end=FIT_END, stop_start=STOP_START, stop_end=STOP_END):
    """按年读特征，取训练、早停的行，直接拼成 float32 数组（不留整段的 DataFrame，省内存）。"""
    fit_dates, stop_dates = bw.split_dates(can_buy_wide.index, fit_end, stop_start, stop_end)
    can_buy = _stack_bool(can_buy_wide)
    parts = {"fit": ([], []), "stop": ([], [])}
    names = None
    for year in train_years:
        df = year_features(year, codes, True, cache_dir, load)
        x, ret5 = df.drop(columns="ret5"), df["ret5"]
        names = list(x.columns)
        for key, dates in (("fit", fit_dates), ("stop", stop_dates)):
            xs, ys = bw.training_rows(x, ret5, can_buy, dates)
            if len(xs):
                parts[key][0].append(xs.to_numpy(np.float32))
                parts[key][1].append(ys.to_numpy(np.float32))
        del df, x, ret5
    cat = lambda key, i: np.concatenate(parts[key][i]) if parts[key][i] else np.empty((0, len(names)), np.float32)
    return cat("fit", 0), cat("fit", 1), cat("stop", 0), cat("stop", 1), names


def predict(model, pred_years, codes, cache_dir, load, dates, columns) -> pd.DataFrame:
    frames = []
    for year in pred_years:
        x = year_features(year, codes, False, cache_dir, load)
        frames.append(bw.predict_wide(model, x[[c for c in x.columns if c != "ret5"]]))
    return pd.concat(frames).reindex(index=dates, columns=columns)


def backtest(valid: dict, train: dict, score: pd.DataFrame, trials_path: Path | None, version: str) -> pd.DataFrame:
    target = pc.vol_target(train)
    variants = [(NO_CONTROL, AccountRules(drawdown_stop=None), None, {"control": "none"})]
    variants += [(pc.a_name(w, low), AccountRules(), pc.trend(valid, w, low), {"control": "A+C", "window": w, "low": low})
                 for w in pc.A_WINDOWS for low in pc.A_LOWS]
    variants += [(B_NAME, AccountRules(), pc.vol(valid, target), {"control": "B+C", "window": pc.B_WINDOW})]
    rows = []
    for name, rules, exposure, params in variants:
        m, _ = pc.run_variant(valid, score, rules, exposure)
        rows.append({"variant": name, **m})
        if trials_path is not None:
            log_trial("baseline", {"model": "alpha158_lgbm_full_a", "variant": name, "segment": "valid",
                                   "train": "daily_2010_2016", "data_version": version, **params},
                      {k: m[k] for k in ("annual_return", "excess_annual", "max_drawdown", "sharpe")}, path=trials_path)
    return pd.DataFrame(rows).set_index("variant")


def valid_rank_ic(score: pd.DataFrame, valid: dict) -> float:
    """验证段 5 天 RankIC（只在第二天能买进的股票里；验证段最后几天的远期收益缺失，不会看到测试段）。"""
    fwd = forward_returns(valid["close"], [5])[5]
    return float(rank_ic(score, fwd, valid["can_buy"].astype(bool)).mean())


def run(train: dict, valid: dict, codes_by_year: dict[int, list[str]], cache_dir: Path, load,
        trials_path: Path | None = TRIALS_FILE, version: str = "", train_years=TRAIN_YEARS, pred_years=PRED_YEARS,
        fit_end=FIT_END, stop_start=STOP_START, stop_end=STOP_END) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("训练集 ...", flush=True)
    xf, yf, xs, ys, names = build_training(train_years, codes_by_year, cache_dir, load, train["can_buy"],
                                           fit_end, stop_start, stop_end)
    print(f"  训练 {len(yf):,} 行、早停 {len(ys):,} 行、{len(names)} 个特征；开始训练 LightGBM ...", flush=True)
    t0 = time.time()
    model, best = bw.fit_arrays(xf, yf, xs, ys, names)
    del xf, yf, xs, ys
    print(f"  早停在第 {best} 轮，用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    model.save_model(str(cache_dir / "model.txt"), num_iteration=best)
    score = predict(model, pred_years, codes_by_year, cache_dir, load, valid["close"].index, valid["close"].columns)
    score.astype("float32").to_parquet(cache_dir / "score_valid.parquet")
    metrics = backtest(valid, train, score, trials_path, version)
    importance = pd.Series(model.feature_importance("gain"), index=names).sort_values(ascending=False)
    ic = {"对照基准": valid_rank_ic(score, valid), "合成分": valid_rank_ic(valid["composite"], valid)}
    return {"metrics": metrics, "best_iteration": best, "importance": importance, "valid_ic": ic, "score": score}


def write_report(res: dict, out_dir: Path = OUT_DIR, ours: pd.DataFrame | None = None, note: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = res["metrics"]
    m.to_csv(out_dir / "baseline.csv", float_format="%.6f")
    pct = lambda x: "" if pd.isna(x) else f"{x:.1%}"
    cols = ["annual_return", "bench_annual", "max_drawdown", "bench_max_drawdown", "avg_exposure", "turnover", "cost_annual"]
    show = m[cols].copy()
    if ours is not None:
        o = ours[ours["segment"] == "valid"].set_index("variant")
        show.insert(1, "合成分年化", o["annual_return"].reindex(show.index))
        show.insert(4, "合成分最大回撤", o["max_drawdown"].reindex(show.index))
    show = show.map(pct).rename(columns={"annual_return": "对照年化", "bench_annual": "沪深300全收益", "max_drawdown": "对照最大回撤",
                                         "bench_max_drawdown": "沪深300同期最大回撤", "avg_exposure": "对照平均仓位",
                                         "turnover": "对照日均换手", "cost_annual": "对照年成本"})
    ic = res["valid_ic"]
    lines = ["# 对照基准：Alpha158 + LightGBM（全 A 主板，验证段）", "",
             "规则见 `docs/DECISIONS.md` 2026-11-09。账户规则、仓位控制和合成分完全一样，只换选股打分。"
             "训练 2010–2016 每天、2017–2018 早停（各去掉最后 21 个交易日）。", ""]
    if note:
        lines += [note, ""]
    lines += [f"LightGBM 早停在第 {res['best_iteration']} 轮。", "",
              f"验证段 5 天 RankIC（第二天能买进的股票里）：对照基准 {ic['对照基准']:.4f}，合成分 {ic['合成分']:.4f}。", "",
              "## 验证段回测（2019–2021）", "", show.to_markdown(), "",
              "## 特征重要性前 20（按增益）", "",
              res["importance"].head(20).round(1).to_frame("增益").to_markdown(), ""]
    (out_dir / "baseline.md").write_text("\n".join(lines) + "\n")


def qlib_loader(codes_by_year: dict[int, list[str]]):
    """真实数据：按年用 Qlib 算 Alpha158（和标签）。"""
    from qlib.data import D

    from qt.data import init_qlib
    from qt.factors import alpha158, forward_return_expr

    init_qlib()
    feats = {f.name: f.expr for f in alpha158()}

    def load(_codes, year: int, with_label: bool) -> pd.DataFrame:
        exprs = feats | ({"ret5": forward_return_expr(5)} if with_label else {})
        start, end = bw.year_window(year)
        df = D.features(codes_by_year[year], list(exprs.values()), start_time=start, end_time=end)
        df.columns = list(exprs)
        df = df.swaplevel().sort_index()
        df.index.names = ["datetime", "instrument"]
        df = df[df.index.get_level_values("datetime").year == year]
        return df.replace([np.inf, -np.inf], np.nan).astype("float32")

    return load


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", help="data/cache/composite/ 下的数据版本目录，默认最新的")
    args = ap.parse_args()
    from build_composite import load_segment
    from run_position_control import find_cache

    cache = find_cache(args.version)
    print(f"读取 {cache}", flush=True)
    segs = {s: load_segment(cache / s) for s in ("train", "valid")}
    codes = {y: list(segs["train" if y in TRAIN_YEARS else "valid"]["close"].columns) for y in (*TRAIN_YEARS, *PRED_YEARS)}
    res = run(segs["train"], segs["valid"], codes, CACHE_ROOT / cache.name, qlib_loader(codes), version=cache.name)
    ours_path = OUT_DIR / "backtest.csv"
    ours = pd.read_csv(ours_path) if ours_path.exists() else None
    write_report(res, ours=ours, note=f"数据版本 `{cache.name}`。")
    print(res["metrics"][["annual_return", "bench_annual", "max_drawdown", "bench_max_drawdown"]].round(4).to_string())
    print(f"验证段 RankIC：{json.dumps({k: round(v, 4) for k, v in res['valid_ic'].items()}, ensure_ascii=False)}")
    print(f"报告：{(OUT_DIR / 'baseline.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
