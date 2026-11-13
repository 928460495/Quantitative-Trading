"""计划 3c 任务 10：验证段（2019–2021）只跑一次，按五条成功标准预演。

用法（先跑过 scripts/run_plan3c_train.py 且所有检查通过、写出了 final.json；单进程）：
    python scripts/run_plan3c_valid.py

reports/plan3c/valid/README.md 已存在就拒绝运行：验证段在计划 4 已看过两次，3c 只能最后确认一次。
规则见 docs/DECISIONS.md 2026-11-10「计划 3c 事先登记的规则」：
- 固定方案 + 回撤分档；另报"不做回撤控制"、成本翻倍
- 对照 Alpha158 + LightGBM：打分用同样的平滑天数，账户规则（N、调仓间隔、卖出名次、行业上限、回撤分档）完全一样
- 五条标准；DSR 的试验次数 = 试验日志里不同配置数（含计划 4）
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt import walkforward as wf  # noqa: E402
from qt.account import TRADING_DAYS, run_account  # noqa: E402
from qt.diagnostics import BENCH, annualize, criteria, diagnose, write_report  # noqa: E402
from qt.fastsim import SimParams  # noqa: E402
from qt.trials import count_unique, deflated_sharpe, load_unique, log_trial  # noqa: E402

OUT_DIR = ROOT / "reports" / "plan3c" / "valid"
FINAL = ROOT / "reports" / "plan3c" / "train" / "final.json"
TRIALS = ROOT / "reports" / "trials.jsonl"
VALID = ("2019-01-01", "2021-12-31")


def load_final(path: Path = FINAL) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"没有 {path}：训练段脚本没跑完，或有检查没通过")
    return json.loads(path.read_text())


def _sharpe_var(trials: Path) -> float:
    """各次试验每日夏普的方差（日志里记的是年化夏普）。"""
    s = [r["metrics"]["sharpe"] for r in load_unique(trials) if r.get("metrics", {}).get("sharpe") is not None]
    s = np.asarray([x for x in s if np.isfinite(x)], dtype=float) / np.sqrt(TRADING_DAYS)
    return float(s.var(ddof=1)) if len(s) > 1 else 0.0


def run(valid: dict, pool: pd.DataFrame, final: dict, baseline_score: pd.DataFrame, out_dir: Path, trials: Path) -> int:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"{out_dir} 里已经有东西（跑过，或上次跑到一半）：验证段只能跑一次，拒绝运行")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "STARTED").write_text("验证段已开始运行；之后不能再跑\n")   # 先占位：中途崩了也不能重跑
    p = SimParams(**final["params"])
    k, w, factors = int(final["smooth"]), float(final["lowvol_weight"]), final["factors"]
    rules = wf.account_rules(p)
    score = wf.score_for(valid, pool, factors, VALID, k, w)
    data = wf.market(valid, score, VALID)

    main = run_account(data, rules)
    bare = run_account(data, dataclasses.replace(rules, drawdown_tiers=None))
    cost2x = run_account(data, dataclasses.replace(rules, cost_multiplier=2.0))
    base = baseline_score.loc[VALID[0]:VALID[1]]
    base = base.rolling(k, min_periods=1).mean().where(base.notna())
    baseline = run_account(wf.market(valid, base, VALID), rules)

    nav = main.daily["nav"]
    bench = valid["index"][BENCH]
    navs = {"方案（回撤分档）": nav, "方案（不做回撤控制，只报告）": bare.daily["nav"],
            "方案（成本翻倍）": cost2x.daily["nav"], "对照 Alpha158 + LightGBM": baseline.daily["nav"]}
    table = pd.DataFrame({name: {"年化": annualize(s), "最大回撤": float((s / s.cummax() - 1).min())}
                          for name, s in navs.items()}).T
    crit = criteria(nav, bench, cost2x.daily["nav"], annualize(baseline.daily["nav"]))

    pd.DataFrame(navs).to_csv(out_dir / "backtest.csv")
    for name, s in navs.items():
        log_trial("plan3c_valid", {"variant": name, "segment": "valid", **final},
                  {"annual_return": annualize(s), "max_drawdown": float((s / s.cummax() - 1).min())}, path=trials)
    n_trials = count_unique(trials)
    dsr = deflated_sharpe(nav.pct_change().dropna(), n_trials, _sharpe_var(trials))
    sections = {"五条成功标准（验证段预演）": crit, "各方案": table,
                "DSR": pd.Series({**dsr, "试验次数": n_trials}).to_frame("值"),
                **{f"诊断：{n}": t for n, t in diagnose(main, valid, score, p.rebalance_every).items()}}
    notes = ["规则见 docs/DECISIONS.md 2026-11-10「计划 3c 事先登记的规则」",
             "**验证段在计划 4 已看过两次**，这里的结论会偏乐观；之后只有最终测试是干净的检验",
             f"方案：{', '.join(factors)}；{wf.label(p, k, w)}",
             "五条都满足：" + ("是" if crit["满足"].all() else "否")]
    write_report(sections, out_dir / "README.md", "计划 3c 验证段结果（只跑一次）", notes)
    print(crit.to_string())
    return 0


def main() -> None:
    from build_composite import load_segment
    from build_pool import read_pool
    from run_plan3c_train import POOL_CACHE, latest_version

    final = load_final()
    version = latest_version(POOL_CACHE)
    valid = load_segment(version / "valid")
    baseline = pd.read_parquet(ROOT / "data" / "cache" / "baseline" / version.name / "score_valid.parquet")
    sys.exit(run(valid, read_pool(), final, baseline, OUT_DIR, TRIALS))


if __name__ == "__main__":
    main()
