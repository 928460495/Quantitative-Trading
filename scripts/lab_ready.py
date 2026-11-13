"""实验室：能否进最终测试的检查（docs/FRAMEWORK.md 第 8 节第 2、3 条；具体做法见 DECISIONS 2026-11-13）。

用法（单进程，约半小时）：
    python scripts/lab_ready.py

对现行基线：
  （框架 2.1 起比夏普比率，扣 2% 无风险收益；基准是满仓的能持有等权和沪深300 全收益）
  2a 四个起点平均，三个阶段的夏普都高于同期两个基准的夏普
  2b 四个起点各自，全段夏普都高于各自同期能持有等权的夏普
  2c 成本翻倍，四个起点平均，全段夏普高于两个基准
  3  邻近参数各一档（持股 10 / 20、每 10 / 40 天调仓、跌出前 2N / 4N 名才卖），全段夏普都高于能持有等权
结果写进 lab/reports/READY-r<第几次>.md；每个版本记一条试验日志（kind = lab_ready）。
这里只判第 2、3 条；第 4 条（独立复查）和第 5 条（用户批准）另做。不碰封存段。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt.lab import procedure as pc  # noqa: E402
from qt.lab import scorecard as sc  # noqa: E402
from qt.lab.protocol import ERAS, FRAMEWORK_VERSION, FULL, noise_sharpe  # noqa: E402
from qt.trials import count_unique, log_trial  # noqa: E402

TRIALS = ROOT / "reports" / "trials.jsonl"
NEIGHBOURS = {"持股 10 只": {"n_hold": 10}, "持股 20 只": {"n_hold": 20},
              "每 10 天调仓": {"rebalance_every": 10}, "每 40 天调仓": {"rebalance_every": 40},
              "跌出前 2N 名才卖": {"sell_mult": 2}, "跌出前 4N 名才卖": {"sell_mult": 4}}


def readiness(dev: dict, base: pc.Procedure, log: Path | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """返回（每条检查一行：最小余量、满足；各版本的成绩单）。"""
    cache: dict = {}
    procs = {"基线": base, "成本翻倍": base.replace(cost_multiplier=2.0 * base.cost_multiplier),
             **{name: base.replace(**change) for name, change in NEIGHBOURS.items()}}
    versions = {}
    for name, proc in procs.items():
        card = sc.scorecard(dev, sc.run_phases(dev, proc, cache=cache))
        versions[name] = card
        if log is not None:
            log_trial("lab_ready", {"version": name, "framework": FRAMEWORK_VERSION, **proc.to_dict()},
                      {"annual_return": float(card.loc[FULL, "年化"]), "excess": float(card.loc[FULL, "超额"]),
                       "max_drawdown": float(card.loc[FULL, "最大回撤"])}, path=log)
    b, c2 = versions["基线"], versions["成本翻倍"]
    eras = [e for e in ERAS if e in b.index]
    # 框架 2.1（DECISIONS 2026-11-13）：比夏普比率（扣 2% 无风险收益），基准是满仓的能持有等权和沪深300 全收益
    margin = lambda card, row: min(card.loc[row, "夏普"] - card.loc[row, "等权夏普"],
                                   card.loc[row, "夏普"] - card.loc[row, "沪深300夏普"])
    rows = {
        "2a 三个阶段的夏普都高于两个基准": min(margin(b, e) for e in eras),
        "2b 四个起点各自的夏普高于能持有等权": float(b.loc[FULL, "最差起点夏普差"]),
        "2c 成本翻倍后夏普高于两个基准": margin(c2, FULL),
        "3 邻近参数的夏普都高于能持有等权": min(float(versions[n].loc[FULL, "夏普"] - versions[n].loc[FULL, "等权夏普"])
                                       for n in NEIGHBOURS),
    }
    checks = pd.DataFrame({"最小余量": pd.Series(rows, dtype=float)})
    checks["满足"] = checks["最小余量"] > 0
    return checks, versions


def report(checks: pd.DataFrame, versions: dict[str, pd.DataFrame], n_trials: int) -> str:
    md = lambda df: df.to_markdown(floatfmt=".4f")
    ok = bool(checks["满足"].all())
    summary = pd.DataFrame({name: card.loc[FULL, ["夏普", "等权夏普", "沪深300夏普", "最差起点夏普差", "年化", "能持有等权", "沪深300全收益",
                                                  "最大回撤", "信息比率", "M²", "索提诺", "卡尔玛", "年成本", "大亏只数"]]
                            for name, card in versions.items()}).T
    lines = [f"# 能否进最终测试（FRAMEWORK 第 8 节第 2、3 条，夏普口径）：{'全部满足' if ok else '没有全部满足'}", "",
             f"- 框架版本 {FRAMEWORK_VERSION}；具体做法见 DECISIONS 2026-11-13",
             f"- 累计试验 {n_trials} 次；这么多次试验、12 年数据，纯噪声里最好的年化夏普期望约 {noise_sharpe(max(n_trials, 1), 12):.2f}",
             "- 这些数都来自开发段（2010–2021），已经被看过很多次，不算干净的证据；第 4 条（独立复查）、第 5 条（用户批准）另做", "",
             "## 检查", "", md(checks), "", "## 各版本全段（四个起点平均）", "", md(summary), "",
             "## 基线分阶段", "", md(versions["基线"]), "", "## 成本翻倍分阶段", "", md(versions["成本翻倍"]), "",
             "## 邻近参数分阶段", ""]
    for name in NEIGHBOURS:
        lines += [f"### {name}", "", md(versions[name]), ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    from qt.lab import cards as cd
    from qt.lab.store import load_dev
    from run_plan3c_train import POOL_CACHE, latest_version

    base = cd.load_baseline(ROOT / "lab" / "baseline.yaml")
    if not cd.is_committed(ROOT / "lab" / "baseline.yaml"):
        sys.exit("lab/baseline.yaml 有没提交的改动")
    if not cd.code_is_committed(ROOT):
        sys.exit("qt/ 或 scripts/ 有没提交的改动：先提交代码再运行")
    print("读取开发段数据…", flush=True)
    dev = load_dev(latest_version(POOL_CACHE), base.factors())
    checks, versions = readiness(dev, base, log=TRIALS)
    out = ROOT / "lab" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    k = len(list(out.glob("READY-r*.md"))) + 1
    (out / f"READY-r{k}.md").write_text(report(checks, versions, count_unique(TRIALS)), encoding="utf-8")
    print(checks.to_string())
    sys.exit(0 if checks["满足"].all() else 1)


if __name__ == "__main__":
    main()
