"""实验室：跑一张假设卡的账户检验，按冻结的规则算出采纳与否（docs/FRAMEWORK.md 第 4–6 节）。

用法（单进程，约十几分钟）：
    python scripts/lab_run.py H001

卡片必须已经提交、且提交后没改过，否则拒绝运行。候选流程和基线各跑四个错开的调仓起点，再各跑一遍成本翻倍；
成绩单、配对比较、规则逐条的结果写进 lab/reports/<卡号>-r<第几次>.md，并在 lab/runs.jsonl 追加一条。
脚本只算结论，不改基线、不改卡片：采纳后的落盘（改 lab/baseline.yaml 和卡片状态）另行提交。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt.lab import cards as cd  # noqa: E402
from qt.lab import evidence as ev  # noqa: E402
from qt.lab import gates  # noqa: E402
from qt.lab import procedure as pc  # noqa: E402
from qt.lab import scorecard as sc  # noqa: E402
from qt.lab.protocol import FRAMEWORK_VERSION, FULL, noise_sharpe  # noqa: E402
from qt.trials import _git_commit, count_unique, log_trial  # noqa: E402

LAB = ROOT / "lab"
TRIALS = ROOT / "reports" / "trials.jsonl"
DEV_YEARS = 12


def proc_hash(proc: pc.Procedure) -> str:
    return hashlib.sha1(json.dumps(proc.to_dict(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]


def _tail(dev: dict, proc: pc.Procedure, cache: dict) -> pd.DataFrame:
    data = pc.market(dev, proc, 0, cache)
    frame = lambda a: pd.DataFrame(a, index=data.dates, columns=data.instruments)
    return ev.top_tail(frame(data.score), dev["close"], frame(data.can_buy))


def evaluate(dev: dict, card: dict, baseline: pc.Procedure) -> dict:
    """候选和基线各跑标准成本、成本翻倍两遍；返回成绩单、配对比较、规则表和是否采纳。"""
    cand = cd.candidate(card, baseline)
    if cand == baseline:
        raise cd.CardError(f"{card['id']} 的 variant 和基线一模一样，没有改动")
    cache: dict = {}
    twice = lambda p: p.replace(cost_multiplier=2.0 * p.cost_multiplier)
    runs = {name: sc.run_phases(dev, p, cache=cache)
            for name, p in (("基线", baseline), ("候选", cand), ("基线×2", twice(baseline)), ("候选×2", twice(cand)))}
    card_base, card_cand = sc.scorecard(dev, runs["基线"]), sc.scorecard(dev, runs["候选"])
    pair, pair2 = sc.paired(runs["候选"], runs["基线"]), sc.paired(runs["候选×2"], runs["基线×2"])
    table, adopt = gates.decide(card["kind"], card_cand, card_base, pair, pair2, card["prediction"])
    return {"candidate": cand, "baseline": baseline, "基线": card_base, "候选": card_cand, "pair": pair, "pair2": pair2,
            "gates": table, "adopt": adopt, "tail_base": _tail(dev, baseline, cache), "tail_cand": _tail(dev, cand, cache)}


def report(card: dict, out: dict, n_trials: int, run_no: int) -> str:
    md = lambda df: df.to_markdown(floatfmt=".4f")
    pair = pd.DataFrame({"年化差值": out["pair"]["差值"], "成本翻倍后年化差值": out["pair2"]["差值"]})
    verdict = "采纳" if out["adopt"] else "不采纳"
    failed = "、".join(out["gates"].index[~out["gates"]["满足"]]) or "无"
    return "\n".join([
        f"# {card['id']} 第 {run_no} 次账户检验：{verdict}", "",
        f"- 框架版本 {FRAMEWORK_VERSION}；类型：{card['kind']}；方向：{card['family']}",
        f"- 改动：{card['change']}", f"- 事先的预期：{card['prediction']['metric']} {card['prediction']['direction']}",
        f"- 没满足的规则：{failed}",
        f"- 累计试验 {n_trials} 次；这么多次试验、{DEV_YEARS} 年数据，纯噪声里最好的年化夏普期望约 "
        f"{noise_sharpe(max(n_trials, 1), DEV_YEARS):.2f}",
        "- 这些数都来自开发段（2010–2021），已经被看过很多次，不算干净的证据", "",
        "## 规则逐条", "", md(out["gates"]), "",
        f"## 配对比较（候选 − 基线，四个起点平均；全段 t 值 {out['pair']['t']:.2f}）", "", md(pair), "",
        "## 成绩单：候选", "", md(out["候选"]), "", "## 成绩单：基线", "", md(out["基线"]), "",
        "## 最顶端各档之后 20 天的收益（年化，不扣成本）：候选", "", md(out["tail_cand"]), "",
        "## 最顶端各档：基线", "", md(out["tail_base"]), ""])


def run_card(dev: dict, card_path: Path, lab: Path, trials: Path | None, check_git: bool = True) -> int:
    card = cd.load_card(card_path)
    if check_git and not cd.is_committed(card_path):
        raise cd.CardError(f"{card_path.name} 还没提交，或者提交后被改过：先提交卡片再运行")
    if check_git and not cd.is_committed(lab / "baseline.yaml"):
        raise cd.CardError("lab/baseline.yaml 有没提交的改动")
    runs = cd.load_runs(lab / "runs.jsonl")
    cd.check_budget(card, cd.load_cards(card_path.parent), runs)
    baseline = cd.load_baseline(lab / "baseline.yaml")
    out = evaluate(dev, card, baseline)
    run_no = cd.next_run_number(runs, card["id"])
    full = lambda df, col: float(df.loc[FULL, col])
    record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": _git_commit(),
              "framework": FRAMEWORK_VERSION, "card": card["id"], "run": run_no, "kind": card["kind"],
              "baseline": proc_hash(baseline), "candidate": out["candidate"].to_dict(), "adopt": out["adopt"],
              "failed": list(out["gates"].index[~out["gates"]["满足"]]), "t": out["pair"]["t"],
              "diff": out["pair"]["差值"].to_dict(),
              "候选": {c: full(out["候选"], c) for c in ("年化", "超额", "最大回撤", "大亏只数")},
              "基线": {c: full(out["基线"], c) for c in ("年化", "超额", "最大回撤", "大亏只数")}}
    cd.append_run(lab / "runs.jsonl", record)
    if trials is not None:
        d = out["pair"]["daily"]
        log_trial("lab", {"card": card["id"], "framework": FRAMEWORK_VERSION, "baseline": proc_hash(baseline),
                          **out["candidate"].to_dict()},
                  {"annual_return": full(out["候选"], "年化"), "max_drawdown": full(out["候选"], "最大回撤"),
                   "paired_t": out["pair"]["t"], "paired_diff": float(d.mean() * 252)}, path=trials)
    n_trials = count_unique(trials) if trials is not None else 0
    (lab / "reports").mkdir(parents=True, exist_ok=True)
    (lab / "reports" / f"{card['id']}-r{run_no}.md").write_text(report(card, out, n_trials, run_no), encoding="utf-8")
    print(out["gates"].to_string())
    print(f"{card['id']}：{'采纳' if out['adopt'] else '不采纳'}")
    return 0


def main() -> None:
    from qt.lab.store import load_dev
    from run_plan3c_train import POOL_CACHE, latest_version

    if len(sys.argv) != 2:
        sys.exit(__doc__)
    card_path = LAB / "cards" / f"{sys.argv[1]}.yaml"
    card = cd.load_card(card_path)
    baseline = cd.load_baseline(LAB / "baseline.yaml")
    names = sorted(set(baseline.factors()) | set(cd.candidate(card, baseline).factors()))
    print(f"读取开发段数据（{len(names)} 个因子）…", flush=True)
    dev = load_dev(latest_version(POOL_CACHE), names)
    sys.exit(run_card(dev, card_path, LAB, TRIALS))


if __name__ == "__main__":
    main()
