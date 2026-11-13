"""最终测试：在封存段上检验登记好的流程，只跑一次（DECISIONS 2026-11-13「最终测试的程序」）。

用法（用户批准之后；单进程）：
    QT_UNLOCK_TEST=1 python scripts/lab_final.py

运行前必须：lab/FINAL.yaml 已提交（procedure、code_commit、window、approval）；qt/、scripts/ 没有没提交的改动，
并且和 code_commit 相比没有任何改动；lab/reports/FINAL.md 还不存在。数据由 build_pool.py、build_plan3d_flags.py、
lab_build_flags.py 加 --segment test 事先生成。
判定：全段四个起点平均的夏普高于沪深300 全收益的夏普，并且高于能持有等权的夏普；其余只报告。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt.config import is_test_unlocked  # noqa: E402
from qt.lab import cards as cd  # noqa: E402
from qt.lab import final  # noqa: E402
from qt.lab import procedure as pc  # noqa: E402
from qt.lab import scorecard as sc  # noqa: E402
from qt.lab.protocol import FRAMEWORK_VERSION, RISK_FREE, noise_sharpe  # noqa: E402
from qt.trials import count_unique, deflated_sharpe, log_trial  # noqa: E402

TRIALS = ROOT / "reports" / "trials.jsonl"
REQUIRED = ("procedure", "code_commit", "window", "approval")
DEV_DAYS = 2917                                   # 开发段 2010–2021 的交易日数（DSR 的噪声方差用 1/DEV_DAYS）


class FinalError(RuntimeError):
    pass


def check_ready(root: Path) -> dict:
    root = Path(root)
    if not is_test_unlocked():
        raise FinalError("测试段没有解锁：用户批准后用 QT_UNLOCK_TEST=1 运行")
    if (root / "lab" / "reports" / "FINAL.md").exists():
        raise FinalError("最终测试只跑一次：lab/reports/FINAL.md 已经存在")
    reg_path = root / "lab" / "FINAL.yaml"
    if not reg_path.exists() or not cd.is_committed(reg_path):
        raise FinalError("lab/FINAL.yaml 不存在、没提交，或者提交后被改过")
    reg = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if not reg.get(k)]
    if missing:
        raise FinalError(f"lab/FINAL.yaml 缺少：{missing}（approval 要写用户批准的原话和日期）")
    if not cd.code_is_committed(root):
        raise FinalError("qt/ 或 scripts/ 有没提交的改动")
    diff = subprocess.run(["git", "diff", "--quiet", reg["code_commit"], "HEAD", "--", "qt", "scripts"], cwd=root)
    if diff.returncode != 0:
        raise FinalError(f"qt/、scripts/ 和冻结的提交 {reg['code_commit']} 不一样")
    return reg


def run_final(dev: dict, reg: dict, out: Path, trials: Path | None) -> bool:
    proc = pc.Procedure.from_dict(reg["procedure"])
    final.check_window(dev["close"].index, str(reg["window"]["start"]), str(reg["window"]["end"]))
    cache: dict = {}
    res = sc.run_phases(dev, proc, cache=cache)
    res2 = sc.run_phases(dev, proc.replace(cost_multiplier=2.0 * proc.cost_multiplier), cache=cache)
    card, card2 = final.final_scorecard(dev, res), final.final_scorecard(dev, res2)
    passed, table = final.verdict(card)
    n = count_unique(trials) if trials is not None and trials.exists() else 0
    rf = (1 + RISK_FREE) ** (1 / 252) - 1
    daily = sc.daily_returns(res).mean(axis=1) - rf
    dsr = deflated_sharpe(daily.to_numpy(), max(n, 2), 1.0 / DEV_DAYS)
    md = lambda df: df.to_markdown(floatfmt=".4f")
    lines = [f"# 最终测试：{'通过' if passed else '不通过'}", "",
             f"- 窗口 {reg['window']['start']} ~ {reg['window']['end']}；框架 {FRAMEWORK_VERSION}；冻结代码 {reg['code_commit']}",
             f"- 用户批准：{reg['approval']}",
             "- 判定：全段四个起点平均的夏普高于沪深300 全收益的夏普，并且高于能持有等权的夏普；其余只报告", "",
             "## 判定", "", md(table), "",
             "## 成绩单（四个起点平均；分年只报告）", "", md(card), "",
             "## 成本翻倍（只报告）", "", md(card2), "",
             "## 试验次数的折扣（只报告）", "",
             f"- 累计试验 {n} 次；这么多次、按封存段的年数，纯噪声里最好的年化夏普期望约 "
             f"{noise_sharpe(max(n, 1), len(daily) / 252):.2f}",
             f"- 通缩夏普（DSR，各次试验的夏普方差取开发段长度下的估计误差 1/{DEV_DAYS}）：{dsr['dsr']:.3f}", ""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    if trials is not None:
        full = card.loc[final.FULL]
        log_trial("final", {"framework": FRAMEWORK_VERSION, **proc.to_dict()},
                  {"passed": passed, "sharpe": float(full["夏普"]), "sharpe_ew": float(full["等权夏普"]),
                   "sharpe_hs300": float(full["沪深300夏普"]), "annual_return": float(full["年化"]),
                   "max_drawdown": float(full["最大回撤"]), "dsr": float(dsr["dsr"]) if np.isfinite(dsr["dsr"]) else None},
                  path=trials)
    return passed


def main() -> None:
    from qt.lab.store import load_dev
    from run_plan3c_train import POOL_CACHE, latest_version

    try:
        reg = check_ready(ROOT)
    except FinalError as e:
        sys.exit(f"不能运行最终测试：{e}")
    proc = pc.Procedure.from_dict(reg["procedure"])
    dev = load_dev(latest_version(POOL_CACHE), proc.factors(), segments=("test",))
    out = ROOT / "lab" / "reports" / "FINAL.md"
    passed = run_final(dev, reg, out, TRIALS)
    print(out.read_text())
    sys.exit(0 if passed else 3)


if __name__ == "__main__":
    main()
