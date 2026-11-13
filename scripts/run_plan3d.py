"""计划 3d：训练段四条检查，2019–2021 中途检查（规则见 docs/DECISIONS.md 2026-11-12「计划 3d 事先登记的规则」）。

用法（先跑过 scripts/build_pool.py 和 scripts/build_plan3d_flags.py；单进程，各几分钟）：
    python scripts/run_plan3d.py --segment train
    python scripts/run_plan3d.py --segment valid

训练段：主方案和事先列好的全部单项改动各跑一次真实账户；四条检查都满足才写 final.json。
中途检查：只有训练段写出了 final.json 才能跑，而且只能跑一次（输出目录里已有东西就拒绝）。
任何检查不满足：报告里写"停下讨论"、退出码 1，不改规则。单项改动只报告，不换主方案。
输出在 reports/plan3d/{train,valid}/。不碰测试段。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt import plan3d as p3  # noqa: E402
from qt.diagnostics import diagnose, write_report  # noqa: E402
from qt.trials import log_trial  # noqa: E402

OUT_ROOT = ROOT / "reports" / "plan3d"
FINAL = OUT_ROOT / "train" / "final.json"
TRIALS = ROOT / "reports" / "trials.jsonl"
MAX_SHORTFALL = 0.05          # 中途检查：4 个错开版本的平均年化比能持有等权低超过 5 个百分点 = 明显坏掉
RULES = "规则见 docs/DECISIONS.md 2026-11-12「计划 3d 事先登记的规则」"


def _logger(path: Path | None):
    return (lambda kind, params, metrics: log_trial(kind, params, metrics, path=path)) if path else None


def _offsets_mean(table: pd.DataFrame) -> pd.DataFrame:
    cols = ["年化", "能持有等权", "沪深300全收益", "超额", "最大回撤"]
    return table.loc[list(p3.OFFSETS), cols].astype(float).mean().to_frame("错开 0、5、10、15 天的平均")


def _diagnosis(result, seg: dict, spec: p3.Spec) -> dict[str, pd.DataFrame]:
    tables = diagnose(result, seg, p3.composite_score(seg, spec.groups), p3.BENCH_EVERY)
    names = {"summary": "主方案：汇总", "yearly": "主方案：分年", "rolling": "主方案：滚动 3 年相对沪深300"}
    return {names[k]: v for k, v in tables.items()}


def run_train(seg: dict, out_dir: Path, log_path: Path | None, checks=p3.CHECKS) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final.json").unlink(missing_ok=True)       # 旧的通过记录不能留到这次没通过的运行之后
    table, results = p3.evaluate(seg, p3.VARIANTS, _logger(log_path), tag="train")
    table.to_csv(out_dir / "variants.csv")
    main = results["主方案"]
    main.daily.to_csv(out_dir / "backtest.csv")
    chk = p3.check_table(table, checks)
    ok = bool(chk["满足"].all())
    sections = {"检查（每条都要满足）": chk, "错开起点的平均（比单看主方案可靠）": _offsets_mean(table),
                "各版本（除检查用到的，其余只报告，不换主方案）": table, **_diagnosis(main, seg, p3.MAIN)}
    notes = [RULES, "只用训练段（2010–2018）；因子名单和方向也是在这一段上筛出来的，所以这里的数字偏乐观",
             "主方案：" + "；".join(f"{g}（{'、'.join(fs)}）" for g, fs in p3.GROUPS.items()),
             f"持股 {p3.MAIN.n_hold} 只，每 {p3.MAIN.rebalance_every} 个交易日调仓，跌出前 "
             f"{p3.MAIN.sell_mult * p3.MAIN.n_hold} 名才卖，不止损，不择时"]
    if ok:
        final = {"spec": dataclasses.asdict(p3.MAIN), "groups": p3.GROUPS}
        (out_dir / "final.json").write_text(json.dumps(final, ensure_ascii=False, indent=2))
        notes.append("**所有检查都满足**，方案已固定（final.json）")
    else:
        failed = "、".join(chk.index[~chk["满足"]])
        notes.append(f"**停下讨论**：没满足的检查：{failed}")
    write_report(sections, out_dir / "README.md", "计划 3d 训练段结果", notes)
    print(chk.to_string())
    print(table[["年化", "能持有等权", "沪深300全收益", "超额", "最大回撤"]].round(4).to_string())
    return 0 if ok else 1


def load_final(path: Path = FINAL) -> p3.Spec:
    if not path.exists():
        raise FileNotFoundError(f"没有 {path}：训练段脚本没跑，或有检查没满足")
    spec = json.loads(path.read_text())["spec"]
    return p3.Spec(**{**spec, "groups": tuple(spec["groups"])})


def broken(table: pd.DataFrame) -> bool:
    """4 个错开版本的平均超额（相对能持有等权）低于 −MAX_SHORTFALL。"""
    return bool(table.loc[list(p3.OFFSETS), "超额"].astype(float).mean() < -MAX_SHORTFALL)


def run_valid(seg: dict, spec: p3.Spec, out_dir: Path, log_path: Path | None) -> int:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"{out_dir} 里已经有东西（跑过，或上次跑到一半）：中途检查只能跑一次，拒绝运行")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "STARTED").write_text("中途检查已开始运行；之后不能再跑\n")      # 先占位：中途崩了也不能重跑
    variants = {name: dataclasses.replace(spec, offset=p3.VARIANTS[name].offset) for name in p3.OFFSETS}
    variants["成本翻倍"] = dataclasses.replace(spec, cost_multiplier=2.0)
    table, results = p3.evaluate(seg, variants, _logger(log_path), tag="valid")
    table.to_csv(out_dir / "variants.csv")
    results["主方案"].daily.to_csv(out_dir / "backtest.csv")
    bad = broken(table)
    sections = {"错开起点的平均": _offsets_mean(table), "各版本": table, **_diagnosis(results["主方案"], seg, spec)}
    notes = [RULES, "**这一段已经看过三次**（计划 4 两次、3c 一次），而且事先知道 3c 的两因子版本在这段不差："
             "**通过不算证据**，只用来拦明显坏掉的方案",
             f"判定：4 个错开版本的平均年化比能持有等权低超过 {MAX_SHORTFALL:.0%} 才算明显坏掉",
             "**停下讨论**：明显坏掉" if bad else "结论：没有明显坏掉。下一步是最终测试，要用户批准，只跑一次"]
    write_report(sections, out_dir / "README.md", "计划 3d 中途检查（2019–2021，只跑一次）", notes)
    print(table[["年化", "能持有等权", "沪深300全收益", "超额", "最大回撤"]].round(4).to_string())
    return 1 if bad else 0


def load(seg_dir: Path) -> dict:
    from build_composite import load_segment

    seg = load_segment(seg_dir)
    for name in ("is_st", "loss_small"):
        path = seg_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"没有 {path}，先跑 scripts/build_plan3d_flags.py")
        seg[name] = pd.read_parquet(path)
    return seg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", choices=["train", "valid"], required=True)
    args = ap.parse_args()

    from run_plan3c_train import POOL_CACHE, latest_version

    version = latest_version(POOL_CACHE)
    if args.segment == "valid":
        spec = load_final()                                  # 训练段没通过：读数据之前就停
    print(f"读取 {version / args.segment}", flush=True)
    seg = load(version / args.segment)
    if args.segment == "train":
        sys.exit(run_train(seg, OUT_ROOT / "train", TRIALS))
    sys.exit(run_valid(seg, spec, OUT_ROOT / "valid", TRIALS))


if __name__ == "__main__":
    main()
