"""计划 3c 任务 9：训练段（2010–2018）滚动前推、组合网格、CSCV、研究关卡，选出最终方案并固定。

用法（先跑过 scripts/build_pool.py；单进程，约 3–4 小时，中断后重跑会接着做）：
    python scripts/run_plan3c_train.py

规则见 docs/DECISIONS.md 2026-11-10「计划 3c 事先登记的规则」。步骤：
  ① 参考组合：快速模拟器 vs 真实账户，年化相差 > 2 个百分点 → 停
  ② 各大类 IC 随持有天数衰减（只报告）
  ③ 前推 2013–2018：每年用之前的数据逐步加因子、挑组合，跑下一年
  ④ 用 2010–2018 全部数据按同一程序做最终选择
  ⑤ 最终因子下 216 个组合做 CSCV，PBO > 0.3 → 停
  ⑥ 研究关卡：前推年化 > 能买股票等权，且合成分和市值相关 < 0.1；不满足 → 停
  ⑦ 最终方案用真实账户跑训练段，写诊断；壳价值对比（去掉市值最小 30%，只报告）
  ⑧ 写报告和 final.json（任务 10 读它）
"停" = 写报告说明原因、退出码 1、不写 final.json。
输出在 reports/plan3c/train/；中间结果在 data/cache/plan3c/（不进仓库）。
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt import walkforward as wf  # noqa: E402
from qt.account import run_account  # noqa: E402
from qt.composite import category_scores  # noqa: E402
from qt.diagnostics import annualize, diagnose, size_corr, write_report  # noqa: E402
from qt.pbo import cscv_pbo  # noqa: E402
from qt.trials import log_trial  # noqa: E402
from qt.wide_eval import buyable_benchmark, ic_decay  # noqa: E402

OUT_DIR = ROOT / "reports" / "plan3c" / "train"
CACHE_ROOT = ROOT / "data" / "cache" / "plan3c"
POOL_CACHE = ROOT / "data" / "cache" / "pool"
TRIALS = ROOT / "reports" / "trials.jsonl"
TRAIN_END = "2018-12-31"
MAX_GAP = 0.02
MAX_PBO = 0.30
MAX_SIZE_CORR = 0.10
SHELL_CUT = 0.30


def _nav(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def run(train: dict, pool: pd.DataFrame, out_dir: Path, cache_dir: Path, log_path: Path | None,
        grid=wf.GRID, years=range(2013, 2019), reference=wf.REFERENCE, train_end: str = TRAIN_END) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final.json").unlink(missing_ok=True)      # 旧的方案不能留到这次没通过的运行之后
    cache_dir = cache_dir / wf.fingerprint(grid, reference, pool)   # 规则或代码变了就不用旧的中间结果
    log = (lambda kind, params, metrics: log_trial(kind, params, metrics, path=log_path)) if log_path else None
    full = (wf.TRAIN_START, train_end)
    sections: dict[str, pd.DataFrame] = {}
    notes: list[str] = ["规则见 docs/DECISIONS.md 2026-11-10「计划 3c 事先登记的规则」", "验证段在计划 4 已看过两次；本报告只用训练段",
                        "前推结果略偏乐观：59 个候选因子和它们的方向是用整个训练段（2010–2018）选出来的，前推时没有每年重新筛选"]

    def stop(reason: str) -> int:
        write_report(sections, out_dir / "README.md", "计划 3c 训练段结果：停下讨论", notes + [f"**停下讨论**：{reason}"])
        print(f"停下讨论：{reason}")
        return 1

    # ① 一致性
    all_factors = list(pool.index)
    data = wf.market(train, wf.score_for(train, pool, all_factors, full), full)
    fast = annualize(_nav(wf.simulate(data, reference)))
    acct = annualize(run_account(data, wf.account_rules(reference)).daily["nav"])
    sections["① 快速模拟器 vs 真实账户（参考组合、全部候选因子等权）"] = pd.DataFrame(
        {"年化": [fast, acct]}, index=["快速模拟器", "真实账户"])
    print(f"① 快速模拟器 {fast:.2%}，真实账户 {acct:.2%}", flush=True)
    if abs(fast - acct) > MAX_GAP:
        return stop(f"① 两者年化相差 {abs(fast - acct):.2%}，超过 {MAX_GAP:.0%}")

    # ② IC 衰减
    cats = category_scores({f: train["factors"][f].loc[full[0]:full[1]] for f in all_factors}, pool["category"].to_dict())
    close, can_buy = train["close"].loc[full[0]:full[1]], train["can_buy"].loc[full[0]:full[1]]
    decay = pd.DataFrame({c: ic_decay(s, close, can_buy) for c, s in cats.items()}).T
    decay.to_csv(out_dir / "ic_decay.csv")
    sections["② 各大类 5 日 RankIC 随持有天数衰减（只报告）"] = decay

    # ③ 前推
    print("③ 前推 …", flush=True)
    res = wf.walk_forward(train, pool, years=years, cache_dir=cache_dir, grid=grid, reference=reference, log=log)
    res["returns"].to_frame("前推日收益").to_csv(out_dir / "walk_forward.csv")
    sections["③ 前推每年的选择"] = pd.DataFrame(
        {y: {"因子": ", ".join(c["factors"]), **c["config"]} for y, c in res["years"].items()}).T

    # ④ 最终选择
    print("④ 最终选择 …", flush=True)
    final_path = cache_dir / "final_choice.json"
    if final_path.exists():
        factors = json.loads(final_path.read_text())["factors"]
    else:
        factors = wf.forward_select(train, pool, full, reference=reference, log=log)
        final_path.write_text(json.dumps({"factors": factors}, ensure_ascii=False))
    rets, best = wf.grid_select(train, pool, factors, full, grid=grid, log=log)
    rets.to_csv(out_dir / "grid.csv")
    p, k, w = best
    sections["④ 最终方案"] = pd.Series({"因子": ", ".join(factors), **wf.to_dict(best)}).to_frame("值")

    # ⑤ CSCV
    pbo = cscv_pbo(rets)["pbo"]
    sections["⑤ CSCV"] = pd.Series({"PBO": pbo, "上限": MAX_PBO}).to_frame("值")
    print(f"⑤ PBO = {pbo:.3f}", flush=True)
    if pbo > MAX_PBO:
        return stop(f"⑤ PBO = {pbo:.3f}，超过 {MAX_PBO}")

    # ⑥ 研究关卡
    wf_ann = annualize(_nav(res["returns"]))
    ew = buyable_benchmark(train["close"], train["can_buy"]).reindex(res["returns"].index)
    ew_ann = annualize(_nav(ew))
    score = wf.score_for(train, pool, factors, full, k, w)
    corr = size_corr(score, train["log_mcap"].loc[full[0]:full[1]])
    sections["⑥ 研究关卡"] = pd.DataFrame(
        {"值": [wf_ann, ew_ann, corr], "门槛": ["> 能买等权", "", f"< {MAX_SIZE_CORR}"]},
        index=["前推年化", "能买股票等权年化（同期）", "合成分和市值相关"])
    if not (wf_ann > ew_ann and corr < MAX_SIZE_CORR):
        return stop(f"⑥ 研究关卡没通过：前推年化 {wf_ann:.2%}，能买等权 {ew_ann:.2%}，市值相关 {corr:.3f}")

    # ⑦ 真实账户确认 + 壳价值对比
    data = wf.market(train, score, full)
    result = run_account(data, wf.account_rules(p))
    for name, table in diagnose(result, train, score).items():
        sections[f"⑦ 真实账户：{name}"] = table
    small = train["log_mcap"].loc[full[0]:full[1]].rank(axis=1, pct=True) < SHELL_CUT
    shell = run_account(wf.market(train, score.mask(small), full), wf.account_rules(p))
    sections["⑦ 壳价值对比（只报告）"] = pd.Series(
        {"全部": annualize(result.daily["nav"]), f"去掉市值最小 {SHELL_CUT:.0%}": annualize(shell.daily["nav"])}).to_frame("年化")

    # ⑧ 写结果
    final = {"factors": factors, "params": dataclasses.asdict(p), "smooth": k, "lowvol_weight": w}
    (out_dir / "final.json").write_text(json.dumps(final, ensure_ascii=False, indent=2))
    write_report(sections, out_dir / "README.md", "计划 3c 训练段结果", notes + ["所有检查通过，方案已固定（final.json）"])
    print(f"报告：{out_dir / 'README.md'}")
    return 0


def latest_version(root: Path) -> Path:
    dirs = sorted(d for d in root.iterdir() if (d / "train" / "meta.json").exists())
    if not dirs:
        raise FileNotFoundError(f"{root} 下没有缓存，先跑 scripts/build_pool.py")
    return dirs[-1]


def main() -> None:
    from build_composite import load_segment
    from build_pool import read_pool

    version = latest_version(POOL_CACHE)
    print(f"读取 {version}", flush=True)
    train = load_segment(version / "train")
    sys.exit(run(train, read_pool(), OUT_DIR, CACHE_ROOT / version.name, TRIALS))


if __name__ == "__main__":
    main()
