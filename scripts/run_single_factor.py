"""跑单因子检验，写出 reports/single_factor/<market>_<segment>/，并把每次检验记入试验日志。

用法：
    python scripts/run_single_factor.py                       # 沪深300、训练段、全部文献因子
    python scripts/run_single_factor.py --factors rev20 vol20
    python scripts/run_single_factor.py --neutralize          # 行业 + 市值中性化（需要 data/external/）
    python scripts/run_single_factor.py --neutralize --set alpha158 --primary-only --details passed --jobs 4
                                                              # Alpha158 的 158 个因子，4 个进程并行

筛选只在训练段上做。验证段留给因子合成阶段（第 5 个 PR）比较方案用，测试段封存。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qt.config import DATA_DIR, MAX_FORWARD_DAYS, segment  # noqa: E402
from qt.data import drop_embargo, load_features  # noqa: E402
from qt.evaluate import REJECTED, SCORING, VETO, evaluate_factor  # noqa: E402
from qt.factors import FACTOR_SETS, HORIZONS, MAIN_HORIZON, Factor, forward_return_expr  # noqa: E402
from qt.neutralize import MIN_INDUSTRY_SIZE  # noqa: E402
from qt.pipeline import EXPOSURE_EXPRS, exposures, neutral_score  # noqa: E402
from qt.preprocess import standardize  # noqa: E402
from qt.report import summary_row, write_factor_report  # noqa: E402
from qt.simulate import COMPARISON_RULES, PRIMARY_RULE  # noqa: E402
from qt.trials import count_unique, log_trial  # noqa: E402
from qt.universe import FILTER_EXPRS, LIMIT_THRESHOLD, sellable_mask, tradable_mask  # noqa: E402


def _data_version() -> str:
    f = DATA_DIR / ".qt_data_version"
    return f.read_text().strip() if f.exists() else "unknown"


# 并行时子进程通过 fork 继承这些数据，不用重新读
_CTX: dict = {}


def _evaluate_one(f: Factor):
    c = _CTX
    df = c["df"]
    if c["neutralize"]:
        score = neutral_score(df[f.name], c["industry"], c["log_mcap"])
        standardized = score
    else:
        score, standardized = df[f.name], standardize(df[f.name].where(c["can_buy"]))
    return evaluate_factor(
        f.name, score, standardized, c["returns"], c["can_buy"], c["can_sell"], f.expected_sign, MAIN_HORIZON,
        rules=c["rules"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default="csi300")
    ap.add_argument("--segment", default="train", choices=["train", "valid"])
    ap.add_argument("--factors", nargs="*", help="只跑这几个因子，默认全部")
    ap.add_argument("--neutralize", action="store_true", help="对申万一级行业和对数流通市值做中性化")
    ap.add_argument("--set", default="literature", choices=list(FACTOR_SETS), help="因子集，默认 literature")
    ap.add_argument("--primary-only", action="store_true", help=f"只跑主缓冲规则 {PRIMARY_RULE.label}，不跑对照规则")
    ap.add_argument("--details", default="all", choices=["all", "passed"], help="哪些因子写详细报告，默认全部")
    ap.add_argument("--jobs", type=int, default=1, help="并行进程数，默认 1")
    args = ap.parse_args()

    factors = [f for f in FACTOR_SETS[args.set]() if not args.factors or f.name in args.factors]
    start, end = segment(args.segment)
    exprs = {f.name: f.expr for f in factors}
    exprs |= {f"ret{h}": forward_return_expr(h) for h in HORIZONS}
    exprs |= FILTER_EXPRS
    exprs |= EXPOSURE_EXPRS

    print(f"读取 {args.market} {start} ~ {end} ...")
    df = drop_embargo(load_features(exprs, args.market, start, end), MAX_FORWARD_DAYS)
    can_buy = tradable_mask(df)
    can_sell = sellable_mask(df)
    n_days = df.index.get_level_values("datetime").nunique()
    if args.neutralize:
        industry, log_mcap, coverage, quality = exposures(df)
        neutral_note = (
            f"已对申万一级行业（当天不足 {MIN_INDUSTRY_SIZE} 只的行业合并为“其他”）和对数流通市值做中性化；"
            f"行业和市值都有数据的样本占 {coverage:.1%}，其中 {quality['imputed_share']:.1%} 的市值缺股本数据、"
            f"用对数 20 日平均成交额估算（估算值与真实值的每日相关系数平均 {quality['fit_corr_mean']:.2f}、"
            f"最低 {quality['fit_corr_min']:.2f}）"
        )
    else:
        neutral_note = "未做行业和市值中性化"
    context = (
        f"样本：{args.market} 历史成分，{args.segment} 段 {start} ~ {df.index.get_level_values('datetime').max().date()}"
        f"（段尾去掉 {MAX_FORWARD_DAYS} 个交易日），{n_days} 个交易日；能买入的样本占 {can_buy.mean():.1%}；{neutral_note}"
    )

    suffix = ("_neutral" if args.neutralize else "") + ("" if args.set == "literature" else f"_{args.set}")
    out_dir = ROOT / "reports" / "single_factor" / f"{args.market}_{args.segment}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rules = (PRIMARY_RULE,) if args.primary_only else (PRIMARY_RULE, *COMPARISON_RULES)
    _CTX.update(df=df, can_buy=can_buy, can_sell=can_sell, rules=rules, neutralize=args.neutralize,
                returns={h: df[f"ret{h}"] for h in HORIZONS})
    if args.neutralize:
        _CTX.update(industry=industry, log_mcap=log_mcap)

    print(f"检验 {len(factors)} 个因子，{args.jobs} 个进程 ...")
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context("fork")) as pool:
            results = list(pool.map(_evaluate_one, factors))
    else:
        results = [_evaluate_one(f) for f in factors]

    rows = []
    for f, res in zip(factors, results):
        if args.details == "all" or res.category != REJECTED:
            write_factor_report(res, f, out_dir, context)
        rows.append(summary_row(res))
        for rule in rules:
            log_trial(
                "single_factor",
                {"factor": f.name, "expr": f.expr, "direction": res.direction, "market": args.market,
                 "segment": args.segment, "horizon": MAIN_HORIZON, "rule": rule.label,
                 "benchmark": "buyable", "neutralized": args.neutralize, "data_version": _data_version(),
                 **({"min_industry_size": MIN_INDUSTRY_SIZE, "size_imputation": "log_amount20"} if args.neutralize else {})},
                {"rank_ic": res.rank_ic[MAIN_HORIZON]["mean"], "t_nw": res.rank_ic[MAIN_HORIZON]["t_nw"],
                 **res.sim_summary[rule.label], "category": res.category},
            )

    summary = pd.DataFrame(rows).set_index("factor")
    summary.to_csv(out_dir / "summary.csv", float_format="%.6f")
    raw_dir = ROOT / "reports" / "single_factor" / f"{args.market}_{args.segment}"
    baseline = pd.read_csv(raw_dir / "summary.csv", index_col="factor") if args.neutralize and (raw_dir / "summary.csv").exists() else None
    _write_index(summary, out_dir, context, baseline, literature=args.set == "literature")
    print(summary[["category", f"rank_ic_{MAIN_HORIZON}d", f"t_nw_{MAIN_HORIZON}d", "sim_turnover",
                   "sim_excess_ann", "sim_net_ann", "sim_net_ann_all", "bottom_t"]].round(4).to_string())
    print(f"试验日志里不同配置的单因子试验共 {count_unique(kind='single_factor')} 个")


def _comparison(summary: pd.DataFrame, baseline: pd.DataFrame) -> list[str]:
    h = MAIN_HORIZON
    cols = {"category": "分类", f"rank_ic_{h}d": "RankIC", f"t_nw_{h}d": "t 值", "sim_net_ann": "扣成本后"}
    names = [n for n in summary.index if n in baseline.index]
    if not names:
        return []
    both = pd.DataFrame(index=pd.Index(names, name="因子"))
    for col, label in cols.items():
        both[f"{label}（未中性化）"] = baseline.loc[names, col]
        both[f"{label}（中性化）"] = summary.loc[names, col]
    return ["## 和未中性化的对比", "", both.to_markdown(floatfmt=".4f"), ""]


def _write_index(summary: pd.DataFrame, out_dir: Path, context: str, baseline: pd.DataFrame | None = None,
                 literature: bool = True) -> None:
    h = MAIN_HORIZON
    show = summary[["category", "direction", f"rank_ic_{h}d", f"icir_{h}d", f"t_nw_{h}d", "rank_ic_1d",
                    "reversed_year_ratio", "monotonicity", "bottom_t", "sim_turnover", "sim_excess_ann", "sim_net_ann",
                    "sim_net_ann_all"]].copy()
    show.index = [f"[{n}]({n}.md)" if (out_dir / f"{n}.md").exists() else n for n in show.index]
    show.columns = ["分类", "方向", "RankIC", "ICIR", "t 值", "1天 RankIC", "反向年份占比", "单调性",
                    "最低组 t 值", "日均换手", "年化超额", "扣成本后", "扣成本后（全体基准）"]
    counts = summary["category"].value_counts()
    lines = [
        "# 单因子检验汇总",
        "",
        f"- {context}",
        f"- 结果：{SCORING} {counts.get(SCORING, 0)} 个，{VETO} {counts.get(VETO, 0)} 个，{REJECTED} {counts.get(REJECTED, 0)} 个",
        f"- 主指标：{h} 天持有期 RankIC，t 值经 Newey-West 调整（滞后 {h - 1} 阶）",
        f"- {SCORING}：t 值 > 3；方向反了的年份不超过一半；按实际交易规则（每日调仓、持仓 30、缓冲 {PRIMARY_RULE.label}）扣成本后做多超额 > 0",
        f"- {VETO}：t 值 > 3；方向反了的年份不超过一半；最低组年化超额 < 0 且 t 值 < -3。用法是剔除得分最差的股票，不参与打分",
        "- 相关性去冗余在第 4 个 PR 做",
        f"- 入场日收盘涨跌幅 ≥ {LIMIT_THRESHOLD:.1%} 视为买不进，≤ -{LIMIT_THRESHOLD:.1%} 视为卖不掉；对 20% 涨跌停的创业板和科创板偏保守",
        "- 还没有 ST 标记，ST 股没有剔除（沪深300 里很少）",
        "",
        show.to_markdown(floatfmt=".4f"),
        "",
        f"最后四列来自缓冲 {PRIMARY_RULE.label} 的日频组合模拟；对照规则（缓冲 {', '.join(r.label for r in COMPARISON_RULES)}）"
        "如果跑了，结果在各因子的详细报告里。没有链接的因子未通过，没有写详细报告。",
        "超额默认相对第二天能买进的股票（筛选门槛用这个）；最后一列相对所有有打分的股票，包括第二天涨停买不进的，作参考。",
        "",
    ]
    if baseline is not None:
        lines += _comparison(summary, baseline)
    if "neutral" in out_dir.name and literature:
        lines += [
            "## 说明",
            "",
            "- 市值估算质量低于事先约定的 0.8，按 docs/DECISIONS.md（2026-10-26）保留估算并如实报告",
            "- pv_corr20 为**临界**：缺市值改用当天平均值时它是打分因子，用成交额估算时是排雷因子，"
            "扣成本后的收益贴近 0；到因子合成阶段由验证段检验",
            "",
        ]
    (out_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
