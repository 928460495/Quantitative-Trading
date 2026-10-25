"""跑单因子检验，写出 reports/single_factor/<market>_<segment>/，并把每次检验记入试验日志。

用法：
    python scripts/run_single_factor.py                       # 沪深300、训练段、全部文献因子
    python scripts/run_single_factor.py --factors rev20 vol20

筛选只在训练段上做。验证段留给因子合成阶段（第 5 个 PR）比较方案用，测试段封存。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qt.config import DATA_DIR, MAX_FORWARD_DAYS, segment  # noqa: E402
from qt.data import drop_embargo, load_features  # noqa: E402
from qt.evaluate import REJECTED, SCORING, VETO, evaluate_factor  # noqa: E402
from qt.factors import HORIZONS, LITERATURE, MAIN_HORIZON, forward_return_expr  # noqa: E402
from qt.preprocess import standardize  # noqa: E402
from qt.report import summary_row, write_factor_report  # noqa: E402
from qt.simulate import COMPARISON_RULES, PRIMARY_RULE  # noqa: E402
from qt.trials import count_unique, log_trial  # noqa: E402
from qt.universe import FILTER_EXPRS, LIMIT_THRESHOLD, sellable_mask, tradable_mask  # noqa: E402


def _data_version() -> str:
    f = DATA_DIR / ".qt_data_version"
    return f.read_text().strip() if f.exists() else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default="csi300")
    ap.add_argument("--segment", default="train", choices=["train", "valid"])
    ap.add_argument("--factors", nargs="*", help="只跑这几个因子，默认全部")
    args = ap.parse_args()

    factors = [f for f in LITERATURE if not args.factors or f.name in args.factors]
    start, end = segment(args.segment)
    exprs = {f.name: f.expr for f in factors}
    exprs |= {f"ret{h}": forward_return_expr(h) for h in HORIZONS}
    exprs |= FILTER_EXPRS

    print(f"读取 {args.market} {start} ~ {end} ...")
    df = drop_embargo(load_features(exprs, args.market, start, end), MAX_FORWARD_DAYS)
    can_buy = tradable_mask(df)
    can_sell = sellable_mask(df)
    n_days = df.index.get_level_values("datetime").nunique()
    context = (
        f"样本：{args.market} 历史成分，{args.segment} 段 {start} ~ {df.index.get_level_values('datetime').max().date()}"
        f"（段尾去掉 {MAX_FORWARD_DAYS} 个交易日），{n_days} 个交易日；能买入的样本占 {can_buy.mean():.1%}；"
        "未做行业和市值中性化"
    )

    out_dir = ROOT / "reports" / "single_factor" / f"{args.market}_{args.segment}"
    out_dir.mkdir(parents=True, exist_ok=True)
    returns = {h: df[f"ret{h}"] for h in HORIZONS}
    rules = (PRIMARY_RULE, *COMPARISON_RULES)
    rows = []
    for f in factors:
        print(f"检验 {f.name} ...")
        res = evaluate_factor(
            f.name, df[f.name], standardize(df[f.name].where(can_buy)), returns,
            can_buy, can_sell, f.expected_sign, MAIN_HORIZON, rules=rules,
        )
        write_factor_report(res, f, out_dir, context)
        rows.append(summary_row(res))
        for rule in rules:
            log_trial(
                "single_factor",
                {"factor": f.name, "expr": f.expr, "direction": res.direction, "market": args.market,
                 "segment": args.segment, "horizon": MAIN_HORIZON, "rule": rule.label,
                 "neutralized": False, "data_version": _data_version()},
                {"rank_ic": res.rank_ic[MAIN_HORIZON]["mean"], "t_nw": res.rank_ic[MAIN_HORIZON]["t_nw"],
                 **res.sim_summary[rule.label], "category": res.category},
            )

    summary = pd.DataFrame(rows).set_index("factor")
    summary.to_csv(out_dir / "summary.csv", float_format="%.6f")
    _write_index(summary, out_dir, context)
    print(summary[["category", f"rank_ic_{MAIN_HORIZON}d", f"t_nw_{MAIN_HORIZON}d", "sim_turnover",
                   "sim_excess_ann", "sim_net_ann", "bottom_t"]].round(4).to_string())
    print(f"试验日志里不同配置的单因子试验共 {count_unique(kind='single_factor')} 个")


def _write_index(summary: pd.DataFrame, out_dir: Path, context: str) -> None:
    h = MAIN_HORIZON
    show = summary[["category", "direction", f"rank_ic_{h}d", f"icir_{h}d", f"t_nw_{h}d", "rank_ic_1d",
                    "reversed_year_ratio", "monotonicity", "bottom_t", "sim_turnover", "sim_excess_ann", "sim_net_ann"]].copy()
    show.index = [f"[{n}]({n}.md)" for n in show.index]
    show.columns = ["分类", "方向", "RankIC", "ICIR", "t 值", "1天 RankIC", "反向年份占比", "单调性",
                    "最低组 t 值", "日均换手", "年化超额", "扣成本后"]
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
        f"日均换手、年化超额、扣成本后这三列来自缓冲 {PRIMARY_RULE.label} 的日频组合模拟；"
        f"缓冲 {', '.join(r.label for r in COMPARISON_RULES)} 的结果在各因子的详细报告里。",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
