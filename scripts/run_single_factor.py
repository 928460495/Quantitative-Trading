"""跑单因子检验，写出 reports/single_factor/<market>_<segment>/。

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
from qt.config import MAX_FORWARD_DAYS, segment  # noqa: E402
from qt.data import drop_embargo, load_features  # noqa: E402
from qt.evaluate import evaluate_factor  # noqa: E402
from qt.factors import HORIZONS, LITERATURE, MAIN_HORIZON, forward_return_expr  # noqa: E402
from qt.preprocess import standardize  # noqa: E402
from qt.report import summary_row, write_factor_report  # noqa: E402
from qt.universe import FILTER_EXPRS, LIMIT_THRESHOLD, tradable_mask  # noqa: E402


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
    keep = tradable_mask(df)
    df = df[keep]
    n_days = df.index.get_level_values("datetime").nunique()
    context = (
        f"样本：{args.market} 历史成分，{args.segment} 段 {start} ~ {df.index.get_level_values('datetime').max().date()}"
        f"（段尾去掉 {MAX_FORWARD_DAYS} 个交易日），{n_days} 个交易日，过滤后保留 {keep.mean():.1%} 的样本；未做行业和市值中性化"
    )

    out_dir = ROOT / "reports" / "single_factor" / f"{args.market}_{args.segment}"
    out_dir.mkdir(parents=True, exist_ok=True)
    returns = {h: df[f"ret{h}"] for h in HORIZONS}
    rows = []
    for f in factors:
        print(f"检验 {f.name} ...")
        res = evaluate_factor(f.name, df[f.name], standardize(df[f.name]), returns, f.expected_sign, MAIN_HORIZON)
        write_factor_report(res, f, out_dir, context)
        rows.append(summary_row(res))

    summary = pd.DataFrame(rows).set_index("factor")
    summary.to_csv(out_dir / "summary.csv", float_format="%.6f")
    _write_index(summary, out_dir, context)
    print(summary.round(4).to_string())


def _write_index(summary: pd.DataFrame, out_dir: Path, context: str) -> None:
    show = summary[["direction", "rank_ic_5d", "icir_5d", "t_nw_5d", "reversed_year_ratio", "monotonicity",
                    "top_turnover", "long_excess_ann", "long_excess_net_ann", "passed"]].copy()
    show["passed"] = show["passed"].map({True: "是", False: "否"})
    show.index = [f"[{n}]({n}.md)" for n in show.index]
    show.columns = ["方向", "RankIC", "ICIR", "t 值", "反向年份占比", "单调性", "换手率", "年化超额", "扣成本后", "通过"]
    lines = [
        "# 单因子检验汇总",
        "",
        f"- {context}",
        "- 主指标：5 天持有期 RankIC，t 值经 Newey-West 调整（滞后 4 阶）",
        "- 门槛：t 值 > 3；方向反了的年份不超过一半；扣成本后做多一侧超额 > 0。相关性去冗余在第 4 个 PR 做",
        f"- 入场日收盘涨跌幅 ≥ {LIMIT_THRESHOLD:.1%} 视为买不进，对 20% 涨跌停的创业板和科创板偏保守",
        "- 还没有 ST 标记，ST 股没有剔除（沪深300 里很少）",
        "",
        show.to_markdown(floatfmt=".4f"),
        "",
        "年化超额是最高组相对全体等权平均收益的年化值；扣成本后按最高组换手率乘以双边成本扣除。",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
