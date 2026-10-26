"""因子筛选与去冗余：从单因子检验的结果里挑出通过的因子，按相关性去掉重复的，写出最终候选名单。

用法：
    python scripts/screen_factors.py      # 默认读沪深300 训练段、中性化后的文献因子和 Alpha158 结果

规则（docs/DECISIONS.md）：两两相关性 > 0.7 的只留 ICIR 高的。具体顺序：
1. 先在打分因子里按 ICIR 从高到低逐个看，和已留下的打分因子相关性都不超过 0.7 才留下
2. 再对排雷因子做同样的事，比较对象是已留下的打分因子和排雷因子
   这样一个能赚钱的打分因子，不会因为和一个 ICIR 稍高的排雷因子相关就被挤掉

相关性是中性化、按方向调整后的因子值，在每个调仓日（每 5 个交易日）上的横截面 Spearman 相关，再对时间取平均。
结果写到 reports/screening/<market>_<segment>_neutral/，最终名单 selected.csv 给第 5 个 PR 用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qt.config import MAX_FORWARD_DAYS, segment  # noqa: E402
from qt.data import drop_embargo, load_features  # noqa: E402
from qt.evaluate import SCORING, VETO, rebalance_dates  # noqa: E402
from qt.factors import FACTOR_SETS, MAIN_HORIZON  # noqa: E402
from qt.pipeline import EXPOSURE_EXPRS, exposures, neutral_score  # noqa: E402

CORR_THRESHOLD = 0.7


def load_candidates(report_root: Path, market: str, seg: str) -> pd.DataFrame:
    frames = []
    for name in FACTOR_SETS:
        suffix = "_neutral" + ("" if name == "literature" else f"_{name}")
        path = report_root / f"{market}_{seg}{suffix}" / "summary.csv"
        if not path.exists():
            print(f"跳过 {name}：没有 {path.relative_to(ROOT)}")
            continue
        df = pd.read_csv(path, index_col="factor")
        frames.append(df.assign(set=name))
    summary = pd.concat(frames)
    return summary[summary["category"].isin([SCORING, VETO])]


def mean_rank_corr(scores: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    mats = [scores.xs(d, level="datetime").corr(method="spearman") for d in dates]
    return pd.concat(mats).groupby(level=0).mean().loc[scores.columns, scores.columns]


def greedy_select(cands: pd.DataFrame, corr: pd.DataFrame, threshold: float = CORR_THRESHOLD) -> pd.DataFrame:
    """返回每个候选的去留：kept 为 True 表示留下，否则 dropped_by 是和它相关性过高、先被留下的因子。"""
    h = MAIN_HORIZON
    kept: list[str] = []
    rows = []
    for category in (SCORING, VETO):
        group = cands[cands["category"] == category].sort_values(f"icir_{h}d", ascending=False)
        for name in group.index:
            clash = [(k, corr.loc[name, k]) for k in kept if abs(corr.loc[name, k]) > threshold]
            if clash:
                by, r = max(clash, key=lambda kv: abs(kv[1]))
                rows.append({"factor": name, "kept": False, "dropped_by": by, "corr": r})
            else:
                kept.append(name)
                rows.append({"factor": name, "kept": True, "dropped_by": None, "corr": np.nan})
    return pd.DataFrame(rows).set_index("factor").join(cands)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default="csi300")
    ap.add_argument("--segment", default="train", choices=["train", "valid"])
    args = ap.parse_args()

    report_root = ROOT / "reports" / "single_factor"
    cands = load_candidates(report_root, args.market, args.segment)
    print(f"候选：{SCORING} {(cands['category'] == SCORING).sum()} 个，{VETO} {(cands['category'] == VETO).sum()} 个")
    if cands.empty:
        sys.exit("没有通过单因子检验的因子")

    exprs = {f.name: f.expr for name in FACTOR_SETS for f in FACTOR_SETS[name]() if f.name in cands.index}
    start, end = segment(args.segment)
    df = drop_embargo(load_features(exprs | EXPOSURE_EXPRS, args.market, start, end), MAX_FORWARD_DAYS)
    industry, log_mcap, _, _ = exposures(df)
    scores = pd.DataFrame({n: neutral_score(df[n], industry, log_mcap) * cands.loc[n, "direction"] for n in cands.index})
    corr = mean_rank_corr(scores, rebalance_dates(scores.index, every=MAIN_HORIZON))

    result = greedy_select(cands, corr)
    out_dir = ROOT / "reports" / "screening" / f"{args.market}_{args.segment}_neutral"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = result[result["kept"]]
    selected[["set", "category", "direction", f"rank_ic_{MAIN_HORIZON}d", f"icir_{MAIN_HORIZON}d", f"t_nw_{MAIN_HORIZON}d",
              "sim_net_ann", "bottom_t"]].to_csv(out_dir / "selected.csv", float_format="%.6f")
    corr.to_csv(out_dir / "candidate_corr.csv", float_format="%.4f")
    _heatmap(corr.loc[selected.index, selected.index], out_dir / "selected_corr.png")
    _write_report(result, corr, out_dir, args)
    print(result[["category", "kept", "dropped_by", "corr", f"icir_{MAIN_HORIZON}d"]].round(3).to_string())


def _heatmap(corr: pd.DataFrame, path: Path) -> None:
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(4, 0.45 * n + 2), max(3.5, 0.45 * n + 1.5)))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(n), corr.index, fontsize=8)
    ax.set_title("Mean cross-sectional Spearman corr (selected factors)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _write_report(result: pd.DataFrame, corr: pd.DataFrame, out_dir: Path, args) -> None:
    h = MAIN_HORIZON
    cols = {"set": "来源", "category": "分类", f"rank_ic_{h}d": "RankIC", f"icir_{h}d": "ICIR", f"t_nw_{h}d": "t 值",
            "sim_net_ann": "扣成本后", "bottom_t": "最低组 t 值"}
    kept = result[result["kept"]][list(cols)].rename(columns=cols)
    dropped = result[~result["kept"]][["category", "dropped_by", "corr", f"icir_{h}d"]]
    dropped.columns = ["分类", "被谁替代", "相关系数", "ICIR"]
    sel = result.index[result["kept"]]
    off = corr.loc[sel, sel].where(~np.eye(len(sel), dtype=bool)).abs()
    lines = [
        "# 因子筛选与去冗余",
        "",
        f"- 样本：{args.market} 历史成分，{args.segment} 段，行业和市值中性化后",
        f"- 候选：单因子检验里通过的因子（{SCORING} {(result['category'] == SCORING).sum()} 个，"
        f"{VETO} {(result['category'] == VETO).sum()} 个）",
        f"- 规则：两两相关性绝对值 > {CORR_THRESHOLD} 的只留 ICIR 高的；先在{SCORING}里去冗余，再拿{VETO}和已留下的比",
        "- 相关性：中性化、按方向调整后的因子值，每 5 个交易日一次的横截面 Spearman 相关，对时间取平均",
        f"- 结果：留下 {len(sel)} 个（{SCORING} {(result.loc[sel, 'category'] == SCORING).sum()} 个，"
        f"{VETO} {(result.loc[sel, 'category'] == VETO).sum()} 个），去掉 {len(dropped)} 个；"
        f"留下的因子之间相关性绝对值最大 {np.nanmax(off.to_numpy()) if len(sel) > 1 else float('nan'):.2f}",
        "",
        "## 最终名单",
        "",
        kept.to_markdown(floatfmt=".4f"),
        "",
        "![corr](selected_corr.png)",
        "",
        "## 去掉的因子",
        "",
        dropped.to_markdown(floatfmt=".3f") if len(dropped) else "无",
        "",
        "全部候选之间的相关系数在 `candidate_corr.csv`；最终名单在 `selected.csv`，第 5 个 PR 的因子合成从这里读。",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
