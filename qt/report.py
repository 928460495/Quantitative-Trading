"""把单因子检验结果写成 Markdown 报告和图。

图里的文字用英文：图片在没有中文字体的机器上重新生成时也不会变成方框。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from qt.evaluate import PERIODS_PER_YEAR, FactorResult  # noqa: E402
from qt.factors import Factor  # noqa: E402


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plots(res: FactorResult, img_dir: Path) -> dict[str, str]:
    img_dir.mkdir(parents=True, exist_ok=True)
    names = {}

    fig, ax = plt.subplots(figsize=(8, 3))
    res.rank_ic_daily.dropna().cumsum().plot(ax=ax)
    ax.set_title(f"{res.name}: cumulative daily RankIC (5d, direction-adjusted)")
    ax.axhline(0, color="grey", lw=0.5)
    names["cum_ic"] = f"{res.name}_cum_ic.png"
    _save(fig, img_dir / names["cum_ic"])

    fig, ax = plt.subplots(figsize=(8, 3))
    res.yearly_rank_ic.plot.bar(ax=ax, color=["tab:green" if v > 0 else "tab:red" for v in res.yearly_rank_ic])
    ax.set_title(f"{res.name}: mean RankIC by year (5d)")
    ax.axhline(0, color="grey", lw=0.5)
    names["yearly"] = f"{res.name}_yearly.png"
    _save(fig, img_dir / names["yearly"])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    groups = [c for c in res.quantiles.columns if c != "universe"]
    (1 + res.quantiles[groups]).cumprod().plot(ax=ax)
    ax.set_title(f"{res.name}: cumulative excess return by quintile (5 = best)")
    ax.axhline(1, color="grey", lw=0.5)
    names["quantiles"] = f"{res.name}_quantiles.png"
    _save(fig, img_dir / names["quantiles"])

    fig, ax = plt.subplots(figsize=(5, 3))
    pd.Series({f"{h}d": s["mean"] for h, s in res.rank_ic.items()}).plot.bar(ax=ax)
    ax.set_title(f"{res.name}: RankIC decay")
    ax.axhline(0, color="grey", lw=0.5)
    names["decay"] = f"{res.name}_decay.png"
    _save(fig, img_dir / names["decay"])
    return names


def summary_row(res: FactorResult, main_horizon: int = 5) -> dict:
    s = res.rank_ic[main_horizon]
    return {
        "factor": res.name,
        "direction": res.direction,
        "rank_ic_5d": s["mean"],
        "icir_5d": s["icir"],
        "t_nw_5d": s["t_nw"],
        "ic_positive_ratio": s["positive_ratio"],
        "pearson_ic_5d": res.pearson_ic_main["mean"],
        "rank_ic_1d": res.rank_ic[1]["mean"],
        "rank_ic_10d": res.rank_ic[10]["mean"],
        "rank_ic_20d": res.rank_ic[20]["mean"],
        "reversed_year_ratio": res.reversed_year_ratio,
        "monotonicity": res.monotonicity,
        "top_turnover": res.turnover.iloc[1:].mean(),
        "long_excess_ann": res.long_excess.mean() * PERIODS_PER_YEAR,
        "long_excess_net_ann": res.long_excess_net.mean() * PERIODS_PER_YEAR,
        "passed": res.passed,
    }


def write_factor_report(res: FactorResult, factor: Factor, out_dir: Path, context: str) -> None:
    img = _plots(res, out_dir / "img")
    row = summary_row(res)
    ic_table = pd.DataFrame(res.rank_ic).T
    ic_table.index = [f"{h} 天" for h in ic_table.index]
    ic_table["n_days"] = ic_table["n_days"].astype(int).astype(str)
    ic_table = ic_table.rename(columns={"mean": "RankIC 均值", "std": "标准差", "icir": "ICIR", "positive_ratio": "IC>0 占比", "t_nw": "t 值（NW）", "n_days": "天数"})
    groups = [c for c in res.quantiles.columns if c != "universe"]
    q_ann = (res.quantiles[groups].mean() * PERIODS_PER_YEAR).to_frame("年化超额（相对等权全体）")
    q_ann.index = [f"第 {g} 组" for g in q_ann.index]
    checks = "\n".join(f"- {'✅' if ok else '❌'} {k}" for k, ok in res.checks.items())

    lines = [
        f"# {res.name}",
        "",
        f"{factor.description}。来源：{factor.source}。",
        "",
        f"- 表达式：`{factor.expr}`",
        f"- 方向：{'+1（越大越好）' if res.direction > 0 else '-1（越小越好）'}，依据：{res.direction_source}",
        f"- {context}",
        "",
        f"## 结论：{'通过' if res.passed else '未通过'}",
        "",
        checks,
        "",
        "## IC（已按方向调整，正数表示和预期方向一致）",
        "",
        ic_table.to_markdown(floatfmt=".4f"),
        "",
        f"Pearson IC（5 天，去极值和标准化之后）：{row['pearson_ic_5d']:.4f}，t 值 {res.pearson_ic_main['t_nw']:.2f}。",
        "",
        f"![cum ic](img/{img['cum_ic']})",
        "",
        f"![decay](img/{img['decay']})",
        "",
        "## 分年度",
        "",
        f"方向反了的年份占比：{res.reversed_year_ratio:.0%}",
        "",
        res.yearly_rank_ic.rename_axis("年份").to_frame("RankIC 均值").to_markdown(floatfmt=".4f"),
        "",
        f"![yearly](img/{img['yearly']})",
        "",
        "## 分层（每 5 个交易日调仓，5 组）",
        "",
        q_ann.to_markdown(floatfmt=".2%"),
        "",
        f"单调性（组号与平均超额的 Spearman 相关）：{res.monotonicity:.2f}",
        "",
        f"![quantiles](img/{img['quantiles']})",
        "",
        "## 换手与成本",
        "",
        f"- 最高组每期换手率：{row['top_turnover']:.1%}",
        f"- 最高组年化超额（扣成本前）：{row['long_excess_ann']:.2%}",
        f"- 最高组年化超额（扣成本后）：{row['long_excess_net_ann']:.2%}",
        "",
        "成本按 docs/DECISIONS.md：买入 0.125%，卖出佣金 + 印花税 + 滑点，最低 5 元佣金不计。",
        "超额是相对同期全体样本等权平均收益，不是相对沪深300 指数。",
        "",
    ]
    (out_dir / f"{res.name}.md").write_text("\n".join(lines))
