"""把单因子检验结果写成 Markdown 报告和图。

图里的文字用英文：图片在没有中文字体的机器上重新生成时也不会变成方框。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from qt.evaluate import FactorResult  # noqa: E402
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
    ax.set_title(f"{res.name}: cumulative daily RankIC ({res.main_horizon}d, direction-adjusted)")
    ax.axhline(0, color="grey", lw=0.5)
    names["cum_ic"] = f"{res.name}_cum_ic.png"
    _save(fig, img_dir / names["cum_ic"])

    fig, ax = plt.subplots(figsize=(8, 3))
    res.yearly_rank_ic.plot.bar(ax=ax, color=["tab:green" if v > 0 else "tab:red" for v in res.yearly_rank_ic])
    ax.set_title(f"{res.name}: mean RankIC by year ({res.main_horizon}d)")
    ax.axhline(0, color="grey", lw=0.5)
    names["yearly"] = f"{res.name}_yearly.png"
    _save(fig, img_dir / names["yearly"])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    groups = [c for c in res.quantiles.columns if c != "universe"]
    (1 + res.quantiles[groups]).cumprod().plot(ax=ax)
    ax.set_title(f"{res.name}: cumulative excess return by quintile (5 = best, {res.main_horizon}d non-overlapping)")
    ax.axhline(1, color="grey", lw=0.5)
    names["quantiles"] = f"{res.name}_quantiles.png"
    _save(fig, img_dir / names["quantiles"])

    fig, ax = plt.subplots(figsize=(6, 3))
    pd.Series({f"{h}d": s["mean"] for h, s in res.rank_ic.items()}).plot.bar(ax=ax)
    ax.set_title(f"{res.name}: RankIC decay")
    ax.axhline(0, color="grey", lw=0.5)
    names["decay"] = f"{res.name}_decay.png"
    _save(fig, img_dir / names["decay"])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for label, sim in res.simulations.items():
        (1 + sim["net"]).cumprod().plot(ax=ax, label=f"buffer {label}")
    ax.set_title(f"{res.name}: daily buffered portfolio, cumulative net excess")
    ax.axhline(1, color="grey", lw=0.5)
    ax.legend()
    names["sim"] = f"{res.name}_sim.png"
    _save(fig, img_dir / names["sim"])
    return names


def summary_row(res: FactorResult) -> dict:
    h = res.main_horizon
    s = res.rank_ic[h]
    primary_label = next(iter(res.sim_summary))
    sim = res.sim_summary[primary_label]
    row = {
        "factor": res.name,
        "category": res.category,
        "direction": res.direction,
        f"rank_ic_{h}d": s["mean"],
        f"icir_{h}d": s["icir"],
        f"t_nw_{h}d": s["t_nw"],
        "ic_positive_ratio": s["positive_ratio"],
        f"pearson_ic_{h}d": res.pearson_ic_main["mean"],
        f"factor_return_t_{h}d": res.factor_return_main["t_nw"],
    }
    row |= {f"rank_ic_{k}d": v["mean"] for k, v in res.rank_ic.items() if k != h}
    row |= {
        "reversed_year_ratio": res.reversed_year_ratio,
        "monotonicity": res.monotonicity,
        "bottom_excess_ann": res.bottom_excess_ann,
        "bottom_t": res.bottom_t,
        "sim_rule": primary_label,
        "sim_turnover": sim["turnover"],
        "sim_excess_ann": sim["excess_ann"],
        "sim_cost_ann": sim["cost_ann"],
        "sim_net_ann": sim["net_ann"],
        "sim_net_ann_all": sim["net_ann_all"],
    }
    return row


def _checklist(checks: dict[str, bool]) -> str:
    return "\n".join(f"- {'✅' if ok else '❌'} {k}" for k, ok in checks.items())


def write_factor_report(res: FactorResult, factor: Factor, out_dir: Path, context: str) -> None:
    img = _plots(res, out_dir / "img")
    h = res.main_horizon

    ic_table = pd.DataFrame(res.rank_ic).T
    ic_table["n_days"] = ic_table["n_days"].astype(int).astype(str)
    ic_table.index = [f"{k} 天" for k in ic_table.index]
    ic_table = ic_table.rename(columns={"mean": "RankIC 均值", "std": "标准差", "icir": "ICIR", "positive_ratio": "IC>0 占比", "t_nw": "t 值（NW）", "n_days": "天数"})

    groups = [c for c in res.quantiles.columns if c != "universe"]
    q_ann = (res.quantiles[groups].mean() * res.periods_per_year).to_frame("年化超额（相对等权全体）")
    q_ann.index = [f"第 {g} 组" for g in q_ann.index]

    sim_table = pd.DataFrame(res.sim_summary).T
    sim_table.index = [f"缓冲 {k}" for k in sim_table.index]
    sim_table = sim_table[["turnover", "excess_ann", "cost_ann", "net_ann", "net_t", "net_ann_all"]]
    sim_table.columns = ["日均单边换手", "年化超额（扣成本前）", "年化成本", "年化超额（扣成本后）", "扣成本后 t 值",
                         "扣成本后（参考：全体基准）"]

    lines = [
        f"# {res.name}",
        "",
        f"{factor.description}。来源：{factor.source}。",
        "",
        f"- 表达式：`{factor.expr}`",
        f"- 方向：{'+1（越大越好）' if res.direction > 0 else '-1（越小越好）'}，依据：{res.direction_source}",
        f"- {context}",
        "",
        f"## 结论：{res.category}",
        "",
        "作为打分因子：",
        "",
        _checklist(res.checks),
        "",
        "作为排雷因子：",
        "",
        _checklist(res.veto_checks),
        "",
        "## IC（已按方向调整，正数表示和预期方向一致）",
        "",
        ic_table.to_markdown(floatfmt=".4f"),
        "",
        f"- Pearson IC（{h} 天，去极值和标准化之后）：{res.pearson_ic_main['mean']:.4f}，t 值 {res.pearson_ic_main['t_nw']:.2f}",
        f"- 回归法（{h} 天远期收益对标准化因子做横截面回归）：因子收益率均值 {res.factor_return_main['mean']:.5f}，"
        f"t 值 {res.factor_return_main['t_nw']:.2f}。只作参考，不设门槛",
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
        f"## 分层（每 {h} 个交易日一次、互不重叠，5 组）",
        "",
        "只用来看信号质量和单调性，不扣成本。",
        "",
        q_ann.to_markdown(floatfmt=".2%"),
        "",
        f"- 单调性（组号与平均超额的 Spearman 相关）：{res.monotonicity:.2f}",
        f"- 最低组年化超额 {res.bottom_excess_ann:.2%}，t 值 {res.bottom_t:.2f}",
        "",
        f"![quantiles](img/{img['quantiles']})",
        "",
        "## 按实际交易规则模拟",
        "",
        "每天调仓，持仓 30 只等权；排名进前 30 才买，已持有的排名掉出缓冲线才卖。",
        "买不进（入场日停牌或涨停）跳过，卖不掉（入场日停牌或跌停）继续持有。",
        "成本：买入 0.125%，卖出佣金 + 印花税 + 滑点（docs/DECISIONS.md），最低 5 元佣金不计。",
        "超额默认相对当天有打分、且第二天能买进的股票等权平均（筛选门槛用这个）；",
        "最后一列相对所有有打分的股票，包括第二天涨停买不进的，作参考。两者都不是相对沪深300 指数。",
        "",
        sim_table.to_markdown(floatfmt=".4f"),
        "",
        f"![sim](img/{img['sim']})",
        "",
    ]
    (out_dir / f"{res.name}.md").write_text("\n".join(lines))
