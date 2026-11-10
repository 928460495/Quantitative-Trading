"""计划 3c 任务 8：每次回测的标准诊断和五条成功标准（docs/DECISIONS.md 2026-11-10）。

seg 是 scripts/build_composite.load_segment 读回来的一段数据（close、can_buy、index、log_mcap …）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qt.account import TRADING_DAYS, AccountResult
from qt.wide_eval import buyable_benchmark, rank_ic

BENCH = "H00300"
MAX_SHORTFALL = 0.05


def annualize(nav: pd.Series) -> float:
    nav = nav.dropna()
    return float((nav.iloc[-1] / nav.iloc[0]) ** (TRADING_DAYS / (len(nav) - 1)) - 1)


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min())


def _align(bench: pd.Series, nav: pd.Series) -> pd.Series:
    return bench.reindex(nav.index).ffill().bfill()


def rolling_shortfall(nav: pd.Series, bench: pd.Series, months: int = 36) -> pd.DataFrame:
    """按月滑动的每个完整 months 个月窗口：起点是第一天或某个月末，终点是 months 个月后的月末。

    shortfall = 基准年化 − 策略年化（第 ④ 条：每个窗口都不能超过 5 个百分点）。
    """
    bench = _align(bench, nav)
    idx = nav.index
    month_end = pd.Series(np.arange(len(idx)), index=idx).groupby([idx.year, idx.month]).max().to_numpy()
    anchors = np.unique(np.concatenate([[0], month_end]))
    rows = []
    for k in range(len(anchors) - months):
        a, b = anchors[k], anchors[k + months]
        ann = lambda s: (s.iloc[b] / s.iloc[a]) ** (TRADING_DAYS / (b - a)) - 1
        rows.append({"start": idx[a], "end": idx[b], "策略年化": ann(nav), "基准年化": ann(bench)})
    out = pd.DataFrame(rows, columns=["start", "end", "策略年化", "基准年化"])
    out["shortfall"] = out["基准年化"] - out["策略年化"]
    return out


def criteria(nav: pd.Series, bench: pd.Series, nav_cost2x: pd.Series, baseline_annual: float) -> pd.DataFrame:
    """五条成功标准：数值（策略这边）、门槛、是否满足。"""
    bench = _align(bench, nav)
    bench_ann, bench_dd = annualize(bench), max_drawdown(bench)
    roll = rolling_shortfall(nav, bench)
    worst = float(roll["shortfall"].max()) if len(roll) else float("nan")
    rows = {
        "①": (annualize(nav), bench_ann, annualize(nav) > bench_ann),
        "②": (max_drawdown(nav), bench_dd, max_drawdown(nav) >= bench_dd),
        "③": (annualize(nav_cost2x), bench_ann, annualize(nav_cost2x) > bench_ann),
        "④": (worst, MAX_SHORTFALL, bool(len(roll)) and worst <= MAX_SHORTFALL),
        "⑤": (annualize(nav), baseline_annual, annualize(nav) > baseline_annual),
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["数值", "门槛", "满足"]).astype({"满足": bool})


def size_corr(score: pd.DataFrame, log_mcap: pd.DataFrame) -> float:
    """打分和对数流通市值每天横截面 Spearman 相关的绝对值，对时间取平均（研究关卡：< 0.1）。"""
    return float(rank_ic(score, log_mcap).abs().mean())


def _yearly(nav: pd.Series) -> pd.Series:
    end = nav.groupby(nav.index.year).last()
    start = end.shift(1)
    start.iloc[0] = nav.iloc[0]
    return end / start - 1


def diagnose(result: AccountResult | pd.Series, seg: dict, score: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """result：真实账户的 AccountResult，或快速模拟器的日收益。返回 summary / yearly / rolling 三张表。"""
    nan = float("nan")
    if isinstance(result, AccountResult):
        daily = result.daily
        nav = daily["nav"]
        gross = ((nav + daily["cost"]) / nav.shift(1) - 1).fillna(0.0)
        gross_nav = nav.iloc[0] * (1 + gross).cumprod()
        trades = result.trades
        sold = trades[trades["side"] == "sell"].groupby("date")["value"].sum().reindex(nav.index, fill_value=0.0)
        n_trades, turnover = float(len(trades)), float((sold / nav).iloc[1:].mean())
        cost_annual = float((daily["cost"] / nav).iloc[1:].mean() * TRADING_DAYS)
        tier_days = float((daily["slot_limit"] < daily["slot_limit"].max()).sum())
    else:
        nav = 100_000 * (1 + result).cumprod()
        gross_nav, n_trades, turnover, cost_annual, tier_days = None, nan, nan, nan, nan
    bench = _align(seg["index"][BENCH], nav)
    ew = (1 + buyable_benchmark(seg["close"], seg["can_buy"])).cumprod().reindex(nav.index).ffill()
    summary = pd.Series({
        "扣成本后年化": annualize(nav),
        "扣成本前年化": annualize(gross_nav) if gross_nav is not None else nan,
        "沪深300全收益年化": annualize(bench),
        "能买股票等权年化": annualize(ew),
        "最大回撤": max_drawdown(nav),
        "沪深300最大回撤": max_drawdown(bench),
        "夏普": float(nav.pct_change().mean() / nav.pct_change().std() * np.sqrt(TRADING_DAYS)),
        "交易笔数": n_trades,
        "日均换手": turnover,
        "年成本": cost_annual,
        "分档减仓天数": tier_days,
        "合成分和市值相关": size_corr(score, seg["log_mcap"]) if "log_mcap" in seg else nan,
    }).to_frame("值")
    yearly = pd.DataFrame({"策略": _yearly(nav), "能买股票等权": _yearly(ew), "沪深300全收益": _yearly(bench)})
    yearly["相对能买等权"] = yearly["策略"] - yearly["能买股票等权"]
    return {"summary": summary, "yearly": yearly, "rolling": rolling_shortfall(nav, bench)}


def write_report(sections: dict[str, pd.DataFrame], path: Path, title: str, notes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""] + [f"- {n}" for n in notes] + [""]
    for name, table in sections.items():
        lines += [f"## {name}", "", table.to_markdown(floatfmt=".4f"), ""]
    path.write_text("\n".join(lines) + "\n")
