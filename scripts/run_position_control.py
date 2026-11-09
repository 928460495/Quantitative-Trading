"""计划 4 任务 5：合成分 + 真实账户规则 + 仓位控制，训练段、验证段各跑一遍，按事先写定的规则挑 A。

用法（先跑过 scripts/build_composite.py）：
    python scripts/run_position_control.py

规则见 docs/DECISIONS.md 2026-11-09：
- 不控仓（也不用 C）、A 的 6 个组合（都带 C）、对照 B（带 C）
- 挑 A：训练段和验证段最大回撤都 ≤ 20% 的组合里，选验证段年化最高的；都不达标就停下，不放宽
- 选中之后只报告：持仓 10 只（跌出前 30 名卖）、去掉短周期价量、成本翻倍
输出 reports/combination/full_a/：README.md、backtest.csv、yearly.csv、chosen.json、nav_<段>.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt import position_control as pc  # noqa: E402
from qt.account import AccountRules  # noqa: E402
from qt.trials import TRIALS_FILE, log_trial  # noqa: E402

OUT_DIR = ROOT / "reports" / "combination" / "full_a"
NO_CONTROL = "不控仓"
B_NAME = "B_波动率控制"
SHORT_PV = "短周期价量"
SEGMENTS = ("train", "valid")
SEG_CN = {"train": "训练段", "valid": "验证段"}
METRIC_COLS = ["annual_return", "bench_annual", "excess_annual", "max_drawdown", "volatility", "sharpe",
               "avg_exposure", "days_in_cash", "turnover", "cost_annual", "n_stops", "c_triggers"]


def run_all(segs: dict[str, dict], trials_path: Path | None = TRIALS_FILE, data_version: str = "") -> dict:
    """segs：{"train": 段, "valid": 段}。返回 metrics（每个变体 × 段一行）、yearly、navs、chosen、b_target。"""
    target = pc.vol_target(segs["train"])
    rows, yearly, navs = [], [], {}

    def run(name: str, seg_name: str, score, rules: AccountRules, exposure, params: dict) -> None:
        seg = segs[seg_name]
        m, result = pc.run_variant(seg, score, rules, exposure)
        rows.append({"variant": name, "segment": seg_name, **{k: m[k] for k in METRIC_COLS}})
        yearly.append(pc.yearly(result).assign(variant=name, segment=seg_name))
        navs[(name, seg_name)] = result.daily["nav"]
        if trials_path is not None:
            log_trial("position_control", {"variant": name, "segment": seg_name, "market": "full_a_main",
                                           "data_version": data_version, **params},
                      {k: m[k] for k in ("annual_return", "excess_annual", "max_drawdown", "sharpe")}, path=trials_path)

    for seg_name in SEGMENTS:
        seg = segs[seg_name]
        run(NO_CONTROL, seg_name, seg["composite"], AccountRules(drawdown_stop=None), None, {"control": "none"})
        for w in pc.A_WINDOWS:
            for low in pc.A_LOWS:
                run(pc.a_name(w, low), seg_name, seg["composite"], AccountRules(), pc.trend(seg, w, low),
                    {"control": "A+C", "window": w, "low": low})
        run(B_NAME, seg_name, seg["composite"], AccountRules(), pc.vol(seg, target),
            {"control": "B+C", "window": pc.B_WINDOW, "target": round(target, 6)})

    metrics = pd.DataFrame(rows)
    a_rows = metrics[metrics["variant"].str.startswith("A_")]
    wide = a_rows.pivot(index="variant", columns="segment", values=["max_drawdown", "annual_return"])
    table = pd.DataFrame({"train_max_drawdown": wide[("max_drawdown", "train")],
                          "valid_max_drawdown": wide[("max_drawdown", "valid")],
                          "valid_annual": wide[("annual_return", "valid")]})
    chosen = pc.choose_a(table)

    if chosen is not None:
        w, low = next((w, low) for w in pc.A_WINDOWS for low in pc.A_LOWS if pc.a_name(w, low) == chosen)
        for seg_name in SEGMENTS:
            seg = segs[seg_name]
            exp = pc.trend(seg, w, low)
            base = {"control": "A+C", "window": w, "low": low}
            run(f"{chosen}｜持仓10只", seg_name, seg["composite"], AccountRules(n_hold=10, sell_rank=30), exp,
                base | {"n_hold": 10})
            run(f"{chosen}｜去掉{SHORT_PV}", seg_name, pc.composite_without(seg, SHORT_PV), AccountRules(), exp,
                base | {"drop_category": SHORT_PV})
            run(f"{chosen}｜成本翻倍", seg_name, seg["composite"], AccountRules(cost_multiplier=2.0), exp,
                base | {"cost_multiplier": 2.0})
        metrics = pd.DataFrame(rows)

    return {"metrics": metrics, "yearly": pd.concat(yearly), "navs": navs, "chosen": chosen, "a_table": table,
            "b_target": target, "bench": {s: segs[s]["index"][pc.BENCH].reindex(segs[s]["close"].index).ffill()
                                          for s in SEGMENTS}}


def _pct(x) -> str:
    return "" if pd.isna(x) else f"{x:.1%}"


def write_report(res: dict, out_dir: Path = OUT_DIR, note: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, chosen = res["metrics"], res["chosen"]
    metrics.to_csv(out_dir / "backtest.csv", index=False, float_format="%.6f")
    res["yearly"].to_csv(out_dir / "yearly.csv", float_format="%.6f")
    (out_dir / "chosen.json").write_text(json.dumps({"chosen": chosen, "b_target": res["b_target"]},
                                                    ensure_ascii=False, indent=2))

    t = res["a_table"].copy()
    t["两段回撤都 ≤ 20%"] = (t["train_max_drawdown"] >= -pc.MAX_DRAWDOWN - 1e-12) & (t["valid_max_drawdown"] >= -pc.MAX_DRAWDOWN - 1e-12)
    t["选中"] = t.index == chosen
    a = metrics[metrics["variant"].str.startswith("A_") & ~metrics["variant"].str.contains("｜")]
    piv = a.pivot(index="variant", columns="segment", values="annual_return")
    t.insert(0, "训练段年化", piv["train"])
    t = t.rename(columns={"train_max_drawdown": "训练段最大回撤", "valid_max_drawdown": "验证段最大回撤",
                          "valid_annual": "验证段年化"})
    for c in ("训练段年化", "训练段最大回撤", "验证段最大回撤", "验证段年化"):
        t[c] = t[c].map(_pct)

    show = metrics.copy()
    for c in ("annual_return", "bench_annual", "excess_annual", "max_drawdown", "volatility", "avg_exposure",
              "days_in_cash", "turnover", "cost_annual"):
        show[c] = show[c].map(_pct)
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}")
    show = show.rename(columns={"variant": "方案", "segment": "段", "annual_return": "年化", "bench_annual": "沪深300全收益",
                                "excess_annual": "超额", "max_drawdown": "最大回撤", "volatility": "波动",
                                "sharpe": "夏普", "avg_exposure": "平均仓位", "days_in_cash": "空仓天数占比",
                                "turnover": "日均换手", "cost_annual": "年成本", "n_stops": "止损次数",
                                "c_triggers": "回撤兜底触发"})
    show["段"] = show["段"].map(SEG_CN)

    lines = ["# 合成与仓位控制（全 A 主板）", "",
             "规则见 `docs/DECISIONS.md` 2026-11-09。账户：10 万元、15 只、跌出前 45 名卖出、单股 20%、单行业 30%、"
             "止损 −20% 冷却 20 天；T 日收盘出信号、T+1 收盘成交。比较基准：沪深300 全收益指数（H00300）。", ""]
    if note:
        lines += [note, ""]
    lines += ["## 仓位控制 A 的 6 个组合（都带回撤兜底 C）", "",
              "挑选规则：训练段和验证段最大回撤都不超过 20% 的组合里，选验证段年化收益最高的；一个都达不到就停下来讨论，不放宽。", "",
              t.to_markdown(), ""]
    if chosen is None:
        lines += ["**没有组合在两段上回撤都不超过 20%。按事先写定的规则停下来讨论，不放宽，也不跑后面的变体。**", ""]
    else:
        lines += [f"**选中：{chosen}**", ""]
    lines += ["## 全部方案", "",
              f"B 的目标波动率（训练段 20 日波动率中位数）：{res['b_target']:.1%}。名字里带「｜」的是选中组合上的只报告变体。", "",
              show.to_markdown(index=False), ""]
    if chosen is not None:
        y = res["yearly"]
        y = y[y["variant"] == chosen].drop(columns="variant")
        bench_y = pd.concat([(lambda s: (s.groupby(s.index.year).last() / s.groupby(s.index.year).last().shift(1)
                                          .fillna(s.iloc[0]) - 1))(res["bench"][seg]) for seg in SEGMENTS])
        y["沪深300全收益"] = bench_y.reindex(y.index)
        y = y.rename(columns={"return": "收益", "max_drawdown": "年内最大回撤", "segment": "段"})
        y["段"] = y["段"].map(SEG_CN)
        for c in ("收益", "年内最大回撤", "沪深300全收益"):
            y[c] = y[c].map(_pct)
        lines += [f"## {chosen} 分年度", "", y.to_markdown(), ""]
    for seg in SEGMENTS:
        _plot(res, seg, out_dir / f"nav_{seg}.png")
        lines += [f"![{SEG_CN[seg]}](nav_{seg}.png)", ""]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def _plot(res: dict, seg: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bench = res["bench"][seg]
    ax.plot(bench.index, bench / bench.iloc[0], label="沪深300 全收益", color="gray", lw=1)
    names = [NO_CONTROL] + ([res["chosen"]] if res["chosen"] else [])
    for name in names:
        nav = res["navs"][(name, seg)]
        ax.plot(nav.index, nav / nav.iloc[0], label=name, lw=1.3)
    ax.set_title(SEG_CN[seg])
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def find_cache(version: str | None) -> Path:
    from build_composite import CACHE_ROOT

    dirs = sorted([d for d in CACHE_ROOT.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime) if CACHE_ROOT.exists() else []
    if version:
        dirs = [d for d in dirs if d.name == version]
    if not dirs:
        sys.exit(f"{CACHE_ROOT} 下没有合成分，先运行 python scripts/build_composite.py")
    return dirs[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", help="data/cache/composite/ 下的数据版本目录，默认最新的")
    args = ap.parse_args()
    from build_composite import load_segment

    cache = find_cache(args.version)
    print(f"读取 {cache}", flush=True)
    segs = {s: load_segment(cache / s) for s in SEGMENTS}
    res = run_all(segs, data_version=cache.name)
    write_report(res, note=f"数据版本 `{cache.name}`。")
    print(res["a_table"].round(4).to_string())
    print(f"选中：{res['chosen']}" if res["chosen"] else "没有组合两段回撤都 ≤ 20%：按规则停下来讨论")
    print(f"报告：{OUT_DIR.relative_to(ROOT)}/README.md")


if __name__ == "__main__":
    main()
