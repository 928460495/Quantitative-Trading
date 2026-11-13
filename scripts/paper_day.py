"""纸面跟单：一键运行（计划 5 任务 5）。每个交易日收盘、数据更新之后跑一次。

用法（单进程）：
    python scripts/paper_day.py --start 2026-11-13     # 第一次：定跟单开始日（必须是数据里最新的一天）
    python scripts/paper_day.py                        # 以后每天

做的事：读 live 段数据 → 从开始日起用回测引擎重跑纸面账户（T 日信号、T+1 收盘成交，成本、红利税同回测）→
最新一天出下一个交易日的清单（调仓日存档到 orders_log.jsonl）→ 核对以前存档的清单和重跑的成交 →
按停止规则①②检查 → 写 data/live/paper/ 下的 nav.csv、trades.csv、report-<日期>.md 和清单。
跟单的流程固定为 lab/FINAL.yaml 里检验过的版本（第一次运行时写进 paper.json，之后不变）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from live_orders import run as order_sheet  # noqa: E402
from qt.diagnostics import BENCH  # noqa: E402
from qt.lab import procedure as pc  # noqa: E402
from qt.live.daily import state_from_backtest  # noqa: E402
from qt.live.paper import check_log, load_log, log_orders, stop_checks  # noqa: E402

OUT = ROOT / "data" / "live" / "paper"
__all__ = ["run_day", "load_log"]


def run_day(dev: dict, proc: pc.Procedure, out_dir: Path, start: pd.Timestamp | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "paper.json"
    dates = dev["close"].index
    if cfg_path.exists():
        if start is not None:
            raise ValueError(f"纸面跟单已经开始（{cfg_path}），不能再定开始日")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        if start is None:
            raise ValueError("第一次运行要用 --start 定跟单开始日")
        if pd.Timestamp(start) != dates[-1]:
            raise ValueError(f"开始日 {pd.Timestamp(start).date()} 必须是数据里最新的一天 {dates[-1].date()}")
        cfg = {"anchor": str(pd.Timestamp(start).date()), "procedure": proc.to_dict()}
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    proc = pc.Procedure.from_dict(cfg["procedure"])
    anchor = pd.Timestamp(cfg["anchor"])
    a = dates.get_loc(anchor)
    res = pc.run(dev, proc, a)
    data = pc.market(dev, proc, a)
    last = len(data.dates) - 1
    state = state_from_backtest(res, data, last, anchor=anchor)
    sheet, sheet_problems, orders = order_sheet(dev, proc, state, data.dates[last], out_dir)
    rebalance = last % proc.rebalance_every == 0
    if rebalance or orders.sells or orders.buys:
        log_orders(out_dir / "orders_log.jsonl", data.dates[last], orders)
    log_problems = check_log(out_dir / "orders_log.jsonl", res.trades, data.dates)

    nav = res.daily["nav"]
    hs = dev["index"][BENCH].reindex(dates).ffill().loc[anchor:]
    r, h = nav.pct_change().iloc[1:], hs.pct_change().iloc[1:]
    checks = stop_checks(r, h) if len(r) >= 2 else {"回撤": 0.0, "①回撤复查": False, "满几个月": 0,
                                                    "t": float("nan"), "②失效检查": "未满 24 个月"}
    res.daily.to_csv(out_dir / "nav.csv")
    res.trades.to_csv(out_dir / "trades.csv", index=False)
    day = data.dates[last].date()
    lines = [f"# 纸面跟单：{day}", "",
             f"- 开始日 {anchor.date()}；净值 {nav.iloc[-1]:.2f} 元（开始 {nav.iloc[0]:.0f}），"
             f"累计 {nav.iloc[-1] / nav.iloc[0] - 1:+.2%}；同期沪深300 全收益 {hs.iloc[-1] / hs.iloc[0] - 1:+.2%}",
             f"- 持股 {int(res.daily['n_holdings'].iloc[-1])} 只，现金 {res.daily['cash'].iloc[-1]:.2f} 元",
             f"- 今天{'是' if rebalance else '不是'}调仓日；下一个交易日的清单：{sheet.name}"
             + ("（安全检查有问题，按停止规则③今天不交易）" if sheet_problems else ""), "",
             "## 停止规则", "",
             f"- ① 回撤 {checks['回撤']:.1%}（复查线 −54%）：{'**触发，暂停新买入并复查**' if checks['①回撤复查'] else '未触发'}",
             f"- ② 满 {checks['满几个月']} 个月，相对沪深300 的 t 值 {checks['t']:.2f}：{checks['②失效检查']}",
             "", "## 清单存档核对", "", *([f"- **{p}**" for p in log_problems] or ["- 一致"]), ""]
    (out_dir / f"report-{day}.md").write_text("\n".join(lines), encoding="utf-8")
    return {"date": str(day), "rebalance": rebalance, "n_buys": len(orders.buys), "n_sells": len(orders.sells),
            "nav": float(nav.iloc[-1]), "checks": checks, "log_problems": log_problems,
            "sheet_problems": sheet_problems, "report": out_dir / f"report-{day}.md"}


def main() -> None:
    from qt.lab.store import load_dev
    from run_plan3c_train import POOL_CACHE, latest_version

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", help="第一次运行：跟单开始日（数据里最新的一天）")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    reg = yaml.safe_load((ROOT / "lab" / "FINAL.yaml").read_text(encoding="utf-8"))
    proc = pc.Procedure.from_dict(reg["procedure"])
    dev = load_dev(latest_version(POOL_CACHE, "live"), proc.factors(), segments=("live",))
    out = run_day(dev, proc, args.out, pd.Timestamp(args.start) if args.start else None)
    print(Path(out["report"]).read_text())
    sys.exit(1 if out["log_problems"] or out["sheet_problems"] else 0)


if __name__ == "__main__":
    main()
