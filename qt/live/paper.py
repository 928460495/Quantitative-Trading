"""纸面账户的检查和记录（计划 5 任务 5）。

纸面账户本身就是回测引擎：每次从跟单开始日起用 qt/lab/procedure.run 重跑（scripts/paper_day.py）。这里只有：
- stop_checks：停止规则①②（DECISIONS 2026-11-13「实盘和纸面跟单的停止规则」）
- log_orders / check_log：每个信号日出的清单原样存档；之后重跑时核对回测在下一天的成交和当时的清单是否一致
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qt.evaluate import newey_west_t
from qt.live.orders import Orders

RISK_FREE, NW_LAG, FAIL_T, MIN_MONTHS, EVERY_MONTHS = 0.02, 20, -1.0, 24, 6
DD_LINE_STOCK_ONLY = -0.54                     # 只看选股、满仓：开发段最大回撤约 −45% 的 1.2 倍


def _months(index: pd.DatetimeIndex) -> int:
    a, b = index[0], index[-1]
    return (b.year - a.year) * 12 + b.month - a.month - (b.day < a.day)


def stop_checks(r: pd.Series, h: pd.Series, dd_line: float = DD_LINE_STOCK_ONLY, months: int | None = None) -> dict:
    """r：纸面账户日收益；h：同期沪深300 全收益日收益。months：开始以来满几个月（默认按日期算）。"""
    nav = (1 + r).cumprod()
    dd = float(nav.iloc[-1] / max(nav.max(), 1.0) - 1)
    months = _months(r.index) if months is None else months
    rf = (1 + RISK_FREE) ** (1 / 252) - 1
    t = float("nan")
    if len(r) >= 40 and h.std() > 0:
        a = (r.to_numpy() - rf) - (r.std() / h.std()) * (h.to_numpy() - rf)
        t = float(newey_west_t(pd.Series(a), NW_LAG))
    if months < MIN_MONTHS:
        fail = "未满 24 个月"
    elif months % EVERY_MONTHS:
        fail = "不是检查月"
    else:
        fail = "停用" if t < FAIL_T else "继续"
    return {"回撤": dd, "①回撤复查": bool(dd <= dd_line), "满几个月": months, "t": t, "②失效检查": fail}


def load_log(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def log_orders(path: Path, signal_date: pd.Timestamp, orders: Orders) -> bool:
    """存档一个信号日的清单；同一天已经存过就不再记（返回 False）。"""
    day = str(pd.Timestamp(signal_date).date())
    if any(r["signal_date"] == day for r in load_log(path)):
        return False
    rec = {"signal_date": day, "sells": [list(s) for s in orders.sells],
           "buys": [[b.code, b.shares, b.price] for b in orders.buys], "cash_after": orders.cash_after}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return True


def check_log(path: Path, trades: pd.DataFrame, dates: pd.DatetimeIndex) -> list[str]:
    """回测在信号日下一天的成交，卖出（代码、原因）和买入的股票要和存档的清单一致；下一天还没数据的不核对。"""
    problems = []
    dates = pd.DatetimeIndex(dates)
    for rec in load_log(path):
        s = pd.Timestamp(rec["signal_date"])
        later = dates[dates > s]
        if not len(later):
            continue
        day = trades[trades["date"] == later[0]]
        sells = sorted(map(tuple, day.loc[day["side"] == "sell", ["instrument", "reason"]].to_numpy().tolist()))
        buys = sorted(day.loc[day["side"] == "buy", "instrument"])
        if sells != sorted(map(tuple, rec["sells"])) or buys != sorted(b[0] for b in rec["buys"]):
            problems.append(f"信号日 {rec['signal_date']} 的清单和 {later[0].date()} 重跑的成交对不上"
                            f"（清单卖 {rec['sells']} 买 {[b[0] for b in rec['buys']]}；重跑卖 {sells} 买 {buys}）")
    return problems
