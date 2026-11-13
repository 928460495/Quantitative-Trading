"""出单流程：账户状态文件、调仓日和持股只数上限、安全检查、买卖清单（计划 5 任务 2、3）。

调仓日从跟单开始那天（anchor）起每 rebalance_every 个交易日一次，和回测从第一天起算一样。
持股只数上限 = floor(n_hold × 信号日的持股比例)，持股比例来自 qt/lab/procedure.exposure（没有仓位规则时为满仓）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from qt.account import LOT, STAR_MIN_SHARES, AccountResult, AccountRules, MarketData
from qt.live.orders import Orders, plan_orders


@dataclass
class State:
    anchor: pd.Timestamp                                       # 跟单开始的那天（调仓日从这天起算）
    cash: float
    positions: dict[str, dict] = field(default_factory=dict)   # 代码 → {"shares": 股数, "buy_day": "YYYY-MM-DD"}，按买入先后


def save_state(state: State, path: Path) -> None:
    Path(path).write_text(json.dumps({"anchor": str(state.anchor.date()), "cash": state.cash,
                                      "positions": state.positions}, ensure_ascii=False, indent=1), encoding="utf-8")


def load_state(path: Path) -> State:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return State(anchor=pd.Timestamp(d["anchor"]), cash=float(d["cash"]), positions=d["positions"])


def state_from_backtest(result: AccountResult, data: MarketData, s: int, anchor: pd.Timestamp) -> State:
    """回测账户第 s 天收盘后的状态（核对用）。"""
    pos: dict[str, dict] = {}
    for t in result.trades[result.trades["date"] <= data.dates[s]].itertuples():
        if t.side == "sell":
            pos.pop(t.instrument, None)
        else:
            pos[t.instrument] = {"shares": int(t.shares), "buy_day": str(pd.Timestamp(t.date).date())}
    return State(anchor=pd.Timestamp(anchor), cash=float(result.daily["cash"].iloc[s]), positions=pos)


def make_orders(data: MarketData, rules: AccountRules, exposure: pd.Series | None, state: State,
                signal_date: pd.Timestamp) -> tuple[Orders, dict]:
    """信号日收盘后出单：用信号日的打分、能买能卖、行业；股数和金额按信号日收盘价（参考价）算。"""
    dates = data.dates
    s, a = dates.get_loc(pd.Timestamp(signal_date)), dates.get_loc(state.anchor)
    if s < a:
        raise ValueError(f"信号日 {signal_date} 在跟单开始日 {state.anchor.date()} 之前")
    rebalance = (s - a) % rules.rebalance_every == 0
    exp = 1.0 if exposure is None else float(exposure.reindex([dates[s]]).fillna(1.0).iloc[0])
    slot_limit = int(np.floor(rules.n_hold * exp + 1e-9))
    raw_marks = pd.DataFrame(data.raw_close[: s + 1], columns=data.instruments).ffill().iloc[-1]
    row = lambda a: pd.Series(a[s], index=data.instruments)
    price = row(data.raw_close)
    values = {c: p["shares"] * float(raw_marks.get(c, np.nan)) for c, p in state.positions.items()}
    orders = plan_orders(score=row(data.score), can_buy=row(data.can_buy), can_sell=row(data.can_sell),
                         industry=None if data.industry is None else row(data.industry), tradable=price.notna(),
                         holdings=values, cash=state.cash, price=price, slot_limit=slot_limit, rebalance=rebalance,
                         rules=rules, date=dates[s])
    meta = {"signal_date": str(dates[s].date()), "rebalance": rebalance, "slot_limit": slot_limit, "cash": state.cash,
            "n_held": len(state.positions), "nav_estimate": state.cash + float(np.nansum(list(values.values())))}
    return orders, meta


def check_orders(orders: Orders, held: dict[str, int], slot_limit: int) -> list[str]:
    """出单前的安全检查；返回发现的问题（空表示通过）。held：代码 → 现有股数。"""
    problems = []
    if orders.cash_after < 0:
        problems.append(f"买入总额超过现金加卖出所得：按参考价算差 {-orders.cash_after:.2f} 元")
    for b in orders.buys:
        star = b.code.startswith(("SH688", "SH689"))
        if b.shares <= 0 or (star and b.shares < STAR_MIN_SHARES) or (not star and b.shares % LOT):
            problems.append(f"{b.code} 买入 {b.shares} 股不是整手")
    sold = {c for c, _ in orders.sells}
    problems += [f"{c} 要卖但没有持有" for c in sorted(sold - set(held))]
    problems += [f"{c} 又卖又买" for c in sorted(sold & {b.code for b in orders.buys})]
    after = len(set(held) - sold) + len(orders.buys)
    if after > slot_limit:
        problems.append(f"成交后持股只数 {after} 超过上限 {slot_limit}")
    return problems


REASONS = {"rank": "跌出排名", "exposure": "降仓位", "buy": "买入"}


def write_sheet(orders: Orders, meta: dict, problems: list[str], out_dir: Path, held: dict[str, int] | None = None) -> Path:
    """写 orders-<信号日>.csv 和 .md；返回 .md 的路径。held：代码 → 现有股数（卖出按全部持股卖）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    held = held or {}
    rows = [{"side": "sell", "code": c, "shares": held.get(c), "price": None, "value": None, "cost": None, "reason": r}
            for c, r in orders.sells]
    rows += [{"side": "buy", "code": b.code, "shares": b.shares, "price": b.price, "value": round(b.value, 2),
              "cost": round(b.cost, 2), "reason": "buy"} for b in orders.buys]
    table = pd.DataFrame(rows, columns=["side", "code", "shares", "price", "value", "cost", "reason"])
    stem = f"orders-{meta['signal_date']}"
    table.to_csv(out_dir / f"{stem}.csv", index=False)
    show = pd.DataFrame({"方向": table["side"].map({"sell": "卖", "buy": "买"}), "代码": table["code"],
                         "股数": table["shares"].map(lambda x: "" if pd.isna(x) else f"{int(x)}"),
                         "参考价": table["price"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}"),
                         "预计金额": table["value"].map(lambda x: "" if pd.isna(x) else f"{x:.0f}"),
                         "原因": table["reason"].map(REASONS)})
    lines = [f"# 买卖清单：信号日 {meta['signal_date']}（下一个交易日成交）", "",
             f"- 调仓日：{'是' if meta['rebalance'] else '否（只按持股只数上限卖）'}；持股只数上限 {meta['slot_limit']}",
             f"- 现金 {meta['cash']:.2f} 元；按参考价成交后约剩 {orders.cash_after:.2f} 元",
             "- 先卖后买；卖出按全部持股卖；买入股数按信号日收盘价（参考价）算，成交价不同时按实际资金调整", "",
             "## 安全检查", "", *(["- 通过"] if not problems else [f"- **{p}**" for p in problems]), "",
             "## 清单", "", show.to_markdown(index=False) if len(show) else "（今天没有要做的买卖）", ""]
    path = out_dir / f"{stem}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
