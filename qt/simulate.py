"""按实际交易规则模拟日频组合：持仓 n_hold 只、买卖缓冲、扣成本。

时间约定和标签一致：T 日收盘后算出信号，T+1 收盘成交，组合当天的收益是 T+1 收盘到 T+2 收盘，
也就是 T 日那一行的 1 天远期收益。

缓冲规则（Novy-Marx & Velikov 2016）：排名进前 n_hold 才买入，已持有的排名掉出 sell_rank 才卖出。
- 买不进（入场日停牌或收盘涨停）的股票跳过，顺延到下一名
- 卖不掉（入场日停牌或收盘跌停）的股票被迫继续持有
- 持有期间停牌、没有收益数据的，当天收益记为 0
- 调出指数的股票当天卖出（数据里已经没有它这一行，按可以卖处理）

组合等权，不考虑权重漂移；最低 5 元佣金不计。这是因子检验用的近似，正式回测在第 5 个 PR。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qt.costs import buy_cost, sell_cost


@dataclass(frozen=True)
class BufferRule:
    n_hold: int = 30
    sell_rank: int = 90

    @property
    def label(self) -> str:
        return f"{self.n_hold}/{self.sell_rank}"


PRIMARY_RULE = BufferRule(30, 90)
COMPARISON_RULES = (BufferRule(30, 60), BufferRule(30, 120))


def simulate_buffer(
    score: pd.Series,
    ret1: pd.Series,
    can_buy: pd.Series,
    can_sell: pd.Series,
    rule: BufferRule = PRIMARY_RULE,
) -> pd.DataFrame:
    """返回逐日结果：portfolio、universe（有打分的股票等权平均）、excess、buys、sells、cost、net。

    score 越大越好；四个序列使用同一个 (datetime, instrument) 索引。
    """
    panel = pd.DataFrame({"score": score, "ret1": ret1, "can_buy": can_buy, "can_sell": can_sell})
    held: list[str] = []
    rows = []
    for d, day in panel.groupby(level="datetime"):
        day = day.droplevel("datetime")
        scored = day["score"].dropna()
        rank = scored.rank(ascending=False, method="first")

        keep, sells = [], 0
        for code in held:
            sellable = bool(day["can_sell"].get(code, True))
            leave = code not in rank.index or rank[code] > rule.sell_rank
            if leave and sellable:
                sells += 1
            else:
                keep.append(code)

        buyable = day["can_buy"].reindex(rank.index, fill_value=False)
        candidates = [c for c in rank[buyable].sort_values().index if c not in keep]
        new = candidates[: max(rule.n_hold - len(keep), 0)]
        held = keep + new

        r = day["ret1"]
        portfolio = r.reindex(held).fillna(0.0).mean() if held else 0.0
        universe = r.reindex(scored.index).mean()
        cost = (len(new) * buy_cost(d) + sells * sell_cost(d)) / rule.n_hold
        rows.append((d, portfolio, universe, len(new), sells, cost))

    out = pd.DataFrame(rows, columns=["datetime", "portfolio", "universe", "buys", "sells", "cost"]).set_index("datetime")
    out["excess"] = out["portfolio"] - out["universe"]
    out["net"] = out["excess"] - out["cost"]
    return out


def turnover(sim: pd.DataFrame, n_hold: int) -> pd.Series:
    """单边换手：当天卖出的只数占持仓数的比例。第一天只有买入，不计。"""
    return sim["sells"].iloc[1:] / n_hold
