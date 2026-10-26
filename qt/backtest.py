"""Qlib 正式回测：自写的缓冲规则策略，加上按日期收印花税的交易所。

和 qt/simulate.py 的近似模拟相比，这里按真实账户记账：按 1 手取整、权重随价格漂移、留一点现金、
最低 5 元佣金，基准是沪深300 指数。交易规则相同：
- T 日收盘出信号，T+1 成交（默认收盘价；可以改开盘价做对照）
- 排名进前 n_hold 才买，已持有的掉出 sell_rank 才卖
- 买不进（停牌、涨停、不在 buy_allowed 里）的跳过，顺延到下一名；卖不掉（停牌、跌停）的继续持有
- 新买的股票平分卖出后的现金
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from qt.config import check_range
from qt.costs import buy_cost, sell_cost
from qt.data import init_qlib, trading_calendar
from qt.evaluate import TRADING_DAYS
from qt.simulate import PRIMARY_RULE, BufferRule
from qt.universe import LIMIT_THRESHOLD

ACCOUNT = 1e8
BENCHMARK = "SH000300"
RISK_DEGREE = 0.99  # 买入时最多用掉现金的 99%，留出佣金
MIN_COST = 5.0


def _exchange_cls():
    from qlib.backtest.exchange import Exchange

    class DatedCostExchange(Exchange):
        """成本按成交日期取 qt/costs.py 的值（印花税 2023-08-28 减半），可以整体乘一个倍数。"""

        def __init__(self, *args, cost_multiplier: float = 1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.cost_multiplier = cost_multiplier

        def _calc_trade_info_by_order(self, order, position, dealt_order_amount):
            day = pd.Timestamp(order.start_time)
            self.open_cost = buy_cost(day) * self.cost_multiplier
            self.close_cost = sell_cost(day) * self.cost_multiplier
            return super()._calc_trade_info_by_order(order, position, dealt_order_amount)

    return DatedCostExchange


def _strategy_cls():
    from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
    from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy

    class BufferStrategy(BaseSignalStrategy):
        def __init__(self, *, n_hold: int, sell_rank: int, buy_allowed: dict | None = None, **kwargs):
            """buy_allowed：信号日 → 当天允许新买的股票集合；None 表示不限制（仍受停牌、涨停限制）。"""
            super().__init__(**kwargs)
            self.n_hold, self.sell_rank, self.buy_allowed = n_hold, sell_rank, buy_allowed

        def generate_trade_decision(self, execute_result=None):
            step = self.trade_calendar.get_trade_step()
            start, end = self.trade_calendar.get_step_time(step)
            signal_day, signal_end = self.trade_calendar.get_step_time(step, shift=1)
            score = self.signal.get_signal(start_time=signal_day, end_time=signal_end)
            if score is None:
                return TradeDecisionWO([], self)
            if isinstance(score, pd.DataFrame):
                score = score.iloc[:, 0]
            rank = score.dropna().rank(ascending=False, method="first")
            ex = self.trade_exchange

            position = copy.deepcopy(self.trade_position)
            cash = position.get_cash()
            sells, keep = [], []
            for code in position.get_stock_list():
                leave = code not in rank.index or rank[code] > self.sell_rank
                if leave and ex.is_stock_tradable(stock_id=code, start_time=start, end_time=end, direction=OrderDir.SELL):
                    order = Order(stock_id=code, amount=position.get_stock_amount(code),
                                  start_time=start, end_time=end, direction=Order.SELL)
                    if ex.check_order(order):
                        value, cost, _ = ex.deal_order(order, position=position)
                        cash += value - cost
                        sells.append(order)
                        continue
                keep.append(code)

            allowed = None if self.buy_allowed is None else self.buy_allowed.get(pd.Timestamp(signal_day), set())
            picks = []
            for code in rank.sort_values().index:
                if len(picks) >= self.n_hold - len(keep):
                    break
                if code in keep or (allowed is not None and code not in allowed):
                    continue
                if ex.is_stock_tradable(stock_id=code, start_time=start, end_time=end, direction=OrderDir.BUY):
                    picks.append(code)

            buys = []
            value = cash * self.risk_degree / len(picks) if picks else 0.0
            for code in picks:
                price = ex.get_deal_price(stock_id=code, start_time=start, end_time=end, direction=OrderDir.BUY)
                factor = ex.get_factor(stock_id=code, start_time=start, end_time=end)
                amount = ex.round_amount_by_trade_unit(value / price, factor)
                if amount > 0:
                    buys.append(Order(stock_id=code, amount=amount, start_time=start, end_time=end, direction=Order.BUY))
            return TradeDecisionWO(sells + buys, self)

    return BufferStrategy


def _allowed_sets(buy_allowed: pd.Series) -> dict:
    ok = buy_allowed[buy_allowed]
    return {d: set(g.index.get_level_values("instrument")) for d, g in ok.groupby(level="datetime")}


def run_backtest(
    score: pd.Series,
    buy_allowed: pd.Series | None = None,
    rule: BufferRule = PRIMARY_RULE,
    strategy: str = "buffer",
    n_drop: int = 3,
    deal_price: str = "close",
    cost_multiplier: float = 1.0,
    account: float = ACCOUNT,
) -> pd.DataFrame:
    """score 以 (datetime, instrument) 为索引，datetime 是信号日。返回 Qlib 的逐日账户报告。

    strategy="buffer" 用缓冲规则；"topk_dropout" 用 Qlib 的 TopkDropout（持仓 rule.n_hold，每天最多换 n_drop 只）。
    回测从第一个信号日的下一个交易日开始，到最后一个信号日的下一个交易日结束。
    """
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

    days = score.index.get_level_values("datetime")
    first, last = days.min(), days.max()
    cal = trading_calendar(str(first.date()), str((last + pd.Timedelta(days=15)).date()))
    start, end = cal[1], cal[cal.get_loc(last) + 1]
    check_range(str(start.date()), str(end.date()))
    init_qlib()

    codes = sorted(score.index.get_level_values("instrument").unique())
    exchange = _exchange_cls()(
        freq="day", start_time=start, end_time=end, codes=codes, deal_price=deal_price,
        limit_threshold=LIMIT_THRESHOLD, min_cost=MIN_COST * cost_multiplier, cost_multiplier=cost_multiplier,
    )
    if strategy == "buffer":
        allowed = None if buy_allowed is None else _allowed_sets(buy_allowed.reindex(score.index, fill_value=False))
        strat = _strategy_cls()(signal=score, n_hold=rule.n_hold, sell_rank=rule.sell_rank,
                                buy_allowed=allowed, risk_degree=RISK_DEGREE)
    elif strategy == "topk_dropout":
        strat = TopkDropoutStrategy(signal=score, topk=rule.n_hold, n_drop=n_drop, risk_degree=RISK_DEGREE)
    else:
        raise ValueError(f"未知的策略: {strategy}")
    executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    portfolio, _ = backtest(start, end, strat, executor, benchmark=BENCHMARK, account=account,
                            exchange_kwargs={"exchange": exchange})
    report, _ = portfolio["1day"]
    return report


def report_metrics(report: pd.DataFrame) -> dict[str, float]:
    """扣成本后的收益、相对沪深300 的超额、信息比率、超额最大回撤（按累加计）、换手。"""
    net = report["return"] - report["cost"]
    excess = net - report["bench"]
    curve = excess.cumsum()
    return {
        "return_ann": net.mean() * TRADING_DAYS,
        "bench_ann": report["bench"].mean() * TRADING_DAYS,
        "excess_ann": excess.mean() * TRADING_DAYS,
        "info_ratio": excess.mean() / excess.std() * np.sqrt(TRADING_DAYS) if excess.std() > 0 else np.nan,
        "excess_max_drawdown": float((curve - curve.cummax()).min()),
        "cost_ann": report["cost"].mean() * TRADING_DAYS,
        "turnover": report["turnover"].mean(),
        "n_days": len(report),
    }
