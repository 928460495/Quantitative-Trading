"""候选因子与标签的定义。

因子写成 Qlib 表达式。expected_sign 是文献给出的预期方向：+1 表示因子值越大、未来收益越高，
-1 相反，None 表示文献没有明确结论，按训练段的实际 RankIC 方向确定。
"""

from __future__ import annotations

from dataclasses import dataclass

_RET1 = "($close/Ref($close,1)-1)"


@dataclass(frozen=True)
class Factor:
    name: str
    expr: str
    expected_sign: int | None
    description: str
    source: str


LITERATURE: tuple[Factor, ...] = (
    Factor("rev20", "$close/Ref($close,20)-1", -1,
           "过去 20 个交易日收益（短期反转）", "Chui, Subrahmanyam, Titman；Yue, Li, Ruan"),
    Factor("amt_ratio20", "$amount/Mean($amount,20)", -1,
           "当日成交额 / 20 日均值（换手率异常的近似，数据包里没有流通股本）", "换手率效应"),
    Factor("vol20", f"Std({_RET1},20)", -1,
           "20 日日收益标准差（低波动异象）", "低波动异象"),
    Factor("max20", f"Max({_RET1},20)", -1,
           "20 日内最大单日收益（彩票效应）", "Bali, Cakici, Whitelaw (2011)"),
    Factor("amihud20", f"Mean(Abs({_RET1})/$amount,20)", None,
           "20 日平均 |日收益| / 成交额（非流动性）", "Amihud (2002)；Leippold, Wang, Zhou (2022)"),
    Factor("high52w", "$close/Max($high,250)", 1,
           "收盘价 / 过去 250 日最高价（52 周新高）", "George & Hwang (2004)"),
    Factor("pv_corr20", "Corr($close,Log($volume+1),20)", None,
           "20 日收盘价与对数成交量的相关性", "Alpha158 CORR 类"),
)

# IC 衰减看这几个持有期；主持有期是 5 天
HORIZONS: tuple[int, ...] = (1, 5, 10, 20)
MAIN_HORIZON = 5


def forward_return_expr(horizon: int) -> str:
    """T 日收盘出信号，T+1 收盘买入，持有 horizon 天后收盘卖出。

    Ref 的负偏移表示往未来看：Ref($close, -1) 是 T+1 收盘价。
    """
    if horizon < 1:
        raise ValueError("持有期至少 1 天")
    return f"Ref($close,-{horizon + 1})/Ref($close,-1)-1"
