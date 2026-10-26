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

_ALPHA158_GROUPS = {
    "KMID": "K 线实体", "KLEN": "K 线长度", "KMID2": "K 线实体 / 长度", "KUP": "上影线", "KUP2": "上影线 / 长度",
    "KLOW": "下影线", "KLOW2": "下影线 / 长度", "KSFT": "收盘在 K 线中的位置", "KSFT2": "收盘在 K 线中的位置 / 长度",
    "OPEN": "开盘价 / 收盘价", "HIGH": "最高价 / 收盘价", "LOW": "最低价 / 收盘价", "VWAP": "均价 / 收盘价",
    "ROC": "涨跌幅", "MA": "均线 / 收盘价", "STD": "收盘价标准差", "BETA": "价格斜率", "RSQR": "价格趋势 R²",
    "RESI": "价格趋势残差", "MAX": "最高价 / 收盘价", "MIN": "最低价 / 收盘价", "QTLU": "收盘价 80% 分位",
    "QTLD": "收盘价 20% 分位", "RANK": "收盘价在窗口内的排名", "RSV": "收盘价在高低区间的位置",
    "IMAX": "距最高价的天数", "IMIN": "距最低价的天数", "IMXD": "最高价与最低价出现时间差", "CORR": "价格与成交量相关",
    "CORD": "价格变化与成交量变化相关", "CNTP": "上涨天数占比", "CNTN": "下跌天数占比", "CNTD": "涨跌天数差",
    "SUMP": "涨幅合计占比", "SUMN": "跌幅合计占比", "SUMD": "涨跌幅合计差", "VMA": "成交量均线", "VSTD": "成交量标准差",
    "WVMA": "成交量加权波动", "VSUMP": "放量天数占比", "VSUMN": "缩量天数占比", "VSUMD": "放量缩量差",
}


def alpha158() -> tuple[Factor, ...]:
    """Qlib 自带的 Alpha158 因子（默认配置的 158 个，全是价量因子）。方向按训练段的实际 RankIC 定。"""
    from qlib.contrib.data.loader import Alpha158DL

    fields, names = Alpha158DL.get_feature_config()
    out = []
    for expr, name in zip(fields, names):
        if name in _ALPHA158_GROUPS:  # KMID2、KUP2 这类末尾的 2 不是窗口
            desc = _ALPHA158_GROUPS[name]
        else:
            stem = name.rstrip("0123456789")
            window = name[len(stem):]
            desc = _ALPHA158_GROUPS.get(stem, stem) + (f"，{window} 日窗口" if window and window != "0" else "")
        out.append(Factor(name, expr, None, desc, "Qlib Alpha158"))
    return tuple(out)


FACTOR_SETS = {"literature": lambda: LITERATURE, "alpha158": alpha158}

# IC 衰减看这几个持有期；主持有期是 5 天
HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
MAIN_HORIZON = 5


def forward_return_expr(horizon: int) -> str:
    """T 日收盘出信号，T+1 收盘买入，持有 horizon 天后收盘卖出。

    Ref 的负偏移表示往未来看：Ref($close, -1) 是 T+1 收盘价。
    """
    if horizon < 1:
        raise ValueError("持有期至少 1 天")
    return f"Ref($close,-{horizon + 1})/Ref($close,-1)-1"
