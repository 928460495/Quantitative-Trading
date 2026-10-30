"""因子登记表：因子库里每个因子的名字、来源、大类、公式或计算函数。

    from qt.library import ALL_FACTORS, HELD

因子库只负责"能算出来"，不代表有效：筛选在后续计划里按 docs/DECISIONS.md 的规则做，每次检验记试验日志。

Alpha101、Alpha191 的公式原文在 qt/formulas/*.tsv：`formula` 是实际使用的公式，`original` 是原文，
两者不同的地方在 `note` 里写明改了什么；`status` 为 held 的暂缓纳入，`note` 写原因。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

FORMULA_DIR = Path(__file__).resolve().parent / "formulas"


@dataclass(frozen=True)
class FactorDef:
    name: str
    family: str                      # alpha101 / alpha191 / alpha158 / literature / price_extra / fundamental
    category: str                    # 说明书第 7 节的大类；公式类价量因子记"价量"
    source: str
    formula: str | None = None       # 公式类因子：实际使用的公式
    dialect: str | None = None       # "wq" 或 "gtja"
    func: Callable | None = None     # 函数类因子：func(panel) -> 宽表
    original: str | None = None
    note: str = ""


_SOURCES = {
    "alpha101": ("wq", "Kakushadze (2016), 101 Formulaic Alphas, Wilmott; arXiv:1601.00991"),
    "alpha191": ("gtja", "国泰君安《基于短周期价量特征的多因子选股体系》(2017)；公式文本取自 GitHub popbo/alphas"),
}


def _load(family: str) -> tuple[list[FactorDef], list[FactorDef]]:
    dialect, source = _SOURCES[family]
    table = pd.read_csv(FORMULA_DIR / f"{family}.tsv", sep="\t", keep_default_na=False, dtype=str)
    ok, held = [], []
    for r in table.itertuples(index=False):
        f = FactorDef(name=r.name, family=family, category="价量", source=source, formula=r.formula,
                      dialect=dialect, original=r.original, note=r.note)
        (ok if r.status == "ok" else held).append(f)
    return ok, held


ALPHA101, _HELD101 = _load("alpha101")
ALPHA191, _HELD191 = _load("alpha191")
HELD: list[FactorDef] = _HELD101 + _HELD191


# ---------- 函数形式的因子（qt/factor_funcs.py）----------

from qt import factor_funcs as _ff  # noqa: E402

_LIT = "文献与券商研报"
_BARRA = "Barra CNE5 描述变量"
_FIN = "财务与估值（Hou-Xue-Zhang 异象、Liu-Stambaugh-Yuan 2019）"


def _fn(name, category, func, source, note=""):
    return FactorDef(name=name, family="price_extra" if category in ("动量/反转", "低波动", "流动性") or name in
                     ("log_float_mcap", "nlsize", "price_level") else "fundamental",
                     category=category, source=source, func=func, note=note)


PRICE_EXTRA: list[FactorDef] = [
    _fn("ret5", "动量/反转", _ff.ret5, _LIT), _fn("ret20", "动量/反转", _ff.ret20, _LIT, "1 个月反转"),
    _fn("ret60", "动量/反转", _ff.ret60, _LIT), _fn("ret120", "动量/反转", _ff.ret120, _LIT),
    _fn("mom12_1", "动量/反转", _ff.mom12_1, "Jegadeesh-Titman；Barra RSTR"),
    _fn("overnight20", "动量/反转", _ff.overnight20, _LIT, "A 股隔夜与日内收益的反转"),
    _fn("intraday20", "动量/反转", _ff.intraday20, _LIT),
    _fn("high52w", "动量/反转", _ff.high52w, "George & Hwang (2004)"),
    _fn("low52w", "动量/反转", _ff.low52w, _LIT),
    _fn("ideal_amp20", "动量/反转", _ff.ideal_amp20, "开源证券《因子切割论》理想振幅"),
    _fn("vol20", "低波动", _ff.vol20, _LIT), _fn("vol60", "低波动", _ff.vol60, _LIT),
    _fn("ivol20", "低波动", _ff.ivol20, "Ang et al. (2006) 特质波动率"),
    _fn("ivol60", "低波动", _ff.ivol60, "Ang et al. (2006) 特质波动率"),
    _fn("beta252", "低波动", _ff.beta252, _BARRA),
    _fn("max5", "低波动", _ff.max5, "Bali, Cakici & Whitelaw (2011)"),
    _fn("max20", "低波动", _ff.max20, "Bali, Cakici & Whitelaw (2011)"),
    _fn("skew20", "低波动", _ff.skew20, _LIT), _fn("skew60", "低波动", _ff.skew60, _LIT),
    _fn("range20", "低波动", _ff.range20, _LIT),
    _fn("turn_1m", "流动性", _ff.turn_1m, _BARRA + " STOM"), _fn("turn_3m", "流动性", _ff.turn_3m, _BARRA + " STOQ"),
    _fn("turn_12m", "流动性", _ff.turn_12m, _BARRA + " STOA"),
    _fn("abn_turn", "流动性", _ff.abn_turn, "Liu, Stambaugh & Yuan (2019) 异常换手"),
    _fn("turn_std20", "流动性", _ff.turn_std20, _LIT), _fn("turn_cv20", "流动性", _ff.turn_cv20, _LIT, "换手率稳定性"),
    _fn("amihud5", "流动性", _ff.amihud5, "Amihud (2002)"), _fn("amihud20", "流动性", _ff.amihud20, "Amihud (2002)"),
    _fn("amihud60", "流动性", _ff.amihud60, "Amihud (2002)"),
    _fn("ret_turn_corr20", "流动性", _ff.ret_turn_corr20, _LIT),
    _fn("log_float_mcap", "规模", _ff.log_float_mcap, _BARRA + " LNCAP（流通市值）"),
    _fn("nlsize", "规模", _ff.nlsize, _BARRA + " NLSIZE"),
    _fn("price_level", "规模", _ff.price_level, _LIT, "低价效应"),
]

FUNDAMENTAL: list[FactorDef] = [
    _fn("log_total_mcap", "规模", _ff.log_total_mcap, _BARRA + " LNCAP", "总股本 = 净利润 / 每股收益"),
    _fn("ep_ttm", "价值", _ff.ep_ttm, _FIN, "baostock 市盈率，错开一天"),
    _fn("bp", "价值", _ff.bp, _FIN, "baostock 市净率，错开一天"),
    _fn("sp_ttm", "价值", _ff.sp_ttm, _FIN, "baostock 市销率，错开一天"),
    _fn("cfp_ttm", "价值", _ff.cfp_ttm, _FIN, "baostock 市现率，错开一天"),
    _fn("ep_deduct", "价值", _ff.ep_deduct, _FIN, "扣非净利润 TTM / 总市值"),
    _fn("ep_self", "价值", _ff.ep_self, _FIN, "净利润 TTM / 总市值，用来核对 baostock 的市盈率"),
    _fn("revenue_yoy", "成长", _ff.revenue_yoy, _FIN, "单季营收同比"),
    _fn("profit_yoy", "成长", _ff.profit_yoy, _FIN, "单季归母净利润同比"),
    _fn("deduct_yoy", "成长", _ff.deduct_yoy, _FIN, "单季扣非净利润同比"),
    _fn("sue", "成长", _ff.sue, "Foster, Olsen & Shevlin (1984) SUE", "季节性随机游走，最近 8 个季度的标准差"),
    _fn("roe_ttm", "质量", _ff.roe_ttm, _FIN), _fn("roa_ttm", "质量", _ff.roa_ttm, _FIN),
    _fn("gross_margin_ttm", "质量", _ff.gross_margin_ttm, _FIN), _fn("net_margin_ttm", "质量", _ff.net_margin_ttm, _FIN),
    _fn("cfo_to_np", "质量", _ff.cfo_to_np, _FIN, "经营现金流 / 净利润，净利润 ≤ 0 时缺失"),
    _fn("accruals", "质量", _ff.accruals, "Sloan (1996) 应计项目"),
    _fn("debt_ratio", "质量", _ff.debt_ratio, _FIN), _fn("asset_growth", "质量", _ff.asset_growth, "Cooper et al. (2008)"),
]


# ---------- 已有的 Qlib 表达式因子 ----------

_LITERATURE_CATEGORY = {"rev20": "动量/反转", "amt_ratio20": "流动性", "vol20": "低波动", "max20": "低波动",
                        "amihud20": "流动性", "high52w": "动量/反转", "pv_corr20": "价量"}


def all_factors() -> list[FactorDef]:
    """因子库全部因子。文献因子和 Alpha158 是 Qlib 表达式（dialect="qlib"），名字加前缀避免和函数因子重名。"""
    from qt.factors import LITERATURE, alpha158

    qlib = [FactorDef(name=f"lit_{f.name}", family="literature", category=_LITERATURE_CATEGORY[f.name], source=f.source,
                      formula=f.expr, dialect="qlib", note=f.description) for f in LITERATURE]
    qlib += [FactorDef(name=f"a158_{f.name}", family="alpha158", category="价量", source="Qlib Alpha158",
                       formula=f.expr, dialect="qlib", note=f.description) for f in alpha158()]
    return qlib + ALPHA101 + ALPHA191 + PRICE_EXTRA + FUNDAMENTAL
