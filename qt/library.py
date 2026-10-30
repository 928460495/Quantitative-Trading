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
