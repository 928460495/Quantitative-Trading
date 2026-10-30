"""公式解析器：把 Alpha101（WorldQuant 写法）和 Alpha191（国泰君安写法）的原文公式算成宽表。

    evaluate("(-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))", env)

env 是宽表字典（行是日期、列是股票），键名见 qt/panel.py。函数名、变量名不区分大小写。

两种写法的差别在这里统一：
- 运算：`+ - * / ^`（乘方，右结合）、比较（`=` 和 `==` 都是相等）、`&&`/`||`（也接受 `&`、`|`、`OR`）、三元 `a ? b : c`
- MAX/MIN 两个参数时：第二个参数是 ≥ 2 的整数，按"过去 n 天的最大（小）值"；否则逐元素取大（小）
- 窗口参数不是整数时向下取整（Alpha101 论文的约定）
- 比较、逻辑运算的结果是 1.0 / 0.0，任何一边缺失结果就缺失
- 除法得到无穷大时记为缺失
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qt import ops


class FormulaError(ValueError):
    """公式写错，或用到不认识的函数、变量。"""


# ---------- 词法 ----------

_FULLWIDTH = str.maketrans({"？": "?", "：": ":", "，": ",", "（": "(", "）": ")"})
_TOKEN = re.compile(r"\s*(?:(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+)|([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)"
                    r"|(&&|\|\||<=|>=|==|!=|[-+*/^<>=?:(),&|]))")
_OP_ALIASES = {"&": "&&", "|": "||", "=": "==", "OR": "||", "AND": "&&"}


def _tokens(text: str) -> list[tuple[str, str]]:
    text = text.translate(_FULLWIDTH).strip()
    out, pos = [], 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise FormulaError(f"第 {pos} 个字符无法识别：{text[pos:pos + 10]!r}")
        num, name, op = m.groups()
        if num is not None:
            out.append(("num", num))
        elif name is not None and name.upper() in ("OR", "AND"):
            out.append(("op", _OP_ALIASES[name.upper()]))
        elif name is not None:
            out.append(("name", name))
        elif op is not None:
            out.append(("op", _OP_ALIASES.get(op, op)))
        pos = m.end()
    return out


# ---------- 语法树 ----------

@dataclass
class Node:
    kind: str                 # num / var / call / unary / binary / ternary
    value: object = None      # 数值、变量名、函数名、运算符
    args: tuple = ()

    def variables(self) -> set[str]:
        own = {str(self.value).lower()} if self.kind == "var" else set()
        return own.union(*(a.variables() for a in self.args)) if self.args else own


_INFIX = {"||": 2, "&&": 3, "<": 4, ">": 4, "<=": 4, ">=": 4, "==": 4, "!=": 4,
          "+": 5, "-": 5, "*": 6, "/": 6, "^": 7}
_RIGHT = {"^"}
_TERNARY_BP = 1
_UNARY_BP = 6.5  # 比乘除紧、比乘方松：-x^2 = -(x^2)


class _Parser:
    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else ("end", "")

    def take(self, value=None):
        tok = self.peek()
        if value is not None and tok[1] != value:
            raise FormulaError(f"这里应该是 {value!r}，实际是 {tok[1]!r}")
        self.i += 1
        return tok

    def expr(self, min_bp: float = 0) -> Node:
        kind, val = self.take()
        if kind == "num":
            left = Node("num", float(val))
        elif kind == "name":
            if self.peek() == ("op", "("):
                self.take("(")
                args = []
                if self.peek() != ("op", ")"):
                    args.append(self.expr())
                    while self.peek() == ("op", ","):
                        self.take(",")
                        args.append(self.expr())
                self.take(")")
                left = Node("call", val.upper(), tuple(args))
            else:
                left = Node("var", val)
        elif (kind, val) == ("op", "("):
            left = self.expr()
            self.take(")")
        elif (kind, val) == ("op", "-"):
            left = Node("unary", "-", (self.expr(_UNARY_BP),))
        elif (kind, val) == ("op", "+"):
            left = self.expr(_UNARY_BP)
        else:
            raise FormulaError(f"不该出现在这里：{val!r}")

        while True:
            kind, op = self.peek()
            if kind != "op":
                break
            if op == "?":
                if _TERNARY_BP < min_bp:
                    break
                self.take("?")
                a = self.expr()
                self.take(":")
                b = self.expr(_TERNARY_BP)
                left = Node("ternary", "?", (left, a, b))
                continue
            bp = _INFIX.get(op)
            if bp is None or bp < min_bp or (bp == min_bp and op not in _RIGHT):
                break
            self.take()
            right = self.expr(bp if op in _RIGHT else bp + 0.01)
            left = Node("binary", op, (left, right))
        return left


def parse(text: str, dialect: str = "gtja") -> Node:
    p = _Parser(_tokens(text))
    tree = p.expr()
    if p.peek()[0] != "end":
        raise FormulaError(f"公式末尾多出：{p.peek()[1]!r}")
    return tree


# ---------- 求值 ----------

def _window(x) -> int:
    if isinstance(x, pd.DataFrame):
        raise FormulaError("窗口参数必须是常数")
    return int(np.floor(float(x)))


def _reg_window(args) -> int:
    """REGBETA(y, x, n)；国泰君安里常写成 REGBETA(y, SEQUENCE(n))，窗口就是序列长度。"""
    if len(args) >= 3:
        return _window(args[2])
    if isinstance(args[1], ops.Sequence):
        return args[1].n
    raise FormulaError("REGBETA / REGRESI 缺窗口参数")


def _num_frame(x, like: pd.DataFrame) -> pd.DataFrame:
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame(float(x), index=like.index, columns=like.columns)


def _clean(x):
    return x.replace([np.inf, -np.inf], np.nan) if isinstance(x, pd.DataFrame) else x


def _missing(*xs):
    """任何一边缺失的位置。"""
    masks = [x.isna() for x in xs if isinstance(x, pd.DataFrame)]
    return None if not masks else masks[0].copy() if len(masks) == 1 else np.logical_or.reduce(masks)


def _as_float(result, *operands):
    out = result.astype(float) if isinstance(result, pd.DataFrame) else float(result)
    miss = _missing(*operands)
    if miss is not None and isinstance(out, pd.DataFrame):
        out = out.mask(miss)
    return out


def _max_min(args, take_max: bool, dialect: str):
    a, b = args
    window_like = not isinstance(b, pd.DataFrame) and float(b) >= 2 and float(b).is_integer()
    if dialect == "wq" and not isinstance(b, pd.DataFrame):
        window_like = True  # Alpha101 论文：min(x, d) = ts_min(x, d)
    if window_like:
        return (ops.ts_max if take_max else ops.ts_min)(a, _window(b))
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        like = a if isinstance(a, pd.DataFrame) else b
        a, b = _num_frame(a, like), _num_frame(b, like)
        out = np.maximum(a, b) if take_max else np.minimum(a, b)
        return out.mask(_missing(a, b))
    return max(a, b) if take_max else min(a, b)


def _log(x):
    return np.log(x.where(x > 0)) if isinstance(x, pd.DataFrame) else (np.log(x) if x > 0 else np.nan)


_FUNCS = {
    "RANK": lambda a, d: ops.cs_rank(a[0]),
    "DELAY": lambda a, d: ops.delay(a[0], _window(a[1])),
    "DELTA": lambda a, d: ops.delta(a[0], _window(a[1])),
    "CORR": lambda a, d: ops.ts_corr(a[0], a[1], _window(a[2])),
    "CORRELATION": lambda a, d: ops.ts_corr(a[0], a[1], _window(a[2])),
    "COV": lambda a, d: ops.ts_cov(a[0], a[1], _window(a[2])),
    "COVARIANCE": lambda a, d: ops.ts_cov(a[0], a[1], _window(a[2])),
    "SUM": lambda a, d: ops.ts_sum(a[0], _window(a[1])),
    "MEAN": lambda a, d: ops.ts_mean(a[0], _window(a[1])),
    "MA": lambda a, d: ops.ts_mean(a[0], _window(a[1])),
    "STD": lambda a, d: ops.ts_std(a[0], _window(a[1])),
    "STDDEV": lambda a, d: ops.ts_std(a[0], _window(a[1])),
    "TSMAX": lambda a, d: ops.ts_max(a[0], _window(a[1])),
    "TS_MAX": lambda a, d: ops.ts_max(a[0], _window(a[1])),
    "TSMIN": lambda a, d: ops.ts_min(a[0], _window(a[1])),
    "TS_MIN": lambda a, d: ops.ts_min(a[0], _window(a[1])),
    "TSRANK": lambda a, d: ops.ts_rank(a[0], _window(a[1])),
    "TS_RANK": lambda a, d: ops.ts_rank(a[0], _window(a[1]), pct=(d == "wq")),  # WorldQuant：0–1 百分位
    "TS_ARGMAX": lambda a, d: ops.ts_argmax(a[0], _window(a[1])),
    "TS_ARGMIN": lambda a, d: ops.ts_argmin(a[0], _window(a[1])),
    "HIGHDAY": lambda a, d: ops.highday(a[0], _window(a[1])),
    "LOWDAY": lambda a, d: ops.lowday(a[0], _window(a[1])),
    "DECAYLINEAR": lambda a, d: ops.decay_linear(a[0], _window(a[1])),
    "DECAY_LINEAR": lambda a, d: ops.decay_linear(a[0], _window(a[1])),
    "WMA": lambda a, d: ops.wma(a[0], _window(a[1])),
    "SMA": lambda a, d: ops.sma(a[0], float(a[1]), float(a[2])),
    "COUNT": lambda a, d: ops.count(a[0], _window(a[1])),
    "SUMIF": lambda a, d: ops.sumif(a[0], _window(a[1]), a[2]),
    "REGBETA": lambda a, d: ops.regbeta(a[0], a[1], _reg_window(a)),
    "REGRESI": lambda a, d: ops.regresi(a[0], a[1], _reg_window(a)),
    "SEQUENCE": lambda a, d: ops.sequence(_window(a[0])),
    "PROD": lambda a, d: ops.ts_product(a[0], _window(a[1])),
    "PRODUCT": lambda a, d: ops.ts_product(a[0], _window(a[1])),
    "LOG": lambda a, d: _log(a[0]),
    "ABS": lambda a, d: abs(a[0]),
    "SIGN": lambda a, d: np.sign(a[0]),
    "SIGNEDPOWER": lambda a, d: ops.signed_power(a[0], a[1] if isinstance(a[1], pd.DataFrame) else float(a[1])),
    "SCALE": lambda a, d: ops.cs_scale(a[0], float(a[1]) if len(a) > 1 else 1.0),
    "INDNEUTRALIZE": lambda a, d: ops.ind_neutralize(a[0], a[1]),
    "MAX": lambda a, d: _max_min(a, True, d),
    "MIN": lambda a, d: _max_min(a, False, d),
}

_INDCLASS = {"indclass.sector": "industry_l1", "indclass.industry": "industry_l2", "indclass.subindustry": "industry_l3"}
_ALIASES = {"ret": "returns", "vol": "volume", "cap": "float_mcap", "banchmarkindexclose": "benchmark_close",
            "benchmarkindexclose": "benchmark_close", "banchmarkindexopen": "benchmark_open",
            "benchmarkindexopen": "benchmark_open"}


class _Evaluator:
    def __init__(self, env: dict[str, pd.DataFrame], dialect: str):
        self.env = {k.lower(): v for k, v in env.items()}
        self.dialect = dialect
        self.cache: dict[str, pd.DataFrame] = {}

    def var(self, name: str):
        key = name.lower()
        key = _INDCLASS.get(key, _ALIASES.get(key, key))
        if key in self.env:
            return self.env[key]
        if key in self.cache:
            return self.cache[key]
        m = re.fullmatch(r"adv(\d+)", key)
        if m:
            val = ops.ts_mean(self.env["amount"], int(m.group(1)))
        elif key in ("hd", "ld", "tr", "dtm", "dbm"):
            val = self._gtja_intermediate(key)
        else:
            raise FormulaError(f"不认识的变量：{name}")
        self.cache[key] = val
        return val

    def _gtja_intermediate(self, key: str) -> pd.DataFrame:
        """国泰君安研报里预先定义的几个中间量。"""
        o, h, low, c = (self.env[k] for k in ("open", "high", "low", "close"))
        po, pc = o.shift(1), c.shift(1)
        if key == "hd":
            return h - h.shift(1)
        if key == "ld":
            return low.shift(1) - low
        if key == "tr":
            return np.maximum(np.maximum(h - low, (h - pc).abs()), (low - pc).abs())
        if key == "dtm":
            return np.maximum(h - o, o - po).where(o > po, 0.0).mask(o.isna() | po.isna())
        return np.maximum(o - low, o - po).where(o < po, 0.0).mask(o.isna() | po.isna())  # dbm

    def eval(self, n: Node):
        if n.kind == "num":
            return n.value
        if n.kind == "var":
            return self.var(n.value)
        if n.kind == "unary":
            return -self.eval(n.args[0])
        if n.kind == "call":
            fn = _FUNCS.get(n.value)
            if fn is None:
                raise FormulaError(f"不认识的函数：{n.value}")
            return _clean(fn([self.eval(a) for a in n.args], self.dialect))
        if n.kind == "ternary":
            cond, a, b = (self.eval(x) for x in n.args)
            like = next((x for x in (cond, a, b) if isinstance(x, pd.DataFrame)), None)
            if like is None:
                return a if cond else b
            cond, a, b = (_num_frame(x, like) for x in (cond, a, b))
            return a.where(cond > 0.5, b).mask(cond.isna())
        a, b = (self.eval(x) for x in n.args)
        op = n.value
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return _clean(a / b)
        if op == "^":
            return _clean(a ** b)
        if op in ("&&", "||"):
            fa, fb = (x > 0.5 for x in (a, b))
            return _as_float((fa & fb) if op == "&&" else (fa | fb), a, b)
        cmp = {"<": lambda: a < b, ">": lambda: a > b, "<=": lambda: a <= b, ">=": lambda: a >= b,
               "==": lambda: a == b, "!=": lambda: a != b}[op]
        return _as_float(cmp(), a, b)


def evaluate(text: str, env: dict[str, pd.DataFrame], dialect: str = "gtja") -> pd.DataFrame:
    """dialect："wq"（Alpha101）或 "gtja"（Alpha191）。"""
    out = _Evaluator(env, dialect).eval(parse(text, dialect))
    if not isinstance(out, pd.DataFrame):
        raise FormulaError("公式结果是常数")
    return out.astype(float)
