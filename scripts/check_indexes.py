"""计划 4 任务 2：查指数数据够不够用。只读，到 2021-12-31 为止（不碰测试段）。

- 中证全指（000985）：均线过滤用，要从 2009 年起（250 日均线预热一年）
- 沪深300 全收益（H00300）：比较基准，要覆盖 2010–2021
- 和 Qlib 数据包里的沪深300 价格指数（SH000300）逐年对比：全收益应每年多 1–3 个百分点（分红）

用法：python scripts/check_indexes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

START, END = "2009-01-01", "2021-12-31"


def coverage(s: pd.Series, calendar: pd.DatetimeIndex) -> dict:
    cal = calendar[(calendar >= s.index.min()) & (calendar <= s.index.max())] if len(s) else calendar[:0]
    return {"起": s.index.min().date() if len(s) else None, "止": s.index.max().date() if len(s) else None,
            "天数": len(s), "区间内缺的交易日": int(len(cal.difference(s.index)))}


def yearly(s: pd.Series) -> pd.Series:
    s = s.dropna()
    last = s.groupby(s.index.year).last()
    first_prev = last.shift(1).fillna(s.groupby(s.index.year).first())
    return last / first_prev - 1


def report(indexes: dict[str, pd.Series], price300: pd.Series, calendar: pd.DatetimeIndex) -> str:
    rows = {name: coverage(s, calendar) for name, s in indexes.items()}
    rows["SH000300（Qlib，价格指数）"] = coverage(price300, calendar)
    out = ["覆盖情况：", pd.DataFrame(rows).T.to_string(), ""]
    table = pd.DataFrame({"沪深300 全收益": yearly(indexes["H00300 沪深300 全收益"]),
                          "沪深300 价格": yearly(price300),
                          "中证全指": yearly(indexes["000985 中证全指"])})
    table["全收益 − 价格"] = table["沪深300 全收益"] - table["沪深300 价格"]
    out += ["分年收益（全收益 − 价格 应为正，约 1–3 个百分点）：",
            table.to_string(float_format=lambda x: f"{x:7.3f}")]
    return "\n".join(out)


def main() -> None:
    from qt.data import init_qlib, trading_calendar
    from qt.full_a import load_index
    from qlib.data import D

    init_qlib()
    calendar = trading_calendar(START, END)
    indexes = {"000985 中证全指": load_index("000985", START, END),
               "H00300 沪深300 全收益": load_index("H00300", START, END)}
    price300 = D.features(["SH000300"], ["$close"], START, END)["$close"].droplevel("instrument")
    print(report(indexes, price300, calendar))


if __name__ == "__main__":
    main()
