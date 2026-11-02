"""核对"能买股票等权"基准（计划 3b 通过条件第三条的比较对象）。只读训练段。

试跑时基准年化 −9.86%（2010–2018），比印象中的全 A 等权低得多，怀疑：
1. 复权价有错（除权日没复权，看起来像大跌）
2. 基准算法有错

用法：python scripts/check_benchmark.py

输出：
- 基准、全部有价格股票等权、沪深300 的分年收益
- 能买股票里单日涨跌幅超过 11% 的格子（主板涨跌停 10%，超过多半是复权错误）有多少、最极端的 20 个
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt.config import segment  # noqa: E402
from qt.wide_eval import buyable_benchmark  # noqa: E402
from run_full_a_screen import load_context  # noqa: E402

LIMIT = 0.11


def yearly(daily: pd.Series) -> pd.Series:
    return (1 + daily.fillna(0)).groupby(daily.index.year).prod() - 1


def main() -> None:
    start, end = segment("train")
    panel, ctx, codes, version = load_context(start, end, 1)
    close = ctx.trim(ctx.close)
    buy = ctx.trim(ctx.can_buy).astype(bool)
    r = close / close.shift(1) - 1
    held = buy.shift(2, fill_value=False)

    bench = buyable_benchmark(close, buy)
    every = r.mean(axis=1)                                        # 全部有价格的股票等权
    hs300 = panel["benchmark_close"].iloc[:, 0].reindex(close.index)
    table = pd.DataFrame({"能买股票等权（基准）": yearly(bench), "全部股票等权": yearly(every),
                          "沪深300（价格）": yearly(hs300.pct_change()),
                          "每天能买的只数": held.sum(axis=1).groupby(close.index.year).mean().round(0)})
    n = len(close) - 1
    total = lambda s: (1 + s.fillna(0)).prod() ** (252 / n) - 1
    print(f"数据版本 {version}；{close.index[0].date()} ~ {close.index[-1].date()}")
    print("\n分年收益：")
    print(table.to_string(float_format=lambda x: f"{x:8.3f}"))
    print(f"\n年化：基准 {total(bench):.3f}，全部股票等权 {total(every):.3f}，沪深300 {total(hs300.pct_change()):.3f}")
    print(f"基准每天平均收益 {bench.mean():.5f}，中位数 {bench.median():.5f}")

    rh = r.where(held)
    big = rh.abs() > LIMIT
    print(f"\n能买股票里单日涨跌幅超过 {LIMIT:.0%} 的格子：{int(big.to_numpy().sum())} 个"
          f"（共 {int(rh.notna().to_numpy().sum())} 个）；其中跌的 {int((rh < -LIMIT).to_numpy().sum())} 个")
    drag = rh.where(big).sum(axis=1) / rh.notna().sum(axis=1)
    print(f"这些格子对基准每天平均收益的贡献 {drag.mean():.5f}（年化约 {drag.mean() * 252:.3f}）")

    s = rh.where(big).stack().dropna()
    s.index.names = ["date", "instrument"]
    worst = s.reindex(s.abs().sort_values(ascending=False).index).head(20)
    raw, factor = panel["raw_close"], panel["factor"]
    rows = []
    for (d, c), v in worst.items():
        i = close.index.get_loc(d)
        p = close.index[i - 1]
        rows.append({"日期": d.date(), "股票": c, "复权收益": round(v, 3),
                     "前一天未复权价": round(raw.at[p, c], 2), "当天未复权价": round(raw.at[d, c], 2),
                     "前一天复权因子": round(factor.at[p, c], 4), "当天复权因子": round(factor.at[d, c], 4)})
    print("\n最极端的 20 个：")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
