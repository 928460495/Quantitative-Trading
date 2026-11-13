"""E005（便宜证据，不计试验）：市场整体"便宜程度"能不能预测之后 12 个月的大盘收益。

做法（运行前定好，不看结果改）：
- 市场估值：全 A 主板每天的整体盈利收益率 E/P。两种口径：
  ① 按流通市值加权：Σ(流通市值 ÷ 市盈率TTM) ÷ Σ流通市值（亏损公司的负盈利也计入）
  ② 等权中位数：各股 1 ÷ 市盈率TTM 的中位数
  市盈率用前一天的值（DATA_NOTES：baostock 估值字段要错开一天）；流通市值 = 收盘价 × 由成交量 ÷ 换手率反推的流通股本
- 便宜程度：当天 E/P 在"过去"窗口里排第几个百分位（只和过去比）。两种窗口：过去 5 年（最早 2014 年起）、
  2009 年起的全部过去（至少 3 年，最早 2012 年起）
- 之后 12 个月（252 个交易日）的大盘收益：中证全指（000985，价格指数）。为了不碰封存段，只用终点不晚于 2021-12-31 的窗口，
  所以起点最晚到 2020 年底
- 看三样：每月取一次，百分位和之后 12 个月收益的秩相关（Newey-West t，滞后 12 个月，因为窗口重叠）；
  按便宜程度分三档（贵、中、便宜）的平均之后 12 个月收益；E/P 水平本身和之后收益的相关
样本很小（能用的起点约 2012/2014 – 2020，独立的 12 个月只有 7–9 个），这一步只回答"看不看得到关系"，不能证明有用。
在仓库根目录运行：python lab/evidence/E005_valuation.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qt.evaluate import newey_west_t  # noqa: E402
from qt.full_a import load_daily, load_index  # noqa: E402
from qt.universe import is_main_board  # noqa: E402

START, END = "2009-01-05", "2021-12-31"
HORIZON, STEP = 252, 21
CACHE = ROOT / "lab" / "evidence" / "E005_valuation.parquet"


def market_valuation() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    d = load_daily(["close", "volume", "turn", "peTTM", "tradestatus"], START, END)
    d = d[is_main_board(d.index.get_level_values("instrument"))]
    wide = lambda c: d[c].unstack("instrument").astype(float)
    close, vol, turn, pe = wide("close"), wide("volume"), wide("turn"), wide("peTTM")
    shares = (vol / (turn / 100)).where(turn > 0).ffill()
    mcap = close * shares
    pe = pe.shift(1)                                                        # 估值字段错开一天
    ok = mcap.notna() & pe.notna() & (pe != 0)
    ey_w = (mcap / pe).where(ok).sum(axis=1) / mcap.where(ok).sum(axis=1)
    ey_m = (1 / pe).where(ok).median(axis=1)
    out = pd.DataFrame({"E/P 市值加权": ey_w, "E/P 中位数": ey_m, "股票数": ok.sum(axis=1)})
    out = out[out["股票数"] >= 500]
    out.to_parquet(CACHE)
    return out


def past_percentile(x: pd.Series, window: int | None, min_days: int) -> pd.Series:
    """当天的值在过去窗口（含当天）里的百分位；window=None 用全部过去。只和过去比。"""
    v = x.to_numpy()
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        lo = 0 if window is None else max(0, i - window + 1)
        if i - lo + 1 < min_days:
            continue
        past = v[lo:i + 1]
        out[i] = (past <= v[i]).mean()
    return pd.Series(out, index=x.index)


def main() -> None:
    val = market_valuation()
    idx = load_index("000985", START, END).reindex(val.index).ffill()
    fwd = idx.shift(-HORIZON) / idx - 1                                     # 终点超过 2021-12-31 的自然是空值
    rows, buckets = {}, {}
    for col in ("E/P 市值加权", "E/P 中位数"):
        for name, window, min_days in (("过去 5 年", 1260, 1260), ("2009 年起全部过去", None, 756)):
            pct = past_percentile(val[col], window, min_days)
            df = pd.DataFrame({"便宜程度": pct, "E/P": val[col], "之后12个月": fwd}).dropna()
            m = df.iloc[::STEP]
            rho = m["便宜程度"].rank().corr(m["之后12个月"].rank())
            # 用标准化后的乘积序列做 NW t（秩相关的近似检验，滞后 12 个月）
            z = lambda s: (s.rank() - s.rank().mean()) / s.rank().std()
            t = newey_west_t(z(m["便宜程度"]) * z(m["之后12个月"]), 12)
            lvl = m["E/P"].corr(m["之后12个月"])
            key = f"{col}，{name}"
            rows[key] = {"起点": m.index[0].date(), "终点": m.index[-1].date(), "月度样本": len(m),
                         "独立的12个月约": round(len(m) / 12, 1), "秩相关": rho, "NW t": t, "E/P 水平和之后收益的相关": lvl}
            third = pd.cut(m["便宜程度"], [-0.01, 1 / 3, 2 / 3, 1.0], labels=["贵（前1/3）", "中", "便宜（后1/3）"])
            g = m.groupby(third, observed=False)["之后12个月"]
            buckets[key] = pd.concat([g.mean().rename(lambda x: f"{x} 平均"), g.size().rename(lambda x: f"{x} 月数")])
    res, bk = pd.DataFrame(rows).T, pd.DataFrame(buckets).T
    yearly = val.groupby(val.index.year).last()[["E/P 市值加权", "E/P 中位数"]]
    yearly["对应市盈率（市值加权）"] = 1 / yearly["E/P 市值加权"]
    yearly["当年中证全指涨跌"] = idx.groupby(idx.index.year).last().pct_change()
    pd.set_option("display.width", 250)
    text = "\n".join([
        "# E005：市场整体便宜程度能不能预测之后 12 个月的大盘收益", "",
        "便宜证据：不计试验次数。数据 2009-01 – 2021-12（之后 12 个月的终点不晚于 2021-12-31，不碰封存段）。做法见 `E005_valuation.py` 开头。", "",
        "## 每年年底的市场估值", "", yearly.to_markdown(floatfmt=".4f"), "",
        "## 便宜程度和之后 12 个月中证全指收益的关系（每月取一次）", "", res.to_markdown(floatfmt=".3f"), "",
        "## 按便宜程度分三档的之后 12 个月平均收益", "", bk.to_markdown(floatfmt=".3f"), ""])
    (ROOT / "lab" / "evidence" / "E005.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
