"""E007：估值仓位在 2005–2009 年的检验（规则见 docs/DECISIONS.md 2026-11-13「估值仓位用 2005–2009 年做一次没被看过的检验」，取数据之前写定）。

在仓库根目录运行：python lab/evidence/E007_valuation_oos.py
估值数据：data/cache/external/sse_pe_legulegu.csv（akshare stock_market_pe_lg("上证")，月度）。
文件里有 2022 年以后的行（封存段），读进来就只留 2021-12-31 以前的，之后的值不看、不打印。
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab" / "evidence"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from E004_mix import DAYS, stats  # noqa: E402
from E006_frequency import timed  # noqa: E402

PE_FILE = Path("/Users/ariasleo/qt-full-a/data/cache/external/sse_pe_legulegu.csv")
LAST = pd.Timestamp("2021-12-31")
TEST = ("2005-01-04", "2009-12-31")
REPORT = ("2010-01-04", "2021-12-31")
MONTHS, EVERY = 60, 20


def monthly_pct() -> pd.Series:
    """每月 E/P 在过去 60 个月（含当月）里的百分位，索引是该月数据的日期；不足 60 个月为空。"""
    df = pd.read_csv(PE_FILE)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df[df["日期"] <= LAST].sort_values("日期")                        # 封存段一行都不留
    ep = (1 / df.set_index("日期")["平均市盈率"].astype(float)).dropna()
    v = ep.to_numpy()
    out = [np.nan if i + 1 < MONTHS else float((v[i + 1 - MONTHS:i + 1] <= v[i]).mean()) for i in range(len(v))]
    return pd.Series(out, index=ep.index)


def daily_pct(pct_m: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """第 t 天能用的百分位：日期严格早于 t 的最近一个月的值（月末的值第二个交易日起才用）。"""
    pos = pct_m.index.searchsorted(days, side="left") - 1
    vals = np.where(pos >= 0, pct_m.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=days)


def equity_returns() -> pd.Series:
    from qlib.data import D

    from qt.config import check_range
    from qt.data import init_qlib

    check_range(TEST[0], REPORT[1])
    init_qlib()
    px = D.features(["SH000985"], ["$close"], start_time=TEST[0], end_time=REPORT[1])["$close"]
    px = px.droplevel("instrument").sort_index().astype(float)
    return px.pct_change().fillna(0.0)


def evaluate(r: pd.Series, p: pd.Series) -> dict:
    stock, pct = r.to_numpy(), p.to_numpy()
    acct, w, n = timed(stock, pct, EVERY)
    avg = float(w.mean())
    fixed, wf, nf = timed(stock, np.full(len(stock), (avg - 0.25) / 0.50), 10 ** 9)
    a, f, full = stats(acct), stats(fixed), stats(stock)
    return {"估值仓位：年化": a["年化"], "估值仓位：最大回撤": a["最大回撤"], "平均股票比例": avg, "调整次数": n,
            "固定比例：年化": f["年化"], "固定比例：最大回撤": f["最大回撤"], "固定比例调整次数": nf,
            "一直满仓：年化": full["年化"], "一直满仓：最大回撤": full["最大回撤"],
            "_acct": acct, "_fixed": fixed, "_w": w}


def main() -> None:
    pct_m = monthly_pct()
    assert pct_m.index.max() <= LAST and pct_m.dropna().index.min() <= pd.Timestamp("2004-12-31"), "数据范围不对"
    r = equity_returns()
    p = daily_pct(pct_m, r.index)
    # 自检：第 t 天用到的月份一定早于 t
    used = pct_m.index[np.clip(pct_m.index.searchsorted(r.index, side="left") - 1, 0, None)]
    assert (used < r.index).all()
    out, years = {}, {}
    for name, (a, b) in (("2005–2009（检验）", TEST), ("2010–2021（已看过，只报告）", REPORT)):
        rr, pp = r.loc[a:b], p.loc[a:b]
        assert pp.notna().all(), f"{name} 有日子没有信号"
        res = evaluate(rr, pp)
        idx = rr.index
        yr = pd.DataFrame({"估值仓位": res["_acct"], "固定比例": res["_fixed"], "中证全指": rr.to_numpy(), "股票比例": res["_w"]}, index=idx)
        g = yr.groupby(idx.year)
        years[name] = pd.concat([(1 + yr[["估值仓位", "固定比例", "中证全指"]]).groupby(idx.year).prod() - 1,
                                 g["股票比例"].mean().rename("平均股票比例")], axis=1)
        out[name] = {k: v for k, v in res.items() if not k.startswith("_")}
    table = pd.DataFrame(out).T
    t = out["2005–2009（检验）"]
    passed = t["估值仓位：年化"] > t["固定比例：年化"] and t["估值仓位：最大回撤"] > t["固定比例：最大回撤"]

    # 数据核对（只报告）：乐咕乐股的上证市盈率和我们自己算的全 A 主板 E/P，2014–2021 月度相关
    from E005_valuation import market_valuation
    ours = market_valuation()["E/P 市值加权"].resample("ME").last().loc["2014":"2021"]
    df = pd.read_csv(PE_FILE)
    df["日期"] = pd.to_datetime(df["日期"])
    theirs = (1 / df[df["日期"] <= LAST].set_index("日期")["平均市盈率"]).resample("ME").last().loc["2014":"2021"]
    corr = ours.corr(theirs)

    pd.set_option("display.width", 250)
    text = "\n".join([
        "# E007：估值仓位在 2005–2009 年的检验（没被看过的年份）", "",
        "规则见 `docs/DECISIONS.md` 2026-11-13「估值仓位用 2005–2009 年做一次没被看过的检验」（取数据之前写定）。", "",
        f"**结论：{'通过' if passed else '不通过'}**（通过 = 2005–2009 年估值仓位的年化高于同平均比例的固定比例，并且最大回撤更浅）", "",
        table.to_markdown(floatfmt=".4f"), "",
        "## 分年（2005–2009，检验）", "", years["2005–2009（检验）"].to_markdown(floatfmt=".4f"), "",
        "## 分年（2010–2021，已看过，只报告）", "", years["2010–2021（已看过，只报告）"].to_markdown(floatfmt=".4f"), "",
        f"数据核对：乐咕乐股上证市盈率的倒数和我们自己算的全 A 主板市值加权 E/P，2014–2021 月度相关 {corr:.3f}。", "",
        "如实说明：2005–2009 基本只有一轮牛熊，即使通过也只是一次方向上的证据。", ""])
    (ROOT / "lab" / "evidence" / "E007.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
