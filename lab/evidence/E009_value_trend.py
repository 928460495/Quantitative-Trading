"""E009：估值 + 趋势仓位的检验（规则见 docs/DECISIONS.md 2026-11-13「估值 + 趋势仓位的检验」，运行前写定）。

在仓库根目录运行：python lab/evidence/E009_value_trend.py [下限 上限]（默认 0.25 0.75）
上证月度文件里 2022 年以后的行（封存段）读进来就丢掉，不看、不打印。
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

from E004_mix import CASH_YIELD, DAYS, TRADE_COST, stats  # noqa: E402
from E007_valuation_oos import LAST, PE_FILE, REPORT, TEST, daily_pct, equity_returns, monthly_pct  # noqa: E402

MA_MONTHS, EVERY, BAND = 10, 20, 0.10
LOW, HIGH = (float(x) for x in (sys.argv[1:3] if len(sys.argv) == 3 else ("0.25", "0.75")))   # 股票比例的下限、上限
SPAN = HIGH - LOW


def monthly_trend() -> pd.Series:
    """上证指数月末收盘 ≥ 过去 10 个月（含当月）平均 → 1.0，否则 0.0；不足 10 个月为空。索引是该月的日期。"""
    df = pd.read_csv(PE_FILE)
    df["日期"] = pd.to_datetime(df["日期"])
    px = df[df["日期"] <= LAST].sort_values("日期").set_index("日期")["指数"].astype(float)
    ma = px.rolling(MA_MONTHS, min_periods=MA_MONTHS).mean()
    return (px >= ma).astype(float).where(ma.notna())


def run(stock: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """每 EVERY 天把目标设为当天的 target（target 已经只用前一天能知道的信息），偏离超过 BAND 才调回。"""
    rc = (1 + CASH_YIELD) ** (1 / DAYS) - 1
    goal = target[0]
    s, c = goal, 1 - goal
    out, w, n = np.empty(len(stock)), np.empty(len(stock)), 0
    for t in range(len(stock)):
        if t > 0 and t % EVERY == 0:
            goal = target[t]
        before = s + c
        s, c = s * (1 + stock[t]), c * (1 + rc)
        if abs(s / (s + c) - goal) > BAND:
            nav = s + c
            nav -= abs(goal * nav - s) * TRADE_COST
            s, c = goal * nav, (1 - goal) * nav
            n += 1
        out[t], w[t] = (s + c) / before - 1, s / (s + c)
    return out, w, n


def main() -> None:
    r = equity_returns()
    val = daily_pct(monthly_pct(), r.index)
    trend = daily_pct(monthly_trend(), r.index)                             # 同样"严格早于当天的最近一个月"
    targets = {"估值 + 趋势": LOW + SPAN * (0.5 * val + 0.5 * trend), "只看趋势（只报告）": LOW + SPAN * trend,
               "只看估值（只报告）": LOW + SPAN * val}
    rows, yearly, verdict = {}, {}, {}
    for seg, (a, b) in (("2005–2009", TEST), ("2010–2021", REPORT)):
        rr = r.loc[a:b]
        stock = rr.to_numpy()
        for name, tgt in targets.items():
            tg = tgt.loc[a:b]
            first = tg.first_valid_index()
            sl = slice(first, None)
            acct, w, n = run(rr.loc[sl].to_numpy(), tg.loc[sl].to_numpy())
            fixed, _, nf = run(rr.loc[sl].to_numpy(), np.full(len(w), w.mean()))
            A, F, S = stats(acct), stats(fixed), stats(rr.loc[sl].to_numpy())
            rows[(seg, name)] = {"起点": first.date(), "年化": A["年化"], "最大回撤": A["最大回撤"], "平均股票比例": w.mean(),
                                 "调整次数": n, "同比例固定：年化": F["年化"], "同比例固定：最大回撤": F["最大回撤"],
                                 "中证全指：年化": S["年化"], "中证全指：最大回撤": S["最大回撤"]}
            if name == "估值 + 趋势":
                verdict[seg] = A["年化"] > F["年化"] and A["最大回撤"] > F["最大回撤"]
                idx = rr.loc[sl].index
                df = pd.DataFrame({"估值+趋势": acct, "同比例固定": fixed, "中证全指": rr.loc[sl].to_numpy(), "股票比例": w}, index=idx)
                yearly[seg] = pd.concat([(1 + df[["估值+趋势", "同比例固定", "中证全指"]]).groupby(idx.year).prod() - 1,
                                         df["股票比例"].groupby(idx.year).mean().rename("平均股票比例")], axis=1)
    table = pd.DataFrame(rows).T
    passed = all(verdict.values())
    text = "\n".join([
        f"# E009：估值 + 趋势仓位的检验（股票比例 {LOW:.0%}–{HIGH:.0%}）", "",
        "规则见 `docs/DECISIONS.md` 2026-11-13「估值 + 趋势仓位的检验」（运行前写定）。两段数据都已看过，不是干净的检验。", "",
        f"**结论：{'通过' if passed else '不通过'}**（两段都要年化高于同比例固定、并且最大回撤更浅）："
        + "、".join(f"{k} {'满足' if v else '不满足'}" for k, v in verdict.items()), "",
        table.to_markdown(floatfmt=".4f"), "",
        "## 分年：2005–2009", "", yearly["2005–2009"].to_markdown(floatfmt=".4f"), "",
        "## 分年：2010–2021", "", yearly["2010–2021"].to_markdown(floatfmt=".4f"), ""])
    tag = "" if (LOW, HIGH) == (0.25, 0.75) else f"_{LOW:.0%}-{HIGH:.0%}".replace("%", "")
    (ROOT / "lab" / "evidence" / f"E009{tag}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
