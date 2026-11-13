"""E006（便宜证据，不计试验）：估值仓位多久更新一次——敏感性检查。

运行前写定（不看结果改）：
- 股票部分：现行基线选股的每日收益（E004_returns.parquet，四个调仓起点平均）；现金每年 2%
- 估值信号：E005 的"市值加权 E/P、和过去 5 年比"的百分位（2014-03 起才有）；目标股票比例 = 25% + 50% × 百分位
- 更新频率 k ∈ {1, 5, 10, 20, 60} 个交易日：每 k 天用前一天的百分位重新算目标比例；
  实际比例和目标差超过 10 个百分点才调到目标（和 E004 一样），调整金额扣 0.2% 成本
- 对照：固定比例（等于该版本的平均股票比例）、偏离 10 个百分点调回；以及每 20 天更新的版本
- 判定规则：只有某个更勤的频率比"每 20 天"的配对日收益差显著为正（Newey-West t ≥ 2，滞后 20）时才考虑换；否则用每 20 天。
  这条在运行前写定。区间 2014-03 – 2021-12，独立的年份只有约 8 个
在仓库根目录运行：python lab/evidence/E006_frequency.py
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
from E005_valuation import market_valuation, past_percentile  # noqa: E402
from qt.evaluate import newey_west_t  # noqa: E402

FREQS = (1, 5, 10, 20, 60)
BAND, LOW, SPAN = 0.10, 0.25, 0.50


def timed(stock: np.ndarray, pct: np.ndarray, every: int, band: float = BAND) -> tuple[np.ndarray, np.ndarray, int]:
    """返回（账户日收益，每天的股票比例，调整次数）。第 t 天用第 t−1 天的百分位定目标。"""
    rc = (1 + CASH_YIELD) ** (1 / DAYS) - 1
    target = LOW + SPAN * pct[0]
    s, c = target, 1 - target
    out, weight, n = np.empty(len(stock)), np.empty(len(stock)), 0
    for t in range(len(stock)):
        if t > 0 and t % every == 0:
            target = LOW + SPAN * pct[t - 1]
        before = s + c
        s, c = s * (1 + stock[t]), c * (1 + rc)
        w = s / (s + c)
        if abs(w - target) > band:
            nav = s + c
            nav -= abs(target * nav - s) * TRADE_COST
            s, c = target * nav, (1 - target) * nav
            n += 1
        out[t], weight[t] = (s + c) / before - 1, s / (s + c)
    return out, weight, n


def fixed(stock: np.ndarray, target: float) -> np.ndarray:
    pct = np.full(len(stock), (target - LOW) / SPAN)
    return timed(stock, pct, 10 ** 9)[0]


def main() -> None:
    rets = pd.read_parquet(ROOT / "lab" / "evidence" / "E004_returns.parquet").mean(axis=1)
    val = market_valuation()["E/P 市值加权"]
    pct = past_percentile(val, 1260, 1260).reindex(rets.index).ffill().dropna()
    stock = rets.reindex(pct.index).to_numpy()
    p = pct.to_numpy()
    runs = {k: timed(stock, p, k) for k in FREQS}
    ref = runs[20][0]
    rows = {}
    for k, (r, w, n) in runs.items():
        avg_w = float(w.mean())
        const = fixed(stock, avg_w)
        rows[f"每 {k} 天"] = {**stats(r), "平均股票比例": avg_w, "调整次数": n,
                             "同比例固定：年化": stats(const)["年化"], "同比例固定：最大回撤": stats(const)["最大回撤"],
                             "比同比例固定多（年化）": stats(r)["年化"] - stats(const)["年化"],
                             "对固定的 t": newey_west_t(pd.Series(r - const), 20),
                             "比每20天多（年化）": (r - ref).mean() * DAYS, "对每20天的 t": newey_west_t(pd.Series(r - ref), 20)}
    table = pd.DataFrame(rows).T
    full = stats(stock)
    text = "\n".join([
        "# E006：估值仓位多久更新一次（敏感性检查）", "",
        f"便宜证据：不计试验次数。区间 {pct.index[0].date()} – {pct.index[-1].date()}（约 {len(p) / DAYS:.1f} 年）。做法和判定规则见 `E006_frequency.py` 开头（运行前写定）。", "",
        f"对照：同期一直满仓（100% 股票）年化 {full['年化']:.2%}、最大回撤 {full['最大回撤']:.2%}。", "",
        table.to_markdown(floatfmt=".4f"), ""])
    (ROOT / "lab" / "evidence" / "E006.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
