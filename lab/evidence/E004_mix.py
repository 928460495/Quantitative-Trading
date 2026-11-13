"""E004（便宜证据，不计试验）：在现行基线的选股收益上叠加"股票 / 现金目标比例 + 偏离就调回"，看不同比例的取舍。

做法（运行前定好，不看结果改）：
- 股票部分的每日收益 = 现行基线四个调仓起点的每日收益（已扣选股本身的成本）；现金按每年 2% 计息（货币基金 / 逆回购的量级）
- 目标比例 30%–100%（每 10 个百分点一档）；股票占比偏离目标超过 10 个百分点就调回目标；调整的金额按 0.2% 扣成本（佣金、滑点、卖出印花税，偏保守）
- 一条历史路径：四个起点各算一遍取平均，分阶段报告
- 随机重抽：把四个起点平均后的每日收益切成平均 60 天一段的小块随机重组（平稳块自助法），生成 2000 条 12 年的路径，看每个比例的结果分布
- 凯利公式给出的"最优股票比例"及其不确定性（只作参考）

这是对已经看过的开发段数据的描述，不是检验；它说明的是"每个比例对应什么样的收益和风险"，选哪个取决于能承受多大的回撤，不是数据能替你选的。
在仓库根目录运行：python lab/evidence/E004_mix.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TARGETS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BAND, CASH_YIELD, TRADE_COST = 0.10, 0.02, 0.002
N_PATHS, BLOCK, SEED, DAYS = 2000, 60, 7, 252
CACHE = ROOT / "lab" / "evidence" / "E004_returns.parquet"


def mix(stock: np.ndarray, target: float, band: float = BAND, cash_yield: float = CASH_YIELD,
        cost: float = TRADE_COST) -> np.ndarray:
    """股票部分每日收益 stock，返回账户每日收益。第一天按目标比例建仓。"""
    rc = (1 + cash_yield) ** (1 / DAYS) - 1
    s, c = target, 1 - target
    out = np.empty(len(stock))
    for t, r in enumerate(stock):
        before = s + c
        s, c = s * (1 + r), c * (1 + rc)
        w = s / (s + c)
        if abs(w - target) > band:
            nav = s + c
            trade = abs(target * nav - s)
            nav -= trade * cost
            s, c = target * nav, (1 - target) * nav
        out[t] = (s + c) / before - 1
    return out


def stats(r: np.ndarray) -> dict:
    nav = np.concatenate([[1.0], np.cumprod(1 + r)])
    ann = nav[-1] ** (DAYS / len(r)) - 1
    dd = float((nav / np.maximum.accumulate(nav) - 1).min())
    return {"年化": ann, "最大回撤": dd, "年化÷回撤": ann / -dd if dd < 0 else np.nan}


def block_bootstrap(r: np.ndarray, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """平稳块自助法：每块起点随机、长度服从均值 BLOCK 的几何分布，首尾相接循环取。"""
    out, n = np.empty(n_days), len(r)
    i = 0
    while i < n_days:
        start, length = rng.integers(n), rng.geometric(1 / BLOCK)
        take = min(length, n_days - i)
        out[i:i + take] = r[(start + np.arange(take)) % n]
        i += take
    return out


def load_returns() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    from qt.lab import cards as cd
    from qt.lab import scorecard as sc
    from qt.lab.store import load_dev
    from run_plan3c_train import POOL_CACHE, latest_version

    base = cd.load_baseline(ROOT / "lab" / "baseline.yaml")
    dev = load_dev(latest_version(POOL_CACHE), base.factors())
    rets = sc.daily_returns(sc.run_phases(dev, base))
    rets.columns = [str(c) for c in rets.columns]
    rets.to_parquet(CACHE)
    return rets


def main() -> None:
    rets = load_returns()
    era = np.where(rets.index.year <= 2014, "一", np.where(rets.index.year <= 2018, "二", "三"))
    # 自检：目标 100% 时必须原样还原选股收益；目标 0% 时就是现金
    one = rets.iloc[:, 0].to_numpy()
    assert np.allclose(mix(one, 1.0), one), "100% 股票应当和选股收益完全一样"
    assert np.allclose(mix(one, 0.0, band=1.0), (1.02) ** (1 / DAYS) - 1)

    hist, by_era = {}, {}
    for w in TARGETS:
        per = [mix(rets[c].to_numpy(), w) for c in rets]
        st = pd.DataFrame([stats(p) for p in per]).mean()
        hist[f"{w:.0%}"] = st
        avg = np.mean(per, axis=0)
        by_era[f"{w:.0%}"] = {e: stats(avg[era == e])["年化"] for e in ("一", "二", "三")}
        by_era[f"{w:.0%}"].update({f"{e}回撤": stats(avg[era == e])["最大回撤"] for e in ("一", "二", "三")})
    hist = pd.DataFrame(hist).T
    by_era = pd.DataFrame(by_era).T

    rng = np.random.default_rng(SEED)
    base = rets.mean(axis=1).to_numpy()
    paths = [block_bootstrap(base, len(base), rng) for _ in range(N_PATHS)]
    boot = {}
    for w in TARGETS:
        s = pd.DataFrame([stats(mix(p, w)) for p in paths])
        three = [stats(mix(p[:3 * DAYS], w))["年化"] for p in paths]
        boot[f"{w:.0%}"] = {"年化中位数": s["年化"].median(), "年化差的5%": s["年化"].quantile(0.05),
                            "最大回撤中位数": s["最大回撤"].median(), "最大回撤差的5%": s["最大回撤"].quantile(0.05),
                            "任意三年亏钱的概率": float(np.mean(np.array(three) < 0))}
    boot = pd.DataFrame(boot).T

    mu = base.mean() * DAYS
    var = base.var(ddof=1) * DAYS
    rf = CASH_YIELD
    kelly = (mu - rf) / var
    se_mu = np.sqrt(var / (len(base) / DAYS))
    kelly_rng = ((mu - 1.96 * se_mu - rf) / var, (mu + 1.96 * se_mu - rf) / var)

    pd.set_option("display.width", 250)
    out = ["# E004：股票 / 现金目标比例的取舍（叠加在现行基线的选股收益上）", "",
           "便宜证据：不计试验次数，是对已经看过的开发段（2010–2021）的描述，不是检验。做法见 `E004_mix.py` 开头。", "",
           "## 一条历史路径（四个调仓起点平均）", "", hist.to_markdown(floatfmt=".4f"), "",
           "## 分阶段年化和最大回撤", "", by_era.to_markdown(floatfmt=".4f"), "",
           f"## 随机重抽 {N_PATHS} 条 12 年路径（平均 {BLOCK} 天一块）", "", boot.to_markdown(floatfmt=".4f"), "",
           "## 凯利公式（只作参考）", "",
           f"- 选股收益的年化算术均值 {mu:.2%}、年化波动 {np.sqrt(var):.2%}、现金 {rf:.0%}",
           f"- 全凯利股票比例 (μ−r)/σ² = {kelly:.2f}；按均值的 95% 置信区间算，范围 {kelly_rng[0]:.2f} ~ {kelly_rng[1]:.2f}",
           "- 半凯利 = 全凯利的一半；不能加杠杆，所以超过 1 的都按 100% 算"]
    text = "\n".join(out) + "\n"
    (ROOT / "lab" / "evidence" / "E004.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
