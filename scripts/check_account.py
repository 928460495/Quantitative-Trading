"""核对账户回测引擎（qt/account.py）和 Qlib 回测（qt/backtest.py）：同一个信号、同一套旧设定，结果要对得上。

用法：
    python scripts/check_account.py     # 写 reports/account_check.md

用第 5 个 PR 的沪深300 验证段等权合成分数（data/cache/ 里的缓存，没有就先跑 scripts/run_combination.py），
旧设定：账户 1 亿、30 只、缓冲 30/90、不设上限、不止损、不控仓、不排雷。两个引擎的差别只在记账细节：
Qlib 买入时留 1% 现金（risk_degree 0.99），我们的引擎按"现金 / 空位数"全部用完。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt import combine  # noqa: E402
from qt.account import AccountRules, account_metrics, market_data, run_account  # noqa: E402
from qt.backtest import ACCOUNT, run_backtest  # noqa: E402
from qt.config import check_range  # noqa: E402
from qt.data import init_qlib, trading_calendar  # noqa: E402
from qt.universe import FILTER_EXPRS, sellable_mask, tradable_mask  # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")

OLD_RULES = AccountRules(cash=ACCOUNT, n_hold=30, sell_rank=90, max_stock_weight=1.0, max_industry_weight=1.0,
                         stop_loss=None, drawdown_stop=None)


def _signal(market: str) -> tuple[pd.Series, pd.Series]:
    """第 5 个 PR 选定的等权合成分数，和沪深300 成分股里能买的样本。"""
    import run_combination as rc

    selected = pd.read_csv(ROOT / "reports" / "screening" / f"{market}_train_neutral" / "selected.csv", index_col="factor")
    scoring = list(selected.index[selected["category"] == "打分因子"])
    key = {"market": market, "factors": sorted(selected.index), "directions": selected["direction"].to_dict(),
           "data_version": rc._data_version(), "what": "scores_valid"}
    valid = rc._cached(key, lambda: (_ for _ in ()).throw(FileNotFoundError("没有缓存，先跑 scripts/run_combination.py")), False)
    score = combine.composite(valid["scores"][scoring], combine.equal_weights(scoring))
    return score, valid["can_buy"]


def _prices(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """不按成分股筛选的价格和入场日状态：调出指数的股票也要能估值、能卖。"""
    check_range(start, end)
    init_qlib()
    from qlib.data import D

    exprs = {"adj": "$close", "raw": "$close/$factor"} | FILTER_EXPRS
    df = D.features(codes, list(exprs.values()), start_time=start, end_time=end)
    df.columns = list(exprs)
    df = df.swaplevel().sort_index()
    df.index.names = ["datetime", "instrument"]
    return df.replace([np.inf, -np.inf], np.nan)


def compare(market: str = "csi300") -> dict:
    score, can_buy = _signal(market)
    codes = sorted(score.index.get_level_values("instrument").unique())
    days = score.index.get_level_values("datetime")
    cal = trading_calendar(str(days.min().date()), str((days.max() + pd.Timedelta(days=15)).date()))
    end = cal[cal.get_loc(days.max()) + 1]
    px = _prices(codes, str(days.min().date()), str(end.date()))

    data = market_data(score, px["adj"], px["raw"], can_buy.reindex(px.index, fill_value=False),
                       sellable_mask(px))
    ours = run_account(data, OLD_RULES)
    report = run_backtest(score, buy_allowed=can_buy)

    nav_qlib = (1 + report["return"] - report["cost"]).cumprod()
    common = ours.daily.index.intersection(report.index)
    r_ours = ours.daily["nav"].pct_change().reindex(common)
    r_qlib = (report["return"] - report["cost"]).reindex(common)
    bench = (1 + report["bench"]).cumprod()
    m_ours = account_metrics(ours, bench)
    m_ours["annual_return"] = (1 + r_ours.dropna()).prod() ** (252 / r_ours.notna().sum()) - 1  # 同一段日期比
    m_qlib = {"annual_return": (1 + r_qlib).prod() ** (252 / len(r_qlib)) - 1,
              "cost_annual": float(report["cost"].mean() * 252),
              "max_drawdown": float((nav_qlib / nav_qlib.cummax() - 1).min())}
    return {
        "ours": m_ours,
        "qlib": m_qlib,
        "daily_corr": float(r_ours.corr(r_qlib)),
        "days": len(common),
        "period": (common.min().date(), common.max().date()),
        "avg_cash_ours": float((ours.daily["cash"] / ours.daily["nav"]).iloc[1:].mean()),
        "avg_cash_qlib": float((report["cash"] / report["account"]).mean()) if "cash" in report else np.nan,
    }


def main() -> None:
    out = compare()
    o, q = out["ours"], out["qlib"]
    table = pd.DataFrame({
        "我们的引擎": [o["annual_return"], o["cost_annual"], o["max_drawdown"], out["avg_cash_ours"]],
        "Qlib": [q["annual_return"], q["cost_annual"], q["max_drawdown"], out["avg_cash_qlib"]],
    }, index=["扣成本后年化收益（复合）", "年化成本", "最大回撤", "平均现金比例"])
    lines = [
        "# 账户回测引擎核对",
        "",
        f"- 信号：沪深300 验证段等权合成分数（第 5 个 PR 选定的方案），{out['period'][0]} ~ {out['period'][1]}，{out['days']} 个交易日",
        "- 旧设定：账户 1 亿、30 只、缓冲 30/90、不设上限、不止损、不控仓、不排雷；成本按 `qt/costs.py`（最低 5 元）",
        f"- 两个引擎每日收益的相关系数：**{out['daily_corr']:.4f}**；年化收益相差 "
        f"**{(o['annual_return'] - q['annual_return']) * 100:+.2f} 个百分点**（通过标准：< 1 个百分点、相关系数 > 0.95）",
        "",
        table.map(lambda v: f"{v:.2%}").to_markdown(),
        "",
        "差别来自记账细节：Qlib 买入时只用 99% 的现金（risk_degree 0.99），我们的引擎按\"现金 / 空位数\"用完；"
        "两者都按 1 手取整、收盘价成交、成本相同。",
        "",
    ]
    path = ROOT / "reports" / "account_check.md"
    path.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
