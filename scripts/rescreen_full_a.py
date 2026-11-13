"""按更正后的基准重新判通过条件第 ③ 条，重新筛选（规则见 docs/DECISIONS.md 2026-11-11，运行前写定）。

用法（先跑完过 scripts/run_full_a_screen.py；不重跑账户回测，几分钟）：
    python scripts/rescreen_full_a.py

第 ③ 条原来和"能买股票每天等权"比，那个基准每天把涨停买不进的股票剔除，训练段约低 9 个点 / 年。
这里改成和"每 20 个交易日在能买的股票里等权买入、期间持有"比；①② 两条、账户回测、类内筛选规则都不变。
先在同一区间重算旧基准，和缓存里的数对不上就停（说明区间或数据和当时不一样）。

输出（旧目录不动）：
    reports/single_factor/full_a_train_holdable/summary.csv、README.md
    reports/screening/full_a_train_holdable/selected.csv、all_candidates.csv、corr_selected.csv、README.md
只用训练段（2010–2018）；不记试验日志（同样的因子、同样的回测，只更正了比较用的数）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt.account import TRADING_DAYS  # noqa: E402
from qt.diagnostics import holdable_benchmark  # noqa: E402
from qt.wide_eval import buyable_benchmark  # noqa: E402

SINGLE_DIR = ROOT / "reports" / "single_factor" / "full_a_train_holdable"
SELECT_DIR = ROOT / "reports" / "screening" / "full_a_train_holdable"
EVERY = 20
TOLERANCE = 1e-4
FAIL_COST = "扣成本后跑输能持有等权"


def annualized(returns: pd.Series) -> float:
    """和 wide_eval.account_excess 同样的年化：净值从第一天收盘算起（第一天的收益不计）。"""
    nav = (1 + returns).cumprod()
    return float((nav.iloc[-1] / nav.iloc[0]) ** (TRADING_DAYS / (len(nav) - 1)) - 1)


def rescreen(summary: pd.DataFrame, bench_annual: float) -> pd.DataFrame:
    """只改第 ③ 条：账户年化 − bench_annual > 0。旧基准的数留在 *_old 列。"""
    out = summary.rename(columns={"bench_annual": "bench_annual_old", "excess_annual": "excess_annual_old",
                                  "excess_t": "excess_t_old"}).copy()
    flag = lambda c: out[c].fillna(False).astype(bool)
    ran = out["annual_return"].notna()
    out["bench_annual"] = float("nan")
    out["excess_t"] = float("nan")                      # 没按新基准重算（不是通过条件），留空
    out.loc[ran, "bench_annual"] = bench_annual
    out["excess_annual"] = out["annual_return"] - out["bench_annual"]
    out["pass_cost"] = ran & (out["excess_annual"] > 0)
    first_two = flag("pass_t") & flag("pass_stable")
    out["passed"] = first_two & out["pass_cost"]
    out["fail_reason"] = out["fail_reason"].where(~first_two, "").mask(first_two & ~out["pass_cost"], FAIL_COST)
    return out


def run(summary: pd.DataFrame, subs: dict[str, pd.DataFrame], close: pd.DataFrame, can_buy: pd.DataFrame,
        single_dir: Path, select_dir: Path, context: str, every: int = EVERY) -> int:
    from run_full_a_screen import select, write_reports

    cached = summary["bench_annual"].dropna()
    old = annualized(buyable_benchmark(close, can_buy))
    if cached.empty or (cached - old).abs().max() > TOLERANCE:
        print(f"停：旧基准重算是 {old:.4%}，缓存里是 {cached.iloc[0] if len(cached) else float('nan'):.4%}，"
              "对不上，区间或数据和当时不一样")
        return 1
    new = annualized(holdable_benchmark(close, can_buy, every))
    out = rescreen(summary, new)
    selection, corr = select(out, subs)
    note = (f"**第 ③ 条按更正后的基准重新判定**（`docs/DECISIONS.md` 2026-11-11）：比较对象是每 {every} 个交易日"
            f"在能买的股票里等权买入、期间持有，训练段年化 {new:.2%}（旧基准 {old:.2%}）。①② 和账户回测沿用原结果。"
            f"原来通过 {int(summary['passed'].fillna(False).sum())} 个，现在 {int(out['passed'].sum())} 个")
    write_reports(out, selection, corr, f"{context}\n\n{note}", single_dir, select_dir)
    print(f"旧基准 {old:.2%} → 新基准 {new:.2%}；通过 {int(summary['passed'].fillna(False).sum())} → "
          f"{int(out['passed'].sum())} 个，选中 {int(selection['kept'].sum()) if len(selection) else 0} 个")
    return 0


def main() -> None:
    from qt.config import segment
    from qt.library import all_factors
    from run_full_a_screen import CACHE_ROOT, _load, load_context

    start, end = segment("train")
    _, ctx, codes, version = load_context(start, end, 1)
    cache_dir = CACHE_ROOT / version.replace("/", "_")
    names = [f.name for f in all_factors()]
    rows, subs = _load(cache_dir, names)
    if len(rows) < len(names):
        sys.exit(f"{cache_dir} 里只有 {len(rows)}/{len(names)} 个因子的结果，先跑完 scripts/run_full_a_screen.py")
    summary = pd.DataFrame(rows).set_index("factor").loc[names]
    close, can_buy = ctx.trim(ctx.close), ctx.trim(ctx.can_buy).fillna(False).astype(bool)
    context = (f"样本：全 A 主板 {len(codes)} 只（含退市），训练段 {ctx.dates[0].date()} ~ {ctx.dates[-1].date()}"
               f"（段尾去掉 {ctx.tail} 个交易日），{len(ctx.dates)} 个交易日；数据版本 `{version}`")
    sys.exit(run(summary, subs, close, can_buy, SINGLE_DIR, SELECT_DIR, context))


if __name__ == "__main__":
    main()
