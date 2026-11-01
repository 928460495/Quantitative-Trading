"""全 A 因子的类内筛选（docs/DECISIONS.md 2026-11-01）。

每个大类里，通过检验的因子按 5 天 ICIR 从高到低（并列按名字）逐个看：和已留下的因子平均相关性的绝对值
> 0.7 就跳过，每类最多留 3 个；不足就有几个留几个，没有就空着。跨类不去重。

相关性 = 中性化、按方向调整后的因子值，每 5 个交易日一次的横截面 Spearman 相关（两个因子都有值的股票），
对时间取平均。只算贪心过程中真正要比较的那几对，不算整个相关矩阵：短周期价量一类可能有上百个候选。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from qt.wide_eval import rank_ic

MAX_PER_CATEGORY = 3
CORR_THRESHOLD = 0.7
FULL = f"已满 {MAX_PER_CATEGORY} 个，未比较"


def every_nth_day(x: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """从第一个有值的日子起，每 n 个交易日取一天。"""
    return x.loc[x.index[x.notna().any(axis=1)][::n]]


def pair_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """两个因子逐日横截面 Spearman 相关的平均（不足 30 只的日子不算）。"""
    return float(rank_ic(a, b).mean())


def pick(table: pd.DataFrame, scores: Mapping[str, pd.DataFrame], max_n: int = MAX_PER_CATEGORY,
         threshold: float = CORR_THRESHOLD) -> pd.DataFrame:
    """table：一个大类里通过检验的因子（索引是名字，要有 icir 列）。scores：名字 → 调好方向的因子值。

    返回按考察顺序排列的表：kept、status（留下 / 和谁太像 / 已满）、dropped_by、corr。
    """
    cols = ["icir", "kept", "status", "dropped_by", "corr"]
    if table.empty:
        return pd.DataFrame(columns=cols)
    order = table.assign(_name=table.index).sort_values(["icir", "_name"], ascending=[False, True]).index
    kept: list[str] = []
    rows = []
    for name in order:
        row = {"factor": name, "icir": table.loc[name, "icir"], "kept": False, "dropped_by": None, "corr": np.nan}
        if len(kept) >= max_n:
            row["status"] = FULL
        else:
            corrs = {k: pair_corr(scores[name], scores[k]) for k in kept}
            clash = {k: r for k, r in corrs.items() if abs(r) > threshold}
            if clash:
                by = max(clash, key=lambda k: abs(clash[k]))
                row.update(status=f"和 {by} 太像", dropped_by=by, corr=clash[by])
            else:
                kept.append(name)
                row.update(kept=True, status="留下")
        rows.append(row)
    return pd.DataFrame(rows).set_index("factor")[cols]


def select_by_category(table: pd.DataFrame, scores: Mapping[str, pd.DataFrame], **kw) -> pd.DataFrame:
    """table 要有 icir、category 两列；每个大类各自 pick，结果拼在一起。"""
    parts = [pick(g, scores, **kw).assign(category=cat) for cat, g in table.groupby("category", sort=True)]
    return pd.concat(parts) if parts else pd.DataFrame(columns=["icir", "kept", "status", "dropped_by", "corr", "category"])


def corr_matrix(scores: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """选中因子之间的平均相关性矩阵（报告用，跨类不去重）。"""
    names = list(scores)
    m = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m.loc[a, b] = m.loc[b, a] = pair_corr(scores[a], scores[b])
    return m
