"""计划 3c 任务 7：CSCV 估计回测过拟合概率 PBO（Bailey、Borwein、López de Prado、Zhu，Journal of Computational
Finance 2017）。

把日收益按时间切成 n_blocks 块，取所有"一半块作样本内、另一半作样本外"的组合。每个组合里找样本内夏普最高的方案，
看它在样本外排第几：相对名次 ω = 名次 / (方案数 + 1)，logit λ = ln(ω / (1 − ω))。PBO = λ ≤ 0 的比例，
也就是"样本内最好的方案在样本外排在中位数以下"的概率。

docs/DECISIONS.md 2026-11-10：16 块、216 个组合，PBO > 0.3 停下讨论。
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def cscv_pbo(returns: pd.DataFrame, n_blocks: int = 16) -> dict:
    """returns：index = 日期，columns = 方案。返回 {"pbo", "n_combos", "logits"}。"""
    if n_blocks % 2:
        raise ValueError("块数必须是偶数")
    r = returns.to_numpy(dtype=float)
    n_days, n_cfg = r.shape
    if n_days < 2 * n_blocks:
        raise ValueError(f"只有 {n_days} 天，不够切 {n_blocks} 块")
    blocks = np.array_split(np.arange(n_days), n_blocks)
    # 每块每个方案的收益和、平方和、天数：组合的夏普由这些加总算出，不用每次重切数据
    s1 = np.stack([np.nansum(r[b], axis=0) for b in blocks])
    s2 = np.stack([np.nansum(r[b] ** 2, axis=0) for b in blocks])
    cnt = np.stack([np.sum(~np.isnan(r[b]), axis=0) for b in blocks]).astype(float)

    combos = list(combinations(range(n_blocks), n_blocks // 2))
    is_mask = np.zeros((len(combos), n_blocks))
    for k, c in enumerate(combos):
        is_mask[k, list(c)] = 1.0

    def sharpe(mask: np.ndarray) -> np.ndarray:
        n, a, b = mask @ cnt, mask @ s1, mask @ s2
        mean = a / n
        var = (b - n * mean**2) / (n - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(var > 0, mean / np.sqrt(var), -np.inf)

    sr_is, sr_oos = sharpe(is_mask), sharpe(1.0 - is_mask)
    best = np.argmax(sr_is, axis=1)
    best_oos = sr_oos[np.arange(len(combos)), best]
    rank = (sr_oos < best_oos[:, None]).sum(axis=1) + 1          # 样本外名次：1 = 最差
    omega = rank / (n_cfg + 1)
    logits = np.log(omega / (1 - omega))
    return {"pbo": float(np.mean(logits <= 0)), "n_combos": len(combos), "logits": logits}
