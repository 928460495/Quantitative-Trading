"""按回撤分档减仓（计划 3c，docs/DECISIONS.md 2026-11-10「计划 3c 事先登记的规则」）。

账户净值相对过去 window 个信号日（含当天，不足用已有天数）最高点的回撤 d：
d < 10% 满仓；10% ≤ d < 15% 持股上限 2/3；15% ≤ d < 20% 上限 1/3；d ≥ 20% 空仓。
加深立即生效；回到较浅一档要求 d 低于"该档上面那一档的下沿"再减 buffer，避免在边界来回买卖。

真实账户（qt/account.run_account）和快速模拟器（qt/fastsim）共用这一个状态机，保证两边规则一样。
"""

from __future__ import annotations

import math
from collections import deque

DEFAULT_TIERS: tuple[tuple[float, float], ...] = ((0.10, 2 / 3), (0.15, 1 / 3), (0.20, 0.0))


class DrawdownTiers:
    def __init__(self, tiers: tuple[tuple[float, float], ...] = DEFAULT_TIERS, window: int = 250, buffer: float = 0.02):
        self.lowers = [t for t, _ in tiers]                 # 第 j 档（j ≥ 1）的下沿
        self.fractions = [1.0] + [f for _, f in tiers]      # 第 0 档 = 满仓
        if self.lowers != sorted(self.lowers):
            raise ValueError("回撤分档的下沿要从小到大")
        self.buffer = buffer
        self.navs: deque[float] = deque(maxlen=window)
        self.level = 0

    def update(self, nav: float) -> float:
        """喂入信号日净值，返回持股比例。"""
        self.navs.append(nav)
        d = 1 - nav / max(self.navs)
        deepest = sum(d >= t - 1e-12 for t in self.lowers)  # d 落在的档位（容差：1 − 90/100 算出来略小于 0.1）
        if deepest > self.level:
            self.level = deepest
        else:
            for lvl in range(self.level):                   # 从最浅的档位找能回去的
                if d < self.lowers[lvl] - self.buffer:
                    self.level = lvl
                    break
        return self.fractions[self.level]

    @staticmethod
    def slots(n_hold: int, fraction: float) -> int:
        return int(math.floor(n_hold * fraction + 1e-9))
