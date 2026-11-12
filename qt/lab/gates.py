"""采纳规则（docs/FRAMEWORK.md 第 6 节）：由这里算出"采纳 / 不采纳"，不由人判断。一个世代内冻结。"""

from __future__ import annotations

import pandas as pd

from qt.lab.protocol import BIG_LOSS_CUT, ERAS, FULL, LATEST_ERA, T_ADOPT, T_NOT_WORSE

KINDS = ("收益型", "防雷型")
DIRECTIONS = ("升", "降")
BIG = "大亏只数"


def prediction_holds(prediction: dict, cand: pd.DataFrame, base: pd.DataFrame, pair: dict) -> bool:
    """prediction = {"metric": "三.大亏只数" 或 "配对.全段", "direction": "升" / "降"}。

    "阶段.列名"：比较候选和基线成绩单里的这个数；"配对.阶段"：看候选减基线的年化差值的正负。
    """
    where, _, what = prediction["metric"].partition(".")
    direction = prediction["direction"]
    if direction not in DIRECTIONS:
        raise ValueError(f"方向只能是 {DIRECTIONS}：{direction}")
    change = float(pair["差值"][what]) if where == "配对" else float(cand.loc[where, what] - base.loc[where, what])
    return change > 0 if direction == "升" else change < 0


def decide(kind: str, cand: pd.DataFrame, base: pd.DataFrame, pair: dict, pair_cost2: dict, prediction: dict
           ) -> tuple[pd.DataFrame, bool]:
    """返回（每条规则一行：数值、门槛、满足；是否采纳）。全部满足才采纳。"""
    if kind not in KINDS:
        raise ValueError(f"类型只能是 {KINDS}：{kind}")
    diff, t = pair["差值"], pair["t"]
    predicted = prediction_holds(prediction, cand, base, pair)
    earlier = [e for e in ERAS if e != LATEST_ERA]
    if kind == "收益型":
        rows = {
            "配对差值的 t 值": (t, f"≥ {T_ADOPT}", t >= T_ADOPT),
            f"阶段{LATEST_ERA}差值": (diff[LATEST_ERA], "> 0", diff[LATEST_ERA] > 0),
            "更早的阶段至少一个差值为正": (max(diff[e] for e in earlier), "> 0", any(diff[e] > 0 for e in earlier)),
            "成本翻倍后全段差值": (pair_cost2["差值"][FULL], "> 0", pair_cost2["差值"][FULL] > 0),
            "大亏只数不比基线多": (cand.loc[FULL, BIG] - base.loc[FULL, BIG], "≤ 0", cand.loc[FULL, BIG] <= base.loc[FULL, BIG]),
        }
    else:
        before, after = float(base.loc[LATEST_ERA, BIG]), float(cand.loc[LATEST_ERA, BIG])
        cut = (before - after) / before if before > 0 else float("nan")
        rows = {
            f"阶段{LATEST_ERA}大亏只数减少比例": (cut, f"≥ {BIG_LOSS_CUT}", before > 0 and cut >= BIG_LOSS_CUT),
            f"阶段{LATEST_ERA}差值": (diff[LATEST_ERA], "> 0", diff[LATEST_ERA] > 0),
            "全段配对差值的 t 值": (t, f"> {T_NOT_WORSE}", t > T_NOT_WORSE),
        }
    rows["预期的方向成立"] = (float(predicted), "是", predicted)
    table = pd.DataFrame.from_dict(rows, orient="index", columns=["数值", "门槛", "满足"]).astype({"满足": bool})
    return table, bool(table["满足"].all())
