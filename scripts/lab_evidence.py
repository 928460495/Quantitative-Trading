"""实验室：便宜的证据（不跑账户、不计试验次数；docs/FRAMEWORK.md 第 4 节第 3 步）。

用法（单进程，几分钟）：
    python scripts/lab_evidence.py E001

对现行基线流程的打分：
  ① 最顶端各档之后 20 天的收益（分阶段）
  ② 打分前 50 名里带各个标记的股票占多少、之后表现差多少（分阶段）
  ③ 股票第一次被戴 ST 前后：之前跌了多少、之后连续几天卖不掉、事前各标记亮了多久（分阶段汇总）
写进 lab/evidence/<编号>.md。看了这些之后写的假设卡，要在 seen_before 里写明看过哪一份。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt.lab import evidence as ev  # noqa: E402
from qt.lab import procedure as pc  # noqa: E402
from qt.lab.protocol import ERAS, FULL, era_of  # noqa: E402

FLAGS = ("deduct_loss", "neg_equity", "loss_small", "small30", "high_vol", "cheap")


def st_summary(events: pd.DataFrame) -> pd.DataFrame:
    """按戴帽所在阶段汇总：事件数、之前 250 / 60 天涨跌的中位数、连续卖不掉天数的中位数和 90% 分位、
    各标记在戴帽前已经亮着的比例和提前天数的中位数。"""
    if events.empty:
        return pd.DataFrame()
    era = era_of(pd.DatetimeIndex(events["date"]))
    out = {}
    for name in [*ERAS, FULL]:
        g = events if name == FULL else events[era == name]
        if not len(g):
            continue
        row = {"事件数": len(g), "前250天涨跌中位数": g["前250天涨跌"].median(), "前60天涨跌中位数": g["前60天涨跌"].median(),
               "连续卖不掉天数中位数": g["戴帽后连续卖不掉的天数"].median(),
               "连续卖不掉天数90%分位": g["戴帽后连续卖不掉的天数"].quantile(0.9)}
        for col in [c for c in g if c.endswith("提前天数")]:
            flag = col.split()[0]
            row[f"{flag} 事前亮着的比例"] = g[col].notna().mean()
            row[f"{flag} 提前天数中位数"] = g[col].median()
        out[name] = row
    return pd.DataFrame(out).T


def build(dev: dict, proc: pc.Procedure) -> dict[str, pd.DataFrame]:
    data = pc.market(dev, proc)
    frame = lambda a: pd.DataFrame(a, index=data.dates, columns=data.instruments)
    score, can_buy = frame(data.score), frame(data.can_buy)
    # 看标记时用不排除任何东西的打分：被排除的股票本来就没分数，看不到它们在顶端的样子
    raw = pc.composite(dev, proc)
    raw_buy = dev["can_buy"].reindex(index=raw.index, columns=raw.columns).fillna(False)
    flags = {f: dev["flags"][f] for f in FLAGS if f in dev["flags"]}
    events = ev.st_events(dev["flags"]["is_st"], dev["close"], dev["can_sell"], flags)
    return {"顶端各档（现行基线，含排除）": ev.top_tail(score, dev["close"], can_buy),
            "前50名里带标记的股票（不做任何排除的打分）": ev.flag_split(raw, dev["close"], raw_buy, flags, top=50),
            "第一次戴 ST 前后（按戴帽所在阶段）": st_summary(events)}


def write(sections: dict[str, pd.DataFrame], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "便宜的证据：不跑账户、不计试验次数。数据是开发段（2010–2021），已被看过很多次。", ""]
    for name, table in sections.items():
        lines += [f"## {name}", "", table.to_markdown(floatfmt=".4f") if len(table) else "（无）", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from qt.lab import cards as cd
    from qt.lab.store import load_dev
    from run_plan3c_train import POOL_CACHE, latest_version

    if len(sys.argv) != 2:
        sys.exit(__doc__)
    proc = cd.load_baseline(ROOT / "lab" / "baseline.yaml")
    dev = load_dev(latest_version(POOL_CACHE), proc.factors())
    out = ROOT / "lab" / "evidence" / f"{sys.argv[1]}.md"
    write(build(dev, proc), out, f"{sys.argv[1]}：现行基线的便宜证据")
    print(out.read_text())


if __name__ == "__main__":
    main()
