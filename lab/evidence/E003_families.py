"""E003（便宜证据，不计试验）：每一组单独打分时，多头端比全体多赚多少，分阶段。

每组用和基线一样的合成方法（组内等权、对行业和市值中性化、缺失记 0），能买的范围用基线的排除名单。
看的是"顶端各档之后 20 天的年化收益（不扣成本）减去全体"。在仓库根目录运行：python lab/evidence/E003_families.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from qt.lab import cards as cd  # noqa: E402
from qt.lab import evidence as ev  # noqa: E402
from qt.lab import procedure as pc  # noqa: E402
from qt.lab.store import load_dev  # noqa: E402

EXTRA = {
    "质量": ("roe_ttm", "roa_ttm", "gross_margin_ttm", "net_margin_ttm"),
    "低波动": ("vol60", "ivol60", "ivol20", "range20", "skew60"),
    "成长": ("profit_yoy", "deduct_yoy", "revenue_yoy"),
}


def main() -> None:
    from run_plan3c_train import POOL_CACHE, latest_version

    base = cd.load_baseline(ROOT / "lab" / "baseline.yaml")
    families = {**dict(base.families), **EXTRA}
    names = sorted({f for fs in families.values() for f in fs})
    dev = load_dev(latest_version(POOL_CACHE), names)
    data = pc.market(dev, base)
    can_buy = pd.DataFrame(data.can_buy, index=data.dates, columns=data.instruments)
    rows = {}
    for name, fs in [("基线合成分", None), *families.items()]:
        proc = base if fs is None else base.replace(families={name: list(fs)})
        score = pc.composite(dev, proc)
        tail = ev.top_tail(score, dev["close"], can_buy)
        for era, r in tail.iterrows():
            rows[(name, era)] = {"全体": r["全体"], "前15名−全体": r["前15名"] - r["全体"], "前50名−全体": r["前50名"] - r["全体"],
                                 "前150名−全体": r["前150名"] - r["全体"], "最好五分之一−全体": r["最好五分之一"] - r["全体"],
                                 "前15名跌超20%": r["前15名跌超20%的只数"], "随机15只": r["随机挑15只的期望"]}
    out = pd.DataFrame(rows).T
    pd.set_option("display.width", 250)
    print(out.round(4).to_string())
    (ROOT / "lab" / "evidence" / "E003_table.md").write_text(out.to_markdown(floatfmt=".4f") + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
