"""E008（诊断，只报告，不改 E007 的结论）：估值仓位和固定比例的差距，有多少来自极端行情的交易日。

运行前写定：
- 沿用 E007 的规则和数据（估值仓位、同平均比例的固定比例、中证全指），两段：2005–2009（E007 的检验段）、2010–2021（已看过）
- "极端行情日"：该段里中证全指日收益最高的 5% 和最低的 5% 的交易日
- 报告：① 全部交易日、② 去掉头尾各 5% 之后，估值仓位、固定比例、中证全指的年化（按剩下的日子复利再年化）；
  ③ 两者的累计差距（对数收益之差）拆成"极端日"和"其余日子"两部分
- 去掉极端日后的收益在实盘里拿不到，这只用来理解差距从哪里来；E007 "不通过"的结论不因此改变（DECISIONS 2026-11-13）
在仓库根目录运行：python lab/evidence/E008_trim.py
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

from E004_mix import DAYS  # noqa: E402
from E007_valuation_oos import REPORT, TEST, daily_pct, equity_returns, evaluate, monthly_pct  # noqa: E402

TAIL = 0.05


def ann(r: np.ndarray) -> float:
    return float(np.prod(1 + r) ** (DAYS / len(r)) - 1)


def main() -> None:
    r = equity_returns()
    p = daily_pct(monthly_pct(), r.index)
    rows = {}
    for name, (a, b) in (("2005–2009（E007 检验段）", TEST), ("2010–2021（已看过）", REPORT)):
        rr = r.loc[a:b]
        res = evaluate(rr, p.loc[a:b])
        idx = rr.to_numpy()
        lo, hi = np.quantile(idx, TAIL), np.quantile(idx, 1 - TAIL)
        extreme = (idx <= lo) | (idx >= hi)
        up, down = idx >= hi, idx <= lo
        acct, fixed = res["_acct"], res["_fixed"]
        gap = np.log1p(acct) - np.log1p(fixed)
        rows[name] = {
            "极端日数": int(extreme.sum()), "总日数": len(idx),
            "全部：估值仓位": ann(acct), "全部：固定比例": ann(fixed), "全部：中证全指": ann(idx),
            "去头尾：估值仓位": ann(acct[~extreme]), "去头尾：固定比例": ann(fixed[~extreme]), "去头尾：中证全指": ann(idx[~extreme]),
            "累计差距（对数）": gap.sum(), "其中大涨日": gap[up].sum(), "其中大跌日": gap[down].sum(), "其中其余日子": gap[~extreme].sum(),
            "大涨日平均股票比例": float(res["_w"][up].mean()), "大跌日平均股票比例": float(res["_w"][down].mean()),
        }
    table = pd.DataFrame(rows).T
    text = "\n".join([
        "# E008：估值仓位的差距有多少来自极端行情日（诊断，只报告）", "",
        "做法见 `E008_trim.py` 开头（运行前写定）。去掉极端日后的收益在实盘里拿不到；E007 \"不通过\"的结论不因此改变。", "",
        "\"累计差距\"= 估值仓位减固定比例的对数收益之和，正数表示估值仓位多赚；拆成大涨日、大跌日、其余日子三部分，三部分加起来等于总差距。", "",
        table.T.to_markdown(floatfmt=".4f"), ""])
    (ROOT / "lab" / "evidence" / "E008.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
