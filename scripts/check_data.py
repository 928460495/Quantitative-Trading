"""检查 Qlib 数据包能不能读、质量怎么样，结果写到 reports/data_check.md。

重点看 2019 年前后：数据包里 Wind / Caihui 两个来源只到 2019 年，之后换成 Tushare 等，
如果有口径跳变，会体现在异常收益、缺失率、零成交比例的年度变化上。

只检查到验证段结束（2021 年底），不碰最终测试段。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qt.config import DATA_DIR, SEGMENTS  # noqa: E402
from qt.data import load_features  # noqa: E402

MARKET = "csi300"
START = "2005-04-08"  # 数据包里沪深300 成分从这一天开始
END = SEGMENTS["valid"][1]


def _instrument_summary() -> pd.DataFrame:
    rows = []
    for f in sorted((DATA_DIR / "instruments").glob("*.txt")):
        spans = pd.read_csv(f, sep="\t", header=None, names=["code", "start", "end"])
        rows.append({"股票池": f.stem, "股票数": spans["code"].nunique(), "成分区间条数": len(spans)})
    return pd.DataFrame(rows)


def _yearly_quality(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["ret"]
    year = df.index.get_level_values("datetime").year
    g = pd.DataFrame(
        {
            "year": year,
            "close_missing": df["close"].isna(),
            "no_volume": df["volume"].isna() | (df["volume"] <= 0),
            "abs_ret_gt_11pct": ret.abs() > 0.11,
            "abs_ret_gt_21pct": ret.abs() > 0.21,
            "change_mismatch": (ret - df["change"]).abs() > 1e-3,
        }
    ).groupby("year")
    out = g.mean() * 100
    out.insert(0, "日均股票数", df.groupby(year).size() / df.index.get_level_values("datetime").to_series().groupby(year).nunique().values)
    out.columns = ["日均股票数", "收盘价缺失%", "无成交%", "|日收益|>11%", "|日收益|>21%", "收益与change不一致%"]
    return out.round(3)


def main() -> None:
    version_file = DATA_DIR / ".qt_data_version"
    version = version_file.read_text().strip() if version_file.exists() else "未知（不是用 download_data.py 下载的）"
    cal = pd.read_csv(DATA_DIR / "calendars" / "day.txt", header=None)[0]

    df = load_features(
        {
            "close": "$close",
            "adjclose": "$adjclose",
            "factor": "$factor",
            "volume": "$volume",
            "change": "$change",
            "ret": "$close/Ref($close,1)-1",
        },
        MARKET,
        START,
        END,
    )
    yearly = _yearly_quality(df)
    ratio = (df["close"] / df["adjclose"]).dropna().groupby(level="instrument")
    ratio_rel_std = (ratio.std() / ratio.mean()).max()

    lines = [
        "# 数据检查",
        "",
        f"- 数据目录：`{DATA_DIR}`",
        f"- 数据版本：{version}",
        f"- 交易日历：{cal.iloc[0]} 至 {cal.iloc[-1]}，共 {len(cal)} 天",
        f"- 检查范围：{MARKET} 历史成分，{START} 至 {END}（不碰最终测试段）",
        "",
        "## 股票池",
        "",
        _instrument_summary().to_markdown(index=False),
        "",
        "## 沪深300 成分股的逐年质量",
        "",
        "日收益用复权收盘价算。主板涨跌停 10%，所以 |日收益| > 11% 主要来自创业板 / 科创板、新股和复牌；",
        "> 21% 基本只可能是复权或数据错误。重点比较 2019 年前后各列有没有跳变。",
        "",
        yearly.to_markdown(),
        "",
        "## 收盘价与复权收盘价",
        "",
        f"`$close / $adjclose` 在每只股票内部的相对标准差，所有股票中最大为 {ratio_rel_std:.2e}。",
        "接近 0 说明两者只差一个常数倍，`$close` 算出的收益就是复权收益。",
        "",
    ]
    out = ROOT / "reports" / "data_check.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"已写入 {out}")
    print(yearly.to_string())


if __name__ == "__main__":
    main()
