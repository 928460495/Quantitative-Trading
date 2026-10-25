"""Qlib 初始化与取数。所有取数都走 load_features()，它会先检查测试段封存。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qt.config import DATA_DIR, check_range

_initialized = False


def init_qlib() -> None:
    global _initialized
    if _initialized:
        return
    import qlib

    if not (DATA_DIR / "calendars" / "day.txt").exists():
        raise FileNotFoundError(f"在 {DATA_DIR} 找不到 Qlib 数据，先运行 python scripts/download_data.py")
    qlib.init(provider_uri=str(DATA_DIR), region="cn", kernels=1)
    _initialized = True


def load_features(exprs: dict[str, str], market: str, start: str, end: str) -> pd.DataFrame:
    """按指数历史成分取因子值。返回以 (datetime, instrument) 为索引的 DataFrame，列名为 exprs 的键。

    Qlib 按成分股的进出日期取数，所以每天只包含当天在指数里的股票。±inf 统一换成 NaN。
    """
    check_range(start, end)
    init_qlib()
    from qlib.data import D

    df = D.features(D.instruments(market), list(exprs.values()), start_time=start, end_time=end)
    df.columns = list(exprs)
    df = df.swaplevel().sort_index()
    df.index.names = ["datetime", "instrument"]
    return df.replace([np.inf, -np.inf], np.nan)


def drop_embargo(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """去掉样本段最后 n 个交易日。

    标签往后看 n 天，Qlib 会为此读取 end 之后的行情，所以段尾这几天的标签用到了下一段的数据。
    去掉它们，保证训练段的标签不碰验证段、验证段的标签不碰测试段。
    """
    dates = df.index.get_level_values("datetime").unique().sort_values()
    if len(dates) <= n:
        raise ValueError(f"样本只有 {len(dates)} 个交易日，不够去掉 {n} 天")
    return df[df.index.get_level_values("datetime") <= dates[-n - 1]]


def trading_calendar(start: str, end: str) -> pd.DatetimeIndex:
    check_range(start, end)
    init_qlib()
    from qlib.data import D

    return pd.DatetimeIndex(D.calendar(start_time=start, end_time=end))
