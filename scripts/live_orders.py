"""每日出单（计划 5 任务 3）：信号日收盘后，按现行基线流程算出下一个交易日的买卖清单。

用法（单进程）：
    python scripts/live_orders.py 2021-12-01 --state data/live/state.json

账户状态文件（JSON）：{"anchor": 跟单开始日, "cash": 现金, "positions": {代码: {"shares": 股数, "buy_day": 日期}}}。
清单写到 data/live/orders/（不进仓库）：orders-<信号日>.csv 和 .md。安全检查不通过时退出码为 1，按停止规则③当天不交易。

现在只能用开发段（2021 年底以前）的数据：最新数据要读封存段，等用户决定（DECISIONS 2026-11-13）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qt.lab import procedure as pc  # noqa: E402
from qt.live.daily import State, check_orders, load_state, make_orders, write_sheet  # noqa: E402
from qt.live.orders import Orders  # noqa: E402

OUT = ROOT / "data" / "live" / "orders"


def run(dev: dict, proc: pc.Procedure, state: State, signal_date: pd.Timestamp, out_dir: Path
        ) -> tuple[Path, list[str], Orders]:
    dates = dev["close"].index
    signal_date = pd.Timestamp(signal_date)
    if signal_date not in dates:
        raise ValueError(f"数据里没有这一天：{signal_date.date()}（不是交易日，或者数据还没更新到这天）")
    if state.anchor not in dates:
        raise ValueError(f"跟单开始日 {state.anchor.date()} 不在数据里")
    data = pc.market(dev, proc, dates.get_loc(state.anchor))          # 从开始日起算，和回测的错开起点一样
    orders, meta = make_orders(data, pc.rules(proc), pc.exposure(dev, proc, data.dates), state, signal_date)
    held = {c: p["shares"] for c, p in state.positions.items()}
    problems = check_orders(orders, held, meta["slot_limit"])
    return write_sheet(orders, meta, problems, out_dir, held), problems, orders


def main() -> None:
    from qt.lab import cards as cd
    from qt.lab.store import load_dev
    from run_plan3c_train import POOL_CACHE, latest_version

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date")
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    proc = cd.load_baseline(ROOT / "lab" / "baseline.yaml")
    dev = load_dev(latest_version(POOL_CACHE), proc.factors())
    path, problems, _ = run(dev, proc, load_state(args.state), pd.Timestamp(args.date), args.out)
    print(path.read_text())
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
