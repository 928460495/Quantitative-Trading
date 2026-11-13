"""假设卡、基线和运行台账（docs/FRAMEWORK.md 第 4、7 节）。

    lab/cards/Hnnn.yaml   假设卡，提交之后才能跑；提交后不许改（改了程序拒绝运行）
    lab/baseline.yaml     现行基线流程
    lab/runs.jsonl        每次账户检验一条，只追加
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from qt.lab import gates
from qt.lab.procedure import Procedure
from qt.lab.protocol import COOLDOWN_AFTER, MAX_CARDS, MAX_CARDS_PER_FAMILY, MAX_RUNS_PER_CARD

REQUIRED = ("id", "parent", "family", "kind", "mechanism", "change", "prediction", "falsification", "variant",
            "seen_before", "status")
REGISTERED, ADOPTED, REJECTED, ABANDONED = "已登记", "已采纳", "未采纳", "放弃"
STATUSES = (REGISTERED, ADOPTED, REJECTED, ABANDONED)


class CardError(ValueError):
    """假设卡写得不对，或者不满足运行条件。"""


def load_card(path: Path) -> dict:
    path = Path(path)
    card = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(card, dict):
        raise CardError(f"{path.name} 不是一张卡")
    missing = [k for k in REQUIRED if k not in card]
    if missing:
        raise CardError(f"{path.name} 缺字段：{missing}")
    if card["id"] != path.stem:
        raise CardError(f"编号 {card['id']} 和文件名 {path.stem} 不一致")
    if card["kind"] not in gates.KINDS:
        raise CardError(f"kind 只能是 {gates.KINDS}：{card['kind']}")
    if card["status"] not in STATUSES:
        raise CardError(f"status 只能是 {STATUSES}：{card['status']}")
    pred = card["prediction"]
    if not isinstance(pred, dict) or set(pred) < {"metric", "direction"} or pred["direction"] not in gates.DIRECTIONS:
        raise CardError("prediction 要写 metric 和 direction（升 / 降）")
    if "." not in str(pred["metric"]):
        raise CardError("prediction.metric 的写法是 '阶段.列名' 或 '配对.阶段'")
    if not isinstance(card["variant"], dict) or not card["variant"]:
        raise CardError("variant 要写对基线流程的字段改动")
    for key in ("mechanism", "change", "falsification", "seen_before"):
        if not str(card[key] or "").strip():
            raise CardError(f"{key} 不能空着")
    return card


def load_cards(cards_dir: Path) -> list[dict]:
    return [load_card(p) for p in sorted(Path(cards_dir).glob("H*.yaml"))]


def is_committed(path: Path) -> bool:
    """文件已经进了仓库，而且和最近一次提交的内容一样（没有没提交的改动）。"""
    path = Path(path).resolve()
    git = lambda *args: subprocess.run(["git", *args], cwd=path.parent, capture_output=True, text=True)
    if git("ls-files", "--error-unmatch", path.name).returncode != 0:
        return False
    return git("status", "--porcelain", "--", path.name).stdout.strip() == ""


def load_runs(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_run(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")


def valid_runs(runs: list[dict], card_id: str) -> list[dict]:
    """这张卡没被作废的账户检验记录。作废记录是 {"card", "void": 第几次, "reason"}，追加在台账里。"""
    void = {r["void"] for r in runs if r.get("card") == card_id and "void" in r}
    return [r for r in runs if r.get("card") == card_id and "run" in r and r["run"] not in void]


def next_run_number(runs: list[dict], card_id: str) -> int:
    return sum(1 for r in runs if r.get("card") == card_id and "run" in r) + 1


def void_record(runs: list[dict], card_id: str, run: int, reason: str) -> dict:
    """作废某一次运行（比如程序错误让结果无效）。只追加记录，不删原来的；作废的不占预算，编号不复用。"""
    from datetime import datetime, timezone

    if not any(r.get("card") == card_id and r.get("run") == run for r in runs):
        raise CardError(f"{card_id} 没有第 {run} 次运行")
    if not str(reason or "").strip():
        raise CardError("作废要写明原因")
    return {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "card": card_id, "void": run,
            "reason": reason}


def check_budget(card: dict, cards: list[dict], runs: list[dict]) -> None:
    """能不能再跑一次这张卡；不能就报 CardError 并说明是哪条预算。"""
    if card["status"] != REGISTERED:
        raise CardError(f"{card['id']} 已经有结论（{card['status']}），不能再跑")
    used = len(valid_runs(runs, card["id"]))
    if used >= MAX_RUNS_PER_CARD:
        raise CardError(f"{card['id']} 已经跑了 {used} 次，一张卡最多 {MAX_RUNS_PER_CARD} 次")
    if len(cards) > MAX_CARDS:
        raise CardError(f"这个世代已经有 {len(cards)} 张卡，最多 {MAX_CARDS} 张")
    same = [c["id"] for c in cards if c["id"] != card["id"] and c["status"] == REJECTED
            and c.get("parent") == card.get("parent") and _canonical(c["variant"]) == _canonical(card["variant"])]
    if same:
        raise CardError(f"同一个改动已经在 {same[0]} 被否决：不能换个类型或说法再提（框架失效模式 F10）")
    family = [c for c in cards if c["family"] == card["family"]]
    if len(family) > MAX_CARDS_PER_FAMILY:
        raise CardError(f"方向「{card['family']}」已经有 {len(family)} 张卡，最多 {MAX_CARDS_PER_FAMILY} 张")
    decided = [c["status"] for c in sorted(family, key=lambda c: c["id"])
               if c["id"] < card["id"] and c["status"] in (ADOPTED, REJECTED)]
    if len(decided) >= COOLDOWN_AFTER and all(s == REJECTED for s in decided[-COOLDOWN_AFTER:]):
        raise CardError(f"方向「{card['family']}」连续 {COOLDOWN_AFTER} 张卡没被采纳，本世代冷却")


def _canonical(variant: dict) -> str:
    """改动的规范写法：字典按键排序、列表按内容排序，写法不同但改动一样的会得到同一个字符串。"""
    def norm(x):
        if isinstance(x, dict):
            return {k: norm(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return sorted((norm(v) for v in x), key=lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True))
        return x
    return json.dumps(norm(variant), ensure_ascii=False, sort_keys=True)


def load_baseline(path: Path) -> Procedure:
    return Procedure.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["procedure"])


def candidate(card: dict, baseline: Procedure) -> Procedure:
    try:
        return baseline.replace(**card["variant"])
    except (ValueError, TypeError) as e:
        raise CardError(f"{card['id']} 的 variant 写得不对：{e}") from e
