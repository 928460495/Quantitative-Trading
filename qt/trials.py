"""试验日志：每次检验一个因子、换一组参数、换一种合成方法，都记一条。

第 5 个 PR 用它统计试验次数，算通缩夏普比率（Bailey & López de Prado 2014）。重复运行完全相同
的配置不算新试验，所以每条记录带一个配置哈希，统计时按哈希去重。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIALS_FILE = Path(os.environ.get("QT_TRIALS_FILE", ROOT / "reports" / "trials.jsonl"))


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def config_hash(kind: str, params: dict) -> str:
    blob = json.dumps({"kind": kind, "params": params}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def log_trial(kind: str, params: dict, metrics: dict, path: Path = TRIALS_FILE) -> None:
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "kind": kind,
        "config_hash": config_hash(kind, params),
        "params": params,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")


def count_unique(path: Path = TRIALS_FILE, kind: str | None = None) -> int:
    if not path.exists():
        return 0
    hashes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if kind is None or r["kind"] == kind:
            hashes.add(r["config_hash"])
    return len(hashes)
