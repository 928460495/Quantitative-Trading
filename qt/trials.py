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

import numpy as np

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


def load_unique(path: Path = TRIALS_FILE) -> list[dict]:
    """按配置哈希去重后的试验记录，同一配置保留最后一次。"""
    if not path.exists():
        return []
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        latest[r["config_hash"]] = r
    return list(latest.values())


EULER_GAMMA = 0.5772156649


def expected_max_sharpe(n_trials: int, sharpe_var: float) -> float:
    """N 次独立试验、真实夏普都为 0 时，最大夏普比率的期望（Bailey & López de Prado 2014）。"""
    from scipy.stats import norm

    if n_trials < 2:
        return 0.0
    return np.sqrt(sharpe_var) * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
    )


def deflated_sharpe(returns, n_trials: int, sharpe_var: float) -> dict[str, float]:
    """通缩夏普比率：观察到的夏普比率超过"试了 N 次碰运气能拿到的最大夏普"的概率。

    returns 是每日收益；sharpe_var 是各次试验每日夏普比率的方差（和 returns 同一频率）。
    考虑了收益的偏度和峰度。返回每日夏普、门槛夏普和 DSR。
    """
    from scipy.stats import kurtosis, norm, skew

    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    t = len(r)
    sr = r.mean() / r.std(ddof=1)
    sr0 = expected_max_sharpe(n_trials, sharpe_var)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return {"sharpe": float(sr), "sharpe_threshold": float(sr0), "dsr": float(norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom))}
