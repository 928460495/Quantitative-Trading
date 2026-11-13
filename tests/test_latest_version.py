"""缓存版本：开发段和 live 段可能在不同的数据版本目录下。"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_latest_version_per_segment(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_plan3c_train import latest_version
    for d in ("2026-10-25+local-20261101/train", "2026-11-13+local-20261113/live"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "meta.json").write_text("{}")
    assert latest_version(tmp_path).name == "2026-10-25+local-20261101"
    assert latest_version(tmp_path, "live").name == "2026-11-13+local-20261113"
