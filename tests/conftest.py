import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _test_segment_not_spent(monkeypatch):
    """qt/config.TEST_SEGMENT_SPENT 在真实环境里是 True（最终测试已做）；测试默认按"锁着"跑。"""
    import qt.config
    monkeypatch.setattr(qt.config, "TEST_SEGMENT_SPENT", False)
