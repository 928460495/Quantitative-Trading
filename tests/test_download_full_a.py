"""全 A 数据包的下载和解包：找 Release 附件、拒绝跳出目标目录的路径、记录版本。不联网。"""

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_full_a", ROOT / "scripts" / "download_full_a.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


def _tar(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_release_asset_url_picks_full_a_tar():
    def fetch(url):
        assert url == "https://api.github.com/repos/928460495/Quantitative-Trading/releases/tags/full-a-20261027"
        return {"assets": [
            {"name": "notes.txt", "browser_download_url": "https://x/notes.txt"},
            {"name": "full_a_20261027.tar", "browser_download_url": "https://x/full_a_20261027.tar"},
        ]}

    assert dl.release_asset_url("full-a-20261027", fetch=fetch) == "https://x/full_a_20261027.tar"


def test_release_asset_url_without_tar_raises():
    with pytest.raises(ValueError, match="full_a_"):
        dl.release_asset_url("full-a-1", fetch=lambda url: {"assets": []})


def test_install_rejects_path_traversal(tmp_path):
    archive = _tar(tmp_path / "evil.tar", {"daily.parquet": b"x", "../evil.txt": b"boom"})
    with pytest.raises(ValueError):
        dl.install(archive, tmp_path / "full_a", "full-a-1")
    assert not (tmp_path / "evil.txt").exists()


def test_install_writes_version(tmp_path):
    archive = _tar(tmp_path / "good.tar", {"daily.parquet": b"x", "reports/income.parquet": b"y", "meta.json": b"{}"})
    dl.install(archive, tmp_path / "full_a", "full-a-20261027")
    assert (tmp_path / "full_a" / ".version").read_text() == "full-a-20261027"
    assert (tmp_path / "full_a" / "reports" / "income.parquet").read_bytes() == b"y"
