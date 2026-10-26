"""下载用户上传到本仓库 GitHub Release 的全 A 数据包，解包到 QT_FULL_A_DIR（默认 ~/.qlib/qt_full_a）。

用法：
    python scripts/download_full_a.py --tag full-a-20261027

数据包由 scripts/collect_full_a.py 在本地生成，内容见那个脚本的说明。解包后写 .version 记录 tag。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qt.config import FULL_A_DIR  # noqa: E402

REPO = "928460495/Quantitative-Trading"
VERSION_FILE = ".version"


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def release_asset_url(tag: str, repo: str = REPO, fetch=_fetch_json) -> str:
    """这个 tag 下以 full_a_ 开头、.tar 结尾的附件的下载地址。"""
    release = fetch(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
    for asset in release.get("assets", []):
        if asset["name"].startswith("full_a_") and asset["name"].endswith(".tar"):
            return asset["browser_download_url"]
    raise ValueError(f"Release {tag} 里没有 full_a_*.tar 附件")


def install(archive: Path, target: Path, tag: str) -> None:
    """解包到 target。只接受普通文件，路径不能跳出 target。"""
    root = target.resolve()
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for m in members:
            dest = (root / m.name).resolve()
            if not (m.isfile() or m.isdir()) or not dest.is_relative_to(root):
                raise ValueError(f"数据包里有不安全的条目：{m.name}")
        target.mkdir(parents=True, exist_ok=True)
        tar.extractall(target, members=members, filter="data")
    (target / VERSION_FILE).write_text(tag)


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
        total, done = int(resp.headers.get("Content-Length", 0)), 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r下载中 {done / total:6.1%}  {done >> 20} / {total >> 20} MB", end="", flush=True)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="Release 的 tag，例如 full-a-20261027")
    ap.add_argument("--target", type=Path, default=FULL_A_DIR, help=f"解包目录，默认 {FULL_A_DIR}")
    args = ap.parse_args()
    target: Path = args.target.expanduser()
    url = release_asset_url(args.tag)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / Path(url).name
        _download(url, archive)
        staging = target.with_name(target.name + ".new")  # 解包成功后再替换，坏包不会弄丢旧数据
        shutil.rmtree(staging, ignore_errors=True)
        install(archive, staging, args.tag)
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    print(f"完成：{args.tag}，位于 {target}")


if __name__ == "__main__":
    main()
