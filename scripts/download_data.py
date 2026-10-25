"""下载 chenditc/investment_data 打包好的 Qlib 数据，解压到 Qlib 数据目录。

用法：
    python scripts/download_data.py                  # 最新一版，解压到 ~/.qlib/qlib_data/cn_data
    python scripts/download_data.py --tag 2026-10-25 # 指定版本
    python scripts/download_data.py --target /path/to/cn_data

数据包约 570MB。解压后在目标目录写一个 .qt_data_version，记录下载的是哪一版。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qt.config import DATA_DIR  # noqa: E402

REPO = "https://github.com/chenditc/investment_data/releases"
ASSET = "qlib_bin.tar.gz"
VERSION_FILE = ".qt_data_version"


class _FirstRedirect(urllib.request.HTTPRedirectHandler):
    """记下第一次跳转的地址：latest 会先跳到 /releases/download/<tag>/...，从里面读出版本号。"""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.urls.append(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _tag_from_url(url: str) -> str | None:
    parts = url.split("/releases/download/")
    return parts[1].split("/")[0] if len(parts) == 2 else None


def download(tag: str | None, dest: Path) -> str:
    url = f"{REPO}/download/{tag}/{ASSET}" if tag else f"{REPO}/latest/download/{ASSET}"
    handler = _FirstRedirect()
    opener = urllib.request.build_opener(handler)
    with opener.open(url) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r下载中 {done / total:6.1%}  {done >> 20} / {total >> 20} MB", end="", flush=True)
    print()
    resolved = tag or next((t for t in map(_tag_from_url, handler.urls) if t), None)
    return resolved or "unknown"


def extract(archive: Path, target: Path) -> None:
    """包里顶层目录是 qlib_bin/，解压时去掉这一层。"""
    with tarfile.open(archive) as tar:
        members = []
        for m in tar.getmembers():
            parts = Path(m.name).parts
            if len(parts) < 2:
                continue
            m.name = str(Path(*parts[1:]))
            members.append(m)
        tar.extractall(target, members=members, filter="data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", help="release 版本号，例如 2026-10-25；不填用最新版")
    ap.add_argument("--target", type=Path, default=DATA_DIR, help=f"解压目录，默认 {DATA_DIR}")
    ap.add_argument("--force", action="store_true", help="目标目录已有数据时先删掉再解压")
    args = ap.parse_args()

    target: Path = args.target.expanduser()
    if (target / "calendars").exists():
        if not args.force:
            sys.exit(f"{target} 里已经有数据。要重新下载请加 --force")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / ASSET
        tag = download(args.tag, archive)
        print("解压中 ...")
        extract(archive, target)
    (target / VERSION_FILE).write_text(tag + "\n")
    print(f"完成：版本 {tag}，位于 {target}")


if __name__ == "__main__":
    main()
