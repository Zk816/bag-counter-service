from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = Path(__file__).with_name("model.lock.json")
TIMEOUT_S = 120
CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dst: Path) -> None:
    part = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, part.open("wb") as f:  # noqa: S310
        for block in iter(lambda: r.read(CHUNK), b""):
            f.write(block)
    part.replace(dst)


def main() -> int:
    lock = json.loads(LOCK.read_text())
    p = argparse.ArgumentParser(description="download the packaged model and verify its checksums")
    p.add_argument("--url", default=lock["url"], help="download URL with a {name} placeholder for each file")
    a = p.parse_args()
    target = ROOT / lock["target"]
    target.mkdir(parents=True, exist_ok=True)
    for name, digest in lock["files"].items():
        dst = target / name
        if dst.is_file() and sha256(dst) == digest:
            sys.stdout.write(f"ok      {name}\n")
            continue
        download(a.url.format(name=name), dst)
        if sha256(dst) != digest:
            dst.unlink()
            sys.stderr.write(f"checksum mismatch for {name}: refusing to use it\n")
            return 1
        sys.stdout.write(f"fetched {name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
