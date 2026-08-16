#!/usr/bin/env python3
"""Download NATL60 OSSE NetCDFs listed in data/natl60.py into data/natl60/.

Files are multi-GB. Existing complete files are skipped. Partial ``.part``
files resume via HTTP Range (curl -C - when available, else urllib).
Do not run this in CI.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.natl60 import NATL60_URLS

CHUNK = 1024 * 1024
UA = "4dvarnet-natl60-downloader"


def _load_destinations() -> dict[str, Path]:
    with open(ROOT / "config" / "paths.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    natl = cfg.get("natl60") or {}
    out = {}
    for key, url in NATL60_URLS.items():
        rel = natl.get(key, f"data/natl60/{Path(url).name}")
        path = Path(rel)
        if not path.is_absolute():
            path = ROOT / path
        out[key] = path
    return out


def _head_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except (urllib.error.URLError, ValueError, TypeError, OSError):
        return None


def _download_curl(url: str, dest: Path, tmp: Path) -> bool:
    curl = shutil.which("curl")
    if not curl:
        return False
    # Quiet progress: avoid flooding terminal logs (can fill small system disks).
    # -C - resumes; -L follows redirects; -f fails on HTTP errors; -S shows errors with -s
    cmd = [
        curl,
        "-L",
        "-f",
        "-s",
        "-S",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "-C",
        "-",
        "-A",
        UA,
        "-o",
        str(tmp),
        url,
    ]
    print(f"   resume via curl ({tmp.stat().st_size if tmp.exists() else 0} bytes already)", flush=True)
    r = subprocess.run(cmd)
    if r.returncode not in (0,):
        # curl exit 33 = HTTP server doesn't support range; retry from scratch once
        if r.returncode == 33 and tmp.exists():
            tmp.unlink()
            r = subprocess.run(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"curl failed with exit {r.returncode} for {url}")
    return True


def _download_urllib(url: str, dest: Path, tmp: Path) -> None:
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": UA}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        print(f"   resume via urllib from byte {existing}")
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        if e.code == 416 and existing > 0:
            # Already complete according to server
            return
        if existing > 0 and e.code in (400, 401, 403, 404, 501):
            print("   server rejected Range; restarting from scratch")
            tmp.unlink(missing_ok=True)
            existing = 0
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            resp = urllib.request.urlopen(req, timeout=120)
        else:
            raise
    mode = "ab" if existing > 0 and resp.status == 206 else "wb"
    if mode == "wb" and existing > 0:
        # Server ignored Range and sent full body
        existing = 0
    with resp, open(tmp, mode) as f:
        downloaded = existing
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (64 * CHUNK) < CHUNK:
                print(f"   ... {downloaded / (1024 ** 3):.2f} GiB", flush=True)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"GET {url}", flush=True)
    print(f" -> {dest}", flush=True)
    expected = _head_size(url)
    if expected is not None:
        print(f"   Content-Length: {expected} bytes ({expected / (1024 ** 3):.2f} GiB)", flush=True)
    if not _download_curl(url, dest, tmp):
        _download_urllib(url, dest, tmp)
    size = tmp.stat().st_size if tmp.exists() else 0
    if expected is not None and size < expected:
        raise RuntimeError(
            f"incomplete download for {dest.name}: got {size} / {expected} bytes "
            f"(re-run the script to resume)"
        )
    if size <= 0:
        raise RuntimeError(f"empty download for {url}")
    tmp.replace(dest)
    print(f"   wrote {dest.stat().st_size} bytes", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Download NATL60 OSSE NetCDFs (multi-GB, resumable).")
    p.add_argument("--dry-run", action="store_true", help="print URLs and destinations only")
    p.add_argument("--only", nargs="*", default=None, help="subset of keys: obs oi ssh_ref sst_ref u_ref v_ref")
    p.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = p.parse_args()

    dests = _load_destinations()
    keys = args.only or list(NATL60_URLS)
    print("NATL60 files are typically several GB each. Prefer --dry-run first.")
    print("Resume: partial *.part files continue via curl -C - or HTTP Range.")
    for key in keys:
        if key not in NATL60_URLS:
            raise SystemExit(f"unknown key {key}; choose from {list(NATL60_URLS)}")
        dest = dests[key]
        url = NATL60_URLS[key]
        exists = dest.exists() and dest.stat().st_size > 0
        if args.dry_run:
            status = "exists" if exists else "missing"
            size = dest.stat().st_size if exists else None
            remote = _head_size(url)
            remote_s = f"{remote} B" if remote is not None else "unknown"
            local_s = f"{size} B" if size is not None else "-"
            print(f"[{status}] {key}: remote={remote_s} local={local_s}")
            print(f"         {url}")
            print(f"      -> {dest}")
            continue
        if exists and not args.force:
            print(f"skip {key} (already at {dest}, {dest.stat().st_size} bytes)")
            continue
        if args.force and dest.exists():
            dest.unlink()
            part = dest.with_suffix(dest.suffix + ".part")
            part.unlink(missing_ok=True)
        download_file(url, dest)
    if args.dry_run:
        print("dry-run complete; no files written")


if __name__ == "__main__":
    main()
