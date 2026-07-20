#!/usr/bin/env python3
"""Download and verify the large runtime artifacts described in artifacts.json.

The model checkpoint and ARCHS4 embedding index total roughly 1.5 GB, which is
more than GitHub's free Git LFS allowance. Hosting them outside the repository
keeps clones small; this script fetches them and checks their integrity.

Usage:

    # Verify whatever is already on disk (no network access)
    python fetch_artifacts.py --verify-only

    # Download anything missing or corrupt, then verify
    python fetch_artifacts.py

    # Re-download everything, ignoring existing files
    python fetch_artifacts.py --force

Downloads resume from a partial ``.part`` file when the server supports HTTP
range requests, so an interrupted 1 GB transfer does not restart from zero.
A file is only moved into place after its SHA-256 matches the manifest, so an
aborted or corrupted run can never leave a bad artifact where the app will
silently load it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "artifacts.json"

CHUNK_BYTES = 1024 * 1024
USER_AGENT = "bridge-rna-fetch-artifacts/1.0"

# Progress uses carriage returns, which turn into megabytes of noise when
# redirected to a file or a CI log. Only animate for an interactive terminal.
INTERACTIVE = sys.stdout.isatty()
PROGRESS_STEP_PCT = 2.0


class _Progress:
    """Throttled single-line progress meter; silent when not interactive."""

    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = total
        self._last_pct = -PROGRESS_STEP_PCT

    def update(self, seen: int) -> None:
        if not INTERACTIVE or not self.total:
            return
        pct = 100.0 * seen / self.total
        if pct - self._last_pct < PROGRESS_STEP_PCT and seen < self.total:
            return
        self._last_pct = pct
        print(
            f"\r  {self.label} {pct:5.1f}% ({human_bytes(seen)} / {human_bytes(self.total)})",
            end="",
            flush=True,
        )

    def clear(self) -> None:
        if INTERACTIVE:
            print("\r" + " " * 64 + "\r", end="", flush=True)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} PB"


def sha256_of(path: Path, show_progress: bool = False) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    total = path.stat().st_size
    seen = 0
    meter = _Progress("verifying", total) if show_progress else None
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            seen += len(chunk)
            if meter is not None:
                meter.update(seen)
    if meter is not None:
        meter.clear()
    return digest.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}")
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def verify(path: Path, expected_sha: str, expected_bytes: int) -> tuple[bool, str]:
    """Return (ok, reason). Size is checked first because it is nearly free."""
    if not path.exists():
        return False, "missing"
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        return False, f"wrong size ({human_bytes(actual_bytes)}, expected {human_bytes(expected_bytes)})"
    actual_sha = sha256_of(path, show_progress=True)
    if actual_sha != expected_sha:
        return False, f"checksum mismatch (got {actual_sha[:16]}...)"
    return True, "ok"


def download(url: str, dest: Path, expected_bytes: int) -> None:
    """Download to a .part file, resuming if a partial transfer exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    existing = part.stat().st_size if part.exists() else 0
    if existing > expected_bytes:
        # A stale .part larger than the target cannot be a prefix of it.
        part.unlink()
        existing = 0

    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"  resuming at {human_bytes(existing)}")

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as exc:
        if existing and exc.code in (200, 416):
            # Server ignored or rejected the range request; start over.
            part.unlink(missing_ok=True)
            return download(url, dest, expected_bytes)
        raise

    # If the server ignored Range and sent the whole body, do not append to it.
    mode = "ab"
    if existing and response.status != 206:
        part.unlink(missing_ok=True)
        existing = 0
        mode = "wb"

    seen = existing
    meter = _Progress("downloading", expected_bytes)
    with response, part.open(mode) as fh:
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            fh.write(chunk)
            seen += len(chunk)
            meter.update(seen)
    meter.clear()

    os.replace(part, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-only", action="store_true", help="Check local files without downloading.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the local file is valid.")
    args = parser.parse_args()

    manifest = load_manifest()
    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        print("Manifest lists no artifacts; nothing to do.")
        return 0

    if manifest.get("doi"):
        print(f"Artifact record: {manifest.get('record_url') or manifest['doi']}\n")

    failures: list[str] = []
    unhosted: list[str] = []

    for entry in artifacts:
        path = ROOT / entry["path"]
        print(f"{entry['path']}  ({human_bytes(entry['bytes'])})")

        if not args.force:
            ok, reason = verify(path, entry["sha256"], entry["bytes"])
            if ok:
                print("  verified\n")
                continue
            print(f"  {reason}")

        if args.verify_only:
            failures.append(entry["path"])
            print()
            continue

        url = entry.get("url")
        if not url:
            # Expected before the files are published; not an integrity failure.
            unhosted.append(entry["path"])
            print("  no download URL in artifacts.json; cannot fetch\n")
            continue

        try:
            download(url, path, entry["bytes"])
        except Exception as exc:  # noqa: BLE001 - report and continue to next artifact
            failures.append(entry["path"])
            print(f"  download failed: {exc}\n")
            continue

        ok, reason = verify(path, entry["sha256"], entry["bytes"])
        if ok:
            print("  downloaded and verified\n")
        else:
            failures.append(entry["path"])
            print(f"  downloaded but {reason}\n")

    if unhosted:
        print("These artifacts have no download URL yet:")
        for name in unhosted:
            print(f"  - {name}")
        print(
            "\nThey are still distributed via Git LFS. Run 'git lfs pull', or add\n"
            "download URLs to artifacts.json once the files are published.\n"
        )

    if failures:
        print("FAILED:")
        for name in failures:
            print(f"  - {name}")
        print("\nRe-run without --verify-only to fetch, or 'git lfs pull' if using LFS.")
        return 1

    if not unhosted:
        print("All artifacts present and verified.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Partial downloads are kept as .part files and will resume.")
        sys.exit(130)
