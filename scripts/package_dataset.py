#!/usr/bin/env python3
"""Create and independently verify the portable Colab dataset archive."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "merged"
ARCHIVE = ROOT / "trash_bin_dataset.zip"
REQUIRED = ("data.yaml", "images", "labels")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace an existing archive.")
    args = parser.parse_args()
    if ARCHIVE.exists() and not args.force:
        raise SystemExit(f"Archive already exists: {ARCHIVE}; use --force to replace it.")
    if not all((SOURCE / item).exists() for item in REQUIRED):
        raise SystemExit("Merged dataset does not have data.yaml, images and labels.")

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(SOURCE.rglob("*")):
            if path.is_file() and path.name != ".gitkeep":
                archive.write(path, path.relative_to(SOURCE).as_posix())

    with zipfile.ZipFile(ARCHIVE) as archive:
        broken = archive.testzip()
        if broken is not None:
            raise SystemExit(f"Archive checksum failed for member: {broken}")
        with tempfile.TemporaryDirectory(prefix="trash-bin-archive-") as temporary:
            extracted = Path(temporary)
            archive.extractall(extracted)
            if not all((extracted / item).exists() for item in REQUIRED):
                raise SystemExit("Archive root layout is invalid after extraction.")
            source_files = sorted(
                path.relative_to(SOURCE).as_posix()
                for path in SOURCE.rglob("*")
                if path.is_file() and path.name != ".gitkeep"
            )
            extracted_files = sorted(path.relative_to(extracted).as_posix() for path in extracted.rglob("*") if path.is_file())
            if source_files != extracted_files:
                raise SystemExit("Archive contents differ from data/merged.")
    print(f"Archive: {ARCHIVE}")
    print(f"SHA-256: {sha256(ARCHIVE)}")
    print("Extraction and contents: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
