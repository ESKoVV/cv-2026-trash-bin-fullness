#!/usr/bin/env python3
"""Download the approved Roboflow datasets through the official SDK.

The API key is deliberately read only from ROBOFLOW_API_KEY.  It is never
printed, persisted, or included in generated reports.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    workspace: str
    project: str
    version: int
    url: str


DATASETS = (
    DatasetSpec(
        slug="garbage-can-overflow",
        workspace="mariswary-deepak-4ajr0",
        project="garbage-can-overflow",
        version=1,
        url="https://universe.roboflow.com/mariswary-deepak-4ajr0/garbage-can-overflow",
    ),
    DatasetSpec(
        slug="waste-bin-fill-level-detect2",
        workspace="image-data-mlcpz",
        project="waste-bin-fill-level-detect2-lxsf4-u3odm",
        version=1,
        url="https://universe.roboflow.com/image-data-mlcpz/waste-bin-fill-level-detect2-lxsf4-u3odm",
    ),
)


def complete_download(target: Path) -> bool:
    """Return whether the SDK has fully extracted a YOLO dataset at target."""
    return (target / "data.yaml").is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace an existing raw download.")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print(
            "Blocked: set ROBOFLOW_API_KEY in the environment, then rerun this command.\n"
            "Example (current shell only): read -rs ROBOFLOW_API_KEY; export ROBOFLOW_API_KEY"
        )
        return 2

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Missing dependency: pip install -r requirements-local.txt", file=sys.stderr)
        return 3

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # The SDK may write its own exception diagnostics, including temporary
    # download URLs.  Keep that output private: those URLs can contain a
    # short-lived download token.
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rf = Roboflow(api_key=api_key)
    except Exception as exc:
        print(f"Unable to authenticate with Roboflow ({type(exc).__name__}).", file=sys.stderr)
        return 4
    for spec in DATASETS:
        target = RAW_DIR / spec.slug
        if target.exists() and any(target.iterdir()):
            if complete_download(target) and not args.force:
                print(f"Already present, skipped: {target}")
                continue
            print(f"Replacing incomplete download: {target}")
            shutil.rmtree(target)

        print(f"Downloading {spec.slug} as YOLOv8 format …")
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                project = rf.workspace(spec.workspace).project(spec.project)
                project.version(spec.version).download("yolov8", location=str(target), overwrite=True)
        except Exception as exc:
            print(f"Failed to download {spec.slug} ({type(exc).__name__}).", file=sys.stderr)
            return 4
        print(f"Downloaded: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
