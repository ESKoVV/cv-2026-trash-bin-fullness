#!/usr/bin/env python3
"""Run an independent integrity check on data/merged after preparation."""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "merged"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    issues: list[str] = []
    class_counts: Counter[int] = Counter()
    all_hashes: dict[str, str] = {}
    image_counts: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        images_dir = DATA / "images" / split
        labels_dir = DATA / "labels" / split
        if not images_dir.is_dir() or not labels_dir.is_dir():
            issues.append(f"missing split directories: {split}")
            continue
        for image_path in sorted(path for path in images_dir.iterdir() if path.suffix.casefold() in EXTENSIONS):
            image_counts[split] += 1
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                issues.append(f"missing label: {image_path}")
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                issues.append(f"corrupt image: {image_path}")
            image_hash = digest(image_path)
            prior = all_hashes.get(image_hash)
            if prior is not None and prior != split:
                issues.append(f"cross-split duplicate: {image_path} also in {prior}")
            all_hashes[image_hash] = split
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                issues.append(f"empty label: {label_path}")
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                fields = line.split()
                try:
                    class_id = int(fields[0])
                    x, y, width, height = map(float, fields[1:])
                except (ValueError, IndexError):
                    issues.append(f"malformed label {label_path}:{line_number}")
                    continue
                if len(fields) != 5 or class_id not in range(4) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1 and x - width / 2 >= 0 and x + width / 2 <= 1 and y - height / 2 >= 0 and y + height / 2 <= 1):
                    issues.append(f"invalid bbox {label_path}:{line_number}")
                    continue
                class_counts[class_id] += 1
        if image_counts[split] == 0:
            issues.append(f"empty split: {split}")

    print("Images:", dict(image_counts))
    print("Objects:", {name: class_counts[index] for index, name in enumerate(("empty", "partial", "full", "overflow"))})
    if issues:
        print("FAILED")
        print("\n".join(issues))
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
