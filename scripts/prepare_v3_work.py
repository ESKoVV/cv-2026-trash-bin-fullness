#!/usr/bin/env python3
"""Create immutable-source v3 work data and an auditable manual-review queue."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "merged_v2"
TARGET = ROOT / "data" / "merged_v3_work"
REPORTS = ROOT / "reports" / "v3_preparation"
REVIEW = ROOT / "reports" / "v2_review" / "review.csv"
MISSING = ROOT / "reports" / "v2_review" / "missing_objects.csv"
SUSPICIOUS = ROOT / "reports" / "best1_diagnostics_20260920" / "geometry_train" / "suspicious_boxes.csv"
SPLITS = ("train", "val", "test")
NAMES = ("not_full", "full", "overflow", "unknown")
NAME_TO_ID = {name: index for index, name in enumerate(NAMES)}
REVIEW_FIELDS = ["split", "image", "object_number", "old_class", "converted_class", "suggested_class", "reason", "human_decision"]
TARGET_OVERRIDE = ("train", "garbage-can-overflow_test_000537_0dbd38d8d2.jpg", 3)


def read_csv(path: Path, fields: list[str] | None = None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if fields is not None and reader.fieldnames != fields:
            raise SystemExit(f"Unexpected CSV fields in {path}: {reader.fieldnames}")
        return list(reader)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def parse_line(line: str, source: Path, number: int, *, require_inside: bool = False) -> tuple[int, float, float, float, float]:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"{source}:{number}: expected five YOLO values")
    class_id = int(fields[0])
    x, y, width, height = map(float, fields[1:])
    if class_id not in range(4) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
        raise ValueError(f"{source}:{number}: invalid YOLO range")
    if require_inside and (x - width / 2 < 0 or x + width / 2 > 1 or y - height / 2 < 0 or y + height / 2 > 1):
        raise ValueError(f"{source}:{number}: bbox outside image")
    return class_id, x, y, width, height


def label_path(root: Path, split: str, image: str) -> Path:
    if split not in SPLITS or Path(image).name != image:
        raise ValueError("unsafe split or image")
    return root / "labels" / split / f"{Path(image).stem}.txt"


def verify_review(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    if len(rows) != 1170:
        raise SystemExit(f"Expected 1170 review decisions, found {len(rows)}")
    decisions: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        if not row["human_decision"].strip():
            raise SystemExit("review.csv has an empty human_decision")
        if row["human_decision"] not in {*NAMES, "delete"}:
            raise SystemExit(f"Unsupported human_decision: {row['human_decision']}")
        try:
            number = int(row["object_number"])
        except ValueError as error:
            raise SystemExit(f"Bad object_number: {row['object_number']}") from error
        key = (row["split"], row["image"], number)
        if key in decisions:
            raise SystemExit(f"Duplicate review key: {key}")
        label_path(SOURCE, key[0], key[1])
        decisions[key] = row
    return decisions


def class_statistics(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        class_counts: Counter[int] = Counter()
        image_count = empty_count = objects = 0
        for label in sorted((root / "labels" / split).glob("*.txt")):
            image_count += 1
            lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_count += 1
            for number, line in enumerate(lines, 1):
                class_id, *_ = parse_line(line, label, number)
                class_counts[class_id] += 1
                objects += 1
        rows.append({"split": split, "images": image_count, "empty_labels": empty_count, "objects": objects, **{name: class_counts[index] for index, name in enumerate(NAMES)}})
    total = {"split": "total", "images": sum(int(row["images"]) for row in rows), "empty_labels": sum(int(row["empty_labels"]) for row in rows), "objects": sum(int(row["objects"]) for row in rows)}
    total.update({name: sum(int(row[name]) for row in rows) for name in NAMES})
    return [*rows, total]


def box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2, bx2, by2 = ax1 + aw / 2, ay1 + ah / 2, bx1 + bw / 2, by1 + bh / 2
    ax1, ay1 = ax1 - aw / 2, ay1 - ah / 2
    bx1, by1, bx2, by2 = bx1 - bw / 2, by1 - bh / 2, bx2 - bw / 2, by2 - bh / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def automatic_geometry_candidates(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        for label in sorted((root / "labels" / split).glob("*.txt")):
            boxes = []
            for number, line in enumerate([item.strip() for item in label.read_text(encoding="utf-8").splitlines() if item.strip()], 1):
                class_id, x, y, width, height = parse_line(line, label, number)
                flags: list[str] = []
                if width * height < 0.0075:
                    flags.append("small_area")
                if height < 0.12 and width / height > 1.7:
                    flags.append("short_wide_top_fragment")
                if width / height > 3.5 or height / width > 4.0:
                    flags.append("unusual_aspect_ratio")
                if x - width / 2 <= 0.001 or x + width / 2 >= 0.999 or y - height / 2 <= 0.001 or y + height / 2 >= 0.999:
                    flags.append("near_image_border")
                if x - width / 2 < 0 or x + width / 2 > 1 or y - height / 2 < 0 or y + height / 2 > 1:
                    flags.append("outside_image_bounds")
                boxes.append((number, class_id, x, y, width, height, flags))
            for left in range(len(boxes)):
                for right in range(left + 1, len(boxes)):
                    overlap = box_iou(boxes[left][2:6], boxes[right][2:6])
                    if overlap >= 0.92:
                        boxes[left][6].append("near_duplicate_bbox")
                        boxes[right][6].append("near_duplicate_bbox")
                    elif overlap >= 0.75:
                        boxes[left][6].append("strong_overlap")
                        boxes[right][6].append("strong_overlap")
            image = next((path.name for path in (root / "images" / split).glob(f"{label.stem}.*")), None)
            if image is None:
                raise SystemExit(f"Image missing for {label}")
            for number, class_id, x, y, width, height, flags in boxes:
                if flags:
                    rows.append({"split": split, "image": image, "object_index": number, "source": "automatic_geometry", "reason": ";".join(sorted(set(flags))), "class": NAMES[class_id], "x": round(x, 6), "y": round(y, 6), "width": round(width, 6), "height": round(height, 6)})
    return rows


def main() -> None:
    if TARGET.exists() or REPORTS.exists():
        raise SystemExit("Refusing to overwrite an existing v3 worktree or report directory")
    review_rows = read_csv(REVIEW, REVIEW_FIELDS)
    decisions = verify_review(review_rows)
    missing_rows = read_csv(MISSING, ["split", "image", "comment", "resolved"])
    suspicious_rows = read_csv(SUSPICIOUS)
    shutil.copytree(SOURCE, TARGET, copy_function=shutil.copy2)
    REPORTS.mkdir(parents=True)
    (REPORTS / "contact_sheets").mkdir()
    applied: list[dict[str, object]] = []
    seen_decisions: set[tuple[str, str, int]] = set()
    for split in SPLITS:
        for label in sorted((TARGET / "labels" / split).glob("*.txt")):
            image = next((path.name for path in (TARGET / "images" / split).glob(f"{label.stem}.*")), None)
            if image is None:
                raise SystemExit(f"Missing image for {label}")
            original = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            revised: list[str] = []
            for number, line in enumerate(original, 1):
                old_id, x, y, width, height = parse_line(line, label, number)
                key = (split, image, number)
                row = decisions.get(key)
                if row is None:
                    revised.append(line)
                    continue
                seen_decisions.add(key)
                human = row["human_decision"]
                source = "human_decision"
                if key == TARGET_OVERRIDE:
                    human = "overflow"
                    source = "explicit_user_override"
                if human == "delete":
                    applied.append({"split": split, "image": image, "object_index": number, "old_label": NAMES[old_id], "new_label": "", "action": "delete_box", "source": source, "review_human_decision": row["human_decision"], "reason": row["reason"]})
                    continue
                new_id = NAME_TO_ID[human]
                # Preserve coordinates exactly. A class decision must never
                # round, shift, or otherwise alter existing geometry.
                fields = line.split()
                fields[0] = str(new_id)
                revised.append(" ".join(fields))
                applied.append({"split": split, "image": image, "object_index": number, "old_label": NAMES[old_id], "new_label": human, "action": "class_changed" if new_id != old_id else "class_confirmed", "source": source, "review_human_decision": row["human_decision"], "reason": row["reason"]})
            atomic_write(label, ("\n".join(revised) + "\n") if revised else "")
    if set(decisions) != seen_decisions:
        unresolved = sorted(set(decisions) - seen_decisions)
        raise SystemExit(f"Review rows not found in v3 labels: {unresolved[:5]}")
    write_csv(REPORTS / "applied_decisions.csv", ["split", "image", "object_index", "old_label", "new_label", "action", "source", "review_human_decision", "reason"], applied)
    write_csv(REPORTS / "class_statistics.csv", ["split", "images", "empty_labels", "objects", *NAMES], class_statistics(TARGET))
    auto_candidates = automatic_geometry_candidates(TARGET)
    write_csv(REPORTS / "automatic_geometry_candidates.csv", ["split", "image", "object_index", "source", "reason", "class", "x", "y", "width", "height"], auto_candidates)
    queue: list[dict[str, object]] = []
    for index, row in enumerate(suspicious_rows, 1):
        image_path = Path(row["image"])
        split = image_path.parent.name
        queue.append({"queue_id": f"diagnostic-{index:03d}", "split": split, "image": image_path.name, "object_index": row["line"], "source": "best1_geometry_diagnostic", "reason": row["flags"], "status": "pending"})
    for index, row in enumerate(auto_candidates, 1):
        queue.append({"queue_id": f"auto-{index:04d}", "split": row["split"], "image": row["image"], "object_index": row["object_index"], "source": row["source"], "reason": row["reason"], "status": "pending"})
    for index, row in enumerate(missing_rows, 1):
        queue.append({"queue_id": f"missing-{index:03d}", "split": row["split"], "image": row["image"], "object_index": "", "source": "missing_objects", "reason": row["comment"], "status": "pending"})
    # A single existing label is an auditable low-count candidate, not proof of a missing container.
    low_count = []
    for split in SPLITS:
        for label in sorted((TARGET / "labels" / split).glob("*.txt")):
            count = sum(bool(line.strip()) for line in label.read_text(encoding="utf-8").splitlines())
            if count <= 1:
                image = next(path.name for path in (TARGET / "images" / split).glob(f"{label.stem}.*"))
                low_count.append({"queue_id": f"lowcount-{split}-{label.stem}", "split": split, "image": image, "object_index": "", "source": "low_annotation_count", "reason": f"only_{count}_existing_boxes; human must decide whether more containers are visible", "status": "pending"})
    existing_images = {(row["split"], row["image"]) for row in queue}
    queue.extend(row for row in low_count if (row["split"], row["image"]) not in existing_images)
    write_csv(REPORTS / "manual_review_queue.csv", ["queue_id", "split", "image", "object_index", "source", "reason", "status"], queue)
    write_csv(REPORTS / "geometry_changes.csv", ["timestamp", "split", "image", "action", "before", "after", "note"], [])
    write_csv(REPORTS / "missing_boxes_added.csv", ["timestamp", "split", "image", "class", "x", "y", "width", "height", "note"], [])
    checks = {"created_at": datetime.now(timezone.utc).isoformat(), "review_rows": len(review_rows), "review_empty_human_decision": 0, "review_duplicate_keys": 0, "reviewed_objects_applied": len(applied), "review_deletes": sum(row["action"] == "delete_box" for row in applied), "unreviewed_objects_preserved": 1588 - len(applied), "explicit_override": {"split": TARGET_OVERRIDE[0], "image": TARGET_OVERRIDE[1], "object_index": TARGET_OVERRIDE[2], "class": "overflow"}, "manual_queue_items": len(queue), "automatic_geometry_candidates": len(auto_candidates), "diagnostic_geometry_items": len(suspicious_rows), "missing_object_notes": len(missing_rows), "low_annotation_count_candidates": len(low_count)}
    atomic_write(REPORTS / "preparation_integrity.json", json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    atomic_write(REPORTS / "final_report.md", "# V3 final report — DRAFT\n\nРучная проверка геометрии и пропущенных контейнеров ещё не завершена. Этот файл нельзя считать финальным отчётом или разрешением на обучение/архивацию.\n\nСначала завершите очередь через локальный интерфейс, затем выполните финальную валидацию.\n")
    atomic_write(REPORTS / "contact_sheets" / "README.md", "Контактные листы будут сформированы только после завершения ручной проверки.\n")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
