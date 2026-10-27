#!/usr/bin/env python3
"""Prepare a conservative, auditable review set for V3 container proposals.

This utility deliberately does *not* train a model and does not alter
``merged_v3_work``.  It only reads the two proposal sources, validates them,
and prepares an immutable snapshot's review queue.  A zero-shot detector was
calibrated against the manually corrected frames: its numerical confidence is
not sufficient evidence that a proposed rectangle covers a whole container.
Consequently no unreviewed automatic proposal is written to a YOLO label by
this script.  This is intentional: every such proposal remains visible in the
dedicated human-review UI.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "merged_v3_work"
TARGET = ROOT / "data" / "merged_v3_autolabeled"
PREP = ROOT / "reports" / "v3_preparation"
REPORT = ROOT / "reports" / "v3_autolabel"
WORLD = REPORT / "raw_world_proposals.csv"
HOMOGRAPHY = REPORT / "raw_homography_proposals.csv"
NAMES = ("not_full", "full", "overflow", "unknown")
SPLITS = ("train", "val", "test")
BOX_FIELDS = ("x", "y", "width", "height")
CHANGE_FIELDS = [
    "change_id", "status", "level", "split", "image", "change_type",
    "old_box", "new_box", "class_id", "class_name", "basis", "confidence",
    "method", "source", "review_reason",
]
QUEUE_FIELDS = [
    "candidate_id", "split", "image", "level", "status", "source",
    "confidence", "change_type", "old_box", "new_box", "class_id",
    "class_name", "basis", "method", "reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def box_from_mapping(row: dict[str, str]) -> dict[str, float]:
    return {key: round(float(row[key]), 6) for key in BOX_FIELDS}


def box_json(box: dict[str, float] | None) -> str:
    return json.dumps(box, ensure_ascii=False, sort_keys=True) if box else ""


def label_path(root: Path, split: str, image: str) -> Path:
    return root / "labels" / split / f"{Path(image).stem}.txt"


def image_path(root: Path, split: str, image: str) -> Path:
    return root / "images" / split / image


def read_labels(root: Path, split: str, image: str) -> list[dict[str, Any]]:
    path = label_path(root, split, image)
    boxes: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 YOLO values")
        class_id = int(values[0]); box = dict(zip(BOX_FIELDS, map(float, values[1:]), strict=True))
        # Existing V3 labels are evidence, including the few geometry defects
        # that were deliberately queued for human correction.  Do not reject or
        # rewrite them merely because this conservative proposal pass sees an
        # out-of-range edge; candidate rectangles themselves are validated.
        if class_id not in range(4):
            raise ValueError(f"{path}:{line_no}: invalid class id")
        boxes.append({"class_id": class_id, **box})
    return boxes


def valid_box(box: dict[str, float]) -> bool:
    x, y, width, height = (box[key] for key in BOX_FIELDS)
    # Six-decimal YOLO serialisation can yield a harmless floating point edge
    # such as 1.0000005.  It is still a frame-boundary rectangle, not an
    # out-of-frame annotation.
    epsilon = 1e-6
    return (0 < width <= 1 and 0 < height <= 1 and -epsilon <= x - width / 2 and
            x + width / 2 <= 1 + epsilon and -epsilon <= y - height / 2 and y + height / 2 <= 1 + epsilon)


def iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax1, ay1 = a["x"] - a["width"] / 2, a["y"] - a["height"] / 2
    ax2, ay2 = a["x"] + a["width"] / 2, a["y"] + a["height"] / 2
    bx1, by1 = b["x"] - b["width"] / 2, b["y"] - b["height"] / 2
    bx2, by2 = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a["width"] * a["height"] + b["width"] * b["height"] - overlap
    return overlap / union if union else 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manual_reference_frames() -> set[tuple[str, str]]:
    return {(row["split"], row["image"]) for row in read_csv(PREP / "geometry_changes.csv")}


def snapshot_if_missing() -> None:
    if TARGET.exists():
        return
    shutil.copytree(SOURCE, TARGET, copy_function=shutil.copy2)


def assert_manual_frames_unchanged(manual: set[tuple[str, str]]) -> None:
    changed = []
    for split, image in sorted(manual):
        source, target = label_path(SOURCE, split, image), label_path(TARGET, split, image)
        if not target.exists() or sha256(source) != sha256(target):
            changed.append(f"{split}/{image}")
    if changed:
        raise RuntimeError("Manual frames differ from v3_work: " + ", ".join(changed))


def proposal_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_csv(WORLD):
        box = box_from_mapping(raw)
        confidence = float(raw["confidence"])
        # The detector sees geometry only; class is safely unknown until a person
        # can inspect the lid/fill level.
        level = "medium" if confidence >= 0.30 else "low"
        if confidence >= 0.90:
            level = "medium"  # visual calibration disproved automatic-high status
        rows.append({"source": "zero_shot_yolo_world", "split": raw["split"], "image": raw["image"],
                     "box": box, "class_id": 3, "class_name": "unknown", "confidence": confidence,
                     "level": level, "method": raw["method"], "basis": "independent zero-shot container detector"})
    for raw in read_csv(HOMOGRAPHY):
        box = box_from_mapping(raw)
        raw_level = raw["level"]
        # Whole-frame visual audit found lids/fragments among high-score transfers.
        level = "low" if raw_level == "low" else "medium"
        rows.append({"source": "manual_reference_transfer", "split": raw["split"], "image": raw["image"],
                     "box": box, "class_id": int(raw["class_id"]), "class_name": NAMES[int(raw["class_id"])],
                     "confidence": float(raw["confidence"]), "level": level, "method": raw["method"],
                     "basis": f"transfer from manually corrected {raw['reference_split']}/{raw['reference_image']}"})
    return rows


def image_order() -> list[tuple[str, str]]:
    return [(split, path.name) for split in SPLITS for path in sorted((TARGET / "images" / split).iterdir())
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]


def render_sheet(path: Path, frames: list[tuple[str, str]], by_frame: dict[tuple[str, str], list[dict[str, Any]]]) -> None:
    tile_w, tile_h = 420, 330
    canvas = Image.new("RGB", (tile_w * 2, tile_h * max(1, math.ceil(len(frames) / 2))), "#10161e")
    for index, (split, image) in enumerate(frames):
        source = Image.open(image_path(TARGET, split, image)).convert("RGB")
        source.thumbnail((400, 270))
        draw = ImageDraw.Draw(source)
        sw, sh = source.size
        for existing in read_labels(TARGET, split, image):
            x1, y1 = (existing["x"] - existing["width"] / 2) * sw, (existing["y"] - existing["height"] / 2) * sh
            x2, y2 = (existing["x"] + existing["width"] / 2) * sw, (existing["y"] + existing["height"] / 2) * sh
            draw.rectangle((x1, y1, x2, y2), outline="#42d37b", width=2)
        for candidate in by_frame[(split, image)]:
            box = candidate["box"]; x1, y1 = (box["x"] - box["width"] / 2) * sw, (box["y"] - box["height"] / 2) * sh
            x2, y2 = (box["x"] + box["width"] / 2) * sw, (box["y"] + box["height"] / 2) * sh
            color = "#ff9f1a" if candidate["level"] == "medium" else "#bf7cff"
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        tile = Image.new("RGB", (tile_w, tile_h), "#182330"); tile.paste(source, (10, 10))
        candidates = by_frame[(split, image)]
        text = f"{split}/{image[:43]}\nзелёный: исходные; оранжевый: medium; фиолетовый: low\nкандидатов: {len(candidates)}"
        ImageDraw.Draw(tile).text((10, 285), text, fill="white")
        canvas.paste(tile, ((index % 2) * tile_w, (index // 2) * tile_h))
    canvas.save(path, quality=88)


def quality_summary(frames: list[tuple[str, str]]) -> dict[str, int]:
    """Audit the snapshot without normalising or silently repairing labels."""
    invalid = duplicate = source_mismatch = image_mismatch = 0
    for split, image in frames:
        target_boxes = read_labels(TARGET, split, image)
        serialised = []
        for item in target_boxes:
            box = {key: float(item[key]) for key in BOX_FIELDS}
            if item["class_id"] not in range(4) or not valid_box(box):
                invalid += 1
            serialised.append((item["class_id"], *(round(box[key], 6) for key in BOX_FIELDS)))
        duplicate += len(serialised) - len(set(serialised))
        if sha256(label_path(SOURCE, split, image)) != sha256(label_path(TARGET, split, image)):
            source_mismatch += 1
        source_image, target_image = image_path(SOURCE, split, image), image_path(TARGET, split, image)
        if source_image.stat().st_size != target_image.stat().st_size or sha256(source_image) != sha256(target_image):
            image_mismatch += 1
    return {"invalid_existing_labels": invalid, "exact_duplicate_boxes": duplicate,
            "label_mismatches_from_source": source_mismatch, "image_mismatches_from_source": image_mismatch}


def main() -> None:
    snapshot_if_missing()
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "contact_sheets").mkdir(exist_ok=True)
    manual = manual_reference_frames()
    assert_manual_frames_unchanged(manual)
    all_frames = image_order()
    labels = {(split, image): read_labels(TARGET, split, image) for split, image in all_frames}
    changes: list[dict[str, str]] = []
    queue: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    accepted_by_frame: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_by_frame: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    candidate_no = 0

    # Explicitly carry unreviewed items from the original queue into the new
    # reviewer, so the full V3 corpus is not treated as completed by a script.
    original_pending: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(PREP / "manual_review_queue.csv"):
        key = (row["split"], row["image"])
        if row["status"] != "reviewed" and key not in manual:
            original_pending.setdefault(key, row)
    for key, row in sorted(original_pending.items()):
        candidate_no += 1
        queue.append({"candidate_id": f"existing-{candidate_no:05d}", "split": key[0], "image": key[1], "level": "medium",
                      "status": "pending", "source": row["source"], "confidence": "", "change_type": "existing_label_review",
                      "old_box": "", "new_box": "", "class_id": "", "class_name": "", "basis": row["reason"],
                      "method": "existing_v3_manual_queue", "reason": "Исходная очередь v3: автоматическая обработка не считается подтверждением."})

    for proposal in sorted(proposal_rows(), key=lambda value: (-value["confidence"], value["source"], value["split"], value["image"])):
        key = (proposal["split"], proposal["image"]); box = proposal["box"]
        candidate_no += 1; change_id = f"auto-{candidate_no:05d}"
        status, review_reason = "queued", "Нужна ручная проверка полного видимого корпуса и класса заполненности."
        if key in manual:
            status, review_reason = "rejected", "Кадр вручную подтверждён пользователем и защищён от автоматических изменений."
        elif not valid_box(box) or box["width"] * box["height"] < 0.0005:
            status, review_reason = "rejected", "Геометрия слишком мала или недопустима для уверенного контейнера."
        elif any(iou(box, existing) >= 0.80 for existing in labels[key]):
            status, review_reason = "rejected", "Дубликат/почти дубликат существующей рамки (IoU ≥ 0.80)."
        elif any(iou(box, accepted["box"]) >= 0.80 for accepted in seen_by_frame[key]):
            status, review_reason = "rejected", "Дубликат другого автоматического кандидата (IoU ≥ 0.80)."
        else:
            seen_by_frame[key].append(proposal)
        change = {"change_id": change_id, "status": status, "level": proposal["level"], "split": key[0], "image": key[1],
                  "change_type": "add_box_candidate", "old_box": "", "new_box": box_json(box), "class_id": str(proposal["class_id"]),
                  "class_name": proposal["class_name"], "basis": proposal["basis"], "confidence": f"{proposal['confidence']:.6f}",
                  "method": proposal["method"], "source": proposal["source"], "review_reason": review_reason}
        changes.append(change)
        if status == "rejected":
            rejected.append(change)
        else:
            queue.append({"candidate_id": change_id, "split": key[0], "image": key[1], "level": proposal["level"], "status": "pending",
                          "source": proposal["source"], "confidence": f"{proposal['confidence']:.6f}", "change_type": "add_box_candidate",
                          "old_box": "", "new_box": box_json(box), "class_id": str(proposal["class_id"]), "class_name": proposal["class_name"],
                          "basis": proposal["basis"], "method": proposal["method"], "reason": review_reason})
            accepted_by_frame[key].append(proposal)

    write_csv(REPORT / "autolabel_changes.csv", CHANGE_FIELDS, changes)
    write_csv(REPORT / "added_boxes.csv", CHANGE_FIELDS, [])
    write_csv(REPORT / "modified_boxes.csv", CHANGE_FIELDS, [])
    write_csv(REPORT / "rejected_candidates.csv", CHANGE_FIELDS, rejected)
    write_csv(REPORT / "manual_check_queue.csv", QUEUE_FIELDS, queue)

    # Contact sheets cover every frame with a retained candidate; reviewing them
    # is still required before acceptance.  Split them into compact pages.
    frames = sorted(accepted_by_frame)
    for start in range(0, len(frames), 4):
        render_sheet(REPORT / "contact_sheets" / f"review_{start // 4 + 1:03d}.jpg", frames[start:start + 4], accepted_by_frame)
    dense = [key for key, proposals in accepted_by_frame.items() if len(proposals) > 5]
    if dense:
        render_sheet(REPORT / "contact_sheets" / "dense_candidates.jpg", sorted(dense), accepted_by_frame)

    assert_manual_frames_unchanged(manual)
    quality = quality_summary(all_frames)
    status_counts = Counter(row["status"] for row in changes)
    level_counts = Counter(row["level"] for row in changes if row["status"] == "queued")
    queue_frames = {(row["split"], row["image"]) for row in queue}
    report = f"""# V3 auto-label preparation report

## Scope and safety

- Processed images: **{len(all_frames)}** (train/val/test; no split was changed).
- Snapshot: `data/merged_v3_autolabeled`, copied from `data/merged_v3_work`.
- Protected manually confirmed frames: **{len(manual)}**; SHA-256 equality with the source was checked before and after this run.
- Applied automatic label changes: **0**. `added_boxes.csv` and `modified_boxes.csv` intentionally contain headers only.
- Image files differing from the source snapshot: **{quality['image_mismatches_from_source']}**.
- Label files differing from the source snapshot: **{quality['label_mismatches_from_source']}**.

## Candidate sources

- YOLO-World geometry candidates: {sum(1 for row in changes if row['source'] == 'zero_shot_yolo_world')}.
- Manual-reference homography candidates: {sum(1 for row in changes if row['source'] == 'manual_reference_transfer')}.
- Retained for review: {status_counts['queued']} ({level_counts['medium']} medium, {level_counts['low']} low).
- High-confidence automatic additions applied: **0** (the calibrated high-confidence tier is intentionally empty).
- Rejected as protected/duplicate/invalid: {status_counts['rejected']}.
- Original unreviewed V3 queue frames retained: {len(original_pending)}.
- Unique frames requiring a person: **{len(queue_frames)}**.

## Validation

- New automatic boxes: 0, therefore no newly introduced invalid coordinates or duplicate boxes.
- Existing labels outside strict YOLO geometry bounds: {quality['invalid_existing_labels']} (left untouched and retained for review).
- Exact duplicate existing boxes: {quality['exact_duplicate_boxes']} (left untouched; no automatic cleanup was authorised).

## Why no boxes were auto-applied

Visual calibration of the detector's numerically strongest proposals showed false positives and partial rectangles (lids, exposed contents, and overlaps with an existing partial box). The detector also cannot determine fill level. Applying those labels would violate the full-container and class rules. The dedicated reviewer displays proposed boxes separately; only a human acceptance can add one to this dataset snapshot.

## Files

- `autolabel_changes.csv` — every automatic candidate with method, confidence, box and disposition.
- `rejected_candidates.csv` — candidates excluded from review, with a reason.
- `manual_check_queue.csv` — original outstanding V3 items plus medium/low automatic candidates.
- `contact_sheets/` — all frames with retained automatic candidates (green = existing, orange = medium, purple = low).
"""
    (REPORT / "AUTOLABEL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"images_processed": len(all_frames), "manual_frames_protected": len(manual),
                      "auto_applied": 0, "queued": status_counts['queued'], "rejected": status_counts['rejected'],
                      "medium": level_counts['medium'], "low": level_counts['low'], "review_frames": len(queue_frames)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
