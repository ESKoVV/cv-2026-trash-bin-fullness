#!/usr/bin/env python3
"""Audit original-vs-final membership and optionally stratify the final splits."""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import prepare_dataset as prep


ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "data" / "merged"
REPORTS = ROOT / "reports"
CLASSES = tuple(prep.TARGET_NAMES)
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_EXTENSIONS = prep.IMAGE_EXTENSIONS
SEED = 20260912


@dataclass(frozen=True)
class OriginalRecord:
    digest: str
    name: str
    relative_path: str
    split: str
    source_classes: tuple[str, ...]


@dataclass(frozen=True)
class FinalRecord:
    image: Path
    label: Path
    boxes: Counter[int]


def original_candidates() -> dict[str, OriginalRecord]:
    """Recreate the 1244-image merged version before semantic visual filtering."""
    old_map, old_ambiguous = prep.CLASS_MAP, prep.AMBIGUOUS
    prep.CLASS_MAP = {**old_map, "trash_flow": "overflow"}
    prep.AMBIGUOUS = old_ambiguous - {"trash_flow"}
    try:
        records: dict[str, OriginalRecord] = {}
        for source_root, names in prep.find_sources():
            for source_split, target_split in prep.SPLIT_ALIASES.items():
                images_dir = source_root / source_split / "images"
                labels_dir = source_root / source_split / "labels"
                if not images_dir.is_dir():
                    continue
                for image in sorted(path for path in images_dir.rglob("*") if path.suffix.casefold() in IMAGE_EXTENSIONS):
                    label = labels_dir / image.relative_to(images_dir).with_suffix(".txt")
                    if not label.is_file() or not prep.valid_image(image):
                        continue
                    _, reason = prep.parse_labels(label, names)
                    if reason:
                        continue
                    digest = prep.image_digest(image)
                    if digest in records:
                        continue
                    ids = [int(line.split()[0]) for line in label.read_text(encoding="utf-8").splitlines()]
                    source_classes = tuple(sorted({names[index] for index in ids}, key=str.casefold))
                    records[digest] = OriginalRecord(
                        digest=digest,
                        name=image.name,
                        relative_path=str(image.relative_to(source_root)),
                        split=target_split,
                        source_classes=source_classes,
                    )
        return records
    finally:
        prep.CLASS_MAP, prep.AMBIGUOUS = old_map, old_ambiguous


def final_records() -> list[FinalRecord]:
    records: list[FinalRecord] = []
    for split in SPLITS:
        images = {path.stem: path for path in (MERGED / "images" / split).iterdir() if path.suffix.casefold() in IMAGE_EXTENSIONS}
        labels = {path.stem: path for path in (MERGED / "labels" / split).glob("*.txt")}
        if images.keys() != labels.keys():
            raise ValueError(f"unpaired images/labels in {split}")
        for stem, image in images.items():
            boxes: Counter[int] = Counter()
            for line in labels[stem].read_text(encoding="utf-8").splitlines():
                boxes[int(line.split()[0])] += 1
            records.append(FinalRecord(image=image, label=labels[stem], boxes=boxes))
    return records


def is_trash_flow(record: OriginalRecord) -> bool:
    return any(prep.normalized_name(name) == "trash_flow" for name in record.source_classes)


def write_diff(original: dict[str, OriginalRecord], final: list[FinalRecord]) -> None:
    final_digests = {prep.image_digest(record.image) for record in final}
    removed = [record for digest, record in original.items() if digest not in final_digests]
    if len(original) != 1244 or len(final_digests) != 896 or len(removed) != 348:
        raise ValueError(f"unexpected audit sizes: original={len(original)}, final={len(final_digests)}, removed={len(removed)}")
    path = REPORTS / "final_dataset_diff.csv"
    columns = ("filename", "original_split", "original_classes", "removal_reason", "removal_stage", "evidence")
    reasons: Counter[str] = Counter()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in sorted(removed, key=lambda item: (item.split, item.name.casefold())):
            if is_trash_flow(record):
                reason = "irrelevant_waste_pile"
                stage = "semantic_class_filter"
                evidence = "scripts/prepare_dataset.py: AMBIGUOUS includes trash_flow"
            elif record.digest[:10] in prep.VISUAL_REVIEW_EXCLUSIONS:
                reason = "manual_visual_review"
                stage = "visual_qa"
                evidence = "reports/suspicious_images.csv"
            else:
                raise ValueError(f"no recorded removal rule for {record.name} ({record.digest})")
            reasons[reason] += 1
            writer.writerow({
                "filename": record.name,
                "original_split": record.split,
                "original_classes": "; ".join(record.source_classes),
                "removal_reason": reason,
                "removal_stage": stage,
                "evidence": evidence,
            })
    (REPORTS / "final_dataset_diff_summary.md").write_text(
        "# Сверка исходного и финального датасетов\n\n"
        "| Причина | Удалено изображений |\n|---|---:|\n"
        f"| Нерелевантная куча мусора (`Trash flow`) | {reasons['irrelevant_waste_pile']} |\n"
        "| Неоднозначный класс между версиями | 0 |\n"
        "| Неправильная рамка между версиями | 0 |\n"
        f"| Ручная визуальная проверка | {reasons['manual_visual_review']} |\n"
        "| Прочее | 0 |\n"
        f"| **Всего** | **{len(removed)}** |\n\n"
        "`Trash flow` не является классом заполненности контейнера: в этом источнике он размечает отдельную кучу/пакеты мусора. "
        "Поэтому 335 таких изображений исключены правилом, а не случайно вместе с корректным `overflow`. "
        "Тринадцать дополнительных изображений исключены только после ручной визуальной проверки.\n",
        encoding="utf-8",
    )
    print(f"Diff: {path} ({len(removed)} removed; {dict(reasons)})")


def target_boxes(records: list[FinalRecord]) -> dict[str, Counter[int]]:
    totals = Counter()
    for record in records:
        totals.update(record.boxes)
    targets = {split: Counter() for split in SPLITS}
    for class_id, total in totals.items():
        raw = {split: total * SPLIT_RATIOS[split] for split in SPLITS}
        allocated = {split: int(raw[split]) for split in SPLITS}
        for split in sorted(SPLITS, key=lambda item: (raw[item] - allocated[item], -SPLITS.index(item)), reverse=True)[: total - sum(allocated.values())]:
            allocated[split] += 1
        for split in SPLITS:
            targets[split][class_id] = allocated[split]
    return targets


def assign_stratified(records: list[FinalRecord]) -> dict[str, list[FinalRecord]]:
    image_targets = {"train": 627, "val": 134, "test": 135}
    box_targets = target_boxes(records)
    current = {split: Counter() for split in SPLITS}
    assigned = {split: [] for split in SPLITS}
    rng = random.Random(SEED)
    shuffled = list(records)
    rng.shuffle(shuffled)
    total_boxes = Counter()
    for record in records:
        total_boxes.update(record.boxes)
    def priority(record: FinalRecord) -> tuple[float, int]:
        rarity = sum(record.boxes[class_id] / total_boxes[class_id] for class_id in record.boxes)
        return (rarity, sum(record.boxes.values()))
    for record in sorted(shuffled, key=priority, reverse=True):
        available = [split for split in SPLITS if len(assigned[split]) < image_targets[split]]
        if not available:
            raise ValueError("stratifier exhausted all split capacities")
        def score(split: str) -> tuple[float, float, int]:
            class_need = sum(
                record.boxes[class_id] * max(0, box_targets[split][class_id] - current[split][class_id]) / max(1, box_targets[split][class_id])
                for class_id in record.boxes
            )
            capacity_need = (image_targets[split] - len(assigned[split])) / image_targets[split]
            return (class_need, capacity_need, -SPLITS.index(split))
        chosen = max(available, key=score)
        assigned[chosen].append(record)
        current[chosen].update(record.boxes)
    return assigned


def stats(assigned: dict[str, list[FinalRecord]]) -> dict[str, Counter[int]]:
    return {split: sum((record.boxes for record in assigned[split]), Counter()) for split in SPLITS}


def apply_assignment(assigned: dict[str, list[FinalRecord]]) -> None:
    seen = {record.image.name for records in assigned.values() for record in records}
    if len(seen) != 896:
        raise ValueError("assignment has duplicate or missing images")
    with tempfile.TemporaryDirectory(prefix="trash-bin-stratify-") as temporary:
        staging = Path(temporary)
        for split in SPLITS:
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
            for record in assigned[split]:
                shutil.copy2(record.image, staging / "images" / split / record.image.name)
                shutil.copy2(record.label, staging / "labels" / split / record.label.name)
        for split in SPLITS:
            images = list((staging / "images" / split).iterdir())
            labels = list((staging / "labels" / split).iterdir())
            if {path.stem for path in images} != {path.stem for path in labels}:
                raise ValueError(f"staged split mismatch: {split}")
        shutil.rmtree(MERGED / "images")
        shutil.rmtree(MERGED / "labels")
        shutil.copytree(staging / "images", MERGED / "images")
        shutil.copytree(staging / "labels", MERGED / "labels")


def write_stratification_report(assigned: dict[str, list[FinalRecord]]) -> None:
    box_stats = stats(assigned)
    lines = ["# Стратифицированное распределение", "", f"Seed: `{SEED}`.", "", "| Split | Изображений | empty | partial | full | overflow |", "|---|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        lines.append(f"| {split} | {len(assigned[split])} | " + " | ".join(str(box_stats[split][index]) for index in range(4)) + " |")
    (REPORTS / "stratification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Replace split folders with a deterministic stratified assignment.")
    args = parser.parse_args()
    REPORTS.mkdir(exist_ok=True)
    original = original_candidates()
    records = final_records()
    write_diff(original, records)
    assigned = assign_stratified(records)
    write_stratification_report(assigned)
    for split in SPLITS:
        print(split, "images=", len(assigned[split]), "boxes=", {CLASSES[index]: stats(assigned)[split][index] for index in range(4)})
    if args.apply:
        apply_assignment(assigned)
        print("Applied stratified split assignment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
