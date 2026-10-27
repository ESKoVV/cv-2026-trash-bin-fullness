#!/usr/bin/env python3
"""Create a non-destructive v2 YOLO dataset and a manual visual-review queue."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageStat
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "merged"
TARGET = ROOT / "data" / "merged_v2"
REPORTS = ROOT / "reports"
REVIEW_DIR = REPORTS / "v2_review"
SHEETS_DIR = REVIEW_DIR / "contact_sheets"
SPLITS = ("train", "val", "test")
OLD_NAMES = ("empty", "partial", "full", "overflow")
NEW_NAMES = ("not_full", "full", "overflow", "unknown")
CLASS_MAP = {0: 0, 1: 0, 2: 1, 3: 2}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    old_class: int
    new_class: int
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class ReviewItem:
    split: str
    image: Path
    object_number: int
    box: Box
    suggested_class: str
    reasons: tuple[str, ...]


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_label(path: Path) -> list[Box]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty label: {path}")
    boxes: list[Box] = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"malformed label {path}:{line_number}")
        try:
            old_class = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"non-numeric label {path}:{line_number}") from exc
        if old_class not in CLASS_MAP:
            raise ValueError(f"unknown class {path}:{line_number}")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"bbox range error {path}:{line_number}")
        if x - width / 2 < 0 or x + width / 2 > 1 or y - height / 2 < 0 or y + height / 2 > 1:
            raise ValueError(f"bbox outside image {path}:{line_number}")
        boxes.append(Box(old_class, CLASS_MAP[old_class], x, y, width, height))
    return boxes


def bbox_pixels(box: Box, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    left = max(0, round((box.x - box.width / 2) * width))
    top = max(0, round((box.y - box.height / 2) * height))
    right = min(width, round((box.x + box.width / 2) * width))
    bottom = min(height, round((box.y + box.height / 2) * height))
    return left, top, right, bottom


def iou(first: Box, second: Box) -> float:
    first_left, first_top = first.x - first.width / 2, first.y - first.height / 2
    first_right, first_bottom = first.x + first.width / 2, first.y + first.height / 2
    second_left, second_top = second.x - second.width / 2, second.y - second.height / 2
    second_right, second_bottom = second.x + second.width / 2, second.y + second.height / 2
    overlap = max(0.0, min(first_right, second_right) - max(first_left, second_left)) * max(0.0, min(first_bottom, second_bottom) - max(first_top, second_top))
    union = first.width * first.height + second.width * second.height - overlap
    return overlap / union if union else 0.0


def crop_metrics(image: Image.Image, box: Box) -> tuple[float, float, float]:
    crop = image.crop(bbox_pixels(box, image.size)).convert("L")
    if crop.width < 2 or crop.height < 2:
        return 0.0, 0.0, 0.0
    mean = ImageStat.Stat(crop).mean[0]
    contrast = ImageStat.Stat(crop).var[0]
    edges = ImageStat.Stat(crop.filter(ImageFilter.FIND_EDGES)).var[0]
    return mean, contrast, edges


def normalized_name(value: str) -> str:
    return value.strip().casefold().replace(" ", "_").replace("-", "_")


def source_class_lists() -> dict[str, list[str]]:
    """Map raw image hashes to original Roboflow class names in label order."""
    mapping: dict[str, list[str]] = {}
    for data_yaml in (ROOT / "data" / "raw").rglob("data.yaml"):
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        names = data.get("names", [])
        if isinstance(names, dict):
            names = {int(key): str(value) for key, value in names.items()}
        else:
            names = {index: str(value) for index, value in enumerate(names)}
        source_root = data_yaml.parent
        for raw_split in ("train", "valid", "test"):
            images = source_root / raw_split / "images"
            labels = source_root / raw_split / "labels"
            if not images.is_dir():
                continue
            for image in images.iterdir():
                if image.suffix.casefold() not in IMAGE_EXTENSIONS:
                    continue
                label = labels / f"{image.stem}.txt"
                if not label.is_file():
                    continue
                mapping[image_sha256(image)] = [names[int(line.split()[0])] for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    return mapping


def review_reasons(image: Image.Image, boxes: list[Box], index: int, source_class: str) -> tuple[str, ...]:
    box = boxes[index]
    reasons: list[str] = []
    left, top, right, bottom = bbox_pixels(box, image.size)
    area_fraction = (right - left) * (bottom - top) / (image.width * image.height)
    mean, contrast, edges = crop_metrics(image, box)
    if box.new_class in {1, 2}:
        reasons.append("full_or_overflow_requires_visual_fill_level_confirmation")
    if normalized_name(source_class) in {"close_empty", "close_full", "closed", "closedfull", "closed_full"}:
        reasons.append("closed_container_contents_not_visible")
    if area_fraction < 0.012 or min(right - left, bottom - top) < 28:
        reasons.append("container_too_distant_or_small")
    if mean < 42:
        reasons.append("container_too_dark")
    elif mean > 225:
        reasons.append("container_overexposed")
    if contrast < 80 or edges < 200:
        reasons.append("container_blurry_or_low_contrast")
    for other_index, other in enumerate(boxes):
        if index == other_index:
            continue
        overlap = iou(box, other)
        if overlap >= 0.72 and box.new_class != other.new_class:
            reasons.append("overlapping_boxes_with_conflicting_classes")
            break
        if overlap >= 0.92 and box.new_class == other.new_class:
            reasons.append("possible_duplicate_box_for_same_container")
            break
    return tuple(dict.fromkeys(reasons))


def suggested_class(box: Box, reasons: tuple[str, ...]) -> str:
    if any(reason in {"container_too_distant_or_small", "container_too_dark", "container_overexposed", "container_blurry_or_low_contrast"} for reason in reasons):
        return "unknown"
    return NEW_NAMES[box.new_class]


def render_sheet_item(item: ReviewItem, destination: Path) -> None:
    with Image.open(item.image) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = bbox_pixels(item.box, image.size)
    draw.rectangle((left, top, right, bottom), outline="lime", width=max(2, image.width // 280))
    label = f"#{item.object_number} {NEW_NAMES[item.box.new_class]} -> {item.suggested_class}"
    draw.text((left + 3, max(22, top - 18)), label, fill="lime", stroke_width=1, stroke_fill="black")
    title = f"{item.image.name} | object {item.object_number}"
    reason = "; ".join(item.reasons)
    canvas = Image.new("RGB", (900, 630), "#181818")
    fitted = ImageOps.contain(image, (880, 525))
    canvas.paste(fitted, ((900 - fitted.width) // 2, 58))
    caption = ImageDraw.Draw(canvas)
    caption.text((10, 8), title, fill="white")
    caption.text((10, 30), reason, fill="#ffd166")
    canvas.save(destination, quality=90)


def create_contact_sheets(items: list[ReviewItem]) -> int:
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    tiles: list[Path] = []
    tiles_dir = REVIEW_DIR / "items"
    tiles_dir.mkdir(exist_ok=True)
    for number, item in enumerate(items, start=1):
        tile = tiles_dir / f"{number:04d}_{item.split}_{item.image.stem}_object_{item.object_number}.jpg"
        render_sheet_item(item, tile)
        tiles.append(tile)
    tiles_per_sheet, columns = 4, 2
    for start in range(0, len(tiles), tiles_per_sheet):
        selected = tiles[start : start + tiles_per_sheet]
        sheet = Image.new("RGB", (1800, 1260), "#101010")
        for index, tile_path in enumerate(selected):
            with Image.open(tile_path) as opened:
                tile = opened.convert("RGB")
            sheet.paste(tile, ((index % columns) * 900, (index // columns) * 630))
        sheet.save(SHEETS_DIR / f"review_{start // tiles_per_sheet + 1:03d}.jpg", quality=88)
    return len(tiles)


def validate_target() -> tuple[Counter[str], Counter[str], Counter[str]]:
    image_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    hashes: dict[str, str] = {}
    for split in SPLITS:
        images = {path.stem: path for path in (TARGET / "images" / split).iterdir() if path.suffix.casefold() in IMAGE_EXTENSIONS}
        labels = {path.stem: path for path in (TARGET / "labels" / split).glob("*.txt")}
        if images.keys() != labels.keys():
            raise ValueError(f"unpaired images and labels in {split}")
        if not images:
            raise ValueError(f"empty split: {split}")
        image_counts[split] = len(images)
        label_counts[split] = len(labels)
        for stem, image_path in images.items():
            try:
                with Image.open(image_path) as opened:
                    opened.verify()
            except Exception as exc:
                raise ValueError(f"corrupt image: {image_path}") from exc
            digest = image_sha256(image_path)
            if digest in hashes and hashes[digest] != split:
                raise ValueError(f"cross-split duplicate: {image_path}")
            hashes[digest] = split
            boxes = parse_label(labels[stem])
            seen: set[tuple[int, float, float, float, float]] = set()
            for box in boxes:
                key = (box.new_class, round(box.x, 10), round(box.y, 10), round(box.width, 10), round(box.height, 10))
                if key in seen:
                    raise ValueError(f"duplicate bbox: {labels[stem]}")
                seen.add(key)
                object_counts[NEW_NAMES[box.new_class]] += 1
    return image_counts, label_counts, object_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace only previously generated v2 files.")
    args = parser.parse_args()
    if TARGET.exists():
        if not args.force:
            raise SystemExit(f"Refusing to overwrite existing target: {TARGET}")
        shutil.rmtree(TARGET)
    if REVIEW_DIR.exists():
        if not args.force:
            raise SystemExit(f"Refusing to overwrite existing review directory: {REVIEW_DIR}")
        shutil.rmtree(REVIEW_DIR)
    raw_classes = source_class_lists()
    old_counts: Counter[str] = Counter()
    converted_counts: Counter[str] = Counter()
    reviews: list[ReviewItem] = []
    for split in SPLITS:
        source_images = SOURCE / "images" / split
        source_labels = SOURCE / "labels" / split
        target_images = TARGET / "images" / split
        target_labels = TARGET / "labels" / split
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        images = {path.stem: path for path in source_images.iterdir() if path.suffix.casefold() in IMAGE_EXTENSIONS}
        labels = {path.stem: path for path in source_labels.glob("*.txt")}
        if images.keys() != labels.keys():
            raise ValueError(f"source image/label mismatch in {split}")
        for stem, image_path in sorted(images.items()):
            boxes = parse_label(labels[stem])
            source_classes = raw_classes.get(image_sha256(image_path))
            if source_classes is None or len(source_classes) != len(boxes):
                raise ValueError(f"cannot align raw source classes to {image_path}")
            shutil.copy2(image_path, target_images / image_path.name)
            (target_labels / f"{stem}.txt").write_text(
                "".join(f"{box.new_class} {box.x!r} {box.y!r} {box.width!r} {box.height!r}\n" for box in boxes),
                encoding="utf-8",
            )
            old_counts.update(OLD_NAMES[box.old_class] for box in boxes)
            converted_counts.update(NEW_NAMES[box.new_class] for box in boxes)
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            for index, box in enumerate(boxes, start=1):
                reasons = review_reasons(image, boxes, index - 1, source_classes[index - 1])
                if reasons:
                    reviews.append(ReviewItem(split, image_path, index, box, suggested_class(box, reasons), reasons))
    (TARGET / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 4\nnames:\n  0: not_full\n  1: full\n  2: overflow\n  3: unknown\n",
        encoding="utf-8",
    )
    REVIEW_DIR.mkdir(parents=True)
    with (REVIEW_DIR / "review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "image", "object_number", "old_class", "converted_class", "suggested_class", "reason", "human_decision"))
        writer.writeheader()
        for item in reviews:
            writer.writerow({
                "split": item.split,
                "image": item.image.name,
                "object_number": item.object_number,
                "old_class": OLD_NAMES[item.box.old_class],
                "converted_class": NEW_NAMES[item.box.new_class],
                "suggested_class": item.suggested_class,
                "reason": "; ".join(item.reasons),
                "human_decision": "",
            })
    rendered = create_contact_sheets(reviews)
    image_counts, label_counts, object_counts = validate_target()
    reason_counts = Counter(reason for item in reviews for reason in item.reasons)
    created = [TARGET / "data.yaml", REVIEW_DIR / "review.csv", SHEETS_DIR, REVIEW_DIR / "items"]
    report = [
        "# Конвертация датасета v2", "",
        "Исходные `data/merged` и все прежние материалы не изменялись.", "",
        "## Изображения и labels", "", "| Split | Изображений | Label-файлов |", "|---|---:|---:|",
        *(f"| {split} | {image_counts[split]} | {label_counts[split]} |" for split in SPLITS), "",
        "## Классы", "", "| Старый класс | Рамок | Новый класс | Рамок после автоматического преобразования |", "|---|---:|---|---:|",
        f"| empty | {old_counts['empty']} | not_full | {converted_counts['not_full']} |",
        f"| partial | {old_counts['partial']} | not_full | {converted_counts['not_full']} |",
        f"| full | {old_counts['full']} | full | {converted_counts['full']} |",
        f"| overflow | {old_counts['overflow']} | overflow | {converted_counts['overflow']} |",
        f"| — | — | unknown | {converted_counts['unknown']} (не назначался автоматически) |", "",
        "## Ручная проверка", "", f"Программно просмотрено изображений: **{sum(image_counts.values())}**; рамок: **{sum(object_counts.values())}**.",
        f"В ручную проверку отправлено объектов: **{len(reviews)}**; отрисовано фрагментов: **{rendered}**.", "",
        "| Причина | Объектов |", "|---|---:|", *(f"| {reason} | {count} |" for reason, count in sorted(reason_counts.items())), "",
        "`unknown` указан только как предложение в `review.csv`; labels класса 3 не создавались автоматически. Проверяющий должен вручную проверить весь кадр на пропущенные контейнеры и мусор рядом, который не должен менять класс контейнера.", "",
        "## Техническая проверка", "", "PASSED: image-label пары, отсутствие лишних/пустых labels, YOLO-формат, class id 0–3, bbox в [0,1], положительные размеры, SHA-256 дубликаты между split и точные повторяющиеся bbox.", "",
        "## Созданные файлы", "", *(f"- `{path.relative_to(ROOT)}`" for path in created),
    ]
    (REPORTS / "v2_conversion_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
    print(f"Manual review items: {len(reviews)}")
    print(f"Converted objects: {dict(converted_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
