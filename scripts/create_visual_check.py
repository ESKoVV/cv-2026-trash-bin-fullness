#!/usr/bin/env python3
"""Render a reproducible random visual QA sample without changing the dataset."""
from __future__ import annotations

import csv
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "merged"
OUTPUT = ROOT / "reports" / "visual_check"
CLASSES = ("empty", "partial", "full", "overflow")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED = 20260911
SAMPLE_SIZE = 50


@dataclass(frozen=True)
class Record:
    split: str
    image: Path
    label: Path
    boxes: tuple[tuple[int, float, float, float, float], ...]


def parse_label(path: Path) -> tuple[tuple[int, float, float, float, float], ...]:
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"malformed label: {path}:{line_number}")
        class_id = int(fields[0])
        x, y, width, height = (float(value) for value in fields[1:])
        if class_id not in range(len(CLASSES)):
            raise ValueError(f"unknown class: {path}:{line_number}")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"bbox range error: {path}:{line_number}")
        if x - width / 2 < 0 or x + width / 2 > 1 or y - height / 2 < 0 or y + height / 2 > 1:
            raise ValueError(f"bbox outside image: {path}:{line_number}")
        boxes.append((class_id, x, y, width, height))
    if not boxes:
        raise ValueError(f"empty label: {path}")
    return tuple(boxes)


def load_records() -> list[Record]:
    records: list[Record] = []
    for split in ("train", "val", "test"):
        images = {
            path.stem: path
            for path in (DATA / "images" / split).iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        }
        labels = {path.stem: path for path in (DATA / "labels" / split).glob("*.txt")}
        if images.keys() != labels.keys():
            raise ValueError(f"image/label pairs differ in {split}")
        records.extend(Record(split, image, labels[stem], parse_label(labels[stem])) for stem, image in sorted(images.items()))
    return records


def pick_records(records: list[Record]) -> list[Record]:
    if len(records) < SAMPLE_SIZE:
        raise ValueError(f"dataset has only {len(records)} images")
    rng = random.Random(RANDOM_SEED)
    selected: list[Record] = []
    selected_paths: set[Path] = set()
    for class_id in range(len(CLASSES)):
        candidates = [record for record in records if any(box[0] == class_id for box in record.boxes)]
        record = rng.choice(candidates)
        if record.image not in selected_paths:
            selected.append(record)
            selected_paths.add(record.image)
    remaining = [record for record in records if record.image not in selected_paths]
    rng.shuffle(remaining)
    selected.extend(remaining[: SAMPLE_SIZE - len(selected)])
    rng.shuffle(selected)
    return selected


def render_record(record: Record, destination: Path) -> None:
    with Image.open(record.image) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for class_id, x, y, box_width, box_height in record.boxes:
        left = round((x - box_width / 2) * width)
        top = round((y - box_height / 2) * height)
        right = round((x + box_width / 2) * width)
        bottom = round((y + box_height / 2) * height)
        draw.rectangle((left, top, right, bottom), outline="lime", width=max(2, width // 300))
        draw.text((left + 3, max(0, top - 18)), CLASSES[class_id], fill="lime", stroke_width=1, stroke_fill="black")
    image.save(destination, quality=94)


def render_contact_sheets(rendered: list[Path]) -> None:
    tiles_per_sheet, columns, tile_size = 10, 2, 480
    for page_start in range(0, len(rendered), tiles_per_sheet):
        page = rendered[page_start : page_start + tiles_per_sheet]
        rows = (len(page) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "#202020")
        for index, path in enumerate(page):
            with Image.open(path) as opened:
                tile = ImageOps.contain(opened.convert("RGB"), (tile_size, tile_size - 24))
            canvas = Image.new("RGB", (tile_size, tile_size), "#202020")
            canvas.paste(tile, ((tile_size - tile.width) // 2, 24 + (tile_size - 24 - tile.height) // 2))
            ImageDraw.Draw(canvas).text((8, 5), path.name.split("_", 1)[0], fill="white")
            sheet.paste(canvas, ((index % columns) * tile_size, (index // columns) * tile_size))
        sheet.save(OUTPUT / f"contact_sheet_{page_start // tiles_per_sheet + 1:02d}.jpg", quality=92)


def main() -> None:
    records = load_records()
    selected = pick_records(records)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    rendered: list[Path] = []
    class_counts = Counter()
    with (OUTPUT / "selection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("number", "split", "source_image", "rendered_image", "classes", "bbox_count"))
        for number, record in enumerate(selected, start=1):
            destination = OUTPUT / f"{number:02d}_{record.split}_{record.image.stem}.jpg"
            render_record(record, destination)
            rendered.append(destination)
            labels = sorted({CLASSES[box[0]] for box in record.boxes})
            class_counts.update(labels)
            writer.writerow((number, record.split, record.image.relative_to(ROOT), destination.name, ",".join(labels), len(record.boxes)))
    render_contact_sheets(rendered)
    with (OUTPUT / "suspicious_images.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(("number", "split", "source_image", "reason", "decision"))
    (OUTPUT / "visual_check_report.md").write_text(
        "# Визуальная проверка\n\n"
        f"- Выборка: **{len(selected)}** изображений; seed: `{RANDOM_SEED}`.\n"
        f"- Представлены классы: {', '.join(f'`{name}` ({class_counts[name]} изображений)' for name in CLASSES)}.\n"
        "- Программно проверены парность image/label, непустая разметка, id классов, диапазон и границы каждой bbox.\n"
        "- В `suspicious_images.csv` заносятся случаи, требующие ручной перепроверки; до решения их нельзя использовать при обучении.\n"
        "- Контактные листы содержат по 10 отрисованных изображений, отдельные файлы — исходный размер с bbox и подписью.\n",
        encoding="utf-8",
    )
    print(f"Rendered {len(rendered)} images to {OUTPUT}")


if __name__ == "__main__":
    main()
