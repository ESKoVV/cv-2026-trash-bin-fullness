#!/usr/bin/env python3
"""Validate Roboflow YOLO downloads, normalize classes, and build one YOLO dataset.

Images with an ambiguous, unknown, missing, empty, or malformed annotation are
excluded as a whole.  This avoids silently teaching the detector that a visible
container is background.  The exclusion log is written to reports/.
"""
from __future__ import annotations

import csv
import hashlib
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MERGED = ROOT / "data" / "merged"
REPORTS = ROOT / "reports"
SAMPLES = REPORTS / "samples"
TARGET_NAMES = ["empty", "partial", "full", "overflow"]
TARGET_ID = {name: index for index, name in enumerate(TARGET_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_MAP = {
    "empty": "empty",
    "open_empty": "empty",
    "close_empty": "empty",
    "half_full": "partial",
    "partial": "partial",
    "partially_full": "partial",
    "medium": "partial",
    "full": "full",
    "open_full": "full",
    "close_full": "full",
    "closedfull": "full",
    "closed_full": "full",
    "overflow": "overflow",
    "overflowing": "overflow",
    "garbage_overflow": "overflow",
}
# ``Trash flow`` in the first source denotes a pile/bags of waste rather than
# a container.  It must not teach a bin detector that a pile alone is a bin.
AMBIGUOUS = {"healthy_trash_can", "broken_trash_can", "closed", "emptyfull", "trash_flow"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
# Images rejected during manual visual QA.  Keys are prefixes of SHA-256 image
# digests, so the decision remains stable even if a source filename changes.
VISUAL_REVIEW_EXCLUSIONS = {
    "7d0014bfee": "target not reliably identifiable",
    "4f6037c79b": "fill-level label contradicts visible empty container",
    "0539e1414b": "target not reliably identifiable",
    "86837f59b1": "bbox covers only a waste pile, not a container",
    "6479a65132": "target not reliably identifiable",
    "208c3ad39a": "target not reliably identifiable",
    "588e3e2be7": "target not reliably identifiable",
    "696c5140e1": "target not reliably identifiable",
    "5fe574571f": "bbox covers waste bags rather than a container",
    "cbe7ef610a": "bbox covers only a waste pile, not a container",
    "d0f89c835d": "target not reliably identifiable",
    "ee65382a1c": "bbox covers waste bags rather than a container",
    "fa0cae910a": "bbox covers only a waste pile, not a container",
}


@dataclass
class Candidate:
    image: Path
    label: Path
    split: str
    source: str
    labels: list[tuple[int, float, float, float, float]]
    digest: str


def normalized_name(value: str) -> str:
    return re.sub(r"[\s_-]+", "_", value.strip().casefold()).strip("_")


def source_names(data_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise ValueError("data.yaml has no usable 'names' field")


def image_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def parse_labels(label_path: Path, names: dict[int, str]) -> tuple[list[tuple[int, float, float, float, float]], str | None]:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return [], "unreadable_label"
    if not lines:
        return [], "empty_label"

    converted: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            return [], f"malformed_label_line:{line_number}"
        try:
            source_id = int(fields[0])
            x_center, y_center, width, height = (float(item) for item in fields[1:])
        except ValueError:
            return [], f"non_numeric_label_line:{line_number}"
        if source_id not in names:
            return [], f"unknown_class_id:{source_id}"
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            return [], f"invalid_bbox_range:{line_number}"
        if x_center - width / 2 < 0 or x_center + width / 2 > 1 or y_center - height / 2 < 0 or y_center + height / 2 > 1:
            return [], f"bbox_outside_image:{line_number}"
        original = normalized_name(names[source_id])
        if original in AMBIGUOUS:
            return [], f"ambiguous_class:{names[source_id]}"
        target = CLASS_MAP.get(original)
        if target is None:
            return [], f"unmapped_class:{names[source_id]}"
        converted.append((TARGET_ID[target], x_center, y_center, width, height))
    return converted, None


def ensure_output_structure() -> None:
    for split in ("train", "val", "test"):
        (MERGED / "images" / split).mkdir(parents=True, exist_ok=True)
        (MERGED / "labels" / split).mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    (MERGED / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 4\nnames:\n  0: empty\n  1: partial\n  2: full\n  3: overflow\n",
        encoding="utf-8",
    )


def clean_generated_output() -> None:
    for directory in (MERGED / "images", MERGED / "labels", SAMPLES):
        if directory.exists():
            for item in directory.iterdir():
                if item.name == ".gitkeep":
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()


def find_sources() -> list[tuple[Path, dict[int, str]]]:
    sources: list[tuple[Path, dict[int, str]]] = []
    for data_yaml in sorted(RAW.rglob("data.yaml")):
        try:
            sources.append((data_yaml.parent, source_names(data_yaml)))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            print(f"Skipping bad source YAML {data_yaml}: {exc}")
    return sources


def write_exclusions(rows: list[dict[str, str]]) -> None:
    path = REPORTS / "excluded_files.csv"
    columns = ["source", "split", "image", "reason"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_suspicious_images(rows: list[dict[str, str]]) -> None:
    path = REPORTS / "suspicious_images.csv"
    columns = ["source", "split", "image", "reason", "decision"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            if row["reason"].startswith("manual_visual_review:"):
                writer.writerow({**row, "decision": "excluded_from_training"})


def render_samples(candidates: list[Candidate]) -> int:
    for file in SAMPLES.glob("*.*"):
        if file.name != ".gitkeep":
            file.unlink()
    selected = candidates[:12]
    for index, candidate in enumerate(selected, start=1):
        with Image.open(candidate.image) as image:
            rendered = image.convert("RGB")
        drawer = ImageDraw.Draw(rendered)
        width, height = rendered.size
        for class_id, x_center, y_center, box_width, box_height in candidate.labels:
            left = int((x_center - box_width / 2) * width)
            top = int((y_center - box_height / 2) * height)
            right = int((x_center + box_width / 2) * width)
            bottom = int((y_center + box_height / 2) * height)
            drawer.rectangle((left, top, right, bottom), outline="lime", width=max(2, width // 300))
            drawer.text((left + 3, max(0, top - 16)), TARGET_NAMES[class_id], fill="lime", stroke_width=1, stroke_fill="black")
        rendered.save(SAMPLES / f"{index:02d}_{candidate.split}_{candidate.digest[:10]}.jpg", quality=92)
    return len(selected)


def write_report(
    sources: list[tuple[Path, dict[int, str]]],
    candidates: list[Candidate],
    exclusions: list[dict[str, str]],
    samples: int,
) -> None:
    image_counts = Counter(candidate.split for candidate in candidates)
    object_counts = Counter()
    for candidate in candidates:
        object_counts.update(TARGET_NAMES[label[0]] for label in candidate.labels)
    excluded_by_reason = Counter(row["reason"].split(":", 1)[0] for row in exclusions)
    status = "Готов к обучению" if candidates else "Ожидает скачивания исходных данных"
    all_classes_present = all(object_counts[name] > 0 for name in TARGET_NAMES)
    all_splits_present = all(image_counts[split] > 0 for split in ("train", "val", "test"))
    lines = [
        "# Отчёт о подготовке датасета",
        "",
        f"## Статус\n\n**{status}.**",
        "",
        "## Источники",
        "",
    ]
    if sources:
        lines.extend(f"- `{path}` — классы: {', '.join(names.values())}" for path, names in sources)
    else:
        lines.append("Исходные YOLO-наборы ещё не находятся в `data/raw`; скачивание заблокировано отсутствием `ROBOFLOW_API_KEY`.")
    lines.extend([
        "",
        "## Итоговые данные",
        "",
        "| Split | Изображений |",
        "|---|---:|",
        *(f"| {split} | {image_counts[split]} |" for split in ("train", "val", "test")),
        "",
        "| Класс | Рамок |",
        "|---|---:|",
        *(f"| {name} | {object_counts[name]} |" for name in TARGET_NAMES),
        "",
        f"Изображений-примеров с рамками: **{samples}**.",
        "",
        "## Исключения и проверки",
        "",
        f"Всего исключено файлов: **{len(exclusions)}**.",
    ])
    if excluded_by_reason:
        lines.extend(["", "| Причина | Количество |", "|---|---:|"])
        lines.extend(f"| {reason} | {count} |" for reason, count in sorted(excluded_by_reason.items()))
    else:
        lines.append("Проблемы не найдены либо проверка ещё не получила исходные файлы.")
    lines.extend([
        "",
        "Проверяются: отсутствие image/label, повреждённые изображения, пустые и некорректные labels, class id, диапазон и границы YOLO bbox, неоднозначные классы, SHA-256-дубликаты и пересечение split. Дубликат сохраняется только один раз, поэтому один и тот же байтовый файл не может попасть в разные split.",
        "",
        "Неоднозначные классы (`healthy trash can`, `broken trash can`, `closed`, `emptyfull`) и любые неизвестные классы исключаются целиком вместе с изображением; подробности находятся в `excluded_files.csv`.",
        "",
        "## Оценка пригодности",
        "",
    ])
    if not candidates:
        lines.append("До успешного скачивания статистическая оценка невозможна.")
    elif all_classes_present and all_splits_present:
        lines.append(
            "Во всех трёх split и всех четырёх целевых классах есть данные. "
            "Набор пригоден для первого обучения после успешного запуска "
            "`python scripts/validate_merged.py`; класс `partial` заметно меньше остальных и требует контроля метрик."
        )
    else:
        lines.append("Набор непригоден к обучению: отсутствует хотя бы один split или целевой класс.")
    (REPORTS / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_structure()
    clean_generated_output()
    ensure_output_structure()
    sources = find_sources()
    exclusions: list[dict[str, str]] = []
    candidates: list[Candidate] = []
    seen_hashes: set[str] = set()

    for source_root, names in sources:
        for source_split, target_split in SPLIT_ALIASES.items():
            images_dir = source_root / source_split / "images"
            labels_dir = source_root / source_split / "labels"
            if not images_dir.is_dir():
                continue
            for image_path in sorted(path for path in images_dir.rglob("*") if path.suffix.casefold() in IMAGE_EXTENSIONS):
                label_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")
                common = {"source": source_root.name, "split": target_split, "image": str(image_path.relative_to(source_root))}
                if not label_path.is_file():
                    exclusions.append({**common, "reason": "missing_label"})
                    continue
                if not valid_image(image_path):
                    exclusions.append({**common, "reason": "corrupt_image"})
                    continue
                labels, reason = parse_labels(label_path, names)
                if reason:
                    exclusions.append({**common, "reason": reason})
                    continue
                digest = image_digest(image_path)
                if digest in seen_hashes:
                    exclusions.append({**common, "reason": "duplicate_image"})
                    continue
                review_reason = VISUAL_REVIEW_EXCLUSIONS.get(digest[:10])
                if review_reason:
                    exclusions.append({**common, "reason": f"manual_visual_review:{review_reason}"})
                    continue
                seen_hashes.add(digest)
                candidates.append(Candidate(image_path, label_path, target_split, source_root.name, labels, digest))

    for index, candidate in enumerate(candidates, start=1):
        suffix = candidate.image.suffix.casefold()
        stem = f"{candidate.source}_{candidate.split}_{index:06d}_{candidate.digest[:10]}"
        output_image = MERGED / "images" / candidate.split / f"{stem}{suffix}"
        output_label = MERGED / "labels" / candidate.split / f"{stem}.txt"
        shutil.copy2(candidate.image, output_image)
        output_label.write_text(
            # ``repr`` is Python's shortest lossless float serialization.  Fixed
            # decimal rounding can move a bbox which originally touched an edge
            # a fraction outside [0, 1] when parsed again by YOLO tooling.
            "".join(f"{class_id} {x!r} {y!r} {width!r} {height!r}\n" for class_id, x, y, width, height in candidate.labels),
            encoding="utf-8",
        )

    write_exclusions(exclusions)
    write_suspicious_images(exclusions)
    sample_count = render_samples(candidates)
    write_report(sources, candidates, exclusions, sample_count)
    print(f"Prepared {len(candidates)} images. Report: {REPORTS / 'dataset_report.md'}")


if __name__ == "__main__":
    main()
