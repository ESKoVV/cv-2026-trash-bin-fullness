#!/usr/bin/env python3
"""Read-only diagnostics for a YOLO detection model and its labels.

The script never writes into a dataset or changes model weights. All generated
files are written only under the explicitly supplied --output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IOU_MATCH = 0.5
DEFAULT_THRESHOLD = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Run a model on a prepared image/label directory")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--images", type=Path, required=True)
    infer.add_argument("--labels", type=Path)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.15, 0.10])
    infer.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU")
    infer.add_argument("--imgsz", type=int, default=640)
    infer.add_argument("--contact-threshold", type=float, default=DEFAULT_THRESHOLD)

    geometry = subparsers.add_parser("geometry", help="Inspect training-set YOLO box geometry")
    geometry.add_argument("--dataset", type=Path, required=True, help="YOLO dataset root containing images/train")
    geometry.add_argument("--output", type=Path, required=True)
    geometry.add_argument("--names", type=str, nargs="+", required=True)
    geometry.add_argument("--limit", type=int, default=40)
    return parser.parse_args()


def image_paths(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def label_path(labels_dir: Path | None, image_path: Path) -> Path | None:
    if labels_dir is None:
        return None
    path = labels_dir / f"{image_path.stem}.txt"
    return path if path.is_file() else None


def load_labels(path: Path | None, width: int, height: int, names: list[str]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    if path is None:
        return labels
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            class_id = int(fields[0])
            x_center, y_center, box_width, box_height = map(float, fields[1:])
        except ValueError:
            continue
        if not 0 <= class_id < len(names):
            continue
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
        labels.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "bbox": [x1, y1, x2, y2],
                "normalized": [x_center, y_center, box_width, box_height],
                "line": line_number,
            }
        )
    return labels


def iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def greedy_match(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]], class_aware: bool) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for prediction_index, prediction in sorted(enumerate(predictions), key=lambda item: item[1]["confidence"], reverse=True):
        best: tuple[int, float] | None = None
        for ground_truth_index, target in enumerate(ground_truth):
            if ground_truth_index in matched_gt:
                continue
            if class_aware and target["class_id"] != prediction["class_id"]:
                continue
            overlap = iou(prediction["bbox"], target["bbox"])
            if overlap >= IOU_MATCH and (best is None or overlap > best[1]):
                best = (ground_truth_index, overlap)
        if best is not None:
            ground_truth_index, overlap = best
            matched_gt.add(ground_truth_index)
            matched_predictions.add(prediction_index)
            matches.append((prediction_index, ground_truth_index, overlap))
    return matches, sorted(set(range(len(predictions))) - matched_predictions), sorted(set(range(len(ground_truth))) - matched_gt)


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def confidence_stats(values: list[float]) -> dict[str, Any]:
    bins = {"0.00-0.10": 0, "0.10-0.15": 0, "0.15-0.25": 0, "0.25-0.50": 0, "0.50-1.00": 0}
    for value in values:
        if value < 0.10:
            bins["0.00-0.10"] += 1
        elif value < 0.15:
            bins["0.10-0.15"] += 1
        elif value < 0.25:
            bins["0.15-0.25"] += 1
        elif value < 0.50:
            bins["0.25-0.50"] += 1
        else:
            bins["0.50-1.00"] += 1
    return {
        "count": len(values),
        "min": round(min(values), 4) if values else None,
        "max": round(max(values), 4) if values else None,
        "mean": round(statistics.mean(values), 4) if values else None,
        "median": round(statistics.median(values), 4) if values else None,
        "bins": bins,
    }


def font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def panel(image_path: Path, boxes: list[dict[str, Any]], color: str, title: str, width: int = 250, height: int = 190) -> Image.Image:
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        scale = min(width / source.width, (height - 28) / source.height)
        rendered = source.resize((max(1, int(source.width * scale)), max(1, int(source.height * scale))))
    canvas = Image.new("RGB", (width, height), "white")
    x_offset = (width - rendered.width) // 2
    y_offset = 24 + ((height - 24 - rendered.height) // 2)
    canvas.paste(rendered, (x_offset, y_offset))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 5), title, fill="black", font=font())
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        rectangle = (x_offset + x1 * scale, y_offset + y1 * scale, x_offset + x2 * scale, y_offset + y2 * scale)
        draw.rectangle(rectangle, outline=color, width=2)
        label = box["class_name"]
        if "confidence" in box:
            label += f" {box['confidence']:.2f}"
        draw.text((rectangle[0] + 2, max(25, rectangle[1] - 12)), label, fill=color, font=font())
    return canvas


def contact_sheet(records: list[dict[str, Any]], output: Path, threshold: float, include_ground_truth: bool) -> None:
    columns = 3 if include_ground_truth else 2
    tile_width, tile_height = 250, 190
    sample_width = columns * tile_width
    samples_per_row = 3
    rows = math.ceil(len(records) / samples_per_row)
    sheet = Image.new("RGB", (samples_per_row * sample_width, rows * (tile_height + 18)), "#e7e7e7")
    for index, record in enumerate(records):
        prediction = record["predictions"][str(threshold)]
        tiles = [panel(Path(record["image"]), [], "#555555", "Исходное")]
        if include_ground_truth:
            tiles.append(panel(Path(record["image"]), record["ground_truth"], "#14804a", "Разметка"))
        tiles.append(panel(Path(record["image"]), prediction, "#d14343", f"Предсказание ≥ {threshold:.2f}"))
        x = (index % samples_per_row) * sample_width
        y = (index // samples_per_row) * (tile_height + 18)
        for tile_index, tile in enumerate(tiles):
            sheet.paste(tile, (x + tile_index * tile_width, y))
        ImageDraw.Draw(sheet).text((x + 4, y + tile_height + 2), Path(record["image"]).name[:95], fill="black", font=font())
    sheet.save(output, quality=92)


def run_inference(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=False)
    images = image_paths(args.images)
    if not images:
        raise SystemExit("No images supplied")
    model = YOLO(str(args.model))
    names_dict = model.names
    names = [str(names_dict[index]) for index in sorted(names_dict)] if isinstance(names_dict, dict) else [str(value) for value in names_dict]
    thresholds = sorted(set(round(value, 2) for value in args.thresholds), reverse=True)
    records: list[dict[str, Any]] = []
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
        records.append({"image": str(image_path), "width": width, "height": height, "ground_truth": load_labels(label_path(args.labels, image_path), width, height, names), "predictions": {}})

    for threshold in thresholds:
        results = model([str(path) for path in images], imgsz=args.imgsz, conf=threshold, iou=args.iou, device="cpu", verbose=False)
        for record, result in zip(records, results, strict=True):
            prediction: list[dict[str, Any]] = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    prediction.append({"class_id": class_id, "class_name": str(result.names[class_id]), "confidence": round(float(box.conf.item()), 6), "bbox": [round(float(value), 2) for value in box.xyxy[0].tolist()]})
            record["predictions"][str(threshold)] = prediction

    metrics: dict[str, Any] = {"model": str(args.model), "imgsz": args.imgsz, "nms_iou": args.iou, "iou_for_evaluation": IOU_MATCH, "classes": names, "images": len(records), "ground_truth_total": sum(len(record["ground_truth"]) for record in records), "thresholds": {}}
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        key = str(threshold)
        totals: Counter[str] = Counter()
        pred_classes: Counter[str] = Counter()
        gt_classes: Counter[str] = Counter()
        confidence_values: list[float] = []
        confusion: Counter[str] = Counter()
        for record in records:
            predictions = record["predictions"][key]
            ground_truth = record["ground_truth"]
            matches, unmatched_predictions, unmatched_ground_truth = greedy_match(predictions, ground_truth, class_aware=True)
            loose_matches, _, _ = greedy_match(predictions, ground_truth, class_aware=False)
            totals.update(predictions=len(predictions), ground_truth=len(ground_truth), true_positive=len(matches), false_positive=len(unmatched_predictions), false_negative=len(unmatched_ground_truth), any_class_overlap=len(loose_matches))
            pred_classes.update(item["class_name"] for item in predictions)
            gt_classes.update(item["class_name"] for item in ground_truth)
            confidence_values.extend(item["confidence"] for item in predictions)
            if unmatched_ground_truth:
                totals["images_with_miss"] += 1
            for prediction_index, ground_truth_index, overlap in loose_matches:
                prediction = predictions[prediction_index]
                target = ground_truth[ground_truth_index]
                if prediction["class_id"] != target["class_id"]:
                    confusion[f"{target['class_name']}→{prediction['class_name']}"] += 1
            rows.append({"threshold": threshold, "image": Path(record["image"]).name, "ground_truth": len(ground_truth), "predictions": len(predictions), "true_positive": len(matches), "false_positive": len(unmatched_predictions), "missed_ground_truth": len(unmatched_ground_truth), "class_agnostic_matches": len(loose_matches), "confidences": ";".join(f"{item['confidence']:.4f}" for item in predictions)})
        precision = ratio(totals["true_positive"], totals["true_positive"] + totals["false_positive"])
        recall = ratio(totals["true_positive"], totals["ground_truth"])
        f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else 0.0
        metrics["thresholds"][key] = {**dict(totals), "precision": precision, "recall": recall, "f1": f1, "predicted_classes": dict(sorted(pred_classes.items())), "ground_truth_classes": dict(sorted(gt_classes.items())), "confidence": confidence_stats(confidence_values), "class_confusions_at_iou_0_5": dict(sorted(confusion.items()))}

    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "predictions.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheet(records, args.output / "contact_sheet.jpg", args.contact_threshold, args.labels is not None)


def run_geometry(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    images_dir = args.dataset / "images" / "train"
    labels_dir = args.dataset / "labels" / "train"
    entries: list[dict[str, Any]] = []
    by_class: Counter[str] = Counter()
    areas: list[float] = []
    for image_path in image_paths(images_dir):
        with Image.open(image_path) as image:
            width, height = image.size
        labels = load_labels(label_path(labels_dir, image_path), width, height, args.names)
        for label in labels:
            _, _, normalized_width, normalized_height = label["normalized"]
            area = normalized_width * normalized_height
            aspect = normalized_width / normalized_height if normalized_height else None
            flags = []
            if area < 0.0075:
                flags.append("very_small_area")
            if normalized_height < 0.12 and normalized_width > normalized_height * 1.7:
                flags.append("short_wide_box")
            if flags:
                entries.append({"image": str(image_path), "class_name": label["class_name"], "line": label["line"], "normalized_width": round(normalized_width, 5), "normalized_height": round(normalized_height, 5), "normalized_area": round(area, 6), "aspect_ratio": round(aspect, 3) if aspect else None, "flags": ";".join(flags), "bbox": label["bbox"]})
            by_class[label["class_name"]] += 1
            areas.append(area)
    entries.sort(key=lambda item: (item["normalized_area"], item["image"], item["line"]))
    with (args.output / "suspicious_boxes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "class_name", "line", "normalized_width", "normalized_height", "normalized_area", "aspect_ratio", "flags"])
        writer.writeheader()
        for entry in entries:
            writer.writerow({key: entry[key] for key in writer.fieldnames})
    summary = {"split": "train", "boxes": sum(by_class.values()), "class_distribution": dict(sorted(by_class.items())), "area_fraction": {"min": round(min(areas), 7) if areas else None, "median": round(statistics.median(areas), 7) if areas else None, "mean": round(statistics.mean(areas), 7) if areas else None}, "heuristic_candidates": len(entries), "heuristics": {"very_small_area": "normalized area < 0.0075", "short_wide_box": "height < 0.12 and width/height > 1.7"}, "note": "Candidates require human review; geometry alone cannot prove that a box is a lid or contents."}
    (args.output / "geometry_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    visual_records = []
    seen: set[str] = set()
    for entry in entries:
        if entry["image"] in seen:
            continue
        seen.add(entry["image"])
        visual_records.append({"image": entry["image"], "ground_truth": [{"class_name": entry["class_name"], "bbox": entry["bbox"]}], "predictions": {"0.25": [{"class_name": entry["class_name"], "bbox": entry["bbox"], "confidence": 1.0}]}})
        if len(visual_records) >= args.limit:
            break
    if visual_records:
        contact_sheet(visual_records, args.output / "geometry_candidates_contact_sheet.jpg", 0.25, include_ground_truth=False)


def main() -> None:
    args = parse_args()
    if args.command == "infer":
        run_inference(args)
    else:
        run_geometry(args)


if __name__ == "__main__":
    main()
