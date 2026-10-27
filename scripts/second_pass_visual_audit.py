#!/usr/bin/env python3
"""Create contact sheets for a read-only second review of v2 decisions.

Each tile contains the complete source frame and an enlarged bbox crop.  This
script only reads ``data/merged_v2`` and ``review.csv``; it never writes labels
or alters manual decisions.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "reports" / "v2_review" / "review.csv"
IMAGES = ROOT / "data" / "merged_v2" / "images"
LABELS = ROOT / "data" / "merged_v2" / "labels"
OUT = ROOT / "reports" / "v2_review" / "second_pass_visuals"
TILE_W, TILE_H = 620, 390
COLUMNS = 3
ROWS = 4


def bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    label = LABELS / row["split"] / f"{Path(row['image']).stem}.txt"
    lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    values = lines[int(row["object_number"]) - 1].split()
    return tuple(float(value) for value in values[1:5])  # type: ignore[return-value]


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#10151c")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def tile(row: dict[str, str], number: int, font: ImageFont.ImageFont) -> Image.Image:
    source = Image.open(IMAGES / row["split"] / row["image"]).convert("RGB")
    cx, cy, bw, bh = bbox(row)
    x1, y1 = (cx - bw / 2) * source.width, (cy - bh / 2) * source.height
    x2, y2 = (cx + bw / 2) * source.width, (cy + bh / 2) * source.height
    full = fit(source, (292, 225))
    scale_x, scale_y = full.width / source.width, full.height / source.height
    full_draw = ImageDraw.Draw(full)
    full_draw.rectangle((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y), outline="#39e68a", width=4)
    pad = max(25, max(x2 - x1, y2 - y1) * 0.3)
    crop = source.crop((max(0, int(x1 - pad)), max(0, int(y1 - pad)), min(source.width, int(x2 + pad)), min(source.height, int(y2 + pad))))
    crop = fit(crop, (292, 225))
    result = Image.new("RGB", (TILE_W, TILE_H), "#1a222d")
    draw = ImageDraw.Draw(result)
    result.paste(full, (12, 22))
    result.paste(crop, (316, 22))
    draw.text((12, 253), f"#{number}  {row['split']}  obj {row['object_number']}  decision: {row['human_decision']}", font=font, fill="white")
    draw.text((12, 275), f"old→converted→suggested: {row['old_class']} → {row['converted_class']} → {row['suggested_class']}", font=font, fill="#c4d5e8")
    name = row["image"]
    draw.text((12, 298), name[:78], font=font, fill="#c4d5e8")
    reason = row["reason"].replace("_", " ")
    draw.text((12, 321), reason[:78], font=font, fill="#c4d5e8")
    draw.text((12, 350), "full frame (green bbox)                                      enlarged crop", font=font, fill="#8fa6be")
    return result


def main() -> None:
    with REVIEW.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["human_decision"]]
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing visual sheets: {OUT}")
    OUT.mkdir(parents=True)
    groups: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for number, row in enumerate(rows, 1):
        if "closed_container_contents_not_visible" in row["reason"] and row["human_decision"] != "unknown":
            group = "closed_non_unknown"
        elif row["human_decision"] == "delete":
            group = "delete"
        elif row["human_decision"] != row["suggested_class"]:
            group = "decision_differs_from_suggestion"
        else:
            group = "decision_matches_suggestion"
        groups[group].append((number, row))
    font = ImageFont.load_default()
    for group, items in groups.items():
        target = OUT / group
        target.mkdir()
        per_page = COLUMNS * ROWS
        for page_idx in range(math.ceil(len(items) / per_page)):
            page_items = items[page_idx * per_page:(page_idx + 1) * per_page]
            page = Image.new("RGB", (TILE_W * COLUMNS, TILE_H * ROWS), "#10151c")
            for offset, (number, row) in enumerate(page_items):
                item_tile = tile(row, number, font)
                page.paste(item_tile, ((offset % COLUMNS) * TILE_W, (offset // COLUMNS) * TILE_H))
            page.save(target / f"{page_idx + 1:03d}.jpg", quality=92)
    print(f"Created {sum(math.ceil(len(items)/(COLUMNS*ROWS)) for items in groups.values())} sheets for {len(rows)} decisions in {OUT}")
    for name, items in sorted(groups.items()):
        print(f"{name}: {len(items)} objects")


if __name__ == "__main__":
    main()
