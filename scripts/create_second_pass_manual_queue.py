#!/usr/bin/env python3
"""Build a manual-only queue from second_pass_changes.csv without relabeling."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "reports" / "v2_review"
REVIEW = REVIEW_DIR / "review.csv"
CHANGES = REVIEW_DIR / "second_pass_changes.csv"
QUEUE = REVIEW_DIR / "second_pass_manual_queue.csv"
REVIEW_FIELDS = ["split", "image", "object_number", "old_class", "converted_class", "suggested_class", "reason", "human_decision"]
CHANGE_FIELDS = ["split", "image", "object_index", "old_human_decision", "new_human_decision", "reason"]
QUEUE_FIELDS = ["split", "image", "object_index", "initial_human_decision", "second_pass_suggestion", "second_pass_reason", "human_recheck_decision"]


def read(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise SystemExit(f"Unexpected columns in {path}")
        return [{field: row.get(field, "") for field in fields} for row in reader]


def main() -> None:
    if QUEUE.exists():
        raise SystemExit(f"Refusing to overwrite existing queue: {QUEUE}")
    review = read(REVIEW, REVIEW_FIELDS)
    changes = read(CHANGES, CHANGE_FIELDS)
    decisions = {(row["split"], row["image"], row["object_number"]): row["human_decision"] for row in review}
    queue: list[dict[str, str]] = []
    for change in changes:
        key = (change["split"], change["image"], change["object_index"])
        if decisions.get(key) != change["old_human_decision"]:
            raise SystemExit(f"Restored decision mismatch for {key}")
        queue.append({
            "split": change["split"], "image": change["image"], "object_index": change["object_index"],
            "initial_human_decision": change["old_human_decision"],
            "second_pass_suggestion": change["new_human_decision"],
            "second_pass_reason": change["reason"],
            "human_recheck_decision": "",
        })
    with QUEUE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS)
        writer.writeheader(); writer.writerows(queue)
    print(f"Created {len(queue)} manual queue rows: {QUEUE}")


if __name__ == "__main__":
    main()
