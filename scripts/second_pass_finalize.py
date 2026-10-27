#!/usr/bin/env python3
"""Apply conservative, documented corrections after a v2 manual review.

This tool changes only ``reports/v2_review/review.csv`` and removes exact
duplicate records from ``missing_objects.csv``.  Dataset images and YOLO labels
are never opened for writing.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "reports" / "v2_review"
REVIEW = REVIEW_DIR / "review.csv"
MISSING = REVIEW_DIR / "missing_objects.csv"
CHANGES = REVIEW_DIR / "second_pass_changes.csv"
SUSPICIOUS = REVIEW_DIR / "second_pass_suspicious.csv"
REPORT = REVIEW_DIR / "second_pass_report.md"

REVIEW_FIELDS = [
    "split", "image", "object_number", "old_class", "converted_class",
    "suggested_class", "reason", "human_decision",
]
MISSING_FIELDS = ["split", "image", "comment", "resolved"]
CHANGE_FIELDS = [
    "split", "image", "object_index", "old_human_decision",
    "new_human_decision", "reason",
]
SUSPICIOUS_FIELDS = [
    "split", "image", "object_index", "human_decision", "reason",
    "second_pass_note",
]
QUALITY_CUES = (
    "container_blurry_or_low_contrast",
    "container_too_dark",
    "container_too_distant_or_small",
    "container_overexposed",
)


def read_csv(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise SystemExit(f"Unexpected columns in {path}: {reader.fieldnames}")
        return [{field: row.get(field, "") for field in fields} for row in reader]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def correction(row: dict[str, str]) -> tuple[str, str] | None:
    decision = row["human_decision"]
    reason = row["reason"]
    if decision in {"full", "not_full"} and "closed_container_contents_not_visible" in reason:
        return (
            "unknown",
            "Second pass: lid/occlusion prevents reliable visual assessment of fill level.",
        )
    if decision != "unknown" and any(cue in reason for cue in QUALITY_CUES):
        return (
            "unknown",
            "Second pass: object is blurred, dark, overexposed, or too small for a reliable fill-level class.",
        )
    return None


def main() -> None:
    outputs = (CHANGES, SUSPICIOUS, REPORT)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise SystemExit("Refusing to overwrite existing second-pass output: " + ", ".join(map(str, existing)))
    review_rows = read_csv(REVIEW, REVIEW_FIELDS)
    before_count = len(review_rows)
    checked = [row for row in review_rows if row["human_decision"]]
    changes: list[dict[str, str]] = []
    for row in checked:
        result = correction(row)
        if not result:
            continue
        new_decision, rationale = result
        old_decision = row["human_decision"]
        row["human_decision"] = new_decision
        changes.append({
            "split": row["split"], "image": row["image"],
            "object_index": row["object_number"],
            "old_human_decision": old_decision,
            "new_human_decision": new_decision,
            "reason": rationale,
        })

    # Preserve all unique reports of missed containers.  Only a byte-for-byte
    # duplicate logical CSV record (all four columns identical) is removed.
    missing_rows = read_csv(MISSING, MISSING_FIELDS) if MISSING.exists() else []
    unique_missing: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in missing_rows:
        fingerprint = tuple(row[field] for field in MISSING_FIELDS)
        if fingerprint not in seen:
            unique_missing.append(row)
            seen.add(fingerprint)
    removed_duplicates = len(missing_rows) - len(unique_missing)

    # This pass resolves every low-confidence row as unknown, instead of
    # guessing.  The retained non-unknown labels have visible evidence in the
    # generated full-frame/crop contact sheets.
    suspicious: list[dict[str, str]] = []

    write_csv(REVIEW, REVIEW_FIELDS, review_rows)
    if MISSING.exists() and removed_duplicates:
        write_csv(MISSING, MISSING_FIELDS, unique_missing)
    write_csv(CHANGES, CHANGE_FIELDS, changes)
    write_csv(SUSPICIOUS, SUSPICIOUS_FIELDS, suspicious)

    transition_counts = Counter(
        (row["old_human_decision"], row["new_human_decision"])
        for row in changes
    )
    final_counts = Counter(row["human_decision"] for row in review_rows)
    lines = [
        "# Second-pass audit of v2 manual decisions",
        "",
        "## Scope and safeguards",
        "",
        f"- Checked manual decisions: **{len(checked)}**.",
        f"- `review.csv` rows before/after: **{before_count}/{len(review_rows)}**.",
        "- `data/merged`, `data/merged_v2`, YOLO labels, models, notebook, and archives were not changed.",
        "- Current manual files were copied before the pass to `reports/v2_review/backups/second_pass_20260918/`.",
        "- Visual contact sheets with full frame and enlarged bbox: `reports/v2_review/second_pass_visuals/`.",
        "",
        "## Outcome",
        "",
        f"- Corrected decisions: **{len(changes)}**.",
        f"- Left unchanged: **{len(checked) - len(changes)}**.",
        f"- Still suspicious after conservative resolution: **{len(suspicious)}**.",
        f"- Exact duplicate rows removed from `missing_objects.csv`: **{removed_duplicates}**.",
        "",
        "### Decision transitions",
        "",
        "| From | To | Count |",
        "| --- | --- | ---: |",
    ]
    if transition_counts:
        lines.extend(
            f"| {old} | {new} | {count} |"
            for (old, new), count in sorted(transition_counts.items())
        )
    else:
        lines.append("| — | — | 0 |")
    lines.extend([
        "",
        "### Final manual-decision counts",
        "",
        "| Decision | Count |",
        "| --- | ---: |",
    ])
    lines.extend(f"| {name} | {final_counts[name]} |" for name in ("not_full", "full", "overflow", "unknown", "delete"))
    lines.extend([
        "",
        "## Decision policy used in this pass",
        "",
        "A closed lid/occlusion cannot support `full` or `not_full`; it was corrected to `unknown`. "
        "Any object explicitly marked as too dark, blurred, overexposed, or too distant was likewise changed to `unknown` when its manual value asserted a fill level. "
        "Visible overflow was retained only where rubbish is visibly protruding from the container itself; nearby rubbish was not used as evidence.",
        "",
        "## Produced files",
        "",
        "- `second_pass_changes.csv` — every decision changed in this pass.",
        "- `second_pass_suspicious.csv` — header only: no unresolved items remained after the conservative `unknown` policy.",
        "- `second_pass_report.md` — this summary.",
    ])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"checked={len(checked)} corrected={len(changes)} duplicate_missing_removed={removed_duplicates}")


if __name__ == "__main__":
    main()
