# V3 auto-label preparation report

## Scope and safety

- Processed images: **896** (train/val/test; no split was changed).
- Snapshot: `data/merged_v3_autolabeled`, copied from `data/merged_v3_work`.
- Protected manually confirmed frames: **18**; SHA-256 equality with the source was checked before and after this run.
- Applied automatic label changes: **0**. `added_boxes.csv` and `modified_boxes.csv` intentionally contain headers only.
- Image files differing from the source snapshot: **0**.
- Label files differing from the source snapshot: **0**.

## Candidate sources

- YOLO-World geometry candidates: 1219.
- Manual-reference homography candidates: 297.
- Retained for review: 999 (372 medium, 627 low).
- High-confidence automatic additions applied: **0** (the calibrated high-confidence tier is intentionally empty).
- Rejected as protected/duplicate/invalid: 517.
- Original unreviewed V3 queue frames retained: 477.
- Unique frames requiring a person: **606**.

## Validation

- New automatic boxes: 0, therefore no newly introduced invalid coordinates or duplicate boxes.
- Existing labels outside strict YOLO geometry bounds: 0 (left untouched and retained for review).
- Exact duplicate existing boxes: 0 (left untouched; no automatic cleanup was authorised).

## Why no boxes were auto-applied

Visual calibration of the detector's numerically strongest proposals showed false positives and partial rectangles (lids, exposed contents, and overlaps with an existing partial box). The detector also cannot determine fill level. Applying those labels would violate the full-container and class rules. The dedicated reviewer displays proposed boxes separately; only a human acceptance can add one to this dataset snapshot.

## Files

- `autolabel_changes.csv` — every automatic candidate with method, confidence, box and disposition.
- `rejected_candidates.csv` — candidates excluded from review, with a reason.
- `manual_check_queue.csv` — original outstanding V3 items plus medium/low automatic candidates.
- `contact_sheets/` — all frames with retained automatic candidates (green = existing, orange = medium, purple = low).
