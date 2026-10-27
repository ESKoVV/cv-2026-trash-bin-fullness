# Final audit — V3

## Verdict: NOT READY

The dataset must **not** be used for training yet. This verdict is based on the
actual audit evidence, not merely on a successful script run.

## Structural checks

- Images: 896; label objects: 1731.
- `data.yaml` exists and maps all four expected class IDs: True.
- Missing labels: 0; orphan labels: 0; corrupt images: 0; invalid YOLO records: 0.
- Exact duplicate boxes / strong overlaps flagged: 1 (heuristic flags, not automatic corrections).

## Blocking visual work

- Remaining unique frames: **604**.
- Pending strict same-scene transfer candidates: **59**. They include background-container rows from manually corrected reference scenes, but are not treated as labels until a person approves them.
- No boxes/classes/manual overrides were changed by this audit. The four corresponding CSVs are intentionally header-only.

## Split integrity

- Cross-split exact duplicates: 0.
- Cross-split near-duplicate dHash flags: 36.
- High-confidence cross-split same-scene leaks: 1.

At least one strict train→test same-camera transfer is present in `split_leakage.csv`. Test results would therefore be optimistic until complete camera sequences are kept in one split.

## Shortest safe path to readiness

1. Review and either accept/correct/reject the 59 strict scene-transfer boxes in the group UI; do not invent uncertain far objects.
2. Resolve the remaining visual-review frames, prioritising yellow-container groups and all geometry flags.
3. Re-split complete duplicate/near-duplicate camera sequences so they do not cross train/val/test.
4. Re-run this audit. Only an empty `remaining_review.csv`, a clean split-leakage review and a visual sign-off can change the verdict.
