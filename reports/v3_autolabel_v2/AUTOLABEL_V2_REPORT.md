# V3 auto-label v2 — calibration and strict scene groups

## Calibration result

Grounding DINO Tiny was evaluated read-only on every unique full-frame manual
reference available in `geometry_changes.csv` (18 images, 191 boxes). It used
the seven container prompts, full frame plus native-resolution 1024 px tiles
with 25% overlap, detector threshold 0.12, and class-agnostic NMS at IoU 0.35.

| Metric | Result |
| --- | ---: |
| Recall at IoU 0.30 | 31.94% |
| Recall at IoU 0.50 | 23.04% |
| Precision at IoU 0.30 | 22.22% |
| Detections after NMS | 288 |

This is insufficient for a general automatic annotation pass. In particular,
it cannot be used as a replacement for manually corrected background rows.
YOLO-World was also evaluated in the preceding pass and was rejected for
partial-box/lid proposals. `best1.pt` was not used.

## Manual-reference transfer

Six near-identical camera/scene pairs passed the strict gate: histogram
similarity >= 0.90, RANSAC inlier ratio >= 0.90, at least 80 inliers, and
homography confidence >= 0.85. Visual contact sheets are available in
`strict_transfer_contact_sheet.jpg` and the local group reviewer.

- Candidate rectangles: 63
- Duplicates skipped (IoU >= 0.80): 4
- Potential high-confidence additions: 59
- Target images in strict groups: 6

The 59 boxes are prepared but deliberately remain dry-run candidates. The
available history contains 20 save events across only 18 unique full-frame
manual references, so it cannot satisfy the requested 30-image independent
manual calibration requirement. No automatic box was written in this v2 pass.

## Safety state

- Source `data/merged_v3_work` was not modified.
- Backup of the safe copy: `data/merged_v3_autolabeled_backup_20260920T1530`.
- The 18 geometry-history manual references remain protected.
- Two other labels in the safe copy were reserialised by a human action in the
  previous reviewer; their object count and geometry are unchanged.

## Next action needed

Complete 12 more independent, full-frame human corrections (not merely class
changes) so a 30-frame calibration can be run. The strict group reviewer can
then be used to confirm or reject each 1-target static-scene transfer without
returning all 606 frames to a per-frame queue.
