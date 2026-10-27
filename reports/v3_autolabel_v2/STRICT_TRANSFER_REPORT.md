# Strict scene-transfer pass

```json
{
  "manual_protected_frames": 18,
  "strict_scene_groups": 6,
  "high_candidates": 63,
  "high_applied": 0,
  "duplicates_skipped": 4,
  "mode": "dry_run"
}
```

Only strict, visually checked static-scene homographies are included. Existing target boxes were never altered; records with IoU ≥ 0.80 were skipped.
