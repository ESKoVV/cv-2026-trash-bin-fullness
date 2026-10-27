# Second-pass audit of v2 manual decisions

## Scope and safeguards

- Checked manual decisions: **1170**.
- `review.csv` rows before/after: **1170/1170**.
- `data/merged`, `data/merged_v2`, YOLO labels, models, notebook, and archives were not changed.
- Current manual files were copied before the pass to `reports/v2_review/backups/second_pass_20260918/`.
- Visual contact sheets with full frame and enlarged bbox: `reports/v2_review/second_pass_visuals/`.

## Outcome

- Corrected decisions: **285**.
- Left unchanged: **885**.
- Still suspicious after conservative resolution: **0**.
- Exact duplicate rows removed from `missing_objects.csv`: **0**.

### Decision transitions

| From | To | Count |
| --- | --- | ---: |
| full | unknown | 180 |
| not_full | unknown | 74 |
| overflow | unknown | 31 |

### Final manual-decision counts

| Decision | Count |
| --- | ---: |
| not_full | 15 |
| full | 283 |
| overflow | 257 |
| unknown | 612 |
| delete | 3 |

## Decision policy used in this pass

A closed lid/occlusion cannot support `full` or `not_full`; it was corrected to `unknown`. Any object explicitly marked as too dark, blurred, overexposed, or too distant was likewise changed to `unknown` when its manual value asserted a fill level. Visible overflow was retained only where rubbish is visibly protruding from the container itself; nearby rubbish was not used as evidence.

## Produced files

- `second_pass_changes.csv` — every decision changed in this pass.
- `second_pass_suspicious.csv` — header only: no unresolved items remained after the conservative `unknown` policy.
- `second_pass_report.md` — this summary.
