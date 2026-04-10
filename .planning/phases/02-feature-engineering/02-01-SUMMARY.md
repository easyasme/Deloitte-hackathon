---
phase: "02-feature-engineering"
plan: "01"
status: "completed"
completed_at: "2026-04-09"
---

## Phase 2: Feature Engineering — Summary

### What Was Built
`Task2_Data/task2_step2_feature_engineering.py` — 158-line reproducible script that transforms the clean panel dataset into a model-ready feature matrix.

### Output
- **File:** `Task2_Data/task2_step2_feature_matrix.csv`
- **Rows:** 22,037 (2019: 9,329 + 2020: 12,708)
- **Columns:** 67

### Features Engineered

| Feature Type | Columns | Notes |
|---|---|---|
| Temporal lags (shift-1) | 6 `_lag1` columns | Fire risk, exposure, premium, CAT/non-CAT fire losses, PPC |
| Expanding mean/std | `expanding_fire_risk_mean`, `expanding_fire_risk_std`, `expanding_exposure_mean` | Per ZIP+Category, shift(1) prevents leakage |
| Category one-hot | `cat_HO` through `cat_NA` (7 columns) | Sum to 1 per row |
| COVID indicator | `is_covid_year` | 1 for 2020, 0 for 2019 |

### Key Data Decisions
- **2018 rows removed** — served as lag source but excluded from final matrix (no prior year available)
- **NaN lags accepted** — 250 ZIP+Category combos first appeared in 2019; 3,293 first appeared in 2020; these have legitimate NaN lags
- **FEAT-05 deferred** — Zero-inflated premium (12.8% zeros) documented; Tweedie vs two-part model decision deferred to Phase 3

### Verification
All assertions passed: 6 lag columns, 3 expanding stats, 7 category columns, COVID flag, no 2018 rows.

### Next
Phase 3 (Modeling) consumes `task2_step2_feature_matrix.csv` with train/test split: 2019=train, 2020=validation.
