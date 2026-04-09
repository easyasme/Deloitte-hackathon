# Phase 1: Data Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-09
**Phase:** 1-data-foundation
**Areas discussed:** Missing value handling, Temporal integrity checks, Dataset structure validation, Duplicate resolution

---

## Missing Value Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Drop if weather missing | Drop rows where weather data (avg_tmax_c, avg_tmin_c, tot_prcp_mm) is missing | |
| Impute weather data | Forward-fill from prior year within same ZIP, then neighbor ZIP average as fallback | |
| Let model handle NaN | Leave weather as-is and let modeling handle NaN naturally (e.g., LightGBM handles NaN) | ✓ |

**User's choice:** Let model handle NaN (tree models)
**Notes:** User confirmed NaN should stay as-is. Tree models handle NaN natively, no imputation needed.

---

## Temporal Integrity Checks

| Option | Description | Selected |
|--------|-------------|----------|
| Strict lag enforcement | All predictors must be lagged — fire risk for 2021 can only use data from 2020 or earlier | ✓ |
| Verify no 2021 leakage | Verify 2021 actuals not in dataset (only 2018-2020 available for training). Validate no 2021 row used when building 2020 features | |
| Allow Year as feature | Use Year column directly as feature (not lagged) — allow model to learn temporal patterns from Year itself | |

**User's choice:** Strict lag enforcement
**Notes:** All concurrent features must be lagged. Explicit validation: no 2021 data when building training features.

---

## Dataset Structure Validation

| Option | Description | Selected |
|--------|-------------|----------|
| Strict uniqueness check | Strict (Year, ZIP, Category) triple uniqueness — if duplicates found, investigate source before any aggregation | ✓ |
| Aggregate on duplicates | Aggregate duplicates by summing numeric columns, keeping first value for categoricals | |
| Keep all rows | Keep all rows (including potential duplicates from fire event joins) and let downstream feature engineering handle | |

**User's choice:** Strict uniqueness check
**Notes:** Confirmed the data is Year × ZIP × Category granularity, not just Year × ZIP. User wants strict uniqueness enforced.

---

## Duplicate Resolution

| Option | Description | Selected |
|--------|-------------|----------|
| Aggregate per Category | Deduplicate to one row per Year × ZIP × Category. Sum numeric columns across fire events for the same insurance record | ✓ |
| Strict Category uniqueness | Verify structure is exactly Year × ZIP × Category. Investigate if row counts deviate from expected | |

**User's choice:** Aggregate per Category
**Notes:** Data has multiple categories per ZIP-year. Duplicates arise from fire event joins per insurance record. Aggregate by summing numeric columns.

---

## Revisit: Missing Value Handling (Depth Dive)

| Option | Description | Selected |
|--------|-------------|----------|
| Keep NaN (tree models) | No imputation — LightGBM/RandomForest handle NaN natively | ✓ |
| Mean/median imputation | Fill NaN with column mean/median — enables linear models without NaN errors | |
| Column-type specific | Different strategies per column type: weather = keep NaN, categorical = mode, numeric = median | |
| Critical vs optional fields | Distinguish critical fields (Earned Premium, Earned Exposure, Fire Risk Score) from optional | |

**User's choice:** Keep NaN (tree models)
**Notes:** Confirmed — tree models handle NaN natively. No imputation for any column type.

---

## Revisit: Missing Value Threshold

| Option | Description | Selected |
|--------|-------------|----------|
| No threshold (defer to modeling) | No hard threshold — all decisions deferred to modeling phase based on feature importance and null impact | ✓ |
| Row completeness threshold | Drop rows where >50% of columns are NaN | |
| Alert only | Alert/warn but keep row | |

**User's choice:** No threshold (defer to modeling)
**Notes:** No missing-value threshold during loading. Quality assessment deferred to modeling phase.

---

## Claude's Discretion

- Missing value handling strategy applies uniformly to all columns — no column-type-specific strategies
- Temporal integrity checks are enforcement-level, not exploratory — any detected leakage should abort with clear error message

## Deferred Ideas

None — all discussion stayed within Phase 1 scope.
