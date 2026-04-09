# Phase 3: Baseline Models + Temporal Validation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-09
**Phase:** 03-baseline-models-temporal-validation
**Areas discussed:** Zero-exposure handling, Linear variant + regularization, Random Forest tuning, MAPE robustness

---

## Zero-Exposure Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Include all rows | Keep all rows including zero-exposure. Linear Reg and RF handle as-is. | ✓ |
| Filter to non-zero exposure | Remove rows where Earned Exposure == 0 before training and evaluation. | |
| Report both ways | Train/evaluate twice — full dataset and non-zero subset — compare metrics. | |

**User's choice:** Include all rows
**Notes:** Zero-exposure rows produce zero or near-zero premium naturally; both Linear Reg and Random Forest handle them without special treatment.

---

## Linear Variant + Regularization

| Option | Description | Selected |
|--------|-------------|----------|
| Vanilla OLS (LinearRegression) | No regularization. Simple, interpretable baseline. May overfit with many correlated features. | ✓ |
| Ridge regression (L2) | Shrinks correlated coefficients. Better for multicollinearity (fire risk ~ exposure ~ lag features). | |
| Lasso regression (L1) | Feature selection via sparsity. Drops redundant features automatically. | |

**User's choice:** Vanilla OLS (LinearRegression)
**Notes:** Simple interpretable baseline. Ridge/Lasso deferred to Phase 4 if overfitting observed.

---

## Random Forest Tuning

| Option | Description | Selected |
|--------|-------------|----------|
| Sensible defaults | n_estimators=100, max_depth=10-15, min_samples_leaf=5. No extensive tuning — baselines stay simple. | ✓ |
| Light tuning (max_depth, n_estimators) | Tune 2-3 key params: max_depth (5, 10, 20, None), n_estimators (100, 200). Quick search. | |
| Default sklearn params | Let sklearn use its defaults (n_estimators=100, max_depth=None, etc.) — pure baseline. | |

**User's choice:** Sensible defaults
**Notes:** Quick to run, reasonable performance. LightGBM in Phase 4 will be the primary tree model.

---

## MAPE Robustness

| Option | Description | Selected |
|--------|-------------|----------|
| Standard MAPE on non-zero subset | Report MAPE only on rows where Earned Premium > 0. Skip zero-premium rows from MAPE denominator. | ✓ |
| SMAPE (symmetric MAPE) | SMAPE = 2|y - ŷ| / (|y| + |ŷ|) — symmetric, bounded [0, 2]. Handles zeros via denominator sum. | |
| MAPE with epsilon floor | Add small epsilon (e.g., 1% of premium range) to denominator to floor division. Simple adjustment. | |

**User's choice:** Standard MAPE on non-zero subset
**Notes:** Zero-premium rows excluded from MAPE denominator; RMSE and MAE computed on full validation set. Zero-premium row count and percentage documented in validation report.

---

## Claude's Discretion

- Exact feature list (excluding Year and non-feature identifiers) determined by planner from `task2_step2_feature_matrix.csv` columns
- Specific random forest `max_depth` value (10, 12, or 15) left to planner's judgment
- Validation report format and structure left to planner's judgment

## Deferred Ideas

None — discussion stayed within Phase 3 scope.
