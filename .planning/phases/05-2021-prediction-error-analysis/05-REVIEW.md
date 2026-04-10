---
phase: 05
reviewed: 2026-04-10T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - Task2_Data/task2_step5_predictions.py
  - Task2_Data/task2_step5_error_analysis.py
  - Task2_Data/task2_step5_fire_risk_ablation.py
  - Task2_Data/task2_step4_ensemble.py
  - Task2_Data/task2_step4_panel_fe.py
  - Task2_Data/task2_step4_lightgbm.py
findings:
  critical: 1
  warning: 5
  info: 4
  total: 10
status: issues_found
---
# Phase 05: Code Review Report

**Reviewed:** 2026-04-10
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Phase 5 scripts implement 2021 premium predictions, error analysis, and fire risk ablation study. The core ensemble logic (80% Panel FE + 20% LightGBM) is sound and follows established patterns from Phase 4. However, there are several correctness and quality issues that need attention, most notably a **silent ZIP-alignment vulnerability** in the ablation study that could produce incorrect metrics without raising any error.

---

## Critical Issues

### CR-01: Silent ZIP Alignment Bug in Ablation Study

**File:** `Task2_Data/task2_step5_fire_risk_ablation.py:275-280`
**Issue:** When loading predictions from `task2_step5_predictions.csv` (line 275), there is no validation that ZIP codes align with the holdout data. Only row count is checked (line 279). If ZIP codes are in a different order or a different subset exists between files, the RMSE computation silently produces garbage results with no error raised.

```python
preds_with_fire = pd.read_csv("Task2_Data/task2_step5_predictions.csv")
y_pred_with_fire = preds_with_fire['predicted_premium'].values

assert len(y_pred_2021_no_fire) == len(y_actual_2021), ...  # Only length check!
assert len(y_pred_with_fire) == len(y_actual_2021), ...     # Only length check!
```

**Fix:** Add ZIP-level alignment verification:

```python
preds_with_fire = pd.read_csv("Task2_Data/task2_step5_predictions.csv")
# Verify ZIP alignment before using predictions
assert np.array_equal(preds_with_fire['ZIP'].values, holdout['ZIP'].values), \
    "ZIP misalignment: task2_step5_predictions.csv ZIPs do not match holdout"
y_pred_with_fire = preds_with_fire['predicted_premium'].values
```

---

## Warnings

### WR-01: Redundant Computation of Top Errors

**File:** `Task2_Data/task2_step5_error_analysis.py:50-54`
**Issue:** `top_errors` DataFrame is computed twice with a trivial modification (addition of `fire_risk_percentile`). The first computation (line 50) is immediately overwritten (line 54) and never used.

```python
# Line 50 - computed but never used before being overwritten
top_errors = df.nlargest(20, 'abs_error')[['ZIP', 'Category', 'predicted_premium', 'actual_premium', 'error', 'abs_error', 'fire_risk_score']].copy()

# Line 54 - recomputed with additional column, overwrites line 50
top_errors = df.nlargest(20, 'abs_error')[['ZIP', 'Category', 'predicted_premium', 'actual_premium', 'error', 'abs_error', 'fire_risk_score', 'fire_risk_percentile']].copy()
```

**Fix:** Remove line 50 or merge the two operations:

```python
top_errors = df.nlargest(20, 'abs_error')[['ZIP', 'Category', 'predicted_premium', 'actual_premium', 'error', 'abs_error', 'fire_risk_score']].copy()
# Compute fire_risk_percentile for all rows first, then filter
df['fire_risk_percentile'] = df['fire_risk_score'].rank(pct=True) * 100
top_errors = df.nlargest(20, 'abs_error')[['ZIP', 'Category', 'predicted_premium', 'actual_premium', 'error', 'abs_error', 'fire_risk_score', 'fire_risk_percentile']].copy()
```

---

### WR-02: MAPE Function Inconsistency

**File:** `Task2_Data/task2_step5_error_analysis.py:19-24`
**Issue:** The `compute_mape` function uses `.values` on `y_true` but not on `y_pred`, creating inconsistency with how `non_zero_mask` is applied:

```python
def compute_mape(y_true, y_pred):
    non_zero_mask = y_true > 0  # boolean array from y_true
    if non_zero_mask.sum() > 0:
        return np.mean(np.abs((y_true.values[non_zero_mask] - y_pred[non_zero_mask]) / y_true.values[non_zero_mask])) * 100
    return np.nan
```

When `y_true` is a pandas Series (line 33-35), `.values` is applied. When `y_true` is a numpy array (which `evaluate_regression` receives via `y_true - y_pred`), `.values` on a numpy array is a no-op but the style is inconsistent.

**Fix:** Normalize input handling:

```python
def compute_mape(y_true, y_pred):
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    non_zero_mask = y_true_arr > 0
    if non_zero_mask.sum() > 0:
        return np.mean(np.abs((y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask])) * 100
    return np.nan
```

---

### WR-03: Potential NaN Predictions Not Handled

**File:** `Task2_Data/task2_step5_predictions.py:281-294`
**Issue:** After computing `y_pred_2021`, there is no check for NaN values before saving. If any input features for 2021 holdout are NaN and not properly imputed, predictions could be NaN without being detected.

```python
holdout['predicted_premium'] = y_pred_2021
# ... no NaN check before to_csv
output_df.to_csv(OUTPUT_FILE, index=False)

nan_count = output_df['predicted_premium'].isna().sum()  # Checked AFTER saving
```

**Fix:** Validate before saving:

```python
nan_count = pd.Series(y_pred_2021).isna().sum()
if nan_count > 0:
    print(f"WARNING: {nan_count} NaN predictions detected!")
    # Consider: y_pred_2021 = np.nan_to_num(y_pred_2021, nan=0.0)
```

---

### WR-04: Inconsistent COVID Year Indicator

**File:** `Task2_Data/task2_step5_predictions.py:87-89`
**Issue:** The `is_covid_year` is hardcoded to 0 for all 2021 rows, but in the training data (2019-2020), `is_covid_year` is 1 for 2020. This creates a train/predict inconsistency - the model has never seen `is_covid_year=1` in prediction mode:

```python
# In training: is_covid_year = 1 for Year==2020, 0 for Year==2019
holdout['is_covid_year'] = 0  # Always 0 for 2021
```

Note: This may be intentional (2021 is post-COVID), but it means the model learned a COVID coefficient from 2020 that will never be activated in prediction.

**Fix:** Document the intentionality or consider whether 2021 should have a different indicator:

```python
# 2021 is post-COVID; if COVID effects persist, may need is_post_covid_year=1
holdout['is_covid_year'] = 0
```

---

### WR-05: Expanding Stats Merge Uses Incomplete Columns

**File:** `Task2_Data/task2_step5_predictions.py:71-73`
**Issue:** The expanding window statistics computed from raw_df (lines 61-66) include `expanding_fire_risk_mean`, `expanding_fire_risk_std`, and `expanding_exposure_mean`. However, when these are merged (line 72), only a subset is selected (line 71). If `expanding_fire_risk_std` has all NaN values for 2020 (due to single-observation groups), it could cause issues downstream.

```python
expanding_2020 = raw_df[raw_df['Year'] == 2020][['ZIP', 'Category', 'expanding_fire_risk_mean', 'expanding_fire_risk_std', 'expanding_exposure_mean']].copy()
holdout = holdout.merge(expanding_2020, on=['ZIP', 'Category'], how='left')
```

---

## Info

### IN-01: No Type Hints

**File:** All phase 5 files
**Issue:** Functions like `compute_mape` and `evaluate_regression` have no type hints, making it harder to verify correctness at a glance.

**Suggestion:** Add type hints following project conventions (no type hints detected in codebase, but they would improve maintainability).

---

### IN-02: Comments Reference Non-Existent D-numbers

**File:** `Task2_Data/task2_step5_error_analysis.py:15, 48`
**Issue:** Comments reference "D-11", "D-14", "D-08", "D-09" without context. These appear to be requirements traceability references that do not exist in the codebase.

```python
# STEP 2: Compute error metrics per model (D-11, D-14)
# STEP 5: Top-20 highest-error ZIPs (D-08)
# STEP 7: Validate high-error ZIPs against fire risk distribution (D-09)
```

**Suggestion:** Either link to actual requirements artifacts or remove the D-numbers.

---

### IN-03: Hardcoded Ensemble Weights Not Validated Against Actual Performance

**File:** `Task2_Data/task2_step5_predictions.py:25-26`
**Issue:** Ensemble weights (0.80 Panel FE, 0.20 LightGBM) are hardcoded based on Phase 4 validation results. These are not re-validated in this script. If the weights are stale or the underlying models change, the weights could be suboptimal.

```python
ENSEMBLE_WEIGHT_PANEL_FE = 0.80
ENSEMBLE_WEIGHT_LGB = 0.20
```

**Suggestion:** Consider loading from Phase 4 output or re-running weight optimization in this step.

---

### IN-04: Code Duplication Across Files

**File:** `Task2_Data/task2_step5_predictions.py`, `Task2_Data/task2_step5_fire_risk_ablation.py`
**Issue:** Both files contain nearly identical code for:
- Panel FE training (lines 130-175 in predictions vs 130-168 in ablation)
- LightGBM training (lines 179-211 in predictions vs 173-199 in ablation)
- 2021 feature preparation (lines 226-259 in predictions vs 220-258 in ablation)

**Suggestion:** Extract common training and prediction logic into a shared module (e.g., `task2_step5_utils.py`) to DRY out the codebase and reduce maintenance burden.

---

## Positive Findings

The following patterns are correctly implemented:

1. **Temporal lag constraint respected** (`task2_step5_predictions.py:41-52`): 2021 features correctly use 2020 values only via lag mapping.

2. **Leakage prevention** (consistent across all files): Current-year loss/claim columns are properly excluded from features.

3. **NaN handling in Panel FE** (`task2_step5_predictions.py:143-163`): Columns with all NaN are dropped, remaining NaNs filled with column means from training data.

4. **COVID-19 awareness**: The codebase correctly identifies 2020 as COVID-affected and applies sample weighting in model training.

---

_Reviewed: 2026-04-10_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
