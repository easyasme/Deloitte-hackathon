---
phase: 04-model-development-ensemble
reviewed: 2026-04-09T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - Task2_Data/task2_step4_ensemble.py
  - Task2_Data/task2_step4_lightgbm.py
  - Task2_Data/task2_step4_panel_fe.py
findings:
  critical: 0
  warning: 3
  info: 4
  total: 7
status: issues_found
---
# Phase 04: Code Review Report

**Reviewed:** 2026-04-09
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Reviewed three ensemble model scripts (Panel FE, LightGBM, and ensemble combination). The code is generally well-structured with clear step markers and appropriate leakage prevention via exclude_cols. However, there are several issues: redundant step numbering, unused COVID weights in the main evaluation path, potential negative predictions not clipped to valid premium range, and inconsistent LightGBM configuration between inline and standalone execution.

## Warnings

### WR-01: Unused COVID-19 sample weights in main metrics

**File:** `Task2_Data/task2_step4_lightgbm.py:52,107`
**Issue:** `val_weights` is created to downweight 2020 (COVID year) observations, but `evaluate_regression()` at line 95 does not use these weights. The weighted RMSE/MAE are computed separately at lines 114-115, but the primary metrics reported at line 107 do not incorporate the COVID adjustment. This means the "COVID-adjusted" evaluation is not applied to the main reported metrics.
**Fix:** Pass `val_weights` into `evaluate_regression()` or apply weights inside the function:
```python
def evaluate_regression(model_name, y_true, y_pred, weights=None):
    ...
    if weights is not None:
        rmse = np.sqrt(np.average((y_true_arr - y_pred_arr)**2, weights=weights))
        mae = np.average(np.abs(y_true_arr - y_pred_arr), weights=weights)
    else:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
```

### WR-02: Negative predictions not clipped to valid premium range

**File:** `Task2_Data/task2_step4_lightgbm.py:91`
**Issue:** LightGBM predictions are generated via `model.predict()` without clipping to valid premium range [0, infinity). Earned premium cannot be negative, yet gradient boosting models can produce negative outputs. These negative predictions will distort MAPE calculations and produce physically impossible results.
**Fix:** Clip predictions before evaluation:
```python
y_pred_lgb = np.clip(model.predict(X_val, num_iteration=model.best_iteration), 0, None)
```

### WR-03: Inconsistent LightGBM configuration between standalone and inline execution

**File:** `Task2_Data/task2_step4_ensemble.py:65-70,71-76`
**Issue:** The standalone `task2_step4_lightgbm.py` uses early stopping with `callbacks=[lgb.early_stopping(50, verbose=False)]` and `num_boost_round=500`. However, the inline LightGBM at lines 71-76 also uses early stopping, but the ensemble script may produce different predictions if run at different times or with different data loading order. The inline version should be a function call to avoid code duplication and ensure consistency.
**Fix:** Extract LightGBM training into a shared function in a utility module, or import and call it from `task2_step4_lightgbm.py`.

## Info

### IN-01: Duplicate step numbering in panel_fe.py

**File:** `Task2_Data/task2_step4_panel_fe.py:100,133`
**Issue:** Step 10 is used twice - first for "Print key results" (line 100) and then for "Save predictions and metrics" (line 133). This is a documentation bug, not a functional issue.
**Fix:** Renumber the second "STEP 10" to "STEP 13" (since STEP 11 and 12 exist between).

### IN-02: Unused variable in panel_fe.py

**File:** `Task2_Data/task2_step4_panel_fe.py:79-83`
**Issue:** The loop at lines 79-83 that fills missing values in X_val_reduced is unnecessary because the loop at lines 70-76 already fills all columns (including validation columns) using training means. The second loop will have no work to do.
**Fix:** Remove lines 78-83 as dead code.

### IN-03: Magic number without explanation

**File:** `Task2_Data/task2_step4_ensemble.py:104`
**Issue:** The weight grid search uses `np.arange(0, 1.05, 0.05)` with no explanation for why 0.05 step size was chosen. While not incorrect, a named constant would improve readability.
**Fix:** Define a constant like `WEIGHT_GRID_STEP = 0.05` at the top with a brief comment.

### IN-04: Ensemble weight optimization limited to two models

**File:** `Task2_Data/task2_step4_ensemble.py:104-113`
**Issue:** The grid search optimizes weights only between PanelFE and LightGBM (`w * y_pred_fe + (1 - w) * y_pred_lgb`), yet the ensemble output includes LinearRegression and RandomForest predictions as well. The final ensemble only uses two of the five available model predictions.
**Fix:** Either expand the weight optimization to include all models (e.g., 5-way blend with scipy.optimize), or clarify in comments that the ensemble is specifically PanelFE+LightGBM and the other models are for comparison only.

---

_Reviewed: 2026-04-09_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
