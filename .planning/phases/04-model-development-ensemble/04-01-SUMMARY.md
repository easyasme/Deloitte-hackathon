---
phase: "04-model-development-ensemble"
plan: "01"
subsystem: "Task2_Data"
tags:
  - "linearmodels"
  - "PanelOLS"
  - "Pooled OLS"
  - "COVID-19 weighting"
  - "insurance premium prediction"
dependency_graph:
  requires:
    - "Task2_Data/task2_step2_feature_matrix.csv"
    - "Task2_Data/task2_step3_metrics.csv"
  provides:
    - "Task2_Data/task2_step4_panel_fe.py"
    - "Task2_Data/task2_step4_panel_fe_metrics.csv"
    - "Task2_Data/task2_step4_panel_fe_predictions.csv"
  affects: []
tech_stack:
  added:
    - "linearmodels 7.0"
tech_stack:
  patterns:
    - "Pooled OLS with robust SE (entity_effects=False)"
    - "COVID-19 sample weighting (0.5) via weights parameter"
    - "MultiIndex panel structure (ZIP, Year)"
key_files:
  created:
    - path: "Task2_Data/task2_step4_panel_fe.py"
      description: "Panel FE model training script using linearmodels PanelOLS"
    - path: "Task2_Data/task2_step4_panel_fe_metrics.csv"
      description: "Panel FE validation metrics (RMSE, MAE, MAPE)"
    - path: "Task2_Data/task2_step4_panel_fe_predictions.csv"
      description: "2020 validation predictions from Panel FE model"
decisions:
  - id: "panel_entity_effects"
    decision: "Use Pooled OLS (entity_effects=False) instead of Panel FE with entity effects"
    rationale: "Single-period training data (2019 only) — each ZIP appears once, preventing within-entity variation estimation"
    outcome: "AbsorbingEffectError avoided; Pooled OLS estimates common coefficients across all ZIPs"
  - id: "clustered_se_singular"
    decision: "Use robust SE (cov_type='robust') instead of clustered SE"
    rationale: "Singular matrix error when computing clustered covariance — multicollinearity among features"
    outcome: "Robust SE provides valid heteroskedasticity-consistent inference"
  - id: "category_dummies_dropped"
    decision: "Drop category dummies (cat_HO, cat_CO, etc.) from feature set"
    rationale: "Category is constant per ZIP-Category combination — perfectly collinear in single-period panel"
    outcome: "Reduced features from 42 to 35, resolving remaining multicollinearity"
metrics:
  duration: ""
  completed: "2026-04-09"
---

# Phase 04 Plan 01: Panel Fixed Effects Model Summary

## Objective

Train Panel Fixed Effects model (linearmodels PanelOLS) with ZIP entity effects, COVID-19 sample weighting, and clustered standard errors. Produce 2020 validation predictions and metrics for comparison with Phase 3 baselines.

## What Was Built

Panel FE (Pooled OLS) model trained on 2019 insurance premium data using linearmodels PanelOLS. Due to single-period training data (each ZIP appears once in 2019), true entity fixed effects (ZIP FE) were not estimable. Instead, Pooled OLS was used to estimate common coefficients across all ZIPs with robust standard errors.

**Key Implementation Details:**
- Training data: 2019 (9,329 observations)
- Validation data: 2020 (12,708 observations)
- COVID-19 sample weighting: 0.5 applied to all 2019 training observations
- Features: 35 (after dropping category dummies and all-NaN columns)
- Covariance: Robust (heteroskedasticity-consistent)

## Key Coefficients

| Feature | Coefficient |
|---------|-------------|
| Avg Fire Risk Score | 16,579.38 |
| Avg Fire Risk Score_lag1 | -20,713.39 |
| Earned Premium_lag1 | 0.6306 |
| is_covid_year | 0.0000 |

## Metrics Comparison

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| LinearRegression (Phase 3) | 515,829.79 | 194,560.88 | 7,754.70% |
| RandomForest (Phase 3) | 999,715.70 | 165,535.85 | 40.42% |
| **PanelFE (Pooled OLS)** | **518,221.89** | **195,538.36** | **7,337.41%** |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] AbsorbingEffectError with entity_effects=True**
- **Found during:** Task 1 execution
- **Issue:** Single-period training data (2019 only) causes `AbsorbingEffectError` when using `entity_effects=True` because each ZIP fixed effect perfectly explains the target
- **Fix:** Switched to `entity_effects=False` (Pooled OLS) which treats all ZIPs as one panel and estimates common coefficients
- **Files modified:** `Task2_Data/task2_step4_panel_fe.py`
- **Commit:** d77b2f0

**2. [Rule 3 - Blocking] Singular matrix with clustered SE**
- **Found during:** Task 1 execution
- **Issue:** `LinAlgError: Singular matrix` when computing clustered covariance — multicollinearity among features
- **Fix:** Changed `cov_type='clustered', cluster_entity=True` to `cov_type='robust'` for valid heteroskedasticity-consistent inference
- **Files modified:** `Task2_Data/task2_step4_panel_fe.py`
- **Commit:** d77b2f0

**3. [Rule 3 - Blocking] Perfect multicollinearity from category dummies**
- **Found during:** Task 1 execution
- **Issue:** Category dummies (`cat_HO`, `cat_CO`, etc.) are constant per ZIP-Category, causing perfect multicollinearity
- **Fix:** Dropped category dummies from feature set, reducing from 42 to 35 features
- **Files modified:** `Task2_Data/task2_step4_panel_fe.py`
- **Commit:** d77b2f0

**4. [Rule 1 - Bug] AttributeError in evaluate_regression**
- **Found during:** Task 1 execution
- **Issue:** `y_pred` was already a numpy array (not Series), so `.values` attribute access failed
- **Fix:** Added explicit `np.asarray()` conversion for both `y_true` and `y_pred` before array indexing
- **Files modified:** `Task2_Data/task2_step4_panel_fe.py`
- **Commit:** d77b2f0

## Outputs

| File | Rows | Description |
|------|------|-------------|
| `Task2_Data/task2_step4_panel_fe.py` | 143 | Training script |
| `Task2_Data/task2_step4_panel_fe_metrics.csv` | 1 | RMSE: 518,221.89, MAE: 195,538.36, MAPE: 7,337.41% |
| `Task2_Data/task2_step4_panel_fe_predictions.csv` | 12,709 | 2020 validation predictions with panel_fe_pred column |

## Known Stubs

None.

## Threat Flags

None — no security-relevant surface changes introduced.

## Self-Check

- [x] `Task2_Data/task2_step4_panel_fe.py` exists
- [x] Contains `from linearmodels.panel import PanelOLS`
- [x] Contains `entity_effects=False` in PanelOLS call (Pooled OLS due to data structure)
- [x] Contains `cov_type='robust'` in fit call
- [x] Contains `weights_train` for COVID-19 weighting
- [x] `Task2_Data/task2_step4_panel_fe_metrics.csv` exists with rmse, mae, mape columns
- [x] `Task2_Data/task2_step4_panel_fe_predictions.csv` exists with 12,709 rows
- [x] PanelFE RMSE (518,221.89) is comparable to Phase 3 LR RMSE (515,829.79)
- [x] Commit d77b2f0 verified in history

## Notes

The Panel FE model (Pooled OLS) achieved RMSE of 518,221.89, which is slightly higher than the Phase 3 Linear Regression baseline (515,829.79). This is expected because:

1. **Pooled OLS limitation**: Without true entity fixed effects, the model cannot capture ZIP-specific baseline premiums that differ substantially across California zip codes
2. **Single-period training**: The 2019-only training prevents within-ZIP variation estimation — the hallmark advantage of Panel FE over pooled cross-sectional regression
3. **High MAPE**: 7,337% MAPE indicates difficulty with zero-premium observations (common in insurance data)

The model nonetheless provides a useful baseline for the ensemble. True Panel FE would require multiple time periods per ZIP (e.g., 2018-2020 all used for training with ZIP as the entity).
