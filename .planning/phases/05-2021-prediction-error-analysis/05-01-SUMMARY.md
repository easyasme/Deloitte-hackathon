---
phase: "05-2021-prediction-error-analysis"
plan: "01"
subsystem: "ml-prediction"
tags: ["panel-fe", "lightgbm", "ensemble", "time-series", "earned-premium"]

# Dependency graph
requires:
  - phase: "04-model-development"
    provides: "Optimal 80/20 PanelFE/LightGBM ensemble weights, trained models"
provides:
  - "2021 Earned Premium predictions for all 12,708 ZIP x Category combinations"
  - "task2_step5_predictions.csv with predicted_premium, actual_premium, fire_risk_score"
affects:
  - "phase-05-plans-02-and-03"
  - "error-analysis"
  - "2021-holdout-evaluation"

# Tech tracking
tech-stack:
  added: ["linearmodels.panel.PanelOLS", "lightgbm"]
  patterns: ["80/20 weighted ensemble", "temporal lag features", "expanding window statistics", "MultiIndex panel prediction"]

key-files:
  created: ["Task2_Data/task2_step5_predictions.py"]
  modified: []

key-decisions:
  - "Used 80/20 PanelFE/LightGBM ensemble weight from Phase 4 optimization"
  - "Created lag features directly from holdout 2020 values (t-1 constraint)"
  - "Computed expanding window stats from raw 2018-2020 panel data"

patterns-established:
  - "Panel FE requires MultiIndex (ZIP, Year) for out-of-sample prediction"
  - "LightGBM uses full feature set; Panel FE uses reduced feature set (excludes category dummies)"

requirements-completed: ["PRED-01"]

# Metrics
duration: "15min"
completed: "2026-04-10"
---

# Phase 05 Plan 01: 2021 Predictions Summary

**2021 Earned Premium predictions using 80% Panel FE + 20% LightGBM ensemble with temporal lag features**

## Performance

- **Duration:** 15 min
- **Started:** 2026-04-10T10:04:17Z (from STATE.md last_updated)
- **Completed:** 2026-04-10
- **Tasks:** 1
- **Files modified:** 1 (Task2_Data/task2_step5_predictions.py)

## Accomplishments
- Generated 2021 Earned Premium predictions for all 12,708 ZIP x Category combinations
- Ensemble: 80% Panel FE (Pooled OLS) + 20% LightGBM with optimal weights from Phase 4
- Strict temporal lag maintained: 2021 features use 2020 values via lag1 columns
- Expanding window stats (fire risk mean, exposure mean) computed from historical 2018-2020 data
- 2020 validation RMSE: 493,988 (ensemble), 518,222 (Panel FE), 861,907 (LightGBM)

## Task Commits

1. **Task 1: Create 2021 feature generation and prediction script** - `121cc4c` (feat)

**Plan metadata:** N/A (orchestrator owns final metadata commit)

## Files Created/Modified
- `Task2_Data/task2_step5_predictions.py` - 2021 predictions script with Panel FE + LightGBM ensemble

## Decisions Made
- Used optimal 80/20 ensemble weight from Phase 4 grid search (w=0.80 PanelFE)
- Created lag features from holdout's own 2020 columns (Avg Fire Risk Score, Earned Exposure, Earned Premium, etc.)
- Expanding window features computed from raw panel (task2_step1_panel_clean.csv) using 2018-2020 data
- COVID indicator set to 0 for 2021 (post-COVID year)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- **Panel FE predict requires MultiIndex format**: linearmodels predict() converts input to PanelData object and requires 2-level MultiIndex (entity, time). Fixed by creating X_2021_panel with set_index(['ZIP', 'Year']) before prediction.
- **Column alignment mismatch**: Panel FE model was trained on 9329 rows (2019 only) but predict was called on 12708 validation rows. Resolved by ensuring X_train_panel and X_val_panel use same column set.
- **Holdout column name**: actual_premium was stored as 'Earned Premium' in holdout CSV. Fixed by renaming during output creation.

## Next Phase Readiness
- Predictions CSV ready for error analysis (Plans 05-02 and 05-03)
- Predicted_premium, panel_fe_pred, lgb_pred all available for decomposition analysis
- 12,708 rows with predicted vs actual premium pairs available for MAPE/RMSE computation

---
*Phase: 05-2021-prediction-error-analysis*
*Completed: 2026-04-10*
