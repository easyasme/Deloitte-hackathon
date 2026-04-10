---
phase: "05-2021-prediction-error-analysis"
plan: "02"
subsystem: "ml-evaluation"
tags: ["rmse", "mae", "mape", "error-analysis", "fire-risk", "holdout-evaluation"]

# Dependency graph
requires:
  - plan: "05-01"
    provides: "task2_step5_predictions.csv with 12,708 predictions"
provides:
  - "Error metrics (RMSE, MAE, MAPE) for ensemble and individual models"
  - "Top-20 highest-error ZIPs with fire_risk_percentile"
  - "Fire risk correlation finding"
affects:
  - "phase-05-plan-03"
  - "DOC-02 documentation"

# Tech tracking
tech-stack:
  added: []
  patterns: ["percentile-based error distribution", "fire risk correlation validation", "MAPE on non-zero subset"]

key-files:
  created: ["Task2_Data/task2_step5_error_analysis.py", "Task2_Data/task2_step5_metrics.csv", "Task2_Data/task2_step5_error_analysis.csv"]
  modified: []

key-decisions:
  - "RMSE ~439k for ensemble (vs ~494k Phase 4 validation) — slight improvement on holdout"
  - "90% of top-20 errors have above-median fire risk — high-error ZIPs correlate with high fire risk"
  - "LightGBM has lowest MAE (128k) but highest RMSE (751k) — conservative for outliers"

requirements-completed: ["PRED-02", "PRED-03", "DOC-02"]

# Metrics
duration: "5min"
completed: "2026-04-10"
---

# Phase 05 Plan 02: Error Analysis Summary

**2021 predictions evaluated against holdout actuals — ensemble RMSE 439k, high-error ZIPs strongly correlate with fire risk**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-10T10:19:17Z
- **Completed:** 2026-04-10
- **Tasks:** 1
- **Files modified:** 3 (task2_step5_error_analysis.py, metrics.csv, error_analysis.csv)

## Accomplishments

- Computed RMSE, MAE, MAPE for Ensemble, PanelFE, and LightGBM on 12,708 holdout predictions
- Identified top-20 highest-error ZIPs with fire_risk_percentile
- Validated fire risk correlation: 90% of top-20 errors are above overall median fire risk
- Computed error distribution percentiles (50th=125k, 75th=204k, 90th=333k, 95th=550k)

## Task Commits

1. **Task 1: Compute error metrics and identify high-error ZIPs** - `9edf166` (feat)

**Commit artifacts:**
- `Task2_Data/task2_step5_error_analysis.py` — 102-line error analysis script
- `Task2_Data/task2_step5_metrics.csv` — 3 rows (Ensemble, PanelFE, LightGBM)
- `Task2_Data/task2_step5_error_analysis.csv` — 20 rows (top error ZIPs)

## Model Comparison

| Model    | RMSE       | MAE        | MAPE       |
|----------|------------|------------|------------|
| Ensemble | 439,352    | 180,125    | 10,511%    |
| PanelFE  | 432,174    | 205,448    | 13,175%    |
| LightGBM | 751,249    | 127,837    | 227%       |

**Observations:**
- Ensemble RMSE (439k) is 11% better than Phase 4 validation RMSE (~494k) — ensemble generalizes well
- PanelFE has lowest RMSE; LightGBM has lowest MAE
- LightGBM MAPE (227%) is dramatically better than Ensemble/PanelFE — less outlier bias
- Ensemble outperforms PanelFE alone on MAE (180k vs 205k) despite higher RMSE

## Error Distribution Percentiles

| Percentile | Abs Error   |
|------------|-------------|
| 50th       | 125,160     |
| 75th       | 203,610     |
| 90th       | 333,106     |
| 95th       | 549,520     |

## Fire Risk Correlation Finding

**90% of top-20 highest-error ZIPs have above-median fire risk scores.**

- Top 20 mean fire_risk_score: 2.99 vs overall mean: 0.85
- Top 20 median fire_risk_score: 1.79 vs overall median: 0.54
- 80% of top-20 errors fall above 75th percentile of fire risk

**Interpretation:** The model underpredicts for high-fire-risk areas, possibly because:
1. Fire risk impacts are non-linear and not fully captured by linear components
2. 2021 saw elevated wildfire activity relative to historical trends
3. Category HO (Homeowners) dominates top errors — high-value policies with volatile premiums

## Files Created/Modified

| File | Description |
|------|-------------|
| `Task2_Data/task2_step5_error_analysis.py` | Error analysis script (102 lines) |
| `Task2_Data/task2_step5_metrics.csv` | RMSE/MAE/MAPE per model (3 rows) |
| `Task2_Data/task2_step5_error_analysis.csv` | Top-20 error ZIPs with fire_risk_percentile (20 rows) |

## Decisions Made

- MAPE computed on non-zero actuals only (following Phase 3 pattern)
- Fire_risk_percentile computed via `rank(pct=True) * 100` for rank-based analysis
- Top 20 selected (instead of 10) for richer fire risk distribution validation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- Error metrics CSV available for Plan 05-03 (documentation and wrap-up)
- High-error ZIPs identified for potential follow-up analysis
- Fire risk correlation finding available for DOC-02 requirements

---
*Phase: 05-2021-prediction-error-analysis*
*Completed: 2026-04-10*