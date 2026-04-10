# Phase 5: 2021 Prediction + Error Analysis - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-10
**Phase:** 05-2021-prediction-error-analysis
**Areas discussed:** Prediction granularity + 2021 feature generation, Error analysis depth + reporting structure, Output format + documentation scope, Model comparison strategy

---

## Area 1: Prediction Granularity + 2021 Feature Generation

### Q1: What prediction granularity for 2021?
| Option | Description | Selected |
|--------|-------------|----------|
| ZIP × Category | One row per ZIP × Category (~11k predictions) | ✓ |
| ZIP level only | One row per ZIP (~1600 predictions), aggregate categories | |
| Both | Produce both granular and aggregated predictions | |

**User's choice:** ZIP × Category (Recommended)

### Q2: How should 2021 lag features be generated?
| Option | Description | Selected |
|--------|-------------|----------|
| Use 2020 features directly | Use 2020 row values from feature matrix as lag features | |
| Generate fresh 2020→2021 lags | Create new lag features using 2021 holdout's own lag columns | |
| Claude decides | Same lag logic as Phase 2 | ✓ |

**User's choice:** Claude decides (D-03: use 2020→2021 lag pattern, same as Phase 2)

### Q3: How should the 2021 holdout be used?
| Option | Description | Selected |
|--------|-------------|----------|
| Evaluation only | Use 2021 actuals ONLY for final evaluation | ✓ |
| Both feature + eval | Use for both feature generation and evaluation | |
| Blind prediction only | Predict 2021 blindly without using actuals | |

**User's choice:** Evaluation only (Recommended)

---

## Area 2: Error Analysis Depth + Reporting Structure

### Q1: What level of error analysis?
| Option | Description | Selected |
|--------|-------------|----------|
| Top-N errors + fire risk validation | Identify top 10-20 worst ZIPs and validate against fire risk distribution | ✓ |
| Segmented analysis | Compute errors by category and fire risk quintile | |
| Root cause analysis | Drill into which features contribute most to error | |

**User's choice:** Top-N errors + fire risk validation

### Q2: What error outputs to produce?
| Option | Description | Selected |
|--------|-------------|----------|
| High-error ZIP list + characteristics | Tabular top-N worst performing ZIPs | |
| Statistical distribution summary | Report percentiles (50th, 75th, 90th, 95th) | |
| Both | Both tabular list AND distribution percentiles | ✓ |

**User's choice:** Both (Recommended)

---

## Area 3: Output Format + Documentation Scope

### Q1: What should the primary output be?
| Option | Description | Selected |
|--------|-------------|----------|
| Predictions CSV | CSV with ZIP, Category, predicted_premium, actual, error, abs_error, fire_risk | ✓ |
| Metrics summary CSV | CSV with metrics (RMSE, MAE, MAPE per model) | |
| Both | Both predictions CSV and metrics summary | |

**User's choice:** Predictions CSV (Recommended)

### Q2: Documentation scope?
| Option | Description | Selected |
|--------|-------------|----------|
| Full documentation | All requirements: approach, hyperparameters, assumptions, Task 1 comparison, metrics | ✓ |
| Minimal (summary only) | Key metrics + fire risk value-add only | |
| Claude decides | Standard summary | |

**User's choice:** Full documentation (all requirements)

---

## Area 4: Model Comparison Strategy

### Q1: What model comparison approach?
| Option | Description | Selected |
|--------|-------------|----------|
| Ensemble vs individual models + vs actuals | Compare ensemble vs PanelFE vs LightGBM vs actuals | ✓ |
| Rank all models + use best | Compute metrics for each, rank, select best | |
| Ensemble only | Produce ensemble predictions only, no per-model comparison | |

**User's choice:** Ensemble vs individual models + vs actuals

---

## Claude's Discretion

- D-03: 2021 lag features generated using 2020→2021 lag logic — same pattern as Phase 2
- D-04: Expanding window stats for 2021 computed from all available prior years (2018-2020)

## Deferred Ideas

None — discussion stayed within Phase 5 scope.
