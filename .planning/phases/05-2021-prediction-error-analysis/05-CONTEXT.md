# Phase 5: 2021 Prediction + Error Analysis - Context

**Gathered:** 2026-04-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate 2021 Earned Premium predictions for all ZIP codes × Category combinations using the Phase 4 ensemble model (80% Panel FE + 20% LightGBM), evaluate predictions against 2021 holdout actuals, perform error analysis identifying top-error ZIPs and validating against fire risk distribution, and produce full documentation per requirements PRED-01 through PRED-04 and DOC-01 through DOC-03.
</domain>

<decisions>
## Implementation Decisions

### Prediction Granularity
- **D-01:** Prediction granularity: ZIP × Category — one row per ZIP × Category (~11k predictions across ~1600 ZIPs and 7 categories)
- **D-02:** Matches data structure exactly; enables per-category analysis

### 2021 Feature Generation (Claude's Discretion)
- **D-03:** 2021 lag features generated using 2020→2021 lag logic — same pattern as Phase 2 (2020 row values used as t-1 features for 2021 predictions)
- **D-04:** Expanding window stats for 2021 computed from all available prior years (2018-2020)

### Holdout Usage
- **D-05:** 2021 holdout used ONLY for evaluation (computing RMSE/MAE/MAPE against predictions)
- **D-06:** No 2021 actuals used in feature generation — strict temporal lag maintained
- **D-07:** Evaluation performed after predictions are generated

### Error Analysis Depth
- **D-08:** Top-N error analysis: identify top 10-20 highest-error ZIP codes by absolute error
- **D-09:** Validate high-error ZIPs against fire risk distribution — do high-error ZIPs correlate with high fire risk?
- **D-10:** Error outputs: both tabular high-error list AND statistical distribution percentiles (50th, 75th, 90th, 95th for RMSE and MAE)

### Model Comparison
- **D-11:** Compare ensemble predictions vs 2021 actuals AND vs each individual model (PanelFE, LightGBM)
- **D-12:** Demonstrate that ensemble outperforms individual models
- **D-13:** All comparison metrics reported: RMSE, MAE, MAPE per model

### Output Format
- **D-14:** Primary output: predictions CSV with columns ZIP, Category, predicted_premium, actual_premium (if available), error, abs_error, fire_risk_score
- **D-15:** Metrics summary CSV alongside predictions CSV
- **D-16:** Error analysis CSV: top-error ZIPs with their characteristics

### Documentation Scope (Full per all requirements)
- **D-17:** Model approach, hyperparameters, and assumptions documented
- **D-18:** Performance metrics (RMSE, MAE, MAPE) reported for all models
- **D-19:** Comparison with classical approaches discussed (vs quantum/Task 1)
- **D-20:** Fire risk feature value-add analysis (compare with/without fire risk)

### COVID-19 / 2020 Structural Break
- **D-21:** 2021 prediction does NOT use the COVID flag (2021 is not a COVID year)
- **D-22:** Models trained on 2019+2020 with COVID flag remain valid; 2021 predictions use flag=0 implicitly

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Core Data
- `Task2_Data/task2_step1_2021_holdout.csv` — 2021 holdout (12,708 rows) for prediction and evaluation
- `Task2_Data/task2_step2_feature_matrix.csv` — Feature matrix (2018-2020) for training and rolling stats computation
- `Task2_Data/task2_step4_predictions.csv` — Phase 4 predictions on 2020 validation for reference

### Prior Phase Contexts
- `.planning/phases/01-data-foundation/01-CONTEXT.md` — NaN handling, temporal integrity, deduplication
- `.planning/phases/02-feature-engineering/02-CONTEXT.md` — Lag strategy, target = raw Earned Premium, COVID flag
- `.planning/phases/03-baseline-models-temporal-validation/03-CONTEXT.md` — Baseline metrics (LR RMSE=515,830, RF feature importance)
- `.planning/STATE.md` — Stack: Panel FE + LightGBM ensemble, optimal weight 0.80/0.20

### Phase 4 Outputs
- `.planning/phases/04-model-development-ensemble/04-03-SUMMARY.md` — Ensemble weight search results (optimal w=0.80)
- `.planning/phases/04-model-development-ensemble/04-03-PLAN.md` — Ensemble approach details
- `Task2_Data/task2_step4_weight_search.csv` — 21 weight candidates with RMSE values

### Requirements & Roadmap
- `.planning/ROADMAP.md` — Phase 5 goal, success criteria, PRED-01 through PRED-04, DOC-01 through DOC-03
- `.planning/REQUIREMENTS.md` — PRED/DOC requirements traceability

### Code Reference
- `Task2_Data/task2_step4_ensemble.py` — Ensemble weight search and prediction pattern
- `Task2_Data/task2_step3_baseline_models.py` — evaluate_model() pattern for metrics computation

</canonical_refs>

<codebase_context>
## Existing Code Insights

### Reusable Assets
- **evaluate_model() pattern** from `Task2_Data/task2_step3_baseline_models.py`: fit/predict/evaluate loop with metric computation and CSV output
- **Ensemble weight search** from `Task2_Data/task2_step4_ensemble.py`: grid search over weight values, optimal w=0.80
- **Panel FE model** from `Task2_Data/task2_step4_panel_fe.py`: statsmodels PanelOLS with entity effects
- **LightGBM model** from `Task2_Data/task2_step4_lightgbm.py`: inline training and prediction

### Established Patterns
- **Sequential pipeline**: `task2_step5_*` scripts follow same conventions
- **Step numbering**: `task2_step5_2021_predictions.py`, `task2_step5_error_analysis.py`
- **Constants at top**: `INPUT_FILE`, `OUTPUT_FILE`, `RANDOM_STATE` style
- **Progress logging**: Print statements with section headers

### Integration Points
- **Input**: task2_step2_feature_matrix.csv (training), task2_step1_2021_holdout.csv (prediction target)
- **Phase 4 models**: Panel FE and LightGBM models need to be re-loaded/retrained for 2021 predictions
- **Output**: task2_step5_predictions.csv, task2_step5_metrics.csv, task2_step5_error_analysis.csv

### Data Characteristics
- 2021 holdout: 12,708 rows (ZIP × Category × Year == 2021)
- Phase 4 ensemble: w=0.80 PanelFE + w=0.20 LightGBM, RMSE=493,988 on 2020 validation
- MAPE computed on non-zero premium subset only

</code_context>

<specifics>
## Specific Ideas

- 2021 lag features: use 2020 row values from feature matrix directly (no 2021 features available for lag)
- Expanding window stats for 2021 computed from 2018, 2019, 2020 history (same pattern as Phase 2)
- Top 10-20 error ZIPs should be validated against their fire risk scores — are high-error ZIPs also high-fire-risk ZIPs?
- Fire risk value-add: compare ensemble RMSE with fire risk vs without (ablation)
- Comparison with Task 1: quantum vs classical approach tradeoffs documented per DOC-03

</specifics>

<deferred>
## Deferred Ideas

None — all decisions stayed within Phase 5 scope.

</deferred>

---

*Phase: 05-2021-prediction-error-analysis*
*Context gathered: 2026-04-10*
