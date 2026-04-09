# Phase 3: Baseline Models + Temporal Validation - Context

**Gathered:** 2026-04-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish validated temporal split and baseline metrics. Train Linear Regression and Random Forest on `task2_step2_feature_matrix.csv` (2019 train, 2020 validation), report RMSE/MAE/MAPE on 2020 holdout, and document COVID-19 anomaly. Output: baseline metrics, predictions, and validation report ready for Phase 4 ensemble.
</domain>

<decisions>
## Implementation Decisions

### Zero-Exposure Handling
- **D-01:** Include ALL rows in training and evaluation — no filtering of zero-exposure rows
- **D-02:** Zero-exposure rows produce zero or near-zero premium naturally; both Linear Reg and Random Forest handle them without special treatment

### Linear Regression Variant
- **D-03:** Use vanilla OLS (`sklearn.linear_model.LinearRegression`) — no regularization
- **D-04:** Rationale: Simple interpretable baseline. Ridge/Lasso deferred to Phase 4 if overfitting observed.

### Random Forest Tuning
- **D-05:** Use sensible defaults: `n_estimators=100`, `max_depth=10-15`, `min_samples_leaf=5`
- **D-06:** Rationale: Quick to run, reasonable performance. LightGBM in Phase 4 will be the primary tree model.

### MAPE Robustness
- **D-07:** Report MAPE on non-zero premium subset only (`Earned Premium > 0`)
- **D-08:** Zero-premium rows excluded from MAPE denominator; RMSE and MAE computed on full validation set
- **D-09:** Document zero-premium row count and percentage in validation report

### Feature Set
- **D-10:** "Same features" = all lag features + rolling stats + COVID flag + category one-hot + demographics + fire risk, excluding Year and non-feature identifiers (ZIP, Category, ZIP_Cat, FIRE_NAME, AGENCY, INC_NUM)
- **D-11:** Exact feature list determined by planner from `task2_step2_feature_matrix.csv` columns

### Temporal Split (BASE-01)
- **D-12:** Train: 2019 rows from feature matrix (2018 served as lag source only)
- **D-13:** Validation: 2020 rows from feature matrix
- **D-14:** No random split — strictly temporal

### COVID-19 Documentation (BASE-04)
- **D-15:** `is_covid_year` flag is already a feature in the matrix; baseline models will learn its coefficient
- **D-16:** Validation report includes: COVID coefficient estimate, 2020 vs 2019 residual analysis, explicit note that 2020 may have elevated error

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Core Data
- `Task2_Data/task2_step2_feature_matrix.csv` — Primary input (Phase 2 output, 20,037 rows, 49 columns)
- `Task2_Data/task2_step1_2021_holdout.csv` — 2021 holdout (used in Phase 5, not Phase 3)

### Prior Phase Contexts
- `.planning/phases/01-data-foundation/01-CONTEXT.md` — NaN handling, temporal integrity, deduplication decisions
- `.planning/phases/02-feature-engineering/02-CONTEXT.md` — Target = raw Earned Premium, lag strategy, COVID flag, zero-exposure kept
- `.planning/STATE.md` — Stack: LightGBM + statsmodels ensemble, COVID structural break flagged
- `.planning/ROADMAP.md` — Phase 3 goal, BASE-01 through BASE-04 success criteria
- `.planning/REQUIREMENTS.md` — BASE-01, BASE-02, BASE-03, BASE-04 acceptance criteria

### Prior Research
- `.planning/phases/02-feature-engineering/02-RESEARCH.md` — Feature matrix characteristics (column list, NaN distribution)
- `.planning/phases/01-data-foundation/01-RESEARCH.md` — Data characteristics (column list, NaN distribution, duplicate structure)

### Task 1 Reference Code
- `Task1_Data/Task1_Weather/classical_ml.py` — `evaluate_model()` pattern: fit → predict → evaluate → save metrics

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **evaluate_model() pattern** from `Task1_Data/Task1_Weather/classical_ml.py`: fit/predict/evaluate loop with metric computation and CSV output — adapt for Phase 3
- **Pipeline structure** from `Task2_Data/task2_step2_feature_engineering.py`: constants at top, STEP comments, assertions, print logging

### Established Patterns
- **Sequential pipeline**: `task2_step3_baseline_models.py` reads `task2_step2_feature_matrix.csv`
- **Step numbering**: `task2_step3_...` to match Task 2 conventions
- **Output naming**: `task2_step3_*_predictions.csv`, `task2_step3_*_metrics.csv`
- **Metrics**: RMSE, MAE, MAPE (on non-zero subset) — output as CSV

### Integration Points
- **Input**: `task2_step2_feature_matrix.csv` (2019 + 2020 rows)
- **Train rows**: Year == 2019
- **Validation rows**: Year == 2020
- **Output feeds Phase 4**: Metrics CSV and predictions CSV used for ensemble comparison

### Data Characteristics
- 20,037 total rows (2019 + 2020)
- 2018 rows removed at Phase 2 step (serve only as lag source)
- 2,940 zero-exposure rows included (D-01 decision)
- 49 columns including lag features, expanding stats, COVID flag, one-hot categories, demographics
- `is_covid_year`: 0 for 2019, 1 for 2020

</code_context>

<specifics>
## Specific Ideas

- Zero-exposure rows: no special handling — both models include all rows
- MAPE denominator excludes zero-premium rows — document count in validation report
- COVID coefficient from Linear Regression provides interpretable estimate of 2020 anomaly magnitude
- Random Forest feature importance gives intuition for Phase 4 feature selection

</specifics>

<deferred>
## Deferred Ideas

None — all decisions stayed within Phase 3 scope.

</deferred>

---

*Phase: 03-baseline-models-temporal-validation*
*Context gathered: 2026-04-09*
