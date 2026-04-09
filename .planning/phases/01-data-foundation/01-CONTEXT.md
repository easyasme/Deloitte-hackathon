# Phase 1: Data Foundation - Context

**Gathered:** 2026-04-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a clean, validated panel dataset (Year × ZIP × Category) ready for feature engineering. Load the insurance + fire + census + weather CSV, enforce structure validation, resolve duplicates, and handle missing values — all without leaking future-year data into training features.
</domain>

<decisions>
## Implementation Decisions

### Missing Value Handling
- **D-01:** Keep NaN values as-is — do not impute
- **D-02:** Tree-based models (LightGBM, Random Forest) handle NaN natively; no imputation required
- **D-03:** No missing-value threshold enforced during loading; defer quality assessment to modeling phase
- **D-04:** No column-type-specific imputation strategies

### Temporal Integrity Checks
- **D-05:** Strict lag enforcement — all concurrent features must be lagged (no future data in predictors)
- **D-06:** Explicit validation: assert no 2021 data appears when building training features for any target year ≤ 2020
- **D-07:** Year column itself is NOT a direct predictor; temporal patterns emerge from lag features and year-indexed features only
- **D-08:** Document any detected leakage incidents in validation output

### Dataset Structure Validation
- **D-09:** Expected granularity: (Year, ZIP, Category) triple uniqueness enforced
- **D-10:** On duplicate triples: aggregate numeric columns by sum, keep first value for categorical
- **D-11:** Strict uniqueness check — if duplicates exist after aggregation, investigate and document
- **D-12:** Expected row counts: ~47k rows (3 years × ~1600 ZIPs × ~7 categories, with some categories absent per ZIP)

### Duplicate Resolution
- **D-13:** Duplicates arise from fire event joins per insurance record — one row per (Year, ZIP, Category)
- **D-14:** Aggregate strategy: sum numeric columns (losses, claims, exposure, premium, fire risk) across fire events for same insurance record
- **D-15:** Categorical columns (Category_*, FIRE_NAME, etc.): keep first value or mode per group
- **D-16:** After aggregation, assert uniqueness — log and abort if duplicate triples remain

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

- `Task2_Data/abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv` — Primary input dataset
- `Task2_Data/abfaw7bci2UF6CTg_FeatureDescription_fire_insurance.csv` — Feature descriptions for insurance data
- `.planning/ROADMAP.md` — Phase 1 goal, success criteria, and DATA-01/02/03 requirements
- `.planning/REQUIREMENTS.md` — DATA-01, DATA-02, DATA-03 acceptance criteria
- `.planning/STATE.md` — Project stack, architecture, and critical pitfalls

### Task 1 Reuse
- `Task1_Data/Task1_Weather/task1_step1_zip_year_ready.csv` — Reference for zip-year aggregation pattern (Phase 1 in Task 1)
- `Task1_Data/Task1_Weather/data_preprocessing.py` — Reference for loading and deduplication logic
- `Task1_Data/abfap7bci2UF6CTY_wildfire_weather.csv` — Raw wildfire weather data

</canonical_refs>

<codebase_context>
## Existing Code Insights

### Reusable Assets
- **Task 1 data preprocessing** (`Task1_Data/Task1_Weather/data_preprocessing.py`): Pattern for loading CSV, column selection, deduplication, zip-year aggregation — can be adapted for Task 2
- **Task 1 feature engineering** (`Task1_Data/Task1_Weather/classical_ml_feature_engineering.py`): Leakage removal logic — reference for temporal integrity checks

### Established Patterns
- **Sequential pipeline**: Each step reads from previous step's output CSV — Task 2 should follow same pattern
- **Step numbering**: `task2_step1_data_load.py`, `task2_step2_...` to match Task 1 conventions
- **Constants at top**: `INPUT_FILE`, `OUTPUT_FILE`, `RANDOM_STATE` — follow same style
- **Progress logging**: Print statements with section headers (`print("\n--- Step 1: Load ---")`)

### Integration Points
- **Output**: `task2_step1_panel_clean.csv` — feeds Phase 2 feature engineering
- **Validation output**: `task2_step1_validation_report.csv` — success criteria evidence

### Data Characteristics Discovered
- 47,033 raw rows, 76 columns
- Years: 2018, 2019, 2020, 2021 (2021 data present — must be excluded from training features)
- 2,251 unique ZIPs; ~8,288 unique Year-ZIP combos (multiple categories per ZIP-year)
- 7 insurance categories: HO, CO, DT, RT, DO, MH, NA (one-hot encoded as columns)
- Critical insurance columns: `Earned Premium`, `Earned Exposure`, `Avg Fire Risk Score`
- Weather columns with NaN: `avg_tmax_c`, `avg_tmin_c`, `tot_prcp_mm`

</code_context>

<specifics>
## Specific Ideas

- Use strict lag enforcement: when building features for year Y, only use data from years ≤ Y−1
- Validation report should explicitly list any 2021 rows detected in training context
- The `Year` column should not be used as a direct feature — temporal patterns come from lags and year-index derived features

</specifics>

<deferred>
## Deferred Ideas

None — all decisions stayed within Phase 1 scope.

</deferred>

---

*Phase: 01-data-foundation*
*Context gathered: 2026-04-09*
