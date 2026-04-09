# Phase 2: Feature Engineering - Context

**Gathered:** 2026-04-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a feature matrix with temporal lags, rolling statistics, and encodings ready for modeling. Load `task2_step1_panel_clean.csv` (2018-2020), build lag features and per-ZIP rolling statistics, one-hot encode categories, and add a COVID-19 year indicator. Output: `task2_step2_feature_matrix.csv` ready for Phase 3 modeling.
</domain>

<decisions>
## Implementation Decisions

### Target Variable
- **D-01:** Target is raw `Earned Premium` (not premium/exposure ratio)
- **D-02:** Zero-exposure rows (2,940 rows) are kept in the dataset — no special handling at feature engineering stage; Phase 3 models handle naturally or filter if needed

### Lag Strategy
- **D-03:** Single lag (t-1 only) — for target year Y, features come only from year Y-1
- **D-04:** 2020 target rows use 2019 features only (2019→2020 lag)
- **D-05:** 2019 target rows use 2018 features only (2018→2019 lag)
- **D-06:** 2018 rows have no lagged features (cannot lag from 2017); these rows serve as lag-source only, not as modeling targets

### Rolling Statistics
- **D-07:** Expanding window — for each ZIP, mean and std are computed across all available years up to the target year
- **D-08:** Expanding mean of `Avg Fire Risk Score` and `Earned Exposure` per ZIP — captures long-term risk and exposure history
- **D-09:** Expanding std of `Avg Fire Risk Score` per ZIP — captures volatility of fire risk over time

### Category Encoding
- **D-10:** One-hot encode the 6 insurance categories (HO, CO, DT, RT, DO, MH, NA) as 6 binary columns
- **D-11:** Category encoding happens before lag/rolling computation; category does not change over time for a given ZIP (no lag needed)

### COVID-19 Handling
- **D-12:** Add binary `is_covid_year` flag: 1 for Year == 2020, 0 otherwise
- **D-13:** COVID flag is a feature, not a target adjustment — models can learn differential 2020 behavior

### Feature Set (from ROADMAP.md requirements)
- **FEAT-01:** Temporal lag features (t-1) for fire risk, exposure, and key loss columns
- **FEAT-02:** Rolling statistics (expanding mean/std) per ZIP for fire risk and exposure
- **FEAT-03:** Category one-hot encoded (6 binary columns)
- **FEAT-04:** Premium normalized by earned exposure as target option (deferred — D-01 chose raw premium)
- **FEAT-05:** Zero-inflated claims distribution documented; Tweedie vs two-part deferred to Phase 3
- **FEAT-06:** `Avg Fire Risk Score` used as primary wildfire risk predictor (already in dataset)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

- `Task2_Data/task2_step1_panel_clean.csv` — Primary input (31,343 rows, 2018-2020)
- `Task2_Data/task2_step1_2021_holdout.csv` — 2021 holdout (12,708 rows)
- `Task2_Data/task2_step1_data_load.py` — Reference pattern for step-by-step pipeline
- `.planning/phases/01-data-foundation/01-CONTEXT.md` — Phase 1 decisions (NaN preserved, no imputation, temporal integrity)
- `.planning/phases/01-data-foundation/01-RESEARCH.md` — Data characteristics (column list, NaN distribution, duplicate structure)
- `.planning/ROADMAP.md` — Phase 2 goal, success criteria, and FEAT-01 through FEAT-06 requirements
- `.planning/REQUIREMENTS.md` — FEAT-01 through FEAT-06 acceptance criteria
- `.planning/STATE.md` — Project stack (LightGBM + statsmodels ensemble), key decisions (COVID structural break handling)

### Task 1 Reuse
- `Task1_Data/Task1_Weather/classical_ml_feature_engineering.py` — Leakage removal and feature engineering patterns

</canonical_refs>

<codebase_context>
## Existing Code Insights

### Reusable Assets
- **Task 2 Step 1 script** (`task2_step1_data_load.py`): Sequential pipeline with STEP 1-7 comments, constants at top, assertions for critical checks
- **Task 1 feature engineering** (`Task1_Data/Task1_Weather/classical_ml_feature_engineering.py`): Lag computation and leakage removal patterns

### Established Patterns
- **Sequential pipeline**: Each step reads from previous step's output CSV
- **Step numbering**: `task2_step2_feature_engineering.py` to match Task 2 conventions
- **Constants at top**: `INPUT_FILE`, `OUTPUT_FILE` style
- **Progress logging**: Print statements with section headers
- **Assertions**: Critical checks that abort if invariants are violated

### Integration Points
- **Input**: `task2_step1_panel_clean.csv` (2018-2020, 31,343 rows)
- **Output**: `task2_step2_feature_matrix.csv` — feeds Phase 3 baseline models
- **COVID flag**: `is_covid_year` column added to feature matrix

### Data Characteristics
- 31,343 training rows, 49 columns
- Years: 2018 (9,306), 2019 (9,329), 2020 (12,708)
- 6 categories: HO (5,841), RT (5,683), DT (5,604), DO (5,517), CO (4,370), MH (4,328)
- 2,940 zero-exposure rows (kept in dataset per D-02)
- `Avg Fire Risk Score`: mean=0.905, std=1.250, range=[-0.21, 42.0]

</codebase_context>

<specifics>
## Specific Ideas

- 2018 rows cannot produce lag features (no 2017 data) — they serve only as lag source for 2019
- COVID flag is a simple 0/1 binary column, not a weighted sample or masked observation
- Zero-exposure rows: kept but noted — Phase 3 models may need to filter these for premium/exposure calculations

</specifics>

<deferred>
## Deferred Ideas

None — all decisions stayed within Phase 2 scope.

</deferred>

---

*Phase: 02-feature-engineering*
*Context gathered: 2026-04-09*
