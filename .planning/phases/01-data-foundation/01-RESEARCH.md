# Phase 1: Data Foundation - Research

**Researched:** 2026-04-09
**Domain:** pandas CSV loading, panel data deduplication, temporal integrity validation
**Confidence:** HIGH

## Summary

Phase 1 requires loading a 47,033-row insurance panel dataset and producing a clean (Year × ZIP × Category) panel where duplicate fire-event-joined records are aggregated and 2021 data is handled separately from the 2018-2020 training corpus. The key technical challenges are: (1) deduplicating on (Year, ZIP, Category) where multiple fire events create multiple rows per insurance record, (2) preserving NaN as NaN for tree-model compatibility rather than imputing, and (3) strictly separating 2021 rows so no future-year data contaminates training features. The panel granularity is approximately 8,288 unique (Year, ZIP) pairs, each containing up to 7 category rows, but actual row count after dedup will differ.

**Primary recommendation:** Use `task2_step1_data_load.py` following the Task 1 `data_preprocessing.py` pattern: load with `low_memory=False`, deduplicate (Year, ZIP, Category) by summing numeric columns (losses, claims, premiums, fire risk scores) and taking first/mode for categoricals (FIRE_NAME, AGENCY), assert strict uniqueness after aggregation, and split the 2021 rows into a separate holdout immediately after loading.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 through D-04:** NaN values kept as-is — no imputation; tree models handle natively
- **D-05 through D-08:** Strict lag enforcement; no 2021 data in training features; Year column not a direct predictor
- **D-09 through D-12:** (Year, ZIP, Category) triple uniqueness enforced; duplicates aggregated by sum for numerics, first/mode for categoricals; strict uniqueness check after aggregation; expected ~47k rows
- **D-13 through D-16:** Duplicates arise from fire event joins; aggregate by summing numerics, mode for categoricals; assert uniqueness after aggregation

### Claude's Discretion

- Script file naming convention (task2_step1_*.py vs task1 pattern)
- Specific progress logging verbosity
- How to structure the validation report output
- Exact column selection list (beyond critical columns)

### Deferred Ideas

None — all decisions stayed within Phase 1 scope.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | Load and explore insurance dataset (2018-2020) — verify structure, row counts, zip code coverage | Section 1: Data loading with `low_memory=False` handles mixed types; verified 47,033 rows, 76 cols, 2,251 ZIPs, 4 years present |
| DATA-02 | Clean dataset — handle missing values, validate data types, remove rows with critical missing fields | Section 2: NaN kept as-is per D-01; critical insurance cols have 0% missing; no row removal required for training data |
| DATA-03 | Validate temporal split integrity — ensure no data leakage from future years into predictors | Section 4: 2021 data (13,344 rows) must be separated; Year 2021 has fire events (1,974 rows) confirming it is real data, not just a prediction target |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | latest (pip resolved) | CSV loading, deduplication, panel aggregation | Primary data manipulation library per CLAUDE.md |
| numpy | latest (pip resolved) | Numeric aggregation fallback | Universal dependency for pandas |

**Installation:**
```bash
pip install pandas numpy
```

### No Additional Libraries Required
Phase 1 is pure pandas/numpy data manipulation — no ML libraries (scikit-learn, statsmodels) needed until Phase 3.

## Architecture Patterns

### Recommended Project Structure
```
Task2_Data/
├── abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv  # Raw input
└── task2_step1_panel_clean.csv                              # Phase 1 output
    task2_step1_validation_report.csv                        # Success criteria evidence
```

### Pattern 1: Sequential Pipeline with Constants
Following Task 1 conventions:
```python
INPUT_FILE = "abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv"
OUTPUT_FILE = "task2_step1_panel_clean.csv"
VALIDATION_OUT = "task2_step1_validation_report.csv"

# 1. Load with low_memory=False
df = pd.read_csv(INPUT_FILE, low_memory=False)

# 2. Basic type cleaning for numerics
# 3. Deduplicate on (Year, ZIP, Category) — sum numerics, mode categoricals
# 4. Assert uniqueness
# 5. Separate 2021 holdout
# 6. Validate temporal integrity
# 7. Save
```

### Pattern 2: Panel Deduplication by Aggregation
**Source:** Verified from Task 1 `data_preprocessing.py` lines 57-65, adapted for Task 2

```python
# Identify Category from one-hot columns
cat_cols = ['Category_CO', 'Category_DO', 'Category_DT', 'Category_HO', 'Category_MH', 'Category_RT']

# Numeric columns to SUM on deduplication (losses, claims, premiums, risk scores)
numeric_sum_cols = [
    'Avg Fire Risk Score', 'Earned Premium', 'Earned Exposure',
    'CAT Cov A Fire - Incurred Losses', 'CAT Cov A Fire - Number of Claims',
    'CAT Cov A Smoke - Incurred Losses', 'CAT Cov A Smoke - Number of Claims',
    'CAT Cov C Fire - Incurred Losses', 'CAT Cov C Fire - Number of Claims',
    'CAT Cov C Smoke - Incurred Losses', 'CAT Cov C Smoke - Number of Claims',
    'Non-CAT Cov A Fire - Incurred Losses', 'Non-CAT Cov A Fire - Number of Claims',
    'Non-CAT Cov A Smoke - Incurred Losses', 'Non-CAT Cov A Smoke - Number of Claims',
    'Non-CAT Cov C Fire - Incurred Losses', 'Non-CAT Cov C Fire - Number of Claims',
    'Non-CAT Cov C Smoke - Incurred Losses', 'Non-CAT Cov C Smoke - Number of Claims',
    'Cov A Amount Weighted Avg', 'Cov C Amount Weighted Avg',
    'Number of High Fire Risk Exposure', 'Number of Low Fire Risk Exposure',
    'Number of Moderate Fire Risk Exposure', 'Number of Negligible Fire Risk Exposure',
    'Number of Very High Fire Risk Exposure',
    'avg_tmax_c', 'avg_tmin_c', 'tot_prcp_mm',
    'total_population', 'median_income', 'total_housing_units',
    'average_household_size', 'educational_attainment_bachelor_or_higher',
    'poverty_status', 'housing_occupancy_number', 'housing_value',
    'year_structure_built', 'housing_vacancy_number', 'median_monthly_housing_costs',
    'owner_occupied_housing_units', 'renter_occupied_housing_units',
]

# Categorical/text columns to keep FIRST (fire event identifiers)
categorical_first_cols = ['FIRE_NAME', 'AGENCY', 'INC_NUM']

# Deduplicate by (Year, ZIP, Category) — first collapse one-hot to single Category
def one_hot_to_category(df, cat_cols):
    """Convert one-hot category columns to single Category string column."""
    df = df.copy()
    df['Category'] = None
    for col in cat_cols:
        mask = df[col] == True
        df.loc[mask, 'Category'] = col.replace('Category_', '')
    return df

# Aggregation
agg_dict = {col: 'sum' for col in numeric_sum_cols if col in df.columns}
for col in categorical_first_cols:
    if col in df.columns:
        agg_dict[col] = 'first'

panel = (
    df
    .pipe(one_hot_to_category, cat_cols)
    .groupby(['Year', 'ZIP', 'Category'], as_index=False, dropna=False)
    .agg(agg_dict)
)

# Assert uniqueness
assert not panel.duplicated(subset=['Year', 'ZIP', 'Category']).any(), "Duplicate triples remain after aggregation"
```

### Pattern 3: Temporal Split by Year Filtering
```python
# Split 2021 holdout immediately after deduplication
train_panel = panel[panel['Year'] <= 2020].copy()
holdout_2021 = panel[panel['Year'] == 2021].copy()

# Temporal integrity assertion: when building training features for year Y,
# only use data from years <= Y-1
# This is enforced in Phase 2 (FEAT-01), but Phase 1 validates Year column distribution
assert train_panel['Year'].max() == 2020, "Training data contains future years"
assert holdout_2021['Year'].nunique() == 1 and holdout_2021['Year'].iloc[0] == 2021
```

## Common Pitfalls

### Pitfall 1: Mixed-Type CSV Columns Cause Silent Type Coercion
**What goes wrong:** `pd.read_csv()` infers column types row-by-row; mixed types (e.g., numeric + string in same column) become `object` dtype or cause unpredictable coercion.
**Why it happens:** The dataset has fire event columns (AGENCY, FIRE_NAME) that are mostly empty (83% NaN) but non-empty rows contain free text. Weather columns are 83% NaN but present as floats when non-null.
**How to avoid:** Use `low_memory=False` and explicit `dtype` specification for known numeric columns. Validate with `df.dtypes` after loading and `df[列].apply(type).nunique()` for suspect columns.
**Warning signs:** `dtype` changes after operations, `pd.to_numeric` fails with `errors='coerce'`, unexpected `object` dtype on numeric columns.

### Pitfall 2: Treating 2021 Rows as Missing/Filtered Rather Than Explicit Holdout
**What goes wrong:** The dataset contains 13,344 rows for 2021 (including 1,974 with fire events). If 2021 is silently filtered or treated as NaN, it corrupts the panel structure.
**Why it happens:** Phase 1 loads 2018-2021 but Phase 2-4 only train on 2018-2020. Without explicit separation, 2021 rows could leak into aggregated features.
**How to avoid:** Explicitly split into `train_panel` (<=2020) and `holdout_2021` (=2021) immediately after deduplication. Document row counts in validation report.
**Warning signs:** `Year.value_counts()` doesn't include 2021, or 2021 rows found in training data context.

### Pitfall 3: Incomplete Column Selection Drops Critical Features
**What goes wrong:** Many columns (fire event IDs, dates, GIS data) are 83%+ NaN. An over-aggressive column selection might drop `Avg Fire Risk Score` or `Earned Exposure`.
**Why it happens:** The naive approach is to drop columns with high NaN, but `Avg Fire Risk Score` has 0% missing — it's always populated at the insurance record level.
**How to avoid:** Define critical columns that must be preserved: `['Year', 'ZIP', 'Category', 'Avg Fire Risk Score', 'Earned Premium', 'Earned Exposure', 'Avg PPC']`. Only drop columns that are 100% NaN or are purely administrative (FIRE_NUM, COMMENTS, COMPLEX_ID).
**Warning signs:** After dropping high-NaN columns, critical insurance metrics disappear.

### Pitfall 4: Aggregation Losing the Category Dimension
**What goes wrong:** Grouping by (Year, ZIP) without Category, then trying to recover category information from one-hot columns, produces wrong granularity.
**Why it happens:** The one-hot Category columns (Category_HO, etc.) are True/False values. A simple `groupby(['Year', 'ZIP']).sum()` would add these as integers (1s and 0s), losing which category each row belonged to.
**How to avoid:** Convert one-hot to a single `Category` string column first, then groupby (Year, ZIP, Category). The one-hot columns can be kept or dropped after conversion.
**Warning signs:** Row count after aggregation is much lower than expected (should be ~8,288 Year-ZIP pairs × categories).

### Pitfall 5: Fire Event Join Producing Row Multiplication
**What goes wrong:** Each insurance record can join to multiple fire events, multiplying rows. Without deduplication, models see the same insurance record multiple times.
**Why it happens:** The raw data has 7948 fire event rows, but 6554 duplicates on (Year, ZIP, OBJECTID). A single insurance record at (Year=2019, ZIP=90003, Category=HO) can appear 3-5 times if 3-5 fire events occurred at that location that year.
**How to avoid:** The deduplication by (Year, ZIP, Category) with SUM aggregation correctly collapses multiple fire events into one insurance record per category.
**Warning signs:** Row count increases after merging fire event data; duplicate detection shows hundreds of duplicates per category.

## Data Characteristics (Verified from CSV)

| Property | Value | Source |
|----------|-------|--------|
| Total rows | 47,033 | `wc -l` verification |
| Columns | 76 | Header inspection |
| Year distribution | 2018: 10,071, 2019: 9,818, 2020: 13,800, 2021: 13,344 | pandas value_counts |
| Unique ZIPs | 2,251 | pandas nunique |
| Unique Year-ZIP combos | 8,288 | pandas groupby size |
| Category encoding | One-hot (Category_HO/CO/DT/RT/DO/MH) | Column inspection |
| Fire event rows | 7,948 (of 47,033) | OBJECTID notna count |
| 2021 with fire events | 1,974 | OBJECTID notna in 2021 subset |

**Critical columns (0% missing):** Year, ZIP, Avg Fire Risk Score, Earned Premium, Earned Exposure, Cov A Amount Weighted Avg, Cov C Amount Weighted Avg, all Category_*, all fire loss and claims columns

**High-NaN columns (droppable):** FIRE_NUM (100%), COMPLEX_ID (99.9%), COMPLEX_NA (99.8%), COMMENTS (95.7%), fire event identifiers (83% NaN — AGENCY, INC_NUM, ALARM_DATE, etc.)

**Medium-NaN columns (keep, tree models handle NaN):** avg_tmax_c, avg_tmin_c, tot_prcp_mm (83% NaN), median_income, housing_value (17-18% NaN), census columns (10% NaN)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Drop rows with any NaN | Keep NaN as-is; tree models handle natively | D-01 (this project) | Preserves data; avoids imputation bias |
| One-hot as separate binary columns | Collapse to single Category string for groupby | D-09-D-12 (this project) | Correct panel granularity |
| Remove high-NaN columns entirely | Keep if critical columns present | D-03 (this project) | Weather and census available when present |
| Filter 2021 out silently | Explicit train/holdout split with documentation | D-06 (this project) | Clear temporal boundary |

**No deprecated approaches relevant to this phase.**

## Assumptions Log

> List all claims tagged `[ASSUMED]` in this research. Planner and discuss-phase use this to identify decisions needing user confirmation.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The 2021 rows are holdout/test data (not raw inputs for training) | Section 4, Temporal Integrity | If 2021 is actually meant to be included in training for some features, the entire temporal split logic would be wrong |
| A2 | Aggregation strategy (sum for numerics, first for categoricals) correctly represents the data semantics | Pattern 2 | If fire losses should be averaged rather than summed, premium predictions would be systematically biased |
| A3 | All Category one-hot columns use True/False (Python bool), not the strings "True"/"False" | Pattern 2 | String "True" would cause groupby to fail or produce wrong categories |

**If any assumption in this table is wrong, the corresponding section must be revised before planning.**

## Open Questions

1. **Should 2021 fire event data be used as features when predicting 2021 premiums?**
   - What we know: 2021 has 1,974 fire event rows and 13,344 total rows. Avg Fire Risk Score is always present.
   - What's unclear: Does "predict 2021 premiums" mean using all available 2021 data as features, or only lagged (2018-2020) features?
   - Recommendation: Use lagged fire risk from 2018-2020 as primary features. If 2021 fire risk is available at prediction time, it could be used as a scenario input — but training must use only <=2020 data.

2. **What is the exact expected row count after deduplication?**
   - What we know: 47,033 raw rows; 8,288 Year-ZIP combos; ~5.67 category rows per Year-ZIP on average; significant duplicates on (Year, ZIP, Category).
   - What's unclear: After aggregation, expected row count could range from ~8,288 (if one category per Year-ZIP) to ~58,000 (if all 7 categories present per Year-ZIP). The CONTEXT says "approximately 47k rows" but that's the raw count.
   - Recommendation: Report actual post-aggregation row count in validation report; do not enforce a fixed expected count.

3. **Do weather columns (avg_tmax_c, avg_tmin_c, tot_prcp_mm) have data for non-fire-event rows?**
   - What we know: Weather columns are 83% NaN overall, matching the fire event row proportion. When fire event data is present, weather is also present.
   - What's unclear: Is weather data missing for insurance-only records (no fire event that year), or is it missing because the join didn't bring it in?
   - Recommendation: If weather is structurally missing for non-fire-event records, Phase 2 needs a strategy — either fill with ZIP-level median or treat as a separate "no fire event" indicator.

## Environment Availability

Step 2.6: SKIPPED (no external dependencies identified — pure pandas/numpy data loading phase).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | none detected |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Dataset loads with 47k rows, 76 columns, 2251 ZIPs | unit | `python -c "import pandas as pd; df=pd.read_csv('...'); assert len(df)==47033"` | NO |
| DATA-01 | Year distribution: 2018-2021 with expected counts | unit | `python -c "df['Year'].value_counts().to_dict()" | grep 2018` | NO |
| DATA-02 | No critical column (Year, ZIP, Premium, Exposure) has NaN | unit | `python -c "assert df[['Year','ZIP','Earned Premium']].notna().all().all()"` | NO |
| DATA-03 | Post-dedup panel has unique (Year, ZIP, Category) triples | unit | `python -c "assert not panel.duplicated(subset=['Year','ZIP','Category']).any()"` | NO |
| DATA-03 | Training panel (2018-2020) contains no 2021 rows | unit | `python -c "assert train_panel['Year'].max() <= 2020"` | NO |

### Sampling Rate
- **Per task commit:** `pytest tests/test_data_foundation.py -x -q` (if test file exists)
- **Per wave merge:** Full suite
- **Phase gate:** All DATA-* tests green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_data_foundation.py` — covers DATA-01, DATA-02, DATA-03
- [ ] `tests/conftest.py` — shared fixtures (raw_df, panel_df, train_panel)
- Framework install: `pip install pytest` — if not detected

## Security Domain

> Omitted — this phase loads and deduplicates a CSV with no authentication, session management, or user input handling. ASVS V2/V3/V4 controls do not apply. ASVS V5 (Input Validation) applies only at the CSV parsing level, which pandas handles by default with `low_memory=False` and explicit dtype coercion. No sensitive data (PII, credentials) is processed.

## Sources

### Primary (HIGH confidence)
- `Task2_Data/abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv` — CSV inspection via pandas read with low_memory=False, 47,033 rows, 76 columns, verified Year distribution and duplicate structure
- `Task2_Data/abfaw7bci2UF6CTg_FeatureDescription_fire_insurance.csv` — Feature descriptions for all insurance columns (Category, fire losses, premiums, exposure)
- `Task1_Data/Task1_Weather/data_preprocessing.py` — Reference pattern for loading, deduplication, zip-year aggregation (lines 1-213)

### Secondary (MEDIUM confidence)
- `.planning/ROADMAP.md` — Success criteria for Phase 1 (row counts, temporal integrity, column presence)
- `.planning/STATE.md` — Critical pitfalls (feature leakage through current-year data)

### Tertiary (LOW confidence)
None — all findings verified from primary sources.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pandas/numpy are verified standard, no alternative needed
- Architecture: HIGH — Task 1 reference pattern verified, deduplication logic confirmed with actual data
- Pitfalls: MEDIUM — all pitfalls inferred from actual data inspection, not from external documentation

**Research date:** 2026-04-09
**Valid until:** 2026-05-09 (30 days — data pipeline approach is stable, CSV format unlikely to change)
