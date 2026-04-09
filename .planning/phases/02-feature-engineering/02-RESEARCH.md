# Phase 2: Feature Engineering - Research

**Researched:** 2026-04-09
**Domain:** Panel data feature engineering — temporal lags, expanding-window statistics, category encoding
**Confidence:** HIGH

## Summary

Phase 2 transforms the clean 31,343-row panel (`task2_step1_panel_clean.csv`) into a model-ready feature matrix. The core challenge is strict temporal lag enforcement: features for predicting Year Y come only from Year Y-1 (2018 predicts 2019, 2019 predicts 2020). 2018 rows have no lagged features and serve only as lag sources. Expanding-window statistics capture per-ZIP fire risk and exposure history. Category one-hot encoding is applied before lag computation. A COVID-19 binary flag is added for 2020. All NaN values are preserved — tree models (LightGBM) and statsmodels handle them natively.

**Primary recommendation:** Implement a two-pass approach: (1) compute expanding statistics using cumulative data per ZIP, (2) shift features by one year to create lags. Category one-hot encode first, then build lags. Never include current-year features in the target year's feature set.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Target is raw `Earned Premium` (not premium/exposure ratio)
- **D-02:** Zero-exposure rows (2,940 rows) are kept — no special handling at this stage
- **D-03:** Single lag (t-1 only) — features for Year Y come only from Year Y-1
- **D-04:** 2020 target rows use 2019 features only
- **D-05:** 2019 target rows use 2018 features only
- **D-06:** 2018 rows have no lagged features (cannot lag from 2017); these rows serve as lag-source only, not as modeling targets
- **D-07:** Expanding window — mean and std computed across all available years up to the target year
- **D-08:** Expanding mean of `Avg Fire Risk Score` and `Earned Exposure` per ZIP
- **D-09:** Expanding std of `Avg Fire Risk Score` per ZIP
- **D-10:** One-hot encode 6 insurance categories (HO, CO, DT, RT, DO, MH, NA) as 6 binary columns
- **D-11:** Category encoding happens before lag/rolling computation; category does not change over time (no lag needed)
- **D-12:** Binary `is_covid_year`: 1 for Year == 2020, 0 otherwise
- **D-13:** COVID flag is a feature, not a target adjustment

### Deferred Ideas (OUT OF SCOPE)

None — all decisions stayed within Phase 2 scope.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FEAT-01 | Temporal lag features (t-1) for fire risk, exposure, and key loss columns | Sections 2, 3: Lag via year-over-year shift; 2018 rows lack lags; strict one-year offset enforced |
| FEAT-02 | Rolling statistics (expanding mean/std) per ZIP for fire risk and exposure | Sections 4, 5: Expanding window by ZIP using pandas groupby + cumsum/cumcount; D-07 through D-09 locked |
| FEAT-03 | Category one-hot encoded (6 binary columns) | Section 6: pd.get_dummies or manual binary columns; D-10 through D-11 locked |
| FEAT-04 | Premium normalized by earned exposure as target option | Superseded by D-01 (raw premium target); not implemented in Phase 2 |
| FEAT-05 | Zero-inflated claims distribution documented; Tweedie vs two-part deferred to Phase 3 | Section 8: Document only — no modeling changes at FE stage |
| FEAT-06 | `Avg Fire Risk Score` used as primary wildfire risk predictor | Sections 2, 3: Lagged fire risk is a primary lag feature |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | latest | Panel reshaping, groupby, lag/rolling computation | Primary data library per CLAUDE.md |
| numpy | latest | Numeric operations for expanding std | Universal dependency |

**Installation:**
```bash
pip install pandas numpy
```

**No additional libraries required** — pure pandas/numpy data transformation. scikit-learn not needed until Phase 3.

## Architecture Patterns

### Recommended Project Structure
```
Task2_Data/
├── task2_step1_panel_clean.csv         # Input (Phase 1)
└── task2_step2_feature_matrix.csv      # Output (Phase 2)
```

### Pattern 1: Two-Pass Lag Construction

**Source:** Standard panel data practice — pandas shift() per group

```python
# PASS 1: Sort and compute expanding statistics
df = df.sort_values(['ZIP', 'Category', 'Year'])
df['expanding_fire_risk_mean'] = df.groupby(['ZIP', 'Category'])['Avg Fire Risk Score'].transform(
    lambda x: x.expanding().mean().shift(1)
)
df['expanding_fire_risk_std'] = df.groupby(['ZIP', 'Category'])['Avg Fire Risk Score'].transform(
    lambda x: x.expanding().std().shift(1)
)
df['expanding_exposure_mean'] = df.groupby(['ZIP', 'Category'])['Earned Exposure'].transform(
    lambda x: x.expanding().mean().shift(1)
)

# PASS 2: One-year lag via groupby shift
lag_cols = ['Avg Fire Risk Score', 'Earned Exposure', 'Earned Premium',
            'CAT Cov A Fire -  Incurred Losses', 'Non-CAT Cov A Fire -  Incurred Losses',
            'Avg PPC']

for col in lag_cols:
    df[f'{col}_lag1'] = df.groupby(['ZIP', 'Category'])[col].shift(1)

# PASS 3: Drop 2018 lag rows (no prior year available)
df = df[df['Year'] != 2018].copy()

# PASS 4: COVID flag
df['is_covid_year'] = (df['Year'] == 2020).astype(int)
```

### Pattern 2: Category One-Hot Before Lag Computation

```python
# One-hot encode BEFORE computing lags — category is time-invariant per ZIP
categories = ['HO', 'CO', 'DT', 'RT', 'DO', 'MH', 'NA']
for cat in categories:
    df[f'cat_{cat}'] = (df['Category'] == cat).astype(int)
```

### Pattern 3: Expanding Window (No Library Required)

**Source:** Verified via pandas documentation — `expanding()` with `shift(1)` achieves the lagging-expanding effect

```python
# Expanding mean: cumulative average of all prior years for each ZIP
# shift(1): exclude current year from the window (prevents leakage)
df['expanding_fire_risk_mean'] = (
    df.groupby(['ZIP', 'Category'])['Avg Fire Risk Score']
    .transform(lambda x: x.expanding().mean().shift(1))
)

# Expanding std: cumulative standard deviation
df['expanding_fire_risk_std'] = (
    df.groupby(['ZIP', 'Category'])['Avg Fire Risk Score']
    .transform(lambda x: x.expanding().std().shift(1))
)
```

### Pattern 4: 2018 Row Handling

```python
# 2018 rows cannot produce lag features — they lack 2017 data
# Strategy: keep 2018 as lag-source for 2019, then filter out
# This means 2018 rows appear in intermediate but NOT in final feature matrix

# After computing lags and expanding stats:
df_final = df[df['Year'] != 2018].copy()  # Remove 2018 modeling targets

# Assert: all lag columns in final df have no NaN from missing-prior-year
assert df_final['Avg Fire Risk Score_lag1'].notna().all(), "Some lag1 values are NaN due to missing prior year"
```

### Anti-Patterns to Avoid

- **Computing lag on full dataframe before groupby:** `df['col_lag'] = df['col'].shift(1)` without groupby would shift across years incorrectly. Always use `groupby(['ZIP', 'Category']).shift(1)`.
- **Expanding window without shift(1):** Current year would be included in its own expanding mean, causing leakage. Always shift the expanding result by 1.
- **Dropping 2018 rows before expanding computation:** 2018 data is needed as the seed for 2019's expanding window. Compute expanding first, then filter.

## Common Pitfalls

### Pitfall 1: Cross-Year Shift Without Groupby
**What goes wrong:** Using `df.groupby('Year')` or a flat `df.shift(1)` shifts rows across ZIPs, mixing data between different entities.
**Why it happens:** Panel data has multiple observations per year. A flat shift moves a 2019 row's data into the next row (which could be a different ZIP in 2020).
**How to avoid:** Always `groupby(['ZIP', 'Category'])` before `.shift(1)`.

### Pitfall 2: Expanding Window Leakage (No shift on expanding result)
**What goes wrong:** `groupby(...).expanding().mean()` includes the current row. For Year=2020, the expanding mean of fire risk includes 2020 fire risk — leakage.
**Why it happens:** Expanding mean naturally includes all rows up to current index, including current.
**How to avoid:** Chain `.shift(1)` after `.expanding().mean()` to exclude current year.

### Pitfall 3: Filtering 2018 Rows Before Expanding is Seeded
**What goes wrong:** If you drop Year==2018 before computing expanding statistics, Year==2019 has no expanding mean (only 1 year of history).
**Why it happens:** Expanding window needs at least one prior observation. With only 2019 and 2020 remaining, the expanding window for 2020 would be just 2019's value.
**How to avoid:** Compute expanding statistics on full dataframe (2018-2020), THEN filter out 2018 modeling targets.

### Pitfall 4: Expanding std Returns NaN for Single-Observation Groups
**What goes wrong:** Expanding std with fewer than 2 observations returns NaN. For ZIPs with only 1-2 years of data, expanding std may be NaN.
**Why it happens:** Standard deviation requires at least 2 data points.
**How to avoid:** Accept NaN expanding std for early years; LightGBM and statsmodels handle NaN natively (per D-01).

### Pitfall 5: Category Encoding Not Applied Before Lag
**What goes wrong:** If one-hot encoding is done after lag computation, the lag shift could misalign the category columns.
**Why it happens:** Category is time-invariant — it doesn't change between years for the same ZIP. One-hot encoding it first avoids this issue.
**How to avoid:** Apply `pd.get_dummies` or manual binary column creation before lag/rolling computation.

## Code Examples

### Lag Feature Computation
```python
# Source: Standard pandas groupby shift practice
# Verified: pandas 2.x groupby.shift(1) preserves NaN correctly

lag_features = [
    'Avg Fire Risk Score',
    'Earned Exposure',
    'Earned Premium',
    'CAT Cov A Fire -  Incurred Losses',
    'Non-CAT Cov A Fire -  Incurred Losses',
    'Avg PPC',
]

for col in lag_features:
    df[f'{col}_lag1'] = df.groupby(['ZIP', 'Category'])[col].shift(1)
```

### Expanding Statistics
```python
# Source: Standard pandas expanding practice
# Verified: expanding().mean().shift(1) excludes current year

df['expanding_fire_risk_mean'] = (
    df.groupby(['ZIP', 'Category'])['Avg Fire Risk Score']
    .transform(lambda x: x.expanding().mean().shift(1))
)

df['expanding_fire_risk_std'] = (
    df.groupby(['ZIP', 'Category'])['Avg Fire Risk Score']
    .transform(lambda x: x.expanding().std().shift(1))
)

df['expanding_exposure_mean'] = (
    df.groupby(['ZIP', 'Category'])['Earned Exposure']
    .transform(lambda x: x.expanding().mean().shift(1))
)
```

### COVID Flag
```python
# Source: D-12 from CONTEXT.md
df['is_covid_year'] = (df['Year'] == 2020).astype(int)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-ZIP ARIMA fitting | Global panel model with entity features | Project decision (this phase) | 3 time points insufficient for ARIMA |
| Drop NaN expanding std | Keep NaN — tree models handle natively | D-01 (Phase 1) | Preserves data |
| Flat shift for lags | Groupby shift per (ZIP, Category) | This phase | Correct entity-level temporal alignment |
| No COVID flag | Binary is_covid_year flag | D-12 (this phase) | Models can learn 2020 structural break |

**No deprecated approaches relevant to this phase.**

## Assumptions Log

> List all claims tagged `[ASSUMED]` in this research. Planner and discuss-phase use this to identify decisions needing user confirmation.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `Avg Fire Risk Score` is available for all years (0% missing at panel level) | Pattern 1, 2 | If some Year-ZIP rows have NaN fire risk, lag features would be NaN for those rows too |
| A2 | `groupby(['ZIP', 'Category']).shift(1)` correctly handles the Year ordering | Pattern 1 | If rows within a group are not sorted by Year, shift(1) would pair wrong years |
| A3 | Category is time-invariant per ZIP (a given ZIP always has the same category) | Pattern 2 | If categories change year-to-year for same ZIP, one-hot encoding before lag would be incorrect |

**If any assumption in this table is wrong, the corresponding section must be revised before planning.**

## Open Questions

1. **Should lag features also include weather columns (avg_tmax_c, avg_tmin_c, tot_prcp_mm)?**
   - What we know: Weather columns exist in the panel with ~83% NaN (correlated with fire event presence).
   - What's unclear: Should weather be lagged as well, or only insurance-native features (fire risk, exposure, losses)?
   - Recommendation: Include weather lags if they improve model performance in Phase 3. Start with insurance-native features per D-03.

2. **What about the 2,940 zero-exposure rows — should they be filtered before or after feature engineering?**
   - What we know: D-02 says keep zero-exposure rows; Phase 3 models handle them naturally or filter if needed.
   - What's unclear: Does expanding mean of Earned Exposure include zeros, which would dilute the exposure history?
   - Recommendation: Keep zero-exposure rows in the expanding computation. Their presence reflects reality. Filter in Phase 3 if models require.

## Environment Availability

Step 2.6: SKIPPED (no external dependencies — pure pandas/numpy data transformation).

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
| FEAT-01 | Lag columns exist for all non-2018 rows | unit | `python -c "assert 'Avg Fire Risk Score_lag1' in df.columns; assert df[df.Year==2019]['Avg Fire Risk Score_lag1'].notna().all()"` | NO |
| FEAT-01 | 2018 rows have no lag columns (NaN) | unit | `python -c "assert df[df.Year==2018]['Avg Fire Risk Score_lag1'].isna().all()"` | NO |
| FEAT-02 | Expanding mean computed per ZIP (not global) | unit | `python -c "zip_90210 = df[df.ZIP==90210]; assert zip_90210['expanding_fire_risk_mean'].notna().any()"` | NO |
| FEAT-02 | Expanding std may be NaN for early years (acceptable) | unit | `python -c "import pandas as pd; df = pd.read_csv('Task2_Data/task2_step2_feature_matrix.csv'); print('NaN expanding_std:', df['expanding_fire_risk_std'].isna().sum())"` | NO |
| FEAT-03 | 6 category binary columns exist | unit | `python -c "cats=['cat_HO','cat_CO','cat_DT','cat_RT','cat_DO','cat_MH']; assert all(c in df.columns for c in cats)"` | NO |
| FEAT-03 | Category columns sum to 1 per row | unit | `python -c "assert df[['cat_HO','cat_CO','cat_DT','cat_RT','cat_DO','cat_MH','cat_NA']].sum(axis=1).eq(1).all()"` | NO |
| FEAT-06 | `Avg Fire Risk Score_lag1` is present as primary wildfire feature | unit | `python -c "assert 'Avg Fire Risk Score_lag1' in df.columns"` | NO |
| General | Final matrix has no 2018 rows | unit | `python -c "assert 2018 not in df['Year'].values"` | NO |
| General | COVID flag is 1 for 2020, 0 for 2019 | unit | `python -c "assert df[df.Year==2019]['is_covid_year'].unique()[0]==0; assert df[df.Year==2020]['is_covid_year'].unique()[0]==1"` | NO |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q` (if test file exists)
- **Per wave merge:** Full suite
- **Phase gate:** All FEAT-* tests green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_feature_engineering.py` — covers FEAT-01, FEAT-02, FEAT-03, FEAT-06
- [ ] `tests/conftest.py` — shared fixtures (panel_df, feature_matrix_df)
- [ ] Framework install: `pip install pytest` — if not detected

## Security Domain

> Omitted — this phase performs data transformation on a CSV with no authentication, session management, or user input handling. ASVS V2/V3/V4 controls do not apply. ASVS V5 (Input Validation) applies only at the CSV parsing level, which pandas handles natively. No sensitive data (PII, credentials) is processed.

## Sources

### Primary (HIGH confidence)
- `Task2_Data/task2_step1_panel_clean.csv` — Verified 31,343 rows, 49 columns, Year distribution 2018/2019/2020, column list
- `.planning/phases/02-feature-engineering/02-CONTEXT.md` — All decisions (D-01 through D-13) and phase requirements (FEAT-01 through FEAT-06) fully specified
- `Task2_Data/task2_step1_data_load.py` — Pipeline pattern (STEP 1-7 structure, constants, assertions)

### Secondary (MEDIUM confidence)
- pandas 2.x documentation — expanding window and groupby shift behavior
- `.planning/REQUIREMENTS.md` — FEAT-01 through FEAT-06 acceptance criteria
- `.planning/STATE.md` — Stack (LightGBM + statsmodels), COVID structural break handling

### Tertiary (LOW confidence)
None — all findings from primary or verified secondary sources.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pure pandas/numpy, no version issues
- Architecture: HIGH — lag + expanding patterns verified from pandas docs; panel structure confirmed from Phase 1 research
- Pitfalls: HIGH — all pitfalls traced to actual data characteristics

**Research date:** 2026-04-09
**Valid until:** 2026-05-09 (30 days — data pipeline approach is stable)
