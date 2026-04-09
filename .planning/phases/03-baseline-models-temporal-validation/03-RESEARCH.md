# Phase 3: Baseline Models + Temporal Validation - Research

**Researched:** 2026-04-09
**Domain:** Classical regression baselines for insurance premium prediction — Linear Regression and Random Forest with temporal validation
**Confidence:** MEDIUM-HIGH

## Summary

Phase 3 trains two baseline regression models (Linear Regression, Random Forest) on the Phase 2 feature matrix using a strict temporal split: 2019 for training, 2020 for validation. Both models predict `Earned Premium` using 43 validated features (lag features, expanding statistics, demographics, fire risk, and COVID flag). RMSE, MAE, and MAPE (on non-zero premium subset) are reported on the 2020 holdout. The `is_covid_year` coefficient from Linear Regression provides an interpretable estimate of the 2020 structural anomaly.

**Primary recommendation:** Use `sklearn.linear_model.LinearRegression` (OLS, no regularization) and `sklearn.ensemble.RandomForestRegressor` with `n_estimators=100, max_depth=10-15, min_samples_leaf=5`. scikit-learn must be installed before Phase 3 execution.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Include ALL rows in training and evaluation — no filtering of zero-exposure rows
- **D-02:** Zero-exposure rows produce zero or near-zero premium naturally; both Linear Reg and Random Forest handle them without special treatment
- **D-03:** Use vanilla OLS (`sklearn.linear_model.LinearRegression`) — no regularization
- **D-04:** Rationale: Simple interpretable baseline. Ridge/Lasso deferred to Phase 4 if overfitting observed.
- **D-05:** Use sensible defaults: `n_estimators=100`, `max_depth=10-15`, `min_samples_leaf=5`
- **D-06:** Rationale: Quick to run, reasonable performance. LightGBM in Phase 4 will be the primary tree model.
- **D-07:** Report MAPE on non-zero premium subset only (`Earned Premium > 0`)
- **D-08:** Zero-premium rows excluded from MAPE denominator; RMSE and MAE computed on full validation set
- **D-09:** Document zero-premium row count and percentage in validation report
- **D-10:** "Same features" = all lag features + rolling stats + COVID flag + category one-hot + demographics + fire risk, excluding Year and non-feature identifiers (ZIP, Category, ZIP_Cat, FIRE_NAME, AGENCY, INC_NUM)
- **D-11:** Exact feature list determined by planner from `task2_step2_feature_matrix.csv` columns
- **D-12:** Train: 2019 rows from feature matrix (2018 served as lag source only)
- **D-13:** Validation: 2020 rows from feature matrix
- **D-14:** No random split — strictly temporal
- **D-15:** `is_covid_year` flag is already a feature in the matrix; baseline models will learn its coefficient
- **D-16:** Validation report includes: COVID coefficient estimate, 2020 vs 2019 residual analysis, explicit note that 2020 may have elevated error

### Deferred Ideas (OUT OF SCOPE)

None — all decisions stayed within Phase 3 scope.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BASE-01 | Temporal train/validation split (2019 train, 2020 validation) — NO random split | Section 2: Temporal split pattern; D-12 through D-14 |
| BASE-02 | Linear regression with lag features and fire risk score | Section 4: sklearn LinearRegression usage; D-03 (vanilla OLS) |
| BASE-03 | Random Forest with same features | Section 4: sklearn RandomForestRegressor usage; D-05 (sensible defaults) |
| BASE-04 | Report RMSE, MAE, MAPE on validation set for both baselines | Section 3: MAPE on non-zero subset; D-07 through D-09 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | not installed | LinearRegression, RandomForestRegressor, metrics | Per CLAUDE.md; sklearn not yet installed in environment |
| pandas | 2.x | Data loading, feature selection, metric aggregation | Primary data library |
| numpy | latest | Array operations, NaN handling | Universal dependency |

**Installation:**
```bash
pip install scikit-learn pandas numpy
```

**CRITICAL:** scikit-learn is NOT currently installed in the environment. Installation required before Phase 3 execution.

### Supporting
| Library | Purpose | When to Use |
|---------|---------|-------------|
| statsmodels | COVID coefficient significance testing (optional) | Phase 4 or if p-values needed in baseline report |

## Architecture Patterns

### Recommended Project Structure
```
Task2_Data/
├── task2_step2_feature_matrix.csv      # Input (Phase 2)
├── task2_step3_baseline_models.py       # Phase 3 script
├── task2_step3_linear_regression_metrics.csv  # Output
├── task2_step3_random_forest_metrics.csv      # Output
└── task2_step3_validation_report.csv    # COVID analysis output
```

### Pattern 1: Temporal Train/Validation Split

**Source:** Standard time series practice — no random shuffling

```python
# Train: 2019 only (2020 used for lag sources)
# Validation: 2020 only
train_df = df[df['Year'] == 2019].copy()
val_df = df[df['Year'] == 2020].copy()

X_train = train_df[feature_cols]
y_train = train_df['Earned Premium']
X_val = val_df[feature_cols]
y_val = val_df['Earned Premium']
```

### Pattern 2: Linear Regression with scikit-learn

**Source:** sklearn documentation — LinearRegression fits intercept by default

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# D-03: Vanilla OLS, no regularization
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_val)

# RMSE
rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))
# MAE
mae_lr = mean_absolute_error(y_val, y_pred_lr)
# MAPE on non-zero subset (D-07)
non_zero_mask = y_val > 0
mape_lr = np.mean(np.abs((y_val[non_zero_mask] - y_pred_lr[non_zero_mask]) / y_val[non_zero_mask])) * 100
```

### Pattern 3: Random Forest Regressor

**Source:** sklearn documentation — RandomForestRegressor for continuous targets

```python
from sklearn.ensemble import RandomForestRegressor

# D-05: Sensible defaults
model_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_val)
```

### Pattern 4: MAPE Calculation on Non-Zero Subset

**Source:** Insurance domain practice — MAPE undefined when denominator is zero

```python
# D-07 through D-09: MAPE only on non-zero premium
non_zero_mask = y_val > 0
mape_lr = np.mean(np.abs((y_val.values[non_zero_mask] - y_pred_lr[non_zero_mask]) / y_val.values[non_zero_mask])) * 100

# Document zero-premium count
zero_count = (~non_zero_mask).sum()
zero_pct = zero_count / len(y_val) * 100
print(f"MAPE computed on {non_zero_mask.sum()} non-zero rows ({zero_count} zero rows excluded, {zero_pct:.1f}%)")
```

### Pattern 5: COVID-19 Coefficient Extraction

**Source:** Linear Regression interpretability advantage

```python
# Get COVID coefficient (index of is_covid_year in feature_cols)
covid_idx = feature_cols.index('is_covid_year')
covid_coef = model_lr.coef_[covid_idx]
print(f"COVID-19 coefficient (Linear Regression): {covid_coef:,.0f}")
print(f"Interpretation: 2020 premium adjustment of {covid_coef:,.0f} relative to 2019 baseline")
```

### Anti-Patterns to Avoid

- **Using classification instead of regression:** `Earned Premium` is continuous — use `RandomForestRegressor`, not `RandomForestClassifier`
- **MAPE on all rows:** Division by zero for zero-premium rows — always filter `y_val > 0` first
- **Including leakage columns:** Current-year loss/claim columns (CAT Cov A Fire, Non-CAT Cov C Smoke, etc.) must be excluded from features
- **Random train/test split:** Temporal structure requires 2019 train / 2020 validation split

## Common Pitfalls

### Pitfall 1: Missing scikit-learn Installation
**What goes wrong:** `ModuleNotFoundError: No module named 'sklearn'`
**Why it happens:** scikit-learn not installed in environment
**How to avoid:** Add `pip install scikit-learn` to setup steps
**Warning signs:** ImportError on `from sklearn.linear_model import LinearRegression`

### Pitfall 2: High NaN Count in Weather Features
**What goes wrong:** `avg_tmax_c`, `avg_tmin_c`, `tot_prcp_mm` have 89% NaN
**Why it happens:** Weather data only available for event-affected ZIPs (Phase 1 pattern)
**How to avoid:** Random Forest handles NaN natively via surrogate splits; Linear Regression requires imputation (mean or 0)
**Warning signs:** `np.nan` values in predictions or coefficient estimation

### Pitfall 3: Expanding Std NaN for Single-Observation Groups
**What goes wrong:** `expanding_fire_risk_std` has 59% NaN (from Phase 2 research)
**Why it happens:** Expanding std requires >= 2 observations; single-observation groups return NaN
**How to avoid:** Random Forest handles natively; for Linear Regression, impute with 0 or column mean
**Warning signs:** Linear Regression fails to fit due to NaN coefficient

### Pitfall 4: Negative Premium Values
**What goes wrong:** Feature matrix has negative `Earned Premium` values (min=-496)
**Why it happens:** Likely data correction or return of premium
**How to avoid:** Keep all rows per D-01; MAPE uses absolute values
**Warning signs:** MAPE computation produces unexpected values

## Code Examples

### Evaluation Helper (adapted from Task1_Data/Task1_Weather/classical_ml.py)

```python
def evaluate_model(model_name, y_true, y_pred, feature_cols=None, model=None):
    """Evaluate regression model and save metrics."""
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    # MAPE on non-zero subset
    non_zero_mask = y_true > 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true.values[non_zero_mask] - y_pred[non_zero_mask]) / y_true.values[non_zero_mask])) * 100
    else:
        mape = np.nan

    # COVID coefficient if Linear Regression
    covid_coef = None
    if model_name == 'LinearRegression' and hasattr(model, 'coef_'):
        covid_idx = feature_cols.index('is_covid_year')
        covid_coef = model.coef_[covid_idx]

    print(f"\n--- {model_name} Results ---")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"MAPE: {mape:.2f}% (on non-zero subset, n={non_zero_mask.sum()})")
    if covid_coef is not None:
        print(f"COVID coefficient: {covid_coef:,.0f}")

    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'mape': mape}
```

### Feature Exclusion for Leakage Prevention

```python
# D-10: Exclude identifiers and current-year loss/claim columns
id_cols = ['Year', 'ZIP', 'Category', 'ZIP_Cat', 'FIRE_NAME', 'AGENCY', 'INC_NUM']
leakage_cols = [
    'CAT Cov A Fire -  Incurred Losses', 'CAT Cov A Fire -  Number of Claims',
    'CAT Cov A Smoke -  Incurred Losses', 'CAT Cov A Smoke -  Number of Claims',
    'CAT Cov C Fire -  Incurred Losses', 'CAT Cov C Fire -  Number of Claims',
    'CAT Cov C Smoke -  Incurred Losses', 'CAT Cov C Smoke -  Number of Claims',
    'Non-CAT Cov A Fire -  Incurred Losses', 'Non-CAT Cov A Fire -  Number of Claims',
    'Non-CAT Cov A Smoke -  Incurred Losses', 'Non-CAT Cov A Smoke -  Number of Claims',
    'Non-CAT Cov C Fire -  Incurred Losses', 'Non-CAT Cov C Fire -  Number of Claims',
    'Non-CAT Cov C Smoke -  Incurred Losses', 'Non-CAT Cov C Smoke -  Number of Claims',
]
target_col = 'Earned Premium'
feature_cols = [c for c in df.columns if c not in id_cols + [target_col] + leakage_cols]
# 43 valid features
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-ZIP ARIMA | Global panel model | Project decision | 3 time points insufficient |
| Drop NaN features | Keep NaN — tree models handle natively | D-01 (Phase 1) | Preserves data |
| Classification models | Regression (RandomForestRegressor) | This phase | Earned Premium is continuous |
| MAPE on all rows | MAPE on non-zero subset only | D-07 (Phase 3) | Avoids division by zero |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `is_covid_year` coefficient correctly captures 2020 structural break | Pattern 5 | If COVID effect is non-linear, Linear Regression cannot capture it |
| A2 | Random Forest default depth (15) is sufficient | D-05 | If overfitting, reduce max_depth; if underfitting, increase |
| A3 | All 43 features provide signal | Common Pitfalls | If collinearity is high, Ridge/Lasso may help (deferred to Phase 4) |

## Open Questions

1. **Should weather columns (avg_tmax_c, avg_tmin_c, tot_prcp_mm) be imputed or dropped?**
   - What we know: 89% NaN — only available for event-affected ZIPs
   - What's unclear: Does weather add predictive value for non-fire-event ZIPs?
   - Recommendation: Keep all features initially; let Random Forest feature importance guide Phase 4 decision

2. **Should Linear Regression impute NaN with mean/median before fitting?**
   - What we know: sklearn LinearRegression does not handle NaN natively
   - What's unclear: Mean imputation may introduce bias; 0 imputation may be more realistic for missing weather
   - Recommendation: For Phase 3, use Random Forest as primary baseline (handles NaN) and only apply mean imputation if Linear Regression fails

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| scikit-learn | LinearRegression, RandomForestRegressor, metrics | NO | — | Must install |
| pandas | Data loading, feature selection | YES | 2.x | — |
| numpy | Array operations | YES | latest | — |

**Missing dependencies with no fallback:**
- scikit-learn: Required for all Phase 3 models. Must install via `pip install scikit-learn`

**Missing dependencies with fallback:**
- None identified

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
| BASE-01 | Temporal split: train=2019, val=2020 | unit | `python -c "df=pd.read_csv('Task2_Data/task2_step2_feature_matrix.csv'); assert set(df[df.Year==2019].Year.unique())=={2019}; assert set(df[df.Year==2020].Year.unique())=={2020}"` | NO |
| BASE-02 | LinearRegression fitted and predicts | unit | `python -c "from sklearn.linear_model import LinearRegression; model=LinearRegression(); model.fit(X_train, y_train); assert len(model.predict(X_val))==len(y_val)"` | NO |
| BASE-03 | RandomForestRegressor fitted and predicts | unit | `python -c "from sklearn.ensemble import RandomForestRegressor; model=RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42); model.fit(X_train, y_train); assert len(model.predict(X_val))==len(y_val)"` | NO |
| BASE-04 | RMSE, MAE, MAPE computed for both models | unit | `python -c "metrics=['rmse','mae','mape']; assert all(m in results for m in metrics)"` | NO |
| General | Zero-premium count documented | unit | `python -c "zero_count=(df['Earned Premium']==0).sum(); print(f'Zero premium: {zero_count}/{len(df)}')"` | NO |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q` (if test file exists)
- **Per wave merge:** Full suite
- **Phase gate:** All BASE-* tests green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_baseline_models.py` — covers BASE-01, BASE-02, BASE-03, BASE-04
- [ ] `tests/conftest.py` — shared fixtures (feature_matrix_df, X_train, X_val, y_train, y_val)
- [ ] Framework install: `pip install scikit-learn pytest` — sklearn NOT currently installed

## Security Domain

> Omitted — this phase performs regression on a CSV with no authentication, session management, or user input handling. ASVS V2/V3/V4 controls do not apply. ASVS V5 (Input Validation) applies only at CSV parsing level, which pandas handles. No PII, credentials, or sensitive data processed.

## Sources

### Primary (HIGH confidence)
- `Task2_Data/task2_step2_feature_matrix.csv` — Verified 22,037 rows (2019: 9,329, 2020: 12,708), 67 columns, 43 valid features
- `.planning/phases/03-baseline-models-temporal-validation/03-CONTEXT.md` — All decisions (D-01 through D-16) and phase requirements (BASE-01 through BASE-04)
- `Task1_Data/Task1_Weather/classical_ml.py` — `evaluate_model()` pattern for adaptation

### Secondary (MEDIUM confidence)
- scikit-learn documentation — LinearRegression and RandomForestRegressor usage
- `.planning/phases/02-feature-engineering/02-RESEARCH.md` — Feature matrix characteristics, NaN distribution, expanding statistics
- `.planning/STATE.md` — Stack (LightGBM + statsmodels ensemble), COVID structural break handling

### Tertiary (LOW confidence)
- sklearn metrics documentation — MAPE calculation (standard practice, not verified via Context7 in this session)

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM — sklearn is project-standard per CLAUDE.md, but NOT currently installed (environment constraint)
- Architecture: HIGH — temporal split, evaluation patterns verified from Task1 code and sklearn docs
- Pitfalls: HIGH — all pitfalls traced to actual data characteristics (NaN distribution, negative premium values)

**Research date:** 2026-04-09
**Valid until:** 2026-05-09 (30 days — baseline modeling approach is stable)

---

## RESEARCH COMPLETE

**Phase:** 03 - baseline-models-temporal-validation
**Confidence:** MEDIUM-HIGH

### Key Findings
- sklearn NOT installed — must add `pip install scikit-learn` to setup
- 22,037 rows: 9,329 train (2019), 12,708 validation (2020)
- 43 valid features after excluding identifiers and leakage columns
- 2,830 zero-premium rows (12.8%) excluded from MAPE calculation
- MAPE on non-zero subset only (D-07) — avoids division by zero
- COVID coefficient extractable from Linear Regression for structural analysis

### File Created
`/home/dwk/code/Deloitte-hackathon/.planning/phases/03-baseline-models-temporal-validation/03-RESEARCH.md`

### Open Questions
1. Weather columns (89% NaN): impute or keep as-is for Random Forest?
2. Linear Regression NaN handling: mean imputation vs 0 imputation?

### Ready for Planning
Research complete. Planner can now create PLAN.md files.
