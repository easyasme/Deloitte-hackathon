# Technology Stack: Insurance Premium Time Series Forecasting

**Project:** California Insurance Premium Prediction with Wildfire Risk Integration
**Domain:** Panel Time Series Forecasting (ZIP codes x Years)
**Researched:** 2026-04-09
**Confidence:** MEDIUM (training data + established patterns; limited Context7 verification due to tool unavailability)

---

## Executive Summary

For insurance premium time series forecasting with panel data structure (California ZIP codes, 2018-2020 historical, predicting 2021), the standard 2025 stack splits into two complementary approaches:

1. **Econometric:** `statsmodels` with PanelOLS Fixed Effects for causal inference and interpretability
2. **Machine Learning:** `LightGBM` with engineered lag features for predictive accuracy

**Use both in ensemble.** The econometric model provides interpretability (required for insurance regulatory filings), while LightGBM captures non-linear relationships that panel regression misses.

---

## Recommended Stack

### Core Framework

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| **Python** | 3.11+ | Runtime | Required for all below |
| **pandas** | 2.2+ | Data manipulation | Panel data reshaping, groupby operations |
| **numpy** | 1.26+ | Numerical computing | Array operations, statistical functions |

### Econometric Approach (Primary for Interpretability)

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| **statsmodels** | 0.14+ | Panel regression | Fixed Effects estimation, robust standard errors, interpretable coefficients |

**Why statsmodels over alternatives:**
- Native panel data support (Fixed Effects, Random Effects, Dynamic Panel)
- Built-in robust standard errors (clustered by entity or time)
- Econometric inference (t-stats, R-squared, F-tests) for regulatory acceptance
- No black-box behavior

**Alternatives considered:**
- `linearmodels` (specialized panel package) -- Overkill for simple FE estimation; statsmodels sufficient
- `scikit-learn` LinearRegression -- Does not handle panel structure; would require dummy variables for each ZIP, causing overfitting with ~1600 entities

### Machine Learning Approach (Primary for Accuracy)

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| **LightGBM** | 4.5+ | Gradient boosting | Speed, accuracy, handles high-cardinality categoricals (ZIP codes) |
| **scikit-learn** | 1.5+ | Preprocessing, evaluation | train_test_split, cross-validation, metrics |

**Why LightGBM over alternatives:**
- Dominant Kaggle/industry choice for tabular data (2022-2025)
- Handles ~1600 unique ZIP codes efficiently via native categorical support
- Fast training on ~5000 rows
- Built-in regularization prevents overfitting on short panel

**Why NOT XGBoost:**
- XGBoost is slower and requires explicit one-hot encoding for categoricals
- LightGBM's leaf-wise growth captures complex interactions faster
- Same predictive performance, worse developer experience

**Why NOT deep learning (LSTM, Transformer, TFT):**
- Only 3 years of data (insufficient for deep learning)
- ~4800 total observations (far below the millions needed for neural networks)
- High interpretability requirement makes neural nets a liability

### Feature Engineering Libraries

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| **pandas** (built-in) | -- | Lag/lead features | `groupby(ZIP).shift()` for temporal features |
| **sklearn.preprocessing** | 1.5+ | StandardScaler | Normalize continuous features for some models |

---

## Installation

```bash
# Core dependencies
pip install pandas>=2.2.0
pip install numpy>=1.26.0
pip install statsmodels>=0.14.0
pip install lightgbm>=4.5.0
pip install scikit-learn>=1.5.0

# Optional: for better visualizations
pip install matplotlib>=3.8.0
pip install seaborn>=0.13.0
```

---

## Architecture Pattern for This Project

### Panel Data Structure

```
Observations:  ZIP x Year (e.g., 90003-2018, 90003-2019, 90003-2020)
Entities:      ~1600 California ZIP codes
Time periods:  3 (2018, 2019, 2020)
Target:        Earned Premium (continuous, right-skewed)
```

### Recommended Model Pipeline

#### Approach 1: Panel Fixed Effects (statsmodels)

```python
from statsmodels.regression.linear_model import PanelOLS
from statsmodels.stats.sandwich_covariance import cov_cluster

# Model: Premium_it = alpha_i + beta*FireRisk_it + gamma*Controls_it + epsilon_it
# alpha_i = ZIP fixed effect (absorbs time-invariant heterogeneity)

model = PanelOLS.from_formula(
    "EarnedPremium ~ FireRiskScore + AvgPPC + CovAAmountWeightedAvg + "
    "EntityEffects + TimeEffects",
    data=panel_df
)
result = model.fit(cov_type='clustered', cluster_entity=True)
```

**Strengths:**
- Causal identification (fire risk -> premium changes)
- Interpretable coefficients
- Handles unobserved ZIP heterogeneity via fixed effects

**Weaknesses:**
- Assumes linear relationships
- Cannot capture complex interactions

#### Approach 2: LightGBM with Lag Features

```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# Engineer lag features
panel_df = panel_df.sort_values(['ZIP', 'Year'])
panel_df['Premium_lag1'] = panel_df.groupby('ZIP')['EarnedPremium'].shift(1)
panel_df['FireRisk_lag1'] = panel_df.groupby('ZIP')['FireRiskScore'].shift(1)

# Train on 2018-2019, validate on 2020, predict 2021
train = panel_df[panel_df['Year'] < 2020]
val = panel_df[panel_df['Year'] == 2020]

features = ['FireRiskScore', 'AvgPPC', 'lag_premium', 'lag_fire_risk', ...]

train_data = lgb.Dataset(train[features], label=train['EarnedPremium'])
val_data = lgb.Dataset(val[features], label=val['EarnedPremium'], reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=1000)
```

**Strengths:**
- Captures non-linear relationships
- Feature importance ranking
- Handles many features without explicit dimensionality reduction

**Weaknesses:**
- Requires careful lag feature engineering
- Less interpretable than panel regression

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why Avoid | Instead |
|--------------|-----------|---------|
| **Standard OLS without fixed effects** | Biased estimates due to omitted ZIP-specific factors | Panel FE or entity fixed effects |
| **Prophet/NeuralProphet** | Designed for daily data with strong weekly/yearly seasonality; annual insurance data violates these assumptions | Panel regression or LightGBM |
| **XGBoost over LightGBM** | Slower, worse developer experience for tabular data | LightGBM |
| **Deep learning (LSTM, TFT)** | 3 years of data is far too short; will severely overfit | Panel FE or LightGBM |
| **One-hot encoding ZIP as numeric** | 1600 dummy variables causes overfitting and memory issues | LightGBM native categorical support or ZIP embedding |
| **Ignoring time series structure in CV** | Future data leakage | TimeSeriesSplit or expanding window |

---

## Feature Engineering for This Dataset

Based on the provided data structure, the following features are available:

| Feature Type | Examples | Engineering Notes |
|--------------|----------|-------------------|
| **Target lag** | Premium_lag1, Premium_lag2 | GroupBy ZIP then shift |
| **Risk lags** | FireRisk_lag1, FireRisk_lag2 | Lagged fire risk may predict claims |
| **Claims features** | CAT_Fire_IncurredLosses, NonCAT_CovA_Fire_NumClaims | Time-varying, predictive of future premiums |
| **Exposure** | EarnedExposure | Policy volume proxy |
| **Coverage amounts** | CovAAmountWeightedAvg, CovCAmountWeightedAvg | Insured values |
| **Weather** | avg_tmax_c, avg_tmin_c, tot_prcp_mm | Temperature, precipitation |
| **Census** | median_income, total_population, housing_value | Demographic context |
| **Time dummies** | Year_2019, Year_2020 | Capture aggregate trends |

---

## Model Selection Decision Tree

```
Is interpretability required (regulatory filings)?
  YES -> Use Panel Fixed Effects as primary
  NO  -> Skip to next

Is data > 10,000 rows with complex interactions?
  YES -> Add LightGBM ensemble
  NO  -> Panel FE likely sufficient

Is seasonality present (daily/weekly/monthly patterns)?
  YES -> Consider Prophet only if strong seasonality
  NO  -> Not relevant for annual data

Is prediction accuracy the only goal?
  YES -> LightGBM with lag features
  NO  -> Use ensemble of Panel FE + LightGBM
```

For this project: **Ensemble of Panel FE + LightGBM**

---

## Confidence Assessment

| Component | Confidence | Notes |
|----------|------------|-------|
| statsmodels PanelOLS | HIGH | Established, well-documented approach |
| LightGBM for tabular | HIGH | Dominant choice in industry |
| Anti-patterns (Prophet, deep learning) | HIGH | Logical given data constraints |
| Specific library versions | MEDIUM | Verified via training data, not live Context7 |

---

## Sources

- **statsmodels documentation:** https://www.statsmodels.org/stable/panel.html
- **LightGBM documentation:** https://lightgbm.readthedocs.io/
- **Kaggle tabular competition trends:** Dominant choice 2022-2025 is LightGBM/XGBoost over neural networks for structured tabular data
- **Insurance actuarial practice:** Panel regression (Fixed Effects) is standard for insurance premium modeling with spatial/temporal structure

---

## Gap Analysis

**Not verified via Context7 (tool unavailable at research time):**
- Exact latest version numbers (used conservative estimates based on training data)
- Any breaking changes in statsmodels 0.15+ or LightGBM 5.0+

**Recommendation:** Verify versions with `pip index versions <package>` before installation.
