# Architecture Research: Insurance Premium Time Series Forecasting

**Domain:** Insurance Premium Time Series Prediction (Panel Data)
**Researched:** 2026-04-09
**Confidence:** MEDIUM (standard ML architecture patterns; limited search access)

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Ingestion Layer                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Raw CSV    │  │  External   │  │  Reference  │         │
│  │  Loader     │  │  Weather    │  │  Data       │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                  │
├─────────┴────────────────┴────────────────┴─────────────────┤
│                    Feature Engineering Layer                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Time-series │  │ Cross-      │  │ Target      │         │
│  │ Lag/Window  │  │ sectional   │  │ Encoding    │         │
│  │ Features    │  │ Features    │  │             │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                  │
├─────────┴────────────────┴────────────────┴─────────────────┤
│                       Modeling Layer                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Model Training & Validation              │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │    │
│  │  │ Base    │  │ Panel   │  │ Ensemble│              │    │
│  │  │ Models  │  │ Models  │  │ Stacking│              │    │
│  │  └─────────┘  └─────────┘  └─────────┘              │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                     Prediction Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Forecast   │  │  Error     │  │  Output     │         │
│  │  Generation │  │  Analysis  │  │  Export     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Raw CSV Loader | Parse insurance, fire, census, weather files; handle missing values | pandas, polars |
| External Data Joiner | Merge external features by ZIP and Year | pandas merge, polars join |
| Time-series Feature Generator | Create lag features, rolling statistics, trend features | custom functions |
| Cross-sectional Feature Generator | Create ZIP-level aggregates, category encodings | groupby, embedding tables |
| Target Encoder | Encode categorical variables (Category) as numeric | Target encoding, one-hot |
| Panel Model | Train model respecting panel structure (ZIP × Year) | statsmodels, sklearn wrappers |
| Base Models | Individual model candidates | LightGBM, XGBoost, linear regression |
| Cross-validator | Time-series aware train/test splits | TimeSeriesSplit, custom panel splits |
| Forecast Generator | Produce 2021 predictions | model.predict() |
| Error Analyzer | Compute and log forecast metrics | MAE, MAPE, RMSE |

## Recommended Project Structure

```
src/
├── data/                    # Data loading and joining
│   ├── loaders.py          # CSV loading functions
│   ├── joiners.py         # Data merging logic
│   └── validators.py      # Schema validation
├── features/               # Feature engineering
│   ├── time_series.py     # Lag, rolling, trend features
│   ├── cross_sectional.py # ZIP-level aggregations
│   ├── encoders.py        # Categorical encoding
│   └── selectors.py       # Feature selection
├── models/                 # Model training
│   ├── base.py            # Base model wrappers
│   ├── panel_models.py    # Panel data models
│   ├── tree_models.py     # LightGBM, XGBoost
│   └── ensembles.py       # Stacking, averaging
├── validation/             # Cross-validation
│   ├── splits.py          # Time-series splits
│   └── metrics.py         # Evaluation metrics
├── prediction/             # Inference
│   ├── forecaster.py      # Main prediction logic
│   └── exporters.py       # Output formatting
└── pipelines/              # End-to-end pipelines
    └── train_predict.py   # Full training + prediction
```

### Structure Rationale

- **data/**: Separating data loading allows easy swapping of data sources and caches raw data separately from features.
- **features/**: Feature engineering is typically the most time-consuming part; isolating it enables iteration and reuse.
- **models/**: Model selection is iterative; having base, panel, and ensemble models allows comparing different approaches.
- **validation/**: Time-series validation is tricky with panel data; centralizing split logic prevents leakage.
- **prediction/**: Keeping inference separate from training enforces clean separation and easier deployment.
- **pipelines/**: Orchestrates all components; can be entry point for experiments.

## Architectural Patterns

### Pattern 1: Global Model with Entity Features

**What:** Train a single model on all ZIP codes with entity embeddings or dummy variables for ZIP.

**When to use:** When there are many ZIP codes (panel is wide) and individual ZIP time series are short (3 time points is too short for per-ZIP ARIMA).

**Trade-offs:**
- Pros: Learns shared patterns across ZIPs; can incorporate cross-sectional features; handles cold-start better
- Cons: May miss ZIP-specific dynamics; requires careful handling of entity features

**Example:**
```python
# Features: Year, ZIP_embedding, fire_risk_score, weather, census_features
model = LightGBMRegressor()
model.fit(X_train, y_train)  # Global model across all ZIPs
predictions = model.predict(X_2021)
```

### Pattern 2: Hierarchical Modeling (Mixed Effects)

**What:** Use mixed-effects models with random intercepts per ZIP to capture entity-specific baselines while sharing slope parameters across the panel.

**When to use:** When you expect different ZIPs to have different baseline premiums but similar responses to features.

**Trade-offs:**
- Pros: Explicitly models panel structure; provides interpretable fixed and random effects
- Cons: More complex to fit; random effects may not converge with few time points

**Example:**
```python
# Using statsmodels MixedLM
model = MixedLM(endog=y, exog=X, groups=zip_codes)
result = model.fit()
```

### Pattern 3: Two-Stage Modeling

**What:** Stage 1 models the trend/exposure relationship; Stage 2 models residuals with cross-sectional features.

**When to use:** When domain knowledge suggests a known functional form (premium = exposure × rate).

**Trade-offs:**
- Pros: Highly interpretable; aligns with actuarial practice
- Cons: Requires correct specification; may miss interactions

**Example:**
```python
# Stage 1: Premium per exposure = f(year, trend)
rate_model = linear_regression(X['exposure'], X['premium'])
# Stage 2: Adjust by risk factors
final_prediction = rate_model * adjust_for_fire_risk(features)
```

### Pattern 4: Ensemble of Global and Local Models

**What:** Combine predictions from a global model (capturing shared patterns) and local adjustments (capturing ZIP-specific signals).

**When to use:** When panel has both shared dynamics and local variation.

**Trade-offs:**
- Pros: Balances bias-variance tradeoff; robust
- Cons: More complex pipeline; requires careful weighting

**Example:**
```python
global_pred = global_model.predict(X_test)
local_adj = local_model_per_zip[zip_code].predict(local_features)
final_pred = 0.7 * global_pred + 0.3 * local_adj
```

## Data Flow

### Request Flow (Batch Training)

```
[Raw CSV Files]
       ↓
[Data Loader] → [Pandas DataFrame: insurance + fire + census + weather]
       ↓
[Feature Engineering] → [Feature Matrix: lags, rolling stats, encodings]
       ↓
[Train/Test Split] → [Training Set (2018-2019), Test Set (2020)]
       ↓
[Model Training] → [Trained Model]
       ↓
[Hyperparameter Tuning] → [Best Model]
       ↓
[Final Training on 2018-2020] → [Production Model]
       ↓
[2021 Prediction] → [Predictions DataFrame]
```

### Request Flow (Prediction for 2021)

```
[2021 Features (known: fire_risk, weather, census at last available)]
       ↓
[Apply Same Feature Engineering Pipeline]
       ↓
[Trained Model.predict()]
       ↓
[Output: ZIP → Predicted Premium 2021]
```

### Key Data Flows

1. **Training Flow:** Raw data → joined dataset → feature engineering → model training → evaluation
2. **Prediction Flow:** New features → same transformation → model.predict() → output
3. **Feedback Flow:** Error analysis → feature adjustment → retraining

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 100-1000 ZIPs | Single machine sufficient; pandas or polars for data; LightGBM for modeling |
| 1000-10000 ZIPs | Consider polars for faster joins; feature store to cache computed features |
| 10000+ ZIPs | Distribute feature computation; consider dask or spark for data processing |

### Scaling Priorities

1. **First bottleneck:** Feature engineering time (lag/rolling calculations on panel). Mitigation: pre-compute and cache features.
2. **Second bottleneck:** Model training time (especially grid search). Mitigation: use early stopping, reduce hyperparameter search space.
3. **Third bottleneck:** Memory for large feature matrices. Mitigation: use sparse representations for one-hot encodings.

## Anti-Patterns

### Anti-Pattern 1: Per-ZIP ARIMA with Only 3 Time Points

**What people do:** Fit individual ARIMA models for each ZIP code.

**Why it's wrong:** With only 3 time points (2018-2020), ARIMA cannot be reliably estimated. No degrees of freedom for model selection. Overfits to noise.

**Do this instead:** Use global or hierarchical models that pool information across ZIPs.

### Anti-Pattern 2: Random Train/Test Split on Panel Data

**What people do:** Randomly sample rows for train/test split.

**Why it's wrong:** Causes data leakage; future information leaks into past. Invalidates time-series evaluation.

**Do this instead:** Always split temporally. Train on 2018-2019, validate on 2020. For final model, train on all 3 years.

### Anti-Pattern 3: Ignoring the Panel Structure

**What people do:** Stack all data and treat each row as independent.

**Why it's wrong:** Violates independence assumption; standard errors will be wrong; may miss ZIP-specific effects.

**Do this instead:** Use clustered standard errors, mixed effects models, or include ZIP fixed effects.

### Anti-Pattern 4: Using Future Information as Features

**What people do:** When predicting 2021, including features that would only be known in 2021 (e.g., 2021 earned exposure is actually the target variable).

**Why it's wrong:** Features must be available at prediction time.

**Do this instead:** Use only features known before the prediction period (lagged features, historical averages, or projections from external sources).

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Weather API | Batch download of historical + forecast weather by ZIP | Use last available year for 2021 projection |
| Census API | Static demographic data by ZIP | Update annually; data typically lags by 1-2 years |
| Fire Risk Scoring Service | Risk scores by ZIP | May need interpolation for missing years |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| data → features | Function calls passing DataFrames | Avoid materializing intermediate data to disk |
| features → models | Feature matrices (numpy/pandas) | Use consistent feature sets across models |
| models → validation | Model objects with predict methods | Use sklearn interface for consistency |
| validation → prediction | Best model selection | Can be config-driven |

## Build Order Recommendations

Given the dependencies between components, build in this order:

1. **Phase 1: Data Loading and Joining**
   - Load CSV files and join on ZIP × Year
   - Validate schema matches expectations
   - Handle missing values (recode NaN appropriately)
   - Output: Clean joined dataset

2. **Phase 2: Exploratory Analysis**
   - Understand distributions of target (Earned Premium)
   - Identify correlations between features and target
   - Check for data quality issues
   - Informs feature engineering decisions

3. **Phase 3: Feature Engineering Pipeline**
   - Implement time-series features (lags, rolling means where possible)
   - Implement cross-sectional features (ZIP aggregates)
   - Implement categorical encoding
   - Output: Feature matrix

4. **Phase 4: Train/Test Split and Baseline Models**
   - Implement temporal split (2018-2019 train, 2020 validation)
   - Train simple baseline (linear regression, mean per ZIP)
   - Evaluate to establish baseline metrics

5. **Phase 5: Model Development and Selection**
   - Train multiple candidate models (LightGBM, XGBoost, panel regression)
   - Compare validation metrics
   - Tune hyperparameters
   - Output: Best model

6. **Phase 6: Final Training and Prediction**
   - Retrain best model on full 2018-2020 data
   - Apply to 2021 features (assuming same schema)
   - Export predictions

7. **Phase 7: Error Analysis and Iteration**
   - Analyze prediction errors by ZIP segment
   - Identify underperforming segments
   - Iterate on features or model

## Sources

- [sklearn TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [LightGBM for time series](https://lightgbm.readthedocs.io/en/latest/FAQ.html#how-to-handle-time-series-data)
- [statsmodels MixedLM documentation](https://www.statsmodels.org/stable/mixed_linear.html)
- [Panel data methods in econometrics](https://en.wikipedia.org/wiki/Panel_data)
- [Actuarial machine learning practices - Industry standard approaches]

---
*Architecture research for: Insurance Premium Time Series Forecasting*
*Researched: 2026-04-09*
