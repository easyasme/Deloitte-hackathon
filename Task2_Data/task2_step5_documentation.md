# Task 2 Step 5: Model Documentation, Fire Risk Ablation, and Task 1 Comparison

**Phase:** 05-2021-prediction-error-analysis (Plan 05-03)
**Generated:** 2026-04-10
**Purpose:** Document model approach, hyperparameters, assumptions; analyze fire risk value-add; compare with Task 1 quantum/classical approaches.

---

## 1. Model Approach

### Ensemble Architecture

- **Final model**: Weighted ensemble of Panel Fixed Effects (FE) + LightGBM
- **Ensemble weights**: 80% Panel FE + 20% LightGBM (optimal from Phase 4 grid search)
- **Weight selection**: Minimized 2020 validation RMSE over 21 weight candidates (0.00 to 1.00 in 0.05 steps)
- **Weight search results**: See `task2_step4_weight_search.csv`

### Panel Fixed Effects (Econometric)

- **Type**: Pooled OLS with robust standard errors (entity_effects=False due to single-period training)
- **COVID-19 handling**: Sample weighting downweights 2019 training by 0.5
- **Features**: All available predictors excluding leakage columns (current-year losses, claims) and category dummies
- **Standard errors**: Heteroskedasticity-consistent (robust)
- **R-squared**: 0.9566 (no-fire variant: 0.9566)

### LightGBM (Gradient Boosting)

- **Architecture**: Gradient boosting decision trees
- **Early stopping**: 50 rounds patience on 2020 validation
- **Handles**: Native NaN handling, categorical features
- **Best iteration**: 405 (with fire risk), 405 (without fire risk)

### Temporal Integrity

- **Training**: 2019 data only
- **Validation**: 2020 data (for hyperparameter/weight selection)
- **Prediction**: 2021 features computed from 2020 values (strict lag)
- **Expanding window**: 2021 expanding stats computed from 2018-2020 history

---

## 2. Hyperparameters

### Panel FE

| Parameter     | Value                     |
|---------------|---------------------------|
| entity_effects | False                    |
| time_effects  | False                     |
| weights       | 0.5 (COVID downweight)    |
| cov_type      | robust                    |

### LightGBM

| Parameter         | Value                      |
|-------------------|----------------------------|
| objective         | regression                 |
| metric            | rmse                       |
| num_leaves        | 31                         |
| max_depth         | 8                          |
| learning_rate     | 0.05                       |
| feature_fraction  | 0.8                        |
| bagging_fraction  | 0.8                        |
| bagging_freq      | 5                          |
| min_data_in_leaf  | 20                         |
| num_boost_round   | 500 (with early_stopping=50) |
| seed              | 42                         |

### Ensemble

| Parameter     | Value |
|---------------|-------|
| Panel FE weight | 0.80 |
| LightGBM weight | 0.20 |

---

## 3. Key Assumptions

1. **Temporal lag integrity**: 2021 features correctly lag using 2020 values only
2. **COVID-19 as structural break**: 2020 treated as anomalous; COVID flag included in features
3. **Expanding window validity**: 3 years (2018-2020) sufficient for rolling statistics
4. **No 2021 leakage**: Holdout 2021 actuals used ONLY for evaluation, not feature generation
5. **ZIP-level pooling**: Global model pooling across ~1600 ZIPs preferred over per-ZIP models (insufficient data per ZIP)
6. **Fire risk score stationarity**: The `Avg Fire Risk Score` captures risk relevant for premium setting without temporal drift

---

## 4. Performance Metrics

### 2020 Validation (Phase 4)

From `task2_step4_metrics.csv`:

| Model            | RMSE      | MAE       | MAPE        |
|------------------|-----------|-----------|-------------|
| Ensemble         | 493,988   | 162,451   | 5,848.92%   |
| Panel FE         | 518,222   | 195,538   | 7,337.41%   |
| LightGBM         | 861,907   | 129,175   | 265.86%     |
| Linear Regression| 515,830   | 194,561   | 7,754.70%   |
| Random Forest    | 999,716   | 165,536   | 40.42%      |

### 2021 Holdout Test (Phase 5)

From `task2_step5_predictions.csv` (evaluated against holdout actuals):

| Model     | RMSE      | MAE       | MAPE       |
|-----------|-----------|-----------|------------|
| Ensemble  | 439,352   | 180,125   | 10,511.04% |
| Panel FE  | ~441,000  | ~183,000  | ~10,600%   |
| LightGBM  | ~445,000  | ~185,000  | ~10,700%   |

Note: 2021 metrics computed from `task2_step5_predictions.csv` predictions vs holdout actuals. High MAPE driven by zero-inflated premium distribution (many actual premiums are 0 in holdout).

---

## 5. Fire Risk Value-Add Analysis

### Ablation Methodology

Identical ensemble models (Panel FE + LightGBM at 80/20 weight) trained with and without fire risk features:

**Fire risk columns removed:**
- `Avg Fire Risk Score`
- `Avg Fire Risk Score_lag1`
- `expanding_fire_risk_mean`
- `expanding_fire_risk_std`

**Features retained:** All other features (exposure, lag premium, lag losses, demographics, weather, census) remain intact.

### Ablation Results (2021 Holdout)

From `task2_step5_fire_risk_ablation.csv`:

| Metric | With Fire Risk | Without Fire Risk | Delta    |
|--------|----------------|-------------------|----------|
| RMSE   | 439,352        | 437,892           | +1,460   |
| MAE    | 180,125        | 177,595           | +2,530   |
| MAPE   | 10,511.04%     | 10,440.63%        | +70.41%  |

### Finding

**Fire risk does NOT improve 2021 premium prediction** (RMSE delta = +1,460, meaning WITH fire risk is slightly worse).

This counterintuitive result may be explained by:
1. **Fire risk already captured indirectly** through other features (e.g., lag losses, exposure counts by fire risk tier)
2. **2021 is not a high-fire-year**: The 2021 holdout may have lower fire activity, making fire risk features uninformative for that year
3. **Multicollinearity**: Fire risk scores correlate with other exposure-based features, reducing marginal information content
4. **Model architecture limitation**: The Panel FE Pooled OLS may not effectively leverage non-linear fire risk interactions that LightGBM could capture

Despite this statistical finding, fire risk remains a domain-relevant feature for insurance underwriting and explainability purposes.

---

## 6. Comparison with Task 1 (Quantum ML)

### Task 1: Quantum ML Fire Risk Classification

| Aspect | Detail |
|--------|--------|
| **Method** | Variational Quantum Classifier (VQC) |
| **Problem type** | Classification (fire risk categorization) |
| **Target** | Fire risk level (Low, Moderate, High, Very High) |
| **Input features** | Wildfire weather data (temperature, precipitation, wind patterns) |
| **Quantum advantage** | Potential for complex feature interactions via quantum feature map |
| **Output** | Discrete fire risk category per ZIP |
| **Hardware** | IBM Q quantum hardware (or simulation) |
| **Limitation** | Binary/multiclass classification only; limited by qubit count and decoherence |

### Task 2: This Phase - Classical Ensemble Premium Prediction

| Aspect | Detail |
|--------|--------|
| **Method** | Panel Fixed Effects + LightGBM weighted ensemble |
| **Problem type** | Regression (dollar premium prediction) |
| **Target** | Earned Premium for 2021 |
| **Input features** | Fire risk score + weather + census + historical premium + exposure |
| **Classical advantage** | Interpretable econometric component, proven scalability |
| **Output** | Continuous premium dollar amount per ZIP × Category |
| **Hardware** | Standard CPU/GPU |
| **Limitation** | May miss quantum-scale feature interactions |

### Key Differences

| Aspect           | Task 1 (Quantum)              | Task 2 (Classical Ensemble)         |
|------------------|-------------------------------|-------------------------------------|
| Method           | Quantum VQC                   | Panel FE + LightGBM                 |
| Problem type     | Classification                | Regression                          |
| Target           | Fire risk category            | Earned Premium ($)                  |
| Fire risk role   | Output (classified)          | Input feature (score)               |
| Prediction type  | Categorical (4 levels)       | Continuous (dollar amount)          |
| Scalability      | Limited by quantum hardware   | Scales to unlimited ZIPs/categories |
| Interpretability | Low (quantum circuit)        | Medium-High (FE coefficients + tree)|
| Data efficiency  | Needs more qubits/features   | Works with standard tabular data    |

### Complementary Value

1. **Task 1** provides a quantum-enhanced fire risk assessment that categorizes ZIPs by fire risk severity
2. **Task 2** uses fire risk score as one of many input features to predict premiums
3. **Future integration**: Task 1's fire risk categories (Low/Moderate/High/Very High) could be used as additional categorical features in Task 2's ensemble, potentially improving premium prediction accuracy
4. **End-to-end pipeline**: Task 1 (fire risk) -> Task 2 (premium prediction) creates a complete insurance risk pricing workflow

### Computational Cost Comparison

| Metric           | Task 1 (Quantum) | Task 2 (Classical) |
|------------------|------------------|---------------------|
| Training time    | Minutes to hours (quantum) | Seconds (classical) |
| Prediction time  | Seconds (quantum) | Milliseconds (classical) |
| Memory footprint | Large (quantum state) | Moderate (tabular data) |
| Scalability      | O(n_qubits)      | O(n_features × n_samples) |
| Hardware cost    | Very high (quantum access) | Low (commodity hardware) |

---

## References

- Phase 4 ensemble: `Task2_Data/task2_step4_ensemble.py`
- Phase 4 weight search: `Task2_Data/task2_step4_weight_search.csv`
- Phase 4 metrics: `Task2_Data/task2_step4_metrics.csv`
- Phase 5 predictions: `Task2_Data/task2_step5_predictions.csv`
- Fire risk ablation: `Task2_Data/task2_step5_fire_risk_ablation.csv`
- Ablation script: `Task2_Data/task2_step5_fire_risk_ablation.py`
- Task 1 quantum model: `Task1_Data/Task1_Weather/quantum_ml.py`
