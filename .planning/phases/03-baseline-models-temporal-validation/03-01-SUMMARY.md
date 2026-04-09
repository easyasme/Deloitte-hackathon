# Phase 3 Summary: Baseline Models with Temporal Validation

## Objective
Train and evaluate two baseline regression models (Linear Regression, Random Forest) on the Phase 2 feature matrix using a strict temporal split (2019 train, 2020 validation).

## Temporal Split
- **Training:** 9,329 rows (2019 only)
- **Validation:** 12,708 rows (2020 only)

## Models Trained

### Linear Regression (vanilla OLS, no regularization)
- Mean imputation applied for NaN values across 42 features
- `expanding_fire_risk_std` dropped (all-NaN in training set)
- `Avg PPC` post-hoc filled with train mean (all-NaN in validation set)

### Random Forest Regressor
- n_estimators=100, max_depth=15, min_samples_leaf=5
- Handles NaN natively via surrogate splits (no imputation needed)

## Metrics

| Model | RMSE | MAE | MAPE (non-zero) |
|-------|------|-----|------------------|
| Linear Regression | 515,829.79 | 194,560.88 | 7754.70% |
| Random Forest | 999,715.70 | 165,535.85 | 40.42% |

**Note:** Linear Regression's MAPE is inflated by near-zero premium values creating extreme percentage errors. Random Forest achieves a much more reasonable 40.42% MAPE.

## Key Findings

### Random Forest Feature Importance (Top 10)
1. `Earned Premium_lag1` — 90.9% (dominant lag feature)
2. `Cov C Amount Weighted Avg` — 2.0%
3. `Earned Exposure` — 1.5%
4. `Cov A Amount Weighted Avg` — 1.0%
5. `median_monthly_housing_costs` — 0.9%
6. `Number of Very High Fire Risk Exposure` — 0.8%
7. `Avg PPC_lag1` — 0.7%
8. `owner_occupied_housing_units` — 0.4%
9. `Number of Moderate Fire Risk Exposure` — 0.4%
10. `avg_tmax_c` — 0.3%

### COVID-19 Coefficient
The `is_covid_year` coefficient from Linear Regression is **0.00**, indicating the simple linear model does not capture a meaningful COVID-19 structural break. This aligns with the plan's documentation of the open question.

### Zero Premium Analysis
- Non-zero premium rows: 9,887 (77.8%)
- Zero premium rows: 2,821 (22.2%)

## Artifacts Produced
- `Task2_Data/task2_step3_metrics.csv` — RMSE, MAE, MAPE for both models
- `Task2_Data/task2_step3_validation_report.csv` — COVID coefficient, zero-premium count, row counts
- `Task2_Data/task2_step3_predictions.csv` — 12,708 validation rows with predictions + residuals
- `Task2_Data/task2_step3_rf_feature_importance.csv` — 42 features with RF importance scores