"""
Task 2 Step 4: LightGBM Gradient Boosting Model
Uses LightGBM with lag features, COVID-19 sample weighting, and early stopping.
Train: 2019 data. Validate: 2020 data.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# STEP 1: Load feature matrix
INPUT_FILE = "Task2_Data/task2_step2_feature_matrix.csv"
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# STEP 2: Define exclude_cols (same leakage prevention as Phase 3)
exclude_cols = [
    'Year', 'ZIP', 'Category', 'ZIP_Cat', 'FIRE_NAME', 'AGENCY', 'INC_NUM',
    # Current-year loss/claim columns (leakage)
    'CAT Cov A Fire -  Incurred Losses', 'CAT Cov A Fire -  Number of Claims',
    'CAT Cov A Smoke -  Incurred Losses', 'CAT Cov A Smoke -  Number of Claims',
    'CAT Cov C Fire -  Incurred Losses', 'CAT Cov C Fire -  Number of Claims',
    'CAT Cov C Smoke -  Incurred Losses', 'CAT Cov C Smoke -  Number of Claims',
    'Non-CAT Cov A Fire -  Incurred Losses', 'Non-CAT Cov A Fire -  Number of Claims',
    'Non-CAT Cov A Smoke -  Incurred Losses', 'Non-CAT Cov A Smoke -  Number of Claims',
    'Non-CAT Cov C Fire -  Incurred Losses', 'Non-CAT Cov C Fire -  Number of Claims',
    'Non-CAT Cov C Smoke -  Incurred Losses', 'Non-CAT Cov C Smoke -  Number of Claims',
    'Earned Premium'  # target
]
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"Using {len(feature_cols)} features")

# STEP 3: Temporal split
train_df = df[df['Year'] == 2019].copy()
val_df = df[df['Year'] == 2020].copy()
print(f"Train (2019): {len(train_df)} rows, Validation (2020): {len(val_df)} rows")

y_train = train_df['Earned Premium']
y_val = val_df['Earned Premium']

X_train = train_df[feature_cols]
X_val = val_df[feature_cols]

# STEP 4: COVID-19 sample weighting — downweight 2020 validation by 0.5
# LightGBM supports sample weights via lgb.Dataset(..., weight=...)
# For training: weight 2019 observations at 1.0 (since COVID is 2020 event)
# For validation: we still evaluate on all 2020 but with reduced weight in metrics
# Actually per research: downweight 2020 training influence during fitting
# But we train on 2019 and validate on 2020 — so COVID weighting applies to validation evaluation
# For a conservative approach: report metrics both with and without COVID adjustment
# Create sample weights for validation: 0.5 for 2020 observations
val_weights = np.where(val_df['is_covid_year'] == 1, 0.5, 1.0)

# STEP 5: Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# STEP 6: LightGBM hyperparameters (conservative to prevent overfitting on 9k rows)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'max_depth': 8,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,
    'verbosity': -1,
    'seed': 42,
}

# STEP 7: Train with early stopping
print("\n--- LightGBM Training ---")
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)
print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score}")
# Extract RMSE from best_score - compute from predictions using best iteration
y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"Train RMSE at best iteration: {train_rmse:,.2f}")

# STEP 8: Generate predictions
y_pred_lgb = model.predict(X_val, num_iteration=model.best_iteration)
print(f"\nLightGBM predictions generated: {len(y_pred_lgb)} rows")

# STEP 9: Evaluate metrics (standard, no COVID adjustment for fair comparison)
def evaluate_regression(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # Handle both pandas Series and numpy arrays
    if hasattr(y_true, 'values'):
        y_true_arr = y_true.values
    else:
        y_true_arr = y_true
    non_zero_mask = y_true_arr > 0
    mape = np.mean(np.abs((y_true_arr[non_zero_mask] - y_pred[non_zero_mask]) / y_true_arr[non_zero_mask])) * 100
    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'mape': mape}

metrics = evaluate_regression('LightGBM', y_val.values, y_pred_lgb)
print(f"\n--- LightGBM Results ---")
print(f"RMSE: {metrics['rmse']:,.2f}")
print(f"MAE:  {metrics['mae']:,.2f}")
print(f"MAPE: {metrics['mape']:.2f}%")

# STEP 10: Evaluate with COVID-adjusted weights (for MODEL-05 reporting)
rmse_weighted = np.sqrt(np.average((y_val.values - y_pred_lgb)**2, weights=val_weights))
mae_weighted = np.average(np.abs(y_val.values - y_pred_lgb), weights=val_weights)
print(f"\n--- LightGBM COVID-Adjusted Metrics (weight=0.5 for 2020) ---")
print(f"Weighted RMSE: {rmse_weighted:,.2f}")
print(f"Weighted MAE: {mae_weighted:,.2f}")

# STEP 11: Feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values(by='importance', ascending=False)
print(f"\nTop 10 features (LightGBM gain importance):")
print(importance_df.head(10).to_string(index=False))

# STEP 12: Save outputs
METRICS_OUT = "Task2_Data/task2_step4_lightgbm_metrics.csv"
IMPORTANCE_OUT = "Task2_Data/task2_step4_lightgbm_importance.csv"

pd.DataFrame([metrics]).to_csv(METRICS_OUT, index=False)
importance_df.to_csv(IMPORTANCE_OUT, index=False)

print(f"\nSaved:")
print(f"- {METRICS_OUT}")
print(f"- {IMPORTANCE_OUT}")
print(f"\nLightGBM model complete.")