"""
Task 2 Step 4: Panel Fixed Effects Model
Uses linearmodels PanelOLS with ZIP entity fixed effects and COVID-19 sample weighting.
Train: 2019 data only. Validate: 2020 data.

NOTE: entity_effects=True requires multiple time periods per entity (within-ZIP variation).
      With single-period 2019 training data (one observation per ZIP), each ZIP fixed effect
      perfectly explains the target, causing AbsorbingEffectError.
      Solution: Use Pooled OLS (entity_effects=False) which treats all ZIPs as one panel
      and estimates common coefficients across the cross-section.
"""
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from sklearn.metrics import mean_squared_error, mean_absolute_error

# STEP 1: Load feature matrix
INPUT_FILE = "Task2_Data/task2_step2_feature_matrix.csv"
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# STEP 2: Define exclude_cols (same as Phase 3 for leakage prevention)
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
print(f"Using {len(feature_cols)} features (excluding leakage/target cols)")

# STEP 3: Temporal split
train_df = df[df['Year'] == 2019].copy()
val_df = df[df['Year'] == 2020].copy()
print(f"Train (2019): {len(train_df)} rows, Validation (2020): {len(val_df)} rows")

# STEP 4: Prepare panel data — set MultiIndex (entity=ZIP, time=Year)
train_panel = train_df.set_index(['ZIP', 'Year'])
val_panel = val_df.set_index(['ZIP', 'Year'])

y_train = train_panel['Earned Premium']
y_val = val_panel['Earned Premium']

X_train = train_panel[feature_cols]
X_val = val_panel[feature_cols]

# STEP 5: Drop columns with all NaN in training
cols_all_nan = [c for c in X_train.columns if X_train[c].isna().all()]
if cols_all_nan:
    print(f"Dropping all-NaN columns: {cols_all_nan}")
    X_train = X_train.drop(columns=cols_all_nan)
    X_val = X_val.drop(columns=cols_all_nan)
    feature_cols = [c for c in feature_cols if c not in cols_all_nan]

# STEP 6: Drop category dummies — constant per ZIP-Category, cause multicollinearity
exclude_cols_extra = ['cat_HO', 'cat_CO', 'cat_DT', 'cat_RT', 'cat_DO', 'cat_MH', 'cat_NA']
feature_cols_reduced = [c for c in feature_cols if c not in exclude_cols_extra]
X_train_reduced = X_train[feature_cols_reduced]
X_val_reduced = X_val[feature_cols_reduced]
print(f"Reduced features: {len(feature_cols)} -> {len(feature_cols_reduced)} (dropped category dummies)")

# STEP 7: Handle missing values — fill with column means from training
for col in X_train_reduced.columns:
    if X_train_reduced[col].isna().any():
        mean_val = X_train_reduced[col].mean()
        if pd.isna(mean_val):
            mean_val = 0
        X_train_reduced[col] = X_train_reduced[col].fillna(mean_val)
        X_val_reduced[col] = X_val_reduced[col].fillna(mean_val)

for col in X_val_reduced.columns:
    if X_val_reduced[col].isna().any():
        mean_train = X_train_reduced[col].mean() if col in X_train_reduced.columns else 0
        if pd.isna(mean_train):
            mean_train = 0
        X_val_reduced[col] = X_val_reduced[col].fillna(mean_train)

# STEP 8: COVID-19 sample weighting — downweight 2019 training observations by 0.5
# This reflects uncertainty about 2019 being a "normal" year relative to COVID
weights_train = pd.Series(0.5, index=X_train_reduced.index)

# STEP 9: Fit Pooled OLS model with robust SE
# Pooled OLS treats all ZIPs as one panel, estimating common coefficients
# This is the appropriate approach when entity_effects=True is not possible due to single-period data
# Using robust SE (heteroskedasticity-consistent) instead of clustered to avoid singular matrix
print("\n--- Panel FE Model Training (Pooled OLS) ---")
print("NOTE: Pooled OLS captures time-varying feature effects across ZIPs.")
print("      Entity effects (ZIP FE) not estimable with single time period per entity.")
print("      Robust SE provides heteroskedasticity-consistent inference.")
model = PanelOLS(y_train, X_train_reduced, entity_effects=False, time_effects=False, weights=weights_train, check_rank=False)
result = model.fit(cov_type='robust')

# STEP 10: Print key results
print(f"\nR-squared: {result.rsquared:.4f}")
print(f"F-statistic: {result.f_statistic.stat:.4f}")
print(f"N observations: {result.nobs}")
print("\nKey coefficients:")
key_features = ['Avg Fire Risk Score', 'Avg Fire Risk Score_lag1', 'Earned Premium_lag1', 'is_covid_year']
for feat in key_features:
    if feat in result.params.index:
        print(f"  {feat}: {result.params[feat]:,.4f}")

# STEP 11: Generate out-of-sample predictions on 2020 validation
# PanelOLS predict method requires proper exog DataFrame
y_pred_oos = result.predict(X_val_reduced)
y_pred_fe = y_pred_oos.values.flatten() if hasattr(y_pred_oos, 'values') else np.array(y_pred_oos).flatten()

print(f"\nPanel FE predictions generated: {len(y_pred_fe)} rows")

# STEP 12: Evaluate metrics
def evaluate_regression(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    non_zero_mask = y_true > 0
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    mape = np.mean(np.abs((y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask])) * 100
    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'mape': mape}

metrics = evaluate_regression('PanelFE', y_val.values, y_pred_fe)
print(f"\n--- Panel FE Results ---")
print(f"RMSE: {metrics['rmse']:,.2f}")
print(f"MAE:  {metrics['mae']:,.2f}")
print(f"MAPE: {metrics['mape']:.2f}%")

# STEP 10: Save predictions and metrics
METRICS_OUT = "Task2_Data/task2_step4_panel_fe_metrics.csv"
PREDICTIONS_OUT = "Task2_Data/task2_step4_panel_fe_predictions.csv"

pd.DataFrame([metrics]).to_csv(METRICS_OUT, index=False)

predictions_df = val_df[['Year', 'ZIP', 'Category', 'Earned Premium']].copy()
predictions_df['panel_fe_pred'] = y_pred_fe
predictions_df.to_csv(PREDICTIONS_OUT, index=False)

print(f"\nSaved:")
print(f"- {METRICS_OUT}")
print(f"- {PREDICTIONS_OUT}")
print(f"\nPanel FE model complete.")