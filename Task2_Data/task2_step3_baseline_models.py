"""
Task 2 Step 3: Baseline Models (Linear Regression + Random Forest)
Trains two baseline regressors using strict temporal split (2019 train, 2020 validation).
Outputs: metrics, predictions, validation report, feature importance.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# STEP 1: Verify sklearn installed
try:
    import sklearn
    print(f"sklearn version: {sklearn.__version__}")
except ImportError:
    raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")

# Constants
INPUT_FILE = "Task2_Data/task2_step2_feature_matrix.csv"
METRICS_OUT = "Task2_Data/task2_step3_metrics.csv"
PREDICTIONS_OUT = "Task2_Data/task2_step3_predictions.csv"
VALIDATION_REPORT_OUT = "Task2_Data/task2_step3_validation_report.csv"
FEATURE_IMPORTANCE_OUT = "Task2_Data/task2_step3_rf_feature_importance.csv"

# STEP 2: Load data
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# STEP 3: Define feature columns — exclude identifiers, target, and current-year leakage
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

# STEP 4: Temporal split
train_df = df[df['Year'] == 2019].copy()
val_df = df[df['Year'] == 2020].copy()
print(f"Train (2019): {len(train_df)} rows")
print(f"Validation (2020): {len(val_df)} rows")

y_train = train_df['Earned Premium']
y_val = val_df['Earned Premium']

# STEP 5: Prepare data for Linear Regression with mean imputation across ALL columns
X_train_lr = train_df[feature_cols].copy()
X_val_lr = val_df[feature_cols].copy()

# Mean imputation for NaN across ALL feature columns (not just weather)
# Handle columns with all-NaN by dropping them (can't impute with mean=NaN)
cols_with_all_nan = [col for col in X_train_lr.columns if X_train_lr[col].isna().all()]
if cols_with_all_nan:
    print(f"Dropping columns with all NaN: {cols_with_all_nan}")
    X_train_lr = X_train_lr.drop(columns=cols_with_all_nan)
    X_val_lr = X_val_lr.drop(columns=cols_with_all_nan)
    feature_cols = [c for c in feature_cols if c not in cols_with_all_nan]

for col in X_train_lr.columns:
    if X_train_lr[col].isna().any():
        mean_val = X_train_lr[col].mean()
        if pd.isna(mean_val):
            mean_val = 0  # Fallback for columns where mean is NaN
        X_train_lr[col] = X_train_lr[col].fillna(mean_val)
        X_val_lr[col] = X_val_lr[col].fillna(mean_val)

# Fill any remaining NaN in validation set (columns that were all-valid in train but all-NaN in val)
for col in X_val_lr.columns:
    if X_val_lr[col].isna().any():
        mean_train = X_train_lr[col].mean() if col in X_train_lr.columns else 0
        if pd.isna(mean_train):
            mean_train = 0
        X_val_lr[col] = X_val_lr[col].fillna(mean_train)
        print(f"  Post-hoc filled {col} with train mean {mean_train:.4f} (was all-NaN in val)")

# STEP 6: Train Linear Regression (vanilla OLS, no regularization)
model_lr = LinearRegression()
model_lr.fit(X_train_lr, y_train)
y_pred_lr = model_lr.predict(X_val_lr)
print(f"Linear Regression fitted on {len(X_train_lr)} rows")

# STEP 7: Train Random Forest (handles NaN natively)
X_train_rf = train_df[feature_cols].copy()
X_val_rf = val_df[feature_cols].copy()

model_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model_rf.fit(X_train_rf, y_train)
y_pred_rf = model_rf.predict(X_val_rf)
print(f"Random Forest fitted on {len(X_train_rf)} rows")

# STEP 8: Evaluate both models
def evaluate_regression(model_name, y_true, y_pred, feature_cols=None, model=None):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE on non-zero subset only
    non_zero_mask = y_true > 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true.values[non_zero_mask] - y_pred[non_zero_mask]) / y_true.values[non_zero_mask])) * 100
    else:
        mape = np.nan

    # COVID coefficient from Linear Regression
    covid_coef = None
    if model_name == 'LinearRegression' and hasattr(model, 'coef_'):
        covid_idx = feature_cols.index('is_covid_year')
        covid_coef = model.coef_[covid_idx]

    print(f"\n--- {model_name} Results ---")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"MAPE: {mape:.2f}% (on non-zero subset, n={non_zero_mask.sum()})")
    if covid_coef is not None:
        print(f"COVID coefficient: {covid_coef:,.2f}")

    return {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'covid_coef': covid_coef,
        'non_zero_count': int(non_zero_mask.sum()),
        'zero_count': int((~non_zero_mask).sum())
    }

results_lr = evaluate_regression('LinearRegression', y_val, y_pred_lr, feature_cols, model_lr)
results_rf = evaluate_regression('RandomForest', y_val, y_pred_rf)

# STEP 9: Save metrics and predictions
metrics_df = pd.DataFrame([results_lr, results_rf])
metrics_df.to_csv(METRICS_OUT, index=False)

predictions_df = val_df[['Year', 'ZIP', 'Category', 'Earned Premium']].copy()
predictions_df['lr_pred'] = y_pred_lr
predictions_df['rf_pred'] = y_pred_rf
predictions_df['lr_residual'] = y_val - y_pred_lr
predictions_df['rf_residual'] = y_val - y_pred_rf
predictions_df.to_csv(PREDICTIONS_OUT, index=False)

validation_report = {
    'total_validation_rows': len(val_df),
    'train_rows': len(train_df),
    'non_zero_premium_count': results_lr['non_zero_count'],
    'zero_premium_count': results_lr['zero_count'],
    'zero_premium_pct': results_lr['zero_count'] / len(val_df) * 100,
    'lr_covid_coef': results_lr['covid_coef'],
    'lr_rmse': results_lr['rmse'],
    'lr_mae': results_lr['mae'],
    'lr_mape': results_lr['mape'],
    'rf_rmse': results_rf['rmse'],
    'rf_mae': results_rf['mae'],
    'rf_mape': results_rf['mape'],
}
report_df = pd.DataFrame([validation_report])
report_df.to_csv(VALIDATION_REPORT_OUT, index=False)

print("\nSaved:")
print(f"- {METRICS_OUT}")
print(f"- {PREDICTIONS_OUT}")
print(f"- {VALIDATION_REPORT_OUT}")

# STEP 10: Feature importance from Random Forest
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_rf.feature_importances_
}).sort_values(by='importance', ascending=False)
importance_df.to_csv(FEATURE_IMPORTANCE_OUT, index=False)
print(f"\nTop 10 features (Random Forest):")
print(importance_df.head(10).to_string(index=False))

print(f"\n--- Feature importance saved to {FEATURE_IMPORTANCE_OUT} ---")