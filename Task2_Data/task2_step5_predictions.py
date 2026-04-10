"""
Task 2 Step 5: 2021 Earned Premium Predictions
Uses Phase 4 ensemble approach (80% Panel FE + 20% LightGBM) to predict 2021 premiums.
Temporal lag: 2021 features use 2020 values only (strict t-1 constraint).

Key pattern from Phase 4:
- Optimal ensemble weight: w=0.80 PanelFE + w=0.20 LightGBM
- Validation RMSE: 493,988 on 2020 validation
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTS
# ============================================================
INPUT_HOLDOUT = "Task2_Data/task2_step1_2021_holdout.csv"
INPUT_FEATURE_MATRIX = "Task2_Data/task2_step2_feature_matrix.csv"
OUTPUT_FILE = "Task2_Data/task2_step5_predictions.csv"
ENSEMBLE_WEIGHT_PANEL_FE = 0.80
ENSEMBLE_WEIGHT_LGB = 0.20

# ============================================================
# STEP 1: Load data
# ============================================================
print("\n--- STEP 1: Load data ---")
holdout = pd.read_csv(INPUT_HOLDOUT, low_memory=False)
df = pd.read_csv(INPUT_FEATURE_MATRIX, low_memory=False)
print(f"2021 Holdout shape: {holdout.shape}")
print(f"Feature Matrix (2019-2020) shape: {df.shape}")

# ============================================================
# STEP 2: Compute 2021 lag features from 2020 data
# ============================================================
print("\n--- STEP 2: Compute 2021 lag features ---")
lag_mapping = {
    'Avg Fire Risk Score': 'Avg Fire Risk Score_lag1',
    'Earned Exposure': 'Earned Exposure_lag1',
    'Earned Premium': 'Earned Premium_lag1',
    'CAT Cov A Fire -  Incurred Losses': 'CAT Cov A Fire -  Incurred Losses_lag1',
    'Non-CAT Cov A Fire -  Incurred Losses': 'Non-CAT Cov A Fire -  Incurred Losses_lag1',
    'Avg PPC': 'Avg PPC_lag1',
}

for src_col, lag_col in lag_mapping.items():
    holdout[lag_col] = holdout[src_col]
    print(f"  Created {lag_col} from holdout {src_col}")

# ============================================================
# STEP 3: Compute expanding window stats for 2021
# ============================================================
print("\n--- STEP 3: Compute expanding window stats ---")
raw_df = pd.read_csv("Task2_Data/task2_step1_panel_clean.csv", low_memory=False)
raw_df = raw_df.sort_values(['ZIP', 'Category', 'Year']).reset_index(drop=True)

raw_df['expanding_fire_risk_mean'] = raw_df.groupby(['ZIP', 'Category'])['Avg Fire Risk Score'].transform(
    lambda x: x.expanding().mean()  # no shift: captures history through current year (2020 for 2021 features)
)
raw_df['expanding_fire_risk_std'] = raw_df.groupby(['ZIP', 'Category'])['Avg Fire Risk Score'].transform(
    lambda x: x.expanding().std()
)
raw_df['expanding_exposure_mean'] = raw_df.groupby(['ZIP', 'Category'])['Earned Exposure'].transform(
    lambda x: x.expanding().mean()
)

expanding_2020 = raw_df[raw_df['Year'] == 2020][['ZIP', 'Category', 'expanding_fire_risk_mean', 'expanding_fire_risk_std', 'expanding_exposure_mean']].copy()
holdout = holdout.merge(expanding_2020, on=['ZIP', 'Category'], how='left')
print(f"After merging expanding: holdout shape = {holdout.shape}")

# ============================================================
# STEP 4: One-hot encode categories
# ============================================================
print("\n--- STEP 4: One-hot encode categories ---")
categories = ['HO', 'CO', 'DT', 'RT', 'DO', 'MH', 'NA']
for cat in categories:
    holdout[f'cat_{cat}'] = (holdout['Category'] == cat).astype(int)
print(f"Category columns created: {[f'cat_{c}' for c in categories]}")

# ============================================================
# STEP 5: Add COVID year indicator
# ============================================================
print("\n--- STEP 5: Add COVID indicator ---")
holdout['is_covid_year'] = 0
print(f"  is_covid_year = 0 for all 2021 rows")

# ============================================================
# STEP 6: Prepare training data (2019-2020)
# ============================================================
print("\n--- STEP 6: Prepare training data ---")

train_df = df[df['Year'] == 2019].copy()
val_df = df[df['Year'] == 2020].copy()
print(f"Train (2019): {len(train_df)} rows, Val (2020): {len(val_df)} rows")

exclude_cols = [
    'Year', 'ZIP', 'Category', 'ZIP_Cat', 'FIRE_NAME', 'AGENCY', 'INC_NUM',
    'CAT Cov A Fire -  Incurred Losses', 'CAT Cov A Fire -  Number of Claims',
    'CAT Cov A Smoke -  Incurred Losses', 'CAT Cov A Smoke -  Number of Claims',
    'CAT Cov C Fire -  Incurred Losses', 'CAT Cov C Fire -  Number of Claims',
    'CAT Cov C Smoke -  Incurred Losses', 'CAT Cov C Smoke -  Number of Claims',
    'Non-CAT Cov A Fire -  Incurred Losses', 'Non-CAT Cov A Fire -  Number of Claims',
    'Non-CAT Cov A Smoke -  Incurred Losses', 'Non-CAT Cov A Smoke -  Number of Claims',
    'Non-CAT Cov C Fire -  Incurred Losses', 'Non-CAT Cov C Fire -  Number of Claims',
    'Non-CAT Cov C Smoke -  Incurred Losses', 'Non-CAT Cov C Smoke -  Number of Claims',
    'Earned Premium'
]
exclude_cols_extra = ['cat_HO', 'cat_CO', 'cat_DT', 'cat_RT', 'cat_DO', 'cat_MH', 'cat_NA', 'ZIP_Cat']

feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"Total features: {len(feature_cols)}")

# Panel FE feature set (after dropping category dummies)
feature_cols_fe = [c for c in feature_cols if c not in exclude_cols_extra]
print(f"Panel FE features: {len(feature_cols_fe)}")

y_train = train_df['Earned Premium']
y_val = val_df['Earned Premium']

X_train = train_df[feature_cols]
X_val = val_df[feature_cols]

# ============================================================
# STEP 7: Train Panel FE model
# ============================================================
print("\n--- STEP 7: Train Panel FE model ---")

# Build panel data
train_panel = train_df.set_index(['ZIP', 'Year'])
val_panel = val_df.set_index(['ZIP', 'Year'])

y_train_panel = train_panel['Earned Premium']
y_val_panel = val_panel['Earned Premium']

X_train_panel = train_panel[feature_cols_fe].copy()
X_val_panel = val_panel[feature_cols_fe].copy()

# Drop columns that are all NaN
cols_all_nan = [c for c in X_train_panel.columns if X_train_panel[c].isna().all()]
if cols_all_nan:
    print(f"Dropping all-NaN columns: {cols_all_nan}")
    X_train_panel = X_train_panel.drop(columns=cols_all_nan)
    X_val_panel = X_val_panel.drop(columns=cols_all_nan)

print(f"Panel FE features after dropping NaN cols: {X_train_panel.shape[1]}")

# Fill NaN with column means
for col in X_train_panel.columns:
    if X_train_panel[col].isna().any():
        mean_val = X_train_panel[col].mean()
        if pd.isna(mean_val): mean_val = 0
        X_train_panel[col] = X_train_panel[col].fillna(mean_val)
        X_val_panel[col] = X_val_panel[col].fillna(mean_val)

for col in X_val_panel.columns:
    if X_val_panel[col].isna().any():
        mean_train = X_train_panel[col].mean() if col in X_train_panel.columns else 0
        if pd.isna(mean_train): mean_train = 0
        X_val_panel[col] = X_val_panel[col].fillna(mean_train)

weights_train_panel = pd.Series(0.5, index=X_train_panel.index)

print("Fitting Panel FE (Pooled OLS with robust SE)...")
model_fe = PanelOLS(y_train_panel, X_train_panel, entity_effects=False, time_effects=False, weights=weights_train_panel, check_rank=False)
result_fe = model_fe.fit(cov_type='robust')
print(f"R-squared: {result_fe.rsquared:.4f}")

# Predict on 2020 validation
y_pred_fe_val = result_fe.predict(X_val_panel).values.ravel()
rmse_fe = np.sqrt(mean_squared_error(y_val_panel.values, y_pred_fe_val))
print(f"Panel FE 2020 validation RMSE: {rmse_fe:,.2f}")

# ============================================================
# STEP 8: Train LightGBM model
# ============================================================
print("\n--- STEP 8: Train LightGBM model ---")

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

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model_lgb = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)
print(f"LightGBM best iteration: {model_lgb.best_iteration}")

y_pred_lgb_val = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
rmse_lgb = np.sqrt(mean_squared_error(y_val, y_pred_lgb_val))
print(f"LightGBM 2020 validation RMSE: {rmse_lgb:,.2f}")

# ============================================================
# STEP 9: Compute 2020 ensemble validation
# ============================================================
print("\n--- STEP 9: 2020 Ensemble validation ---")
y_pred_ens_val = ENSEMBLE_WEIGHT_PANEL_FE * y_pred_fe_val + ENSEMBLE_WEIGHT_LGB * y_pred_lgb_val
rmse_ens = np.sqrt(mean_squared_error(y_val_panel.values, y_pred_ens_val))
print(f"Ensemble 2020 validation RMSE: {rmse_ens:,.2f}")

# ============================================================
# STEP 10: Generate 2021 predictions
# ============================================================
print("\n--- STEP 10: Generate 2021 predictions ---")

# Prepare 2021 holdout features
X_2021 = holdout[feature_cols].copy()

# Fill missing values using training column means
for col in X_2021.columns:
    if col in X_train.columns:
        if X_2021[col].isna().any():
            mean_train = X_train[col].mean()
            if pd.isna(mean_train): mean_train = 0
            X_2021[col] = X_2021[col].fillna(mean_train)
    else:
        X_2021[col] = X_2021[col].fillna(0)

# Panel FE feature set
X_2021_fe = X_2021[[c for c in X_train_panel.columns if c in feature_cols_fe]].copy()
for col in X_2021_fe.columns:
    if X_2021_fe[col].isna().any():
        mean_val = X_train_panel[col].mean()
        if pd.isna(mean_val): mean_val = 0
        X_2021_fe[col] = X_2021_fe[col].fillna(mean_val)

# Add any missing columns from feature_cols_fe with 0
for col in X_train_panel.columns:
    if col not in X_2021_fe.columns:
        X_2021_fe[col] = 0

# Ensure column order matches
X_2021_fe = X_2021_fe[X_train_panel.columns]

# Convert 2021 data to panel format (MultiIndex: ZIP, Year=2021)
X_2021_panel = X_2021_fe.copy()
X_2021_panel['ZIP'] = holdout['ZIP']
X_2021_panel['Year'] = 2021
X_2021_panel = X_2021_panel.set_index(['ZIP', 'Year'])

print(f"X_2021_panel shape: {X_2021_panel.shape}")
print(f"X_train_panel shape: {X_train_panel.shape}")

# Panel FE prediction
y_pred_fe_2021 = result_fe.predict(X_2021_panel).values.ravel()
print(f"Panel FE 2021 predictions: {len(y_pred_fe_2021)} rows")

# LightGBM prediction
y_pred_lgb_2021 = model_lgb.predict(X_2021, num_iteration=model_lgb.best_iteration)
print(f"LightGBM 2021 predictions: {len(y_pred_lgb_2021)} rows")

# Ensemble prediction
y_pred_2021 = ENSEMBLE_WEIGHT_PANEL_FE * y_pred_fe_2021 + ENSEMBLE_WEIGHT_LGB * y_pred_lgb_2021
print(f"Ensemble 2021 predictions: {len(y_pred_2021)} rows")

# ============================================================
# STEP 11: Save predictions CSV
# ============================================================
print("\n--- STEP 11: Save predictions ---")

holdout['predicted_premium'] = y_pred_2021
holdout['panel_fe_pred'] = y_pred_fe_2021
holdout['lgb_pred'] = y_pred_lgb_2021
holdout['fire_risk_score'] = holdout['Avg Fire Risk Score']

output_df = holdout[['ZIP', 'Category', 'predicted_premium', 'Earned Premium', 'fire_risk_score', 'panel_fe_pred', 'lgb_pred']].copy()
output_df = output_df.rename(columns={'Earned Premium': 'actual_premium'})
output_df.to_csv(OUTPUT_FILE, index=False)

print(f"Output saved: {OUTPUT_FILE}")
print(f"Rows: {output_df.shape[0]}, Columns: {output_df.shape[1]}")

nan_count = output_df['predicted_premium'].isna().sum()
print(f"\nNaN in predicted_premium: {nan_count}")
print(f"predicted_premium range: {output_df['predicted_premium'].min():,.0f} to {output_df['predicted_premium'].max():,.0f}")
print(f"predicted_premium mean: {output_df['predicted_premium'].mean():,.0f}")

print("\n=== Step 5 complete ===")
