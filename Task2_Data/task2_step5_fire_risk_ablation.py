"""
Task 2 Step 5: Fire Risk Value-Add Analysis (Ablation Study)
Removes all fire risk features and re-trains the ensemble to measure RMSE delta.
If RMSE increases without fire risk -> fire risk adds value.

Fire risk columns to remove:
- Avg Fire Risk Score
- Avg Fire Risk Score_lag1
- expanding_fire_risk_mean
- expanding_fire_risk_std
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
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_HOLDOUT = os.path.join(SCRIPT_DIR, "task2_step1_2021_holdout.csv")
INPUT_FEATURE_MATRIX = os.path.join(SCRIPT_DIR, "task2_step2_feature_matrix.csv")
INPUT_RAW = os.path.join(SCRIPT_DIR, "task2_step1_panel_clean.csv")
ENSEMBLE_WEIGHT_PANEL_FE = 0.80
ENSEMBLE_WEIGHT_LGB = 0.20

# Fire risk columns to remove (ablated)
FIRE_RISK_COLS = [
    'Avg Fire Risk Score',
    'Avg Fire Risk Score_lag1',
    'expanding_fire_risk_mean',
    'expanding_fire_risk_std',
]

# ============================================================
# STEP 1: Load data
# ============================================================
print("\n--- STEP 1: Load data ---")
holdout = pd.read_csv(INPUT_HOLDOUT, low_memory=False)
df = pd.read_csv(INPUT_FEATURE_MATRIX, low_memory=False)
raw_df = pd.read_csv(INPUT_RAW, low_memory=False)
print(f"2021 Holdout shape: {holdout.shape}")
print(f"Feature Matrix shape: {df.shape}")

# ============================================================
# STEP 2: Derive 2021 lag features from 2020 panel (no leakage)
# Lag1 features must come from the 2018-2020 panel, not same-year holdout
# ============================================================
print("\n--- STEP 2: Compute 2021 lag features from 2020 panel (no leakage) ---")
panel_2020 = raw_df[raw_df['Year'] == 2020][['ZIP', 'Category', 'Earned Exposure',
    'Earned Premium', 'CAT Cov A Fire -  Incurred Losses',
    'Non-CAT Cov A Fire -  Incurred Losses', 'Avg PPC']].copy()
panel_2020 = panel_2020.rename(columns={
    'Earned Exposure': 'Earned Exposure_lag1',
    'Earned Premium': 'Earned Premium_lag1',
    'CAT Cov A Fire -  Incurred Losses': 'CAT Cov A Fire -  Incurred Losses_lag1',
    'Non-CAT Cov A Fire -  Incurred Losses': 'Non-CAT Cov A Fire -  Incurred Losses_lag1',
    'Avg PPC': 'Avg PPC_lag1',
})
holdout = holdout.drop(columns=[c for c in holdout.columns if c.endswith('_lag1')], errors='ignore')
holdout = holdout.merge(panel_2020, on=['ZIP', 'Category'], how='left')
print(f"  Lag features derived from 2020 panel, joined to holdout")

# ============================================================
# STEP 3: Compute expanding window stats for 2021 (no fire risk)
# ============================================================
print("\n--- STEP 3: Compute expanding window stats (no fire risk) ---")
raw_df = raw_df.sort_values(['ZIP', 'Category', 'Year']).reset_index(drop=True)
raw_df['expanding_exposure_mean'] = raw_df.groupby(['ZIP', 'Category'])['Earned Exposure'].transform(
    lambda x: x.expanding().mean()  # no shift: 2020 row captures history through 2020
)
expanding_2020 = raw_df[raw_df['Year'] == 2020][['ZIP', 'Category', 'expanding_exposure_mean']].copy()
holdout = holdout.merge(expanding_2020, on=['ZIP', 'Category'], how='left')
print(f"  After merge: holdout shape = {holdout.shape}")

# ============================================================
# STEP 4: One-hot encode categories
# ============================================================
print("\n--- STEP 4: One-hot encode categories ---")
categories = ['HO', 'CO', 'DT', 'RT', 'DO', 'MH', 'NA']
for cat in categories:
    holdout[f'cat_{cat}'] = (holdout['Category'] == cat).astype(int)

# ============================================================
# STEP 5: Add COVID year indicator
# ============================================================
print("\n--- STEP 5: Add COVID indicator ---")
holdout['is_covid_year'] = 0

# ============================================================
# STEP 6: Prepare training data (2019) and validation (2020)
# ============================================================
print("\n--- STEP 6: Prepare training data ---")
train_df = df[df['Year'] == 2019].copy()
val_df = df[df['Year'] == 2020].copy()
print(f"Train (2019): {len(train_df)} rows, Val (2020): {len(val_df)} rows")

# Define all exclusion columns
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

# ALSO remove fire risk columns from feature set (ABLATION)
exclude_cols_no_fire = exclude_cols + FIRE_RISK_COLS
exclude_cols_extra = ['cat_HO', 'cat_CO', 'cat_DT', 'cat_RT', 'cat_DO', 'cat_MH', 'cat_NA', 'ZIP_Cat']

feature_cols = [c for c in df.columns if c not in exclude_cols_no_fire]
feature_cols_fe = [c for c in feature_cols if c not in exclude_cols_extra]
print(f"Features (no fire risk): {len(feature_cols)}")
print(f"Panel FE features: {len(feature_cols_fe)}")

# Check fire risk cols were actually present
fire_in_original = [c for c in FIRE_RISK_COLS if c in df.columns]
print(f"Fire risk columns removed: {fire_in_original}")

y_train = train_df['Earned Premium']
y_val = val_df['Earned Premium']

X_train = train_df[feature_cols]
X_val = val_df[feature_cols]

# ============================================================
# STEP 7: Train Panel FE model (no fire risk)
# ============================================================
print("\n--- STEP 7: Train Panel FE model (no fire risk) ---")
train_panel = train_df.set_index(['ZIP', 'Year'])
val_panel = val_df.set_index(['ZIP', 'Year'])

y_train_panel = train_panel['Earned Premium']
y_val_panel = val_panel['Earned Premium']

X_train_panel = train_panel[feature_cols_fe].copy()
X_val_panel = val_panel[feature_cols_fe].copy()

# Drop all-NaN columns
cols_all_nan = [c for c in X_train_panel.columns if X_train_panel[c].isna().all()]
if cols_all_nan:
    print(f"Dropping all-NaN columns: {cols_all_nan}")
    X_train_panel = X_train_panel.drop(columns=cols_all_nan)
    X_val_panel = X_val_panel.drop(columns=cols_all_nan)

# Fill NaN
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

print("Fitting Panel FE (Pooled OLS, no fire risk)...")
model_fe = PanelOLS(y_train_panel, X_train_panel, entity_effects=False, time_effects=False, weights=weights_train_panel, check_rank=False)
result_fe = model_fe.fit(cov_type='robust')
print(f"R-squared: {result_fe.rsquared:.4f}")

# ============================================================
# STEP 8: Train LightGBM model (no fire risk)
# ============================================================
print("\n--- STEP 8: Train LightGBM model (no fire risk) ---")
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

# ============================================================
# STEP 9: Validate on 2020 (no fire risk)
# ============================================================
print("\n--- STEP 9: 2020 Validation (no fire risk) ---")
y_pred_fe_val = result_fe.predict(X_val_panel).values.ravel()
y_pred_lgb_val = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
y_pred_ens_val = ENSEMBLE_WEIGHT_PANEL_FE * y_pred_fe_val + ENSEMBLE_WEIGHT_LGB * y_pred_lgb_val

rmse_fe_nf = np.sqrt(mean_squared_error(y_val_panel.values, y_pred_fe_val))
rmse_lgb_nf = np.sqrt(mean_squared_error(y_val.values, y_pred_lgb_val))
rmse_ens_nf = np.sqrt(mean_squared_error(y_val_panel.values, y_pred_ens_val))

print(f"Panel FE (no fire) 2020 RMSE: {rmse_fe_nf:,.2f}")
print(f"LightGBM (no fire) 2020 RMSE: {rmse_lgb_nf:,.2f}")
print(f"Ensemble (no fire) 2020 RMSE: {rmse_ens_nf:,.2f}")

# ============================================================
# STEP 10: Prepare 2021 holdout (no fire risk) and predict
# ============================================================
print("\n--- STEP 10: Generate 2021 predictions (no fire risk) ---")

X_2021 = holdout[[c for c in feature_cols if c in holdout.columns]].copy()

# Fill missing using training column means
for col in X_2021.columns:
    if col in X_train.columns:
        if X_2021[col].isna().any():
            mean_train = X_train[col].mean()
            if pd.isna(mean_train): mean_train = 0
            X_2021[col] = X_2021[col].fillna(mean_train)
    else:
        X_2021[col] = X_2021[col].fillna(0)

# Add missing columns with 0
for col in X_train.columns:
    if col not in X_2021.columns:
        X_2021[col] = 0

X_2021 = X_2021[X_train.columns]

# Panel FE features
X_2021_fe = X_2021[[c for c in X_train_panel.columns if c in feature_cols_fe]].copy()
for col in X_2021_fe.columns:
    if X_2021_fe[col].isna().any():
        mean_val = X_train_panel[col].mean()
        if pd.isna(mean_val): mean_val = 0
        X_2021_fe[col] = X_2021_fe[col].fillna(mean_val)

for col in X_train_panel.columns:
    if col not in X_2021_fe.columns:
        X_2021_fe[col] = 0
X_2021_fe = X_2021_fe[X_train_panel.columns]

# Panel format
X_2021_panel = X_2021_fe.copy()
X_2021_panel['ZIP'] = holdout['ZIP']
X_2021_panel['Year'] = 2021
X_2021_panel = X_2021_panel.set_index(['ZIP', 'Year'])

y_pred_fe_2021 = result_fe.predict(X_2021_panel).values.ravel()
y_pred_lgb_2021 = model_lgb.predict(X_2021, num_iteration=model_lgb.best_iteration)
y_pred_2021_no_fire = ENSEMBLE_WEIGHT_PANEL_FE * y_pred_fe_2021 + ENSEMBLE_WEIGHT_LGB * y_pred_lgb_2021

print(f"2021 predictions generated (no fire risk): {len(y_pred_2021_no_fire)} rows")

# ============================================================
# STEP 11: Compute ablation metrics
# ============================================================
print("\n--- STEP 11: Compute ablation metrics ---")

# Load 2021 actuals
y_actual_2021 = holdout['Earned Premium'].values

# Load with-fire-risk predictions from Plan 05-01
preds_with_fire = pd.read_csv(os.path.join(SCRIPT_DIR, "task2_step5_predictions.csv"))

# Verify ZIP+Category alignment (each ZIP appears multiple times — one per Category)
assert (preds_with_fire['ZIP'].values == holdout['ZIP'].values).all() and \
       (preds_with_fire['Category'].values == holdout['Category'].values).all(), \
    "ZIP/Category misalignment: task2_step5_predictions.csv does not match holdout"
y_pred_with_fire = preds_with_fire['predicted_premium'].values

# Verify length alignment
assert len(y_pred_2021_no_fire) == len(y_actual_2021), f"Length mismatch: {len(y_pred_2021_no_fire)} vs {len(y_actual_2021)}"
assert len(y_pred_with_fire) == len(y_actual_2021), f"Length mismatch: {len(y_pred_with_fire)} vs {len(y_actual_2021)}"

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    non_zero_mask = np.asarray(y_true) > 0
    mape = np.mean(np.abs((np.asarray(y_true)[non_zero_mask] - np.asarray(y_pred)[non_zero_mask]) / np.asarray(y_true)[non_zero_mask])) * 100
    return rmse, mae, mape

rmse_with, mae_with, mape_with = compute_metrics(y_actual_2021, y_pred_with_fire)
rmse_without, mae_without, mape_without = compute_metrics(y_actual_2021, y_pred_2021_no_fire)

delta_rmse = rmse_with - rmse_without
delta_mae = mae_with - mae_without
delta_mape = mape_with - mape_without

print(f"\n{'='*60}")
print(f"FIRE RISK ABLATION RESULTS (2021 Holdout)")
print(f"{'='*60}")
print(f"Ensemble WITH fire risk:")
print(f"  RMSE: {rmse_with:,.2f}")
print(f"  MAE:  {mae_with:,.2f}")
print(f"  MAPE: {mape_with:.2f}%")
print(f"\nEnsemble WITHOUT fire risk:")
print(f"  RMSE: {rmse_without:,.2f}")
print(f"  MAE:  {mae_without:,.2f}")
print(f"  MAPE: {mape_without:.2f}%")
print(f"\nDelta (with - without):")
print(f"  RMSE: {delta_rmse:+,.2f}")
print(f"  MAE:  {delta_mae:+,.2f}")
print(f"  MAPE: {delta_mape:+.2f}%")

if delta_rmse < 0:
    print(f"\n>>> Fire risk IMPROVES 2021 prediction RMSE by {abs(delta_rmse):,.0f}")
    finding = f"Fire risk IMPROVES 2021 prediction RMSE by {abs(delta_rmse):,.0f} (delta={delta_rmse:+,.0f})"
elif delta_rmse > 0:
    print(f"\n>>> Fire risk DOES NOT improve 2021 prediction (RMSE delta = {delta_rmse:+,.0f})")
    finding = f"Fire risk DOES NOT improve 2021 prediction (RMSE delta = {delta_rmse:+,.0f})"
else:
    print(f"\n>>> Fire risk has NO EFFECT on 2021 prediction")
    finding = f"Fire risk has NO EFFECT on 2021 prediction"

# ============================================================
# STEP 12: Save ablation results
# ============================================================
print("\n--- STEP 12: Save ablation results ---")
ABLARESULTS_OUT = os.path.join(SCRIPT_DIR, "task2_step5_fire_risk_ablation.csv")
ablation_df = pd.DataFrame([
    {'metric_name': 'RMSE', 'with_fire_risk': rmse_with, 'without_fire_risk': rmse_without, 'delta': delta_rmse},
    {'metric_name': 'MAE', 'with_fire_risk': mae_with, 'without_fire_risk': mae_without, 'delta': delta_mae},
    {'metric_name': 'MAPE', 'with_fire_risk': mape_with, 'without_fire_risk': mape_without, 'delta': delta_mape},
])
ablation_df.to_csv(ABLARESULTS_OUT, index=False)
print(f"Ablation results saved: {ABLARESULTS_OUT}")

# Also save 2021 predictions without fire risk for reference
preds_no_fire_df = holdout[['ZIP', 'Category']].copy()
preds_no_fire_df['predicted_premium_no_fire'] = y_pred_2021_no_fire
preds_no_fire_df['actual_premium'] = y_actual_2021
preds_no_fire_df.to_csv(os.path.join(SCRIPT_DIR, "task2_step5_predictions_no_fire.csv"), index=False)
print("No-fire predictions saved: " + os.path.join(SCRIPT_DIR, "task2_step5_predictions_no_fire.csv"))

print(f"\n=== Ablation complete: {finding} ===")
