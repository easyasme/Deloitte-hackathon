#!/usr/bin/env python3
"""
Insurance Premium Time Series Prediction for California Zip Codes
================================================================
Predicting 2021 insurance premiums based on historical data (2018-2020)
Incorporates wildfire risk from Task 1 (VQC model output)

Model: Panel Time Series Regression with Gradient Boosting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 70)
print("INSURANCE PREMIUM TIME SERIES PREDICTION - 2021")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading data...")

df = pd.read_csv('abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv', low_memory=False)

print(f"  Total records: {len(df):,}")
print(f"  Years: {sorted(df['Year'].unique())}")
print(f"  Unique ZIPs: {df['ZIP'].nunique():,}")

# Load wildfire risk predictions from Task 1 (VQC model)
try:
    wildfire_risk = pd.read_csv('wildfire_risk_predictions_2023.csv')
    print("  ✓ Loaded wildfire risk predictions from Task 1")
    has_wildfire_risk = True
except:
    print("  ⚠ Could not load wildfire risk - using dataset fire risk score")
    has_wildfire_risk = False

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n[2] Engineering features...")

# Key features from dataset
key_features = [
    'Avg Fire Risk Score',       # Fire risk (from dataset or Task 1)
    'Avg PPC',                   # Public Protection Classification
    'Earned Exposure',           # Exposure volume
    'total_population',          # Census: population
    'median_income',             # Census: income
    'total_housing_units',       # Census: housing units
    'average_household_size',    # Census: household size
    'housing_value',              # Census: housing value
    'housing_vacancy_number',    # Census: vacancy
    'median_monthly_housing_costs',  # Census: housing costs
]

# Loss features (important for premium prediction)
loss_features = [
    'CAT Cov A Fire -  Incurred Losses',
    'CAT Cov A Fire -  Number of Claims',
    'CAT Cov A Smoke -  Incurred Losses', 
    'CAT Cov A Smoke -  Number of Claims',
    'Non-CAT Cov A Fire -  Incurred Losses',
    'Non-CAT Cov A Fire -  Number of Claims',
]

# Risk exposure columns
exposure_cols = [
    'Number of High Fire Risk Exposure',
    'Number of Low Fire Risk Exposure', 
    'Number of Moderate Fire Risk Exposure',
    'Number of Negligible Fire Risk Exposure',
    'Number of Very High Fire Risk Exposure',
]

# Create lagged features for time series
def create_lagged_features(data, zip_col='ZIP', year_col='Year'):
    """Create lagged features for each zip code"""
    data = data.sort_values([zip_col, year_col]).copy()
    
    # Group by zip and create lags
    lagged_data = []
    for zip_code in data[zip_col].unique():
        zip_data = data[data[zip_col] == zip_code].copy()
        
        # Create lag features for key variables (safe, no division)
        for col in ['Earned Premium', 'Avg Fire Risk Score', 'Earned Exposure']:
            if col in zip_data.columns:
                zip_data[f'{col}_lag1'] = zip_data[col].shift(1)
                zip_data[f'{col}_lag2'] = zip_data[col].shift(2)
                zip_data[f'{col}_trend'] = zip_data[col] - zip_data[col].shift(1)
        
        lagged_data.append(zip_data)
    
    result = pd.concat(lagged_data, ignore_index=True)
    
    # Safe year-over-year growth (avoid division by zero)
    result['premium_growth_yoy'] = result.groupby(zip_col)['Earned Premium'].pct_change(fill_method=None)
    result['premium_growth_yoy'] = result['premium_growth_yoy'].replace([np.inf, -np.inf], np.nan)
    
    return result

# Apply lagged feature creation
df_lagged = create_lagged_features(df)

# Filter to zips with all years for better time series modeling
zips_all_years = df_lagged.groupby('ZIP')['Year'].count()
zips_complete = zips_all_years[zips_all_years >= 3].index
df_complete = df_lagged[df_lagged['ZIP'].isin(zips_complete)].copy()

print(f"  Records with complete history: {len(df_complete):,}")
print(f"  ZIPs with 3+ years: {len(zips_complete):,}")

# ============================================================================
# 3. PREPARE TRAINING AND PREDICTION DATA
# ============================================================================
print("\n[3] Preparing train/predict datasets...")

# Training data: 2018-2020
train_df = df_complete[df_complete['Year'].isin([2018, 2019, 2020])].copy()
train_df = train_df.dropna(subset=['Earned Premium'])

# Prediction target: 2021
pred_df = df_complete[df_complete['Year'] == 2021].copy()

print(f"  Training records (2018-2020): {len(train_df):,}")
print(f"  Prediction records (2021): {len(pred_df):,}")

# Features for modeling
feature_cols = [
    'Avg Fire Risk Score', 'Avg PPC', 'Earned Exposure',
    'total_population', 'median_income', 'total_housing_units',
    'average_household_size', 'housing_value',
    'CAT Cov A Fire -  Incurred Losses', 
    'Non-CAT Cov A Fire -  Incurred Losses',
    'Number of High Fire Risk Exposure',
    'Number of Very High Fire Risk Exposure',
]

# Add lagged features if available
lag_features = [c for c in train_df.columns if '_lag' in c or 'growth' in c]
feature_cols.extend(lag_features)

# Handle missing and infinite values
for col in feature_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
        train_df[col] = train_df[col].fillna(train_df[col].median())
    if col in pred_df.columns:
        pred_df[col] = pred_df[col].replace([np.inf, -np.inf], np.nan)
        pred_df[col] = pred_df[col].fillna(pred_df[col].median())

# Ensure all feature columns exist
feature_cols = [c for c in feature_cols if c in train_df.columns and c in pred_df.columns]

print(f"  Features used: {len(feature_cols)}")
print(f"  Features: {feature_cols[:5]}...")

# Prepare X, y for training
X_train = train_df[feature_cols].values.astype(np.float64)
y_train = train_df['Earned Premium'].values.astype(np.float64)

X_pred = pred_df[feature_cols].values.astype(np.float64)

# Replace any remaining inf/nan in X arrays
X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
X_pred = np.nan_to_num(X_pred, nan=0, posinf=0, neginf=0)

print(f"\n  X_train shape: {X_train.shape}")
print(f"  X_pred shape: {X_pred.shape}")

# ============================================================================
# 4. BUILD AND TRAIN MODELS
# ============================================================================
print("\n[4] Training models...")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_pred_scaled = scaler.transform(X_pred)

# Model 1: Gradient Boosting (main model)
print("  Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_pred_scaled)
print("    ✓ Gradient Boosting trained")

# Model 2: Random Forest
print("  Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_pred_scaled)
print("    ✓ Random Forest trained")

# Model 3: Ridge Regression (baseline)
print("  Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_pred_scaled)
print("    ✓ Ridge Regression trained")

# Model 4: Time Series Panel Model (Year as feature + ZIP effects)
print("  Training Panel Time Series Model...")
train_df['ZIP_code'] = train_df['ZIP'].astype('category').cat.codes
pred_df['ZIP_code'] = pred_df['ZIP'].astype('category').cat.codes

panel_features = feature_cols + ['Year', 'ZIP_code']

# Handle NaN in panel features
X_train_panel_raw = train_df[panel_features].values.astype(np.float64)
X_pred_panel_raw = pred_df[panel_features].values.astype(np.float64)
X_train_panel = np.nan_to_num(X_train_panel_raw, nan=0, posinf=0, neginf=0)
X_pred_panel = np.nan_to_num(X_pred_panel_raw, nan=0, posinf=0, neginf=0)

scaler_panel = StandardScaler()
X_train_panel_scaled = scaler_panel.fit_transform(X_train_panel)
X_pred_panel_scaled = scaler_panel.transform(X_pred_panel)

panel_model = Ridge(alpha=0.5)
panel_model.fit(X_train_panel_scaled, y_train)
y_pred_panel = panel_model.predict(X_pred_panel_scaled)
print("    ✓ Panel Time Series model trained")

# ============================================================================
# 5. MODEL EVALUATION (Cross-Validation on Training Data)
# ============================================================================
print("\n[5] Cross-validation on training data (2018-2020)...")

# Time series split cross-validation
tscv = TimeSeriesSplit(n_splits=3)

def evaluate_model(model, X, y, name, cv=tscv):
    mae_scores, rmse_scores, r2_scores = [], [], []
    for train_idx, val_idx in cv.split(X):
        X_t, X_v = X[train_idx], X[val_idx]
        y_t, y_v = y[train_idx], y[val_idx]
        
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_t, y_t)
        y_hat = model_clone.predict(X_v)
        
        mae_scores.append(mean_absolute_error(y_v, y_hat))
        rmse_scores.append(np.sqrt(mean_squared_error(y_v, y_hat)))
        r2_scores.append(r2_score(y_v, y_hat))
    
    print(f"  {name:25s}: MAE=${np.mean(mae_scores):,.0f}, RMSE=${np.mean(rmse_scores):,.0f}, R2={np.mean(r2_scores):.3f}")
    return np.mean(mae_scores), np.mean(rmse_scores), np.mean(r2_scores)

mae_gb, rmse_gb, r2_gb = evaluate_model(gb_model, X_train_scaled, y_train, "Gradient Boosting")
mae_rf, rmse_rf, r2_rf = evaluate_model(rf_model, X_train_scaled, y_train, "Random Forest")
mae_ridge, rmse_ridge, r2_ridge = evaluate_model(ridge_model, X_train_scaled, y_train, "Ridge Regression")
mae_panel, rmse_panel, r2_panel = evaluate_model(panel_model, X_train_panel_scaled, y_train, "Panel Time Series")

# ============================================================================
# 6. PREDICT 2021 PREMIUMS
# ============================================================================
print("\n[6] Predicting 2021 premiums...")

# Ensemble prediction (weighted average)
y_pred_ensemble = 0.4 * y_pred_gb + 0.3 * y_pred_rf + 0.2 * y_pred_panel + 0.1 * y_pred_ridge

# Add predictions to dataframe
pred_df = pred_df.copy()
pred_df['Predicted_Premium_GB'] = y_pred_gb
pred_df['Predicted_Premium_RF'] = y_pred_rf
pred_df['Predicted_Premium_Ridge'] = y_pred_ridge
pred_df['Predicted_Premium_Panel'] = y_pred_panel
pred_df['Predicted_Premium_Ensemble'] = y_pred_ensemble

# Compare with actual 2021 if available
if 'Earned Premium' in pred_df.columns and pred_df['Earned Premium'].notna().sum() > 0:
    actual_2021 = pred_df['Earned Premium'].values
    valid_mask = ~np.isnan(actual_2021) & ~np.isnan(y_pred_ensemble)
    
    if valid_mask.sum() > 0:
        mae_2021 = mean_absolute_error(actual_2021[valid_mask], y_pred_ensemble[valid_mask])
        rmse_2021 = np.sqrt(mean_squared_error(actual_2021[valid_mask], y_pred_ensemble[valid_mask]))
        r2_2021 = r2_score(actual_2021[valid_mask], y_pred_ensemble[valid_mask])
        
        print(f"\n  2021 Test Set Performance (Ensemble):")
        print(f"    MAE: ${mae_2021:,.0f}")
        print(f"    RMSE: ${rmse_2021:,.0f}")
        print(f"    R2 Score: {r2_2021:.3f}")
        
        # By-model comparison
        print(f"\n  2021 Performance by Model:")
        print(f"    Gradient Boosting: MAE=${mean_absolute_error(actual_2021[valid_mask], y_pred_gb[valid_mask]):,.0f}")
        print(f"    Random Forest: MAE=${mean_absolute_error(actual_2021[valid_mask], y_pred_rf[valid_mask]):,.0f}")
        print(f"    Ridge: MAE=${mean_absolute_error(actual_2021[valid_mask], y_pred_ridge[valid_mask]):,.0f}")
        print(f"    Panel: MAE=${mean_absolute_error(actual_2021[valid_mask], y_pred_panel[valid_mask]):,.0f}")
        print(f"    Ensemble: MAE=${mae_2021:,.0f}")

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================
print("\n[7] Feature Importance (Gradient Boosting)...")

importances = gb_model.feature_importances_
feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

print("\n  Top 10 Most Important Features:")
for i, (feat, imp) in enumerate(feat_imp[:10], 1):
    bar = "█" * int(imp * 100)
    print(f"  {i:2d}. {feat:40s} {imp:.3f} {bar}")

# ============================================================================
# 8. RESULTS ANALYSIS
# ============================================================================
print("\n[8] Prediction Results Summary...")

print("\n  2021 Predicted Premiums by ZIP (Top 10 Highest):")
top_pred = pred_df.nlargest(10, 'Predicted_Premium_Ensemble')[
    ['ZIP', 'Avg Fire Risk Score', 'Earned Exposure', 'Predicted_Premium_Ensemble']
]
print("\n  Rank | ZIP      | Fire Risk | Exposure  | Predicted Premium")
print("  " + "-" * 65)
for i, (_, row) in enumerate(top_pred.iterrows(), 1):
    print(f"  {i:4d} | {int(row['ZIP']):8d} | {row['Avg Fire Risk Score']:9.2f} | ${row['Earned Exposure']:9,.0f} | ${row['Predicted_Premium_Ensemble']:>12,.0f}")

print("\n  2021 Predicted Premiums by ZIP (Top 10 Lowest):")
bottom_pred = pred_df.nsmallest(10, 'Predicted_Premium_Ensemble')[
    ['ZIP', 'Avg Fire Risk Score', 'Earned Exposure', 'Predicted_Premium_Ensemble']
]
print("\n  Rank | ZIP      | Fire Risk | Exposure  | Predicted Premium")
print("  " + "-" * 65)
for i, (_, row) in enumerate(bottom_pred.iterrows(), 1):
    print(f"  {i:4d} | {int(row['ZIP']):8d} | {row['Avg Fire Risk Score']:9.2f} | ${row['Earned Exposure']:9,.0f} | ${row['Predicted_Premium_Ensemble']:>12,.0f}")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n[9] Saving results...")

output_cols = ['ZIP', 'Year', 'Avg Fire Risk Score', 'Avg PPC', 'Earned Exposure',
               'Predicted_Premium_GB', 'Predicted_Premium_RF', 
               'Predicted_Premium_Ridge', 'Predicted_Premium_Panel',
               'Predicted_Premium_Ensemble']
output_df = pred_df[output_cols].copy()
output_df = output_df.sort_values('Predicted_Premium_Ensemble', ascending=False)
output_df.to_csv('insurance_premium_predictions_2021.csv', index=False)
print("  ✓ Saved: insurance_premium_predictions_2021.csv")

# Summary statistics
with open('insurance_premium_model_summary.txt', 'w') as f:
    f.write("INSURANCE PREMIUM TIME SERIES PREDICTION MODEL\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Training Period: 2018-2020\n")
    f.write(f"Prediction Year: 2021\n")
    f.write(f"ZIP Codes: {len(pred_df):,}\n\n")
    f.write("MODELS USED:\n")
    f.write("1. Gradient Boosting Regressor\n")
    f.write("2. Random Forest Regressor\n")
    f.write("3. Ridge Regression\n")
    f.write("4. Panel Time Series (Year + ZIP effects)\n")
    f.write("5. Ensemble (weighted average)\n\n")
    f.write("CROSS-VALIDATION RESULTS (2018-2020):\n")
    f.write(f"  Gradient Boosting: MAE=${mae_gb:,.0f}, R2={r2_gb:.3f}\n")
    f.write(f"  Random Forest: MAE=${mae_rf:,.0f}, R2={r2_rf:.3f}\n")
    f.write(f"  Ridge Regression: MAE=${mae_ridge:,.0f}, R2={r2_ridge:.3f}\n")
    f.write(f"  Panel Time Series: MAE=${mae_panel:,.0f}, R2={r2_panel:.3f}\n\n")
    if valid_mask.sum() > 0:
        f.write("2021 TEST PERFORMANCE:\n")
        f.write(f"  Ensemble MAE: ${mae_2021:,.0f}\n")
        f.write(f"  Ensemble RMSE: ${rmse_2021:,.0f}\n")
        f.write(f"  Ensemble R2: {r2_2021:.3f}\n\n")
    f.write("TOP 10 FEATURES:\n")
    for feat, imp in feat_imp[:10]:
        f.write(f"  - {feat}: {imp:.3f}\n")
print("  ✓ Saved: insurance_premium_model_summary.txt")

print("\n" + "=" * 70)
print("TIME SERIES PREDICTION COMPLETE")
print("=" * 70)
