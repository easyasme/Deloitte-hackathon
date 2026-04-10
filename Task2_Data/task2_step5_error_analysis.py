"""
Task 2 Step 5: Error Analysis
Evaluates 2021 predictions against holdout actuals, computes error metrics,
identifies high-error ZIPs, and validates against fire risk distribution.
"""

import pandas as pd
import numpy as np

# STEP 1: Load predictions
INPUT = "Task2_Data/task2_step5_predictions.csv"
df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} predictions")

# STEP 2: Compute error metrics per model (D-11, D-14)
# error = predicted - actual
# abs_error = |error|

def compute_mape(y_true, y_pred):
    """MAPE on non-zero actuals only"""
    non_zero_mask = y_true > 0
    if non_zero_mask.sum() > 0:
        return np.mean(np.abs((y_true.values[non_zero_mask] - y_pred[non_zero_mask]) / y_true.values[non_zero_mask])) * 100
    return np.nan

def evaluate_regression(model_name, y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = compute_mape(y_true, y_pred)
    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'mape': mape}

# Compute errors for each model
results_ensemble = evaluate_regression('Ensemble', df['actual_premium'], df['predicted_premium'])
results_panel_fe = evaluate_regression('PanelFE', df['actual_premium'], df['panel_fe_pred'])
results_lgb = evaluate_regression('LightGBM', df['actual_premium'], df['lgb_pred'])

# STEP 3: Save metrics CSV
METRICS_OUT = "Task2_Data/task2_step5_metrics.csv"
metrics_df = pd.DataFrame([results_ensemble, results_panel_fe, results_lgb])
metrics_df.to_csv(METRICS_OUT, index=False)
print(f"\n--- Metrics saved to {METRICS_OUT} ---")
print(metrics_df.to_string(index=False))

# STEP 4: Add error columns to dataframe for analysis
df['error'] = df['predicted_premium'] - df['actual_premium']
df['abs_error'] = np.abs(df['error'])

# STEP 5: Top-20 highest-error ZIPs (D-08)
# Sort by abs_error descending
top_errors = df.nlargest(20, 'abs_error')[['ZIP', 'Category', 'predicted_premium', 'actual_premium', 'error', 'abs_error', 'fire_risk_score']].copy()

# STEP 6: Compute fire_risk_percentile for all rows
df['fire_risk_percentile'] = df['fire_risk_score'].rank(pct=True) * 100
top_errors = df.nlargest(20, 'abs_error')[['ZIP', 'Category', 'predicted_premium', 'actual_premium', 'error', 'abs_error', 'fire_risk_score', 'fire_risk_percentile']].copy()

# STEP 7: Validate high-error ZIPs against fire risk distribution (D-09)
# Compare fire_risk_score of top 20 to overall distribution
top20_fire_risk_mean = top_errors['fire_risk_score'].mean()
overall_fire_risk_mean = df['fire_risk_score'].mean()
top20_fire_risk_median = top_errors['fire_risk_score'].median()
overall_fire_risk_median = df['fire_risk_score'].median()

# What percentile of top-20 errors fall above median fire risk?
median_fire_risk = df['fire_risk_score'].median()
pct_above_median = (top_errors['fire_risk_score'] > median_fire_risk).mean() * 100

# 75th percentile of fire risk
p75_fire_risk = df['fire_risk_score'].quantile(0.75)
pct_above_75 = (top_errors['fire_risk_score'] > p75_fire_risk).mean() * 100

print(f"\n--- Fire Risk Correlation with High-Error ZIPs ---")
print(f"Top 20 mean fire_risk_score: {top20_fire_risk_mean:.4f} vs overall mean: {overall_fire_risk_mean:.4f}")
print(f"Top 20 median fire_risk_score: {top20_fire_risk_median:.4f} vs overall median: {overall_fire_risk_median:.4f}")
print(f"Pct of top-20 above overall median fire risk: {pct_above_median:.1f}%")
print(f"Pct of top-20 above 75th percentile fire risk: {pct_above_75:.1f}%")

if top20_fire_risk_mean > overall_fire_risk_mean:
    print("FINDING: High-error ZIPs tend to have HIGHER fire risk scores than average.")
else:
    print("FINDING: High-error ZIPs do NOT particularly correlate with high fire risk.")

# STEP 8: Error distribution percentiles (D-10)
percentiles = [50, 75, 90, 95]
print(f"\n--- Error Distribution Percentiles ---")
for p in percentiles:
    val = df['abs_error'].quantile(p / 100)
    print(f"  {p}th percentile abs_error: {val:,.2f}")

# STEP 9: Save error analysis CSV
ERROR_OUT = "Task2_Data/task2_step5_error_analysis.csv"
top_errors.to_csv(ERROR_OUT, index=False)
print(f"\n--- Error analysis saved to {ERROR_OUT} ---")

# STEP 10: Print top 10 high-error ZIPs
print(f"\n--- Top 10 Highest-Error ZIPs ---")
print(top_errors.head(10).to_string(index=False))

print("\n--- Error Analysis Complete ---")