"""
Task 2 Step 4: Ensemble Model (Panel FE + LightGBM)
Creates weighted blend, optimizes weight via 2020 validation RMSE grid search.
Compares all models against Phase 3 baselines.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# STEP 1: Load Phase 3 baseline predictions and metrics
lr_metrics = pd.read_csv("Task2_Data/task2_step3_metrics.csv")
lr_predictions = pd.read_csv("Task2_Data/task2_step3_predictions.csv")
print("Loaded Phase 3 baseline data")

# STEP 2: Load Panel FE predictions
panel_fe_preds = pd.read_csv("Task2_Data/task2_step4_panel_fe_predictions.csv")
print(f"Loaded Panel FE predictions: {len(panel_fe_preds)} rows")

# STEP 3: Load LightGBM predictions (need to run LightGBM first to get lgb_pred)
# Since we need to build ensemble, we need LightGBM predictions alongside Panel FE
# If task2_step4_lightgbm.py outputs its own predictions, load that too
# Otherwise we need to regenerate LightGBM predictions here
# For this implementation: check if LightGBM predictions exist, if not run it inline
try:
    lgb_metrics = pd.read_csv("Task2_Data/task2_step4_lightgbm_metrics.csv")
    # Check if LightGBM predictions were saved separately
    # The LightGBM script only saves importance, not predictions
    # We need to regenerate LightGBM predictions here
    print("LightGBM metrics found, regenerating LightGBM predictions for ensemble...")
    RUN_LGB = True
except FileNotFoundError:
    RUN_LGB = True
    print("LightGBM metrics not found, will run LightGBM inline")

if RUN_LGB:
    # Run LightGBM inline to get predictions for ensemble
    import lightgbm as lgb
    df = pd.read_csv("Task2_Data/task2_step2_feature_matrix.csv", low_memory=False)

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
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    train_df = df[df['Year'] == 2019].copy()
    val_df = df[df['Year'] == 2020].copy()

    y_train = train_df['Earned Premium']
    y_val = val_df['Earned Premium']
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression', 'metric': 'rmse', 'num_leaves': 31,
        'max_depth': 8, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_data_in_leaf': 20,
        'verbosity': -1, 'seed': 42,
    }
    model_lgb = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[train_data, val_data], valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    y_pred_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    print(f"LightGBM inline: best iteration={model_lgb.best_iteration}")

# STEP 4: Align predictions on common index
# Load validation ground truth
val_df_full = pd.read_csv("Task2_Data/task2_step2_feature_matrix.csv", low_memory=False)
val_df_full = val_df_full[val_df_full['Year'] == 2020][['Year', 'ZIP', 'Category', 'Earned Premium']].copy()
val_df_full = val_df_full.reset_index(drop=True)

# Align all predictions
y_val = val_df_full['Earned Premium'].values

# Panel FE predictions
y_pred_fe = panel_fe_preds['panel_fe_pred'].values

# LR predictions from Phase 3
y_pred_lr = lr_predictions['lr_pred'].values

# RF predictions from Phase 3
y_pred_rf = lr_predictions['rf_pred'].values

print(f"\nAligned validation rows: {len(y_val)}")

# STEP 5: Ensemble weight optimization via grid search
print("\n--- Ensemble Weight Optimization ---")
best_w, best_rmse, best_mae, best_mape = None, float('inf'), None, None

weight_results = []
for w in np.arange(0, 1.05, 0.05):
    y_ens = w * y_pred_fe + (1 - w) * y_pred_lgb
    rmse = np.sqrt(mean_squared_error(y_val, y_ens))
    mae = mean_absolute_error(y_val, y_ens)
    non_zero_mask = y_val > 0
    mape = np.mean(np.abs((y_val[non_zero_mask] - y_ens[non_zero_mask]) / y_val[non_zero_mask])) * 100
    weight_results.append({'weight_panel_fe': w, 'weight_lgb': 1-w, 'rmse': rmse, 'mae': mae, 'mape': mape})
    if rmse < best_rmse:
        best_rmse, best_mae, best_mape = rmse, mae, mape
        best_w = w

print(f"Optimal weight (Panel FE): {best_w:.2f}")
print(f"Optimal weight (LightGBM): {1-best_w:.2f}")
print(f"Optimal ensemble RMSE: {best_rmse:,.2f}")

# Save weight search results
weights_df = pd.DataFrame(weight_results)
weights_df.to_csv("Task2_Data/task2_step4_weight_search.csv", index=False)
print(f"Weight search saved to Task2_Data/task2_step4_weight_search.csv")

# STEP 6: Generate final ensemble predictions
y_pred_ensemble = best_w * y_pred_fe + (1 - best_w) * y_pred_lgb

# STEP 7: Compute all metrics
def compute_metrics(y_true, y_pred, model_name):
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

all_models = [
    ('PanelFE', y_pred_fe),
    ('LightGBM', y_pred_lgb),
    ('Ensemble', y_pred_ensemble),
    ('LinearRegression', y_pred_lr),
    ('RandomForest', y_pred_rf),
]

metrics_results = []
for name, pred in all_models:
    m = compute_metrics(y_val, pred, name)
    metrics_results.append(m)
    print(f"{name}: RMSE={m['rmse']:,.0f}, MAE={m['mae']:,.0f}, MAPE={m['mape']:.2f}%")

metrics_df = pd.DataFrame(metrics_results)
METRICS_OUT = "Task2_Data/task2_step4_metrics.csv"
metrics_df.to_csv(METRICS_OUT, index=False)
print(f"\nMetrics saved to {METRICS_OUT}")

# STEP 8: Save ensemble predictions
predictions_out = val_df_full.copy()
predictions_out['panel_fe_pred'] = y_pred_fe
predictions_out['lgb_pred'] = y_pred_lgb
predictions_out['ensemble_pred'] = y_pred_ensemble
predictions_out['lr_pred'] = y_pred_lr
predictions_out['rf_pred'] = y_pred_rf

PREDICTIONS_OUT = "Task2_Data/task2_step4_predictions.csv"
predictions_out.to_csv(PREDICTIONS_OUT, index=False)
print(f"Predictions saved to {PREDICTIONS_OUT}")

# STEP 9: MODEL-04 improvement check
print("\n--- MODEL-04: Ensemble Improvement Check ---")
lr_rmse = metrics_df[metrics_df['model']=='LinearRegression']['rmse'].values[0]
rf_rmse = metrics_df[metrics_df['model']=='RandomForest']['rmse'].values[0]
ens_rmse = metrics_df[metrics_df['model']=='Ensemble']['rmse'].values[0]
panel_rmse = metrics_df[metrics_df['model']=='PanelFE']['rmse'].values[0]
lgb_rmse = metrics_df[metrics_df['model']=='LightGBM']['rmse'].values[0]

print(f"Baseline LR RMSE: {lr_rmse:,.0f}")
print(f"Baseline RF RMSE: {rf_rmse:,.0f}")
print(f"PanelFE RMSE: {panel_rmse:,.0f} (improvement over LR: {(lr_rmse-panel_rmse)/lr_rmse*100:.1f}%)")
print(f"LightGBM RMSE: {lgb_rmse:,.0f} (improvement over LR: {(lr_rmse-lgb_rmse)/lr_rmse*100:.1f}%)")
print(f"Ensemble RMSE: {ens_rmse:,.0f} (improvement over LR: {(lr_rmse-ens_rmse)/lr_rmse*100:.1f}%)")
improvement = min(lr_rmse, rf_rmse) - ens_rmse
print(f"\nEnsemble improvement over best baseline: {improvement:,.0f} RMSE ({improvement/min(lr_rmse,rf_rmse)*100:.1f}%)")

print("\nEnsemble model complete.")