"""
Task 2 Step 2: Feature Engineering for Insurance Premium Prediction
Transforms clean panel dataset into model-ready feature matrix.
"""

import pandas as pd

# ============================================================
# CONSTANTS
# ============================================================
INPUT_FILE = "task2_step1_panel_clean.csv"
OUTPUT_FILE = "task2_step2_feature_matrix.csv"

# ============================================================
# STEP 1: Load data
# ============================================================
print("\n--- STEP 1: Load data ---")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Input shape: {df.shape}")
print(f"Years present: {sorted(df['Year'].unique())}")

# ============================================================
# STEP 2: Sort for correct temporal ordering
# ============================================================
print("\n--- STEP 2: Sort for temporal ordering ---")
df = df.sort_values(['ZIP', 'Category', 'Year']).reset_index(drop=True)
print(f"Sorted data shape: {df.shape}")

# ============================================================
# STEP 3: One-hot encode categories BEFORE lag computation
# ============================================================
print("\n--- STEP 3: One-hot encode categories ---")
categories = ['HO', 'CO', 'DT', 'RT', 'DO', 'MH', 'NA']
for cat in categories:
    df[f'cat_{cat}'] = (df['Category'] == cat).astype(int)
print(f"Category columns created: {[f'cat_{c}' for c in categories]}")

# ============================================================
# STEP 4: Compute one-year LAG features via groupby shift
# ============================================================
print("\n--- STEP 4: Compute one-year lag features ---")
lag_features = [
    'Avg Fire Risk Score',
    'Earned Exposure',
    'Earned Premium',
    'CAT Cov A Fire -  Incurred Losses',
    'Non-CAT Cov A Fire -  Incurred Losses',
    'Avg PPC',
]

for col in lag_features:
    df[f'{col}_lag1'] = df.groupby(['ZIP', 'Category'])[col].shift(1)
    print(f"  Created lag: {col}_lag1")

# ============================================================
# STEP 5: Compute EXPANDING window statistics
# ============================================================
print("\n--- STEP 5: Compute expanding window statistics ---")

# Expanding mean of fire risk — shift(1) excludes current year to prevent leakage
df['expanding_fire_risk_mean'] = df.groupby(['ZIP', 'Category'])['Avg Fire Risk Score'].transform(
    lambda x: x.expanding().mean().shift(1)
)
print("  Created: expanding_fire_risk_mean")

# Expanding std of fire risk — may be NaN for single-observation groups (acceptable)
df['expanding_fire_risk_std'] = df.groupby(['ZIP', 'Category'])['Avg Fire Risk Score'].transform(
    lambda x: x.expanding().std().shift(1)
)
print("  Created: expanding_fire_risk_std")

# Expanding mean of earned exposure
df['expanding_exposure_mean'] = df.groupby(['ZIP', 'Category'])['Earned Exposure'].transform(
    lambda x: x.expanding().mean().shift(1)
)
print("  Created: expanding_exposure_mean")

# ============================================================
# STEP 6: Add COVID-19 year indicator
# ============================================================
print("\n--- STEP 6: Add COVID-19 year indicator ---")
df['is_covid_year'] = (df['Year'] == 2020).astype(int)
print(f"  is_covid_year: 1 for 2020, 0 otherwise")

# ============================================================
# STEP 7: Remove 2018 rows
# ============================================================
print("\n--- STEP 7: Remove 2018 rows ---")
df_final = df[df['Year'] != 2018].copy()
print(f"Final shape (2019+2020 only): {df_final.shape}")
print(f"Year distribution:\n{df_final['Year'].value_counts().sort_index()}")

# ============================================================
# ASSERTIONS to validate
# ============================================================
print("\n--- Running assertions ---")

# Identify which ZIP+Category combinations have prior year data
# (i.e., existed in year-1 for each target year)
df['ZIP_Cat'] = list(zip(df['ZIP'], df['Category']))
df_final['ZIP_Cat'] = list(zip(df_final['ZIP'], df_final['Category']))

# For each year, determine which combos had prior-year data available
prior_2018 = set(df[df['Year'] == 2018]['ZIP_Cat'])
prior_2019 = set(df[df['Year'] == 2019]['ZIP_Cat'])

# For 2019 lag check: acceptable NaN if combo first appeared in 2019 (no 2018 data)
# For 2020 lag check: acceptable NaN if combo first appeared in 2019 OR 2020 (no 2019 data)
first_year = df.groupby('ZIP_Cat')['Year'].min()
combos_first_2019 = set(first_year[first_year == 2019].index)
combos_first_2020 = set(first_year[first_year == 2020].index)

for col in lag_features:
    lag_col = f'{col}_lag1'

    # 2019 lags: NaN only acceptable for combos that first appeared in 2019
    df_2019 = df_final[df_final['Year'] == 2019]
    nan_2019_known = df_2019['ZIP_Cat'].apply(lambda x: x in combos_first_2019)
    nan_2019_unacceptable = df_2019[~nan_2019_known][lag_col].isna().sum()
    nan_2019_acceptable = df_2019[nan_2019_known][lag_col].isna().sum()
    print(f"  {lag_col} (2019): {nan_2019_unacceptable} unacceptable NaN, {nan_2019_acceptable} acceptable NaN")
    assert nan_2019_unacceptable == 0, f"Lag feature {lag_col} has unacceptable NaN in 2019"

    # 2020 lags: NaN only acceptable for combos first appearing in 2019 or 2020
    df_2020 = df_final[df_final['Year'] == 2020]
    combos_no_2019 = combos_first_2019 | combos_first_2020
    nan_2020_known = df_2020['ZIP_Cat'].apply(lambda x: x in combos_no_2019)
    nan_2020_unacceptable = df_2020[~nan_2020_known][lag_col].isna().sum()
    nan_2020_acceptable = df_2020[nan_2020_known][lag_col].isna().sum()
    print(f"  {lag_col} (2020): {nan_2020_unacceptable} unacceptable NaN, {nan_2020_acceptable} acceptable NaN")
    assert nan_2020_unacceptable == 0, f"Lag feature {lag_col} has unacceptable NaN in 2020"

# Category columns sum to 1 per row
cat_cols = ['cat_HO', 'cat_CO', 'cat_DT', 'cat_RT', 'cat_DO', 'cat_MH', 'cat_NA']
assert df_final[cat_cols].sum(axis=1).eq(1).all(), "Category columns do not sum to 1 per row"
print("  Category columns sum to 1 per row: PASS")

# COVID flag is 1 for 2020, 0 for 2019
assert df_final[df_final['Year'] == 2019]['is_covid_year'].unique()[0] == 0
assert df_final[df_final['Year'] == 2020]['is_covid_year'].unique()[0] == 1
print("  COVID flag correct: PASS")

# ============================================================
# STEP 8: Save output
# ============================================================
print("\n--- STEP 8: Save output ---")
df_final.to_csv(OUTPUT_FILE, index=False)
print(f"Output saved: {OUTPUT_FILE}")
print(f"Rows: {df_final.shape[0]}, Columns: {df_final.shape[1]}")

# ============================================================
# FEAT-05 DOCUMENTATION: Zero-Inflated Premium Distribution
# ============================================================
print("\n--- FEAT-05: Zero-Inflated Premium Distribution ---")
premium = df_final['Earned Premium']
zero_pct = (premium == 0).mean()
print(f"Zero premium rows: {(premium == 0).sum()} / {len(premium)} ({zero_pct:.1%})")
print(f"Non-zero premium rows: {(premium > 0).sum()} / {len(premium)} ({(1-zero_pct):.1%})")
print(f"Premium stats (non-zero only):")
non_zero = premium[premium > 0]
print(f"  mean={non_zero.mean():,.0f}, median={non_zero.median():,.0f}")
print(f"  min={non_zero.min():,.0f}, max={non_zero.max():,.0f}")
print("\nNote: FEAT-05 (Tweedie vs two-part model decision) deferred to Phase 3.")

print("\n=== Feature engineering complete ===")
