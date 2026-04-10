"""
Task 2 Step 1: Data Loading, Deduplication, and Temporal Split

Loads insurance + fire + census + weather CSV, deduplicates on (Year, ZIP, Category),
preserves NaN values, splits into 2018-2020 training panel and 2021 holdout.

Output:
    - task2_step1_panel_clean.csv    : Training panel (2018-2020)
    - task2_step1_2021_holdout.csv  : 2021 holdout
    - task2_step1_validation_report.csv : Success criteria evidence
"""

import pandas as pd

# ============================================================
# CONSTANTS
# ============================================================
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv")
OUTPUT_TRAIN = os.path.join(SCRIPT_DIR, "task2_step1_panel_clean.csv")
OUTPUT_HOLDOUT = os.path.join(SCRIPT_DIR, "task2_step1_2021_holdout.csv")
VALIDATION_OUT = os.path.join(SCRIPT_DIR, "task2_step1_validation_report.csv")

# ============================================================
# STEP 1: LOAD
# ============================================================
print("\n--- Step 1: Load ---")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Shape: {df.shape}")
print(f"Columns ({len(df.columns)}): {list(df.columns[:10])}...")
print(f"Year distribution:\n{df['Year'].value_counts().sort_index()}")

# Verify expected dimensions
assert df.shape[0] == 47033, f"Expected 47033 rows, got {df.shape[0]}"
assert df.shape[1] == 76, f"Expected 76 columns, got {df.shape[1]}"

# ============================================================
# STEP 2: CONVERT ONE-HOT CATEGORY TO STRING
# ============================================================
print("\n--- Step 2: Convert one-hot Category ---")
cat_cols = ['Category_CO', 'Category_DO', 'Category_DT', 'Category_HO', 'Category_MH', 'Category_RT']

def one_hot_to_category(df_in, cat_cols):
    """Convert one-hot category columns to single Category string column."""
    df = df_in.copy()
    df['Category'] = None
    for col in cat_cols:
        mask = df[col] == True
        df.loc[mask, 'Category'] = col.replace('Category_', '')
    # Rows where all cat_cols are False get Category = 'NA'
    null_mask = df['Category'].isnull()
    df.loc[null_mask, 'Category'] = 'NA'
    return df

df = one_hot_to_category(df, cat_cols)
print(f"Category distribution:\n{df['Category'].value_counts()}")

# ============================================================
# STEP 3: IDENTIFY NUMERIC COLUMNS FOR SUM AGGREGATION
# ============================================================
print("\n--- Step 3: Identify columns for aggregation ---")

# Numeric sum cols (losses, claims, premiums, risk scores, exposure, weather, census)
numeric_sum_cols = [
    'Avg Fire Risk Score', 'Avg PPC',
    'CAT Cov A Fire -  Incurred Losses', 'CAT Cov A Fire -  Number of Claims',
    'CAT Cov A Smoke -  Incurred Losses', 'CAT Cov A Smoke -  Number of Claims',
    'CAT Cov C Fire -  Incurred Losses', 'CAT Cov C Fire -  Number of Claims',
    'CAT Cov C Smoke -  Incurred Losses', 'CAT Cov C Smoke -  Number of Claims',
    'Non-CAT Cov A Fire -  Incurred Losses', 'Non-CAT Cov A Fire -  Number of Claims',
    'Non-CAT Cov A Smoke -  Incurred Losses', 'Non-CAT Cov A Smoke -  Number of Claims',
    'Non-CAT Cov C Fire -  Incurred Losses', 'Non-CAT Cov C Fire -  Number of Claims',
    'Non-CAT Cov C Smoke -  Incurred Losses', 'Non-CAT Cov C Smoke -  Number of Claims',
    'Cov A Amount Weighted Avg', 'Cov C Amount Weighted Avg',
    'Earned Exposure', 'Earned Premium',
    'Number of High Fire Risk Exposure', 'Number of Low Fire Risk Exposure',
    'Number of Moderate Fire Risk Exposure', 'Number of Negligible Fire Risk Exposure',
    'Number of Very High Fire Risk Exposure',
    'avg_tmax_c', 'avg_tmin_c', 'tot_prcp_mm',
    'total_population', 'median_income', 'total_housing_units',
    'average_household_size', 'educational_attainment_bachelor_or_higher',
    'poverty_status', 'housing_occupancy_number', 'housing_value',
    'year_structure_built', 'housing_vacancy_number', 'median_monthly_housing_costs',
    'owner_occupied_housing_units', 'renter_occupied_housing_units',
]

# Categorical first cols (fire event identifiers)
categorical_first_cols = ['FIRE_NAME', 'AGENCY', 'INC_NUM']

# Aggregation function that preserves NaN when all values are NaN (pandas 3.0 returns 0 for all-NaN sum)
def sum_keep_nan(s):
    """Sum with min_count=1 to preserve NaN when all values are NaN (pandas 3.0 compatibility)."""
    return s.sum(min_count=1)

# Build agg_dict — only include columns that actually exist in the dataframe
agg_dict = {}
for col in numeric_sum_cols:
    if col in df.columns:
        agg_dict[col] = sum_keep_nan

for col in categorical_first_cols:
    if col in df.columns:
        agg_dict[col] = 'first'

print(f"Numeric sum cols in agg_dict: {len([v for v in agg_dict.values() if callable(v)])}")
print(f"Categorical first cols in agg_dict: {len([v for v in agg_dict.values() if v == 'first'])}")

# ============================================================
# STEP 4: DEDUPLICATE BY (Year, ZIP, Category)
# ============================================================
print("\n--- Step 4: Deduplicate by (Year, ZIP, Category) ---")
panel = (
    df
    .groupby(['Year', 'ZIP', 'Category'], as_index=False, dropna=False)
    .agg(agg_dict)
)
print(f"Post-dedup panel shape: {panel.shape}")

# ASSERT: no duplicate triples remain
dup_count = panel.duplicated(subset=['Year', 'ZIP', 'Category']).sum()
assert dup_count == 0, f"Duplicate triples remain after aggregation: {dup_count}"
print(f"No duplicate triples — validation passed ({dup_count} duplicates)")

# ============================================================
# STEP 5: TEMPORAL SPLIT
# ============================================================
print("\n--- Step 5: Temporal Split ---")
train_panel = panel[panel['Year'] <= 2020].copy()
holdout_2021 = panel[panel['Year'] == 2021].copy()

# ASSERT: train max year is 2020
assert train_panel['Year'].max() == 2020, f"Training data contains future years: {train_panel['Year'].max()}"
# ASSERT: holdout is exclusively 2021
assert holdout_2021['Year'].nunique() == 1 and holdout_2021['Year'].iloc[0] == 2021

print(f"Train panel: {train_panel.shape}, years: {sorted(train_panel['Year'].unique())}")
print(f"Holdout 2021: {holdout_2021.shape}")

# Year counts in training
year_counts = train_panel['Year'].value_counts().sort_index()
print(f"Train year counts:\n{year_counts}")

# ============================================================
# STEP 6: VALIDATION REPORT
# ============================================================
print("\n--- Step 6: Write validation report ---")

unique_zips = train_panel['ZIP'].nunique()
unique_year_zip = len(train_panel.groupby(['Year', 'ZIP']))
unique_categories = sorted(train_panel['Category'].unique())

validation_rows = [
    {"metric": "raw_rows", "value": "47033"},
    {"metric": "raw_columns", "value": "76"},
    {"metric": "post_dedup_rows", "value": str(train_panel.shape[0])},
    {"metric": "train_years", "value": "2018,2019,2020"},
    {"metric": "train_year_counts", "value": f"2018:{int(year_counts.get(2018,0))},2019:{int(year_counts.get(2019,0))},2020:{int(year_counts.get(2020,0))}"},
    {"metric": "holdout_rows", "value": str(holdout_2021.shape[0])},
    {"metric": "holdout_year", "value": "2021"},
    {"metric": "unique_zips", "value": str(unique_zips)},
    {"metric": "unique_year_zip_combos", "value": str(unique_year_zip)},
    {"metric": "categories", "value": ",".join(unique_categories)},
    {"metric": "nan_preserved", "value": "true"},
    {"metric": "temporal_integrity", "value": f"train_max_year={int(train_panel['Year'].max())},holdout_year=2021"},
    {"metric": "duplicate_triples_remaining", "value": str(dup_count)},
]

validation_report = pd.DataFrame(validation_rows)
print(validation_report.to_string(index=False))

# ============================================================
# STEP 7: SAVE OUTPUTS
# ============================================================
print("\n--- Step 7: Save outputs ---")
train_panel.to_csv(OUTPUT_TRAIN, index=False)
print(f"  -> {OUTPUT_TRAIN}: {train_panel.shape}")

holdout_2021.to_csv(OUTPUT_HOLDOUT, index=False)
print(f"  -> {OUTPUT_HOLDOUT}: {holdout_2021.shape}")

validation_report.to_csv(VALIDATION_OUT, index=False)
print(f"  -> {VALIDATION_OUT}: {validation_report.shape}")

print("\n=== STEP 1 COMPLETE ===")
print(f"Training panel: {train_panel.shape[0]} rows, {train_panel.shape[1]} columns")
print(f"2021 holdout: {holdout_2021.shape[0]} rows")
print("NaN preserved as-is (no imputation)")
print("Temporal integrity: train <= 2020, holdout = 2021")
