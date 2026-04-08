import pandas as pd
import numpy as np

# STEP 2: Classical ML용 Feature Engineering & Cleaning
# Input : task1_step1_zip_year_ready.csv
# Output: task1_step2_classical_ready.csv

INPUT_FILE = "task1_step1_zip_year_ready.csv"

# 1. Load
df = pd.read_csv(INPUT_FILE)

print("Input data shape:", df.shape)
print("columns:")
print(df.columns.tolist())


# 2. Basic sanity checks
required_cols = ["zip", "Year", "fire_occurred"]
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    raise ValueError(f"no necessary columns: {missing_required}")

print("\ndone checking needed columns")


# 3. Remove leakage / post-event columns
# fire_count: directly connected to target
# GIS_ACRES, fire_duration_days: can be known after fire occurs, not good for feature
drop_cols = [
    "fire_count",
    "GIS_ACRES",
    "fire_duration_days"
]

existing_drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=existing_drop_cols)

print("\nleakage/post-event column removed")
print("removed columns:", existing_drop_cols)
print("current shape:", df.shape)


# 4. Create simple engineered features

# temperature range
if {"avg_tmax_c", "avg_tmin_c"}.issubset(df.columns):
    df["temp_range_c"] = df["avg_tmax_c"] - df["avg_tmin_c"]

# dry proxy: temperature / (precipitation + 1)
if {"avg_tmax_c", "tot_prcp_mm"}.issubset(df.columns):
    df["dryness_proxy"] = df["avg_tmax_c"] / (df["tot_prcp_mm"] + 1)

# year relative index (2018 -> 0, 2019 -> 1 ...)
if "Year" in df.columns:
    min_year = df["Year"].min()
    df["year_index"] = df["Year"] - min_year

print("\nfeature engineering done")
new_features = [c for c in ["temp_range_c", "dryness_proxy", "year_index"] if c in df.columns]
print("new feature:", new_features)


# 5. Clean data types
numeric_candidates = [
    "zip", "Year", "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm",
    "temp_range_c", "dryness_proxy", "year_index", "fire_occurred"
]

for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nconverged to int")


# 6. Handle station column
USE_STATION = False

if "station" in df.columns:
    if USE_STATION:
        df["station"] = df["station"].astype(str)
        station_dummies = pd.get_dummies(df["station"], prefix="station", drop_first=True)
        df = pd.concat([df.drop(columns=["station"]), station_dummies], axis=1)
        print("\nstation one-hot encoding done")
    else:
        df = df.drop(columns=["station"])
        print("\nstation column removed (baseline simplified)")


# 7. Missing-value handling
# median
feature_cols_for_fill = [c for c in df.columns if c != "fire_occurred"]

missing_before = df.isna().sum().sum()

for col in feature_cols_for_fill:
    if df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_vals = df[col].mode()
            if not mode_vals.empty:
                df[col] = df[col].fillna(mode_vals.iloc[0])

missing_after = df.isna().sum().sum()

print("\nmissing value handled")
print("before:", int(missing_before))
print("after:", int(missing_after))


# 8. Select final baseline feature set

recommended_features = [
    "avg_tmax_c",
    "avg_tmin_c",
    "tot_prcp_mm",
    "temp_range_c",
    "dryness_proxy",
    "year_index"
]

recommended_features = [c for c in recommended_features if c in df.columns]

print("\nrecommended baseline features:")
print(recommended_features)


# 9. Final column ordering
front_cols = [c for c in ["zip", "Year", "fire_occurred"] if c in df.columns]
other_cols = [c for c in df.columns if c not in front_cols]
df = df[front_cols + other_cols]

print("\nfinal data shape:", df.shape)
print("final columns:")
print(df.columns.tolist())


# 10. Save
OUTPUT_FILE = "task1_step2_classical_ready.csv"
df.to_csv(OUTPUT_FILE, index=False)

print("\nsaved:", OUTPUT_FILE)


# 11. Preview
print("\nsample")
print(df.head())

print("\ntarget distribution")
print(df["fire_occurred"].value_counts())