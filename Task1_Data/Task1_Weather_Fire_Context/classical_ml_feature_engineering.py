import pandas as pd
import numpy as np

# STEP 2 v2: Classical ML Feature Engineering (Weather + Fire Context)
# Input : task1_step1_v2_zip_year_ready.csv
# Output: task1_step2_v2_classical_ready.csv
#
# Goal:
#   Build a modeling-ready dataset using:
#   - Weather features
#   - Fire context features (CAUSE, AGENCY_ID, OBJECTIVE, C_METHOD)

INPUT_FILE = "task1_step1_v2_zip_year_ready.csv"
OUTPUT_FILE = "task1_step2_v2_classical_ready.csv"

# 1. Load
df = pd.read_csv(INPUT_FILE)

print("input data shape:", df.shape)
print("columns:")
print(df.columns.tolist())


# 2. Required columns check
required_cols = ["zip", "Year", "fire_occurred"]
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    raise ValueError(f"no necessary columns: {missing_required}")

print("\nDone")


# 3. Remove leakage / post-event columns
# fire_count: label leakage
# GIS_ACRES, fire_duration_days: post-event information
drop_cols = [
    "fire_count",
    "GIS_ACRES",
    "fire_duration_days"
]

existing_drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=existing_drop_cols)

print("\n[3] leakage / post-event 컬럼 제거 완료")
print("제거된 컬럼:", existing_drop_cols)
print("현재 shape:", df.shape)


# 4. Create engineered weather features
if {"avg_tmax_c", "avg_tmin_c"}.issubset(df.columns):
    df["temp_range_c"] = df["avg_tmax_c"] - df["avg_tmin_c"]

if {"avg_tmax_c", "tot_prcp_mm"}.issubset(df.columns):
    df["dryness_proxy"] = df["avg_tmax_c"] / (df["tot_prcp_mm"] + 1)

if "Year" in df.columns:
    min_year = df["Year"].min()
    df["year_index"] = df["Year"] - min_year

print("\nengineered feature generated")
engineered_cols = [c for c in ["temp_range_c", "dryness_proxy", "year_index"] if c in df.columns]
print("added feature:", engineered_cols)


# 5. Clean / normalize context feature values
# OBJECTIVE -> use 1 or 2
if "OBJECTIVE" in df.columns:
    df.loc[~df["OBJECTIVE"].isin([1, 2]), "OBJECTIVE"] = np.nan

# CAUSE는 -> use 1 ~ 18
if "CAUSE" in df.columns:
    df.loc[~df["CAUSE"].isin(list(range(1, 19))), "CAUSE"] = np.nan

# C_METHOD -> use 1 ~ 8
if "C_METHOD" in df.columns:
    df.loc[~df["C_METHOD"].isin(list(range(1, 9))), "C_METHOD"] = np.nan

# AGENCY_ID -> numeric context
print("\ncontext features organized")


# 6. Convert context variables to categorical dummies
# No zip, station
context_categorical_cols = [c for c in ["CAUSE", "OBJECTIVE", "C_METHOD", "AGENCY_ID"] if c in df.columns]

# category string converted and then one-hot
for col in context_categorical_cols:
    df[col] = df[col].astype("Int64").astype(str)
    df.loc[df[col] == "<NA>", col] = "missing"

if context_categorical_cols:
    context_dummies = pd.get_dummies(
        df[context_categorical_cols],
        prefix=context_categorical_cols,
        drop_first=True
    )
    df = pd.concat([df.drop(columns=context_categorical_cols), context_dummies], axis=1)

print("\ncontext feature one-hot encoding done")
print("number of encoded context columns:", len([c for c in df.columns if any(c.startswith(prefix + "_") for prefix in ["CAUSE", "OBJECTIVE", "C_METHOD", "AGENCY_ID"])]))


# 7. Optionally drop station
if "station" in df.columns:
    df = df.drop(columns=["station"])
    print("\nstation column dropped")


# 8. Missing-value handling
missing_before = int(df.isna().sum().sum())

for col in df.columns:
    if col == "fire_occurred":
        continue

    if df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_vals = df[col].mode()
            if not mode_vals.empty:
                df[col] = df[col].fillna(mode_vals.iloc[0])

missing_after = int(df.isna().sum().sum())

print("\nmissing-value handled")
print("missing before:", missing_before)
print("missing after:", missing_after)


# -----------------------------
# 9. Recommended feature set preview
# -----------------------------
recommended_weather_cols = [
    "avg_tmax_c",
    "avg_tmin_c",
    "tot_prcp_mm",
    "temp_range_c",
    "dryness_proxy",
    "year_index"
]
recommended_weather_cols = [c for c in recommended_weather_cols if c in df.columns]

recommended_context_cols = [
    c for c in df.columns
    if c.startswith("CAUSE_")
    or c.startswith("OBJECTIVE_")
    or c.startswith("C_METHOD_")
    or c.startswith("AGENCY_ID_")
]

print("\nrecommended weather feature:")
print(recommended_weather_cols)

print("\nrecommended context feature (dummy):")
print(recommended_context_cols[:20])
if len(recommended_context_cols) > 20:
    print(f"... total {len(recommended_context_cols)}개")


# 10. Final column ordering
front_cols = [c for c in ["zip", "Year", "fire_occurred"] if c in df.columns]
other_cols = [c for c in df.columns if c not in front_cols]
df = df[front_cols + other_cols]

print("\nfinal data shape:", df.shape)
print("final columns:")
print(df.columns.tolist())


# 11. Save
df.to_csv(OUTPUT_FILE, index=False)

print("\nsaved:", OUTPUT_FILE)


# 12. Preview
print("\nsample")
print(df.head())

print("\ntarget distribution")
print(df["fire_occurred"].value_counts())