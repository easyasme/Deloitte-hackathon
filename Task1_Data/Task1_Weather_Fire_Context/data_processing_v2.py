import pandas as pd
import numpy as np

# STEP 1 v2: Weather + Fire Context
# Input : abfap7bci2UF6CTY_wildfire_weather.csv
# Output:
#   - task1_step1_v2_event_cleaned.csv
#   - task1_step1_v2_zip_year_ready.csv
#
# Purpose:
#   Build a modeling-ready zip-year dataset using:
#   - weather variables
#   - fire context variables (CAUSE, AGENCY_ID)


INPUT_FILE = "abfap7bci2UF6CTY_wildfire_weather.csv"

EVENT_OUTPUT = "task1_step1_v2_event_cleaned.csv"
ZIP_YEAR_OUTPUT = "task1_step1_v2_zip_year_ready.csv"


# 1. Load
df = pd.read_csv(INPUT_FILE, low_memory=False)

print("input data shape:", df.shape)
print("input columns:")
print(df.columns.tolist())


# 2. Keep needed columns
candidate_cols = [
    "OBJECTID",
    "Year",
    "ALARM_DATE",
    "CONT_DATE",
    "CAUSE",
    "C_METHOD",
    "OBJECTIVE",
    "GIS_ACRES",
    "latitude",
    "longitude",
    "zip",
    "year_month",
    "avg_tmax_c",
    "avg_tmin_c",
    "tot_prcp_mm",
    "station",
    "AGENCY_ID",
    "FIRE_NAME_ID",
]

keep_cols = [c for c in candidate_cols if c in df.columns]
df = df[keep_cols].copy()

print("\nselected columns:")
print(df.columns.tolist())
print("shape:", df.shape)


# 3. Type cleaning
date_cols = [c for c in ["ALARM_DATE", "CONT_DATE"] if c in df.columns]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

numeric_cols = [
    "OBJECTID", "Year", "CAUSE", "C_METHOD", "OBJECTIVE", "GIS_ACRES",
    "latitude", "longitude", "zip",
    "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm", "station",
    "AGENCY_ID", "FIRE_NAME_ID"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\ntype converted")


# 3.5. Guide-based data fixes
# AGENCY_ID = 9 means missing
if "AGENCY_ID" in df.columns:
    count_9 = (df["AGENCY_ID"] == 9).sum()
    df["AGENCY_ID"] = df["AGENCY_ID"].replace(9, np.nan)
    print(f"\nAGENCY_ID=9 -> NaN: {count_9}개")

# OBJECTIVE sometimes contains bad values per guide
if "OBJECTIVE" in df.columns:
    df.loc[~df["OBJECTIVE"].isin([1, 2]), "OBJECTIVE"] = np.nan
    print("OBJECTIVE value (different from 1,2 -> NaN)")


# 4. Remove duplicates
before = len(df)
if "OBJECTID" in df.columns:
    df = df.drop_duplicates(subset=["OBJECTID"])
    print(f"\nOBJECTID duplicate removed: {before} -> {len(df)}")
else:
    df = df.drop_duplicates()
    print(f"\nrow duplicate removed: {before} -> {len(df)}")


# 5. Fill ZIP by lat/lon mode
def fill_zip_by_latlon_mode(group):
    if "zip" in group.columns and group["zip"].notna().any():
        zip_mode = group["zip"].dropna().mode()
        if not zip_mode.empty:
            group["zip"] = group["zip"].fillna(zip_mode.iloc[0])
    return group

if {"latitude", "longitude", "zip"}.issubset(df.columns):
    before_missing = df["zip"].isna().sum()

    df = (
        df.groupby(["latitude", "longitude"], dropna=False, group_keys=False)
          .apply(fill_zip_by_latlon_mode)
          .reset_index(drop=True)
    )

    after_missing = df["zip"].isna().sum()
    print("before:", before_missing)
    print("after :", after_missing)


# 6. Basic filtering
required_cols = [c for c in ["Year", "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm"] if c in df.columns]

before = len(df)
df = df.dropna(subset=required_cols)
print(f"\nmissing value removed: {before} -> {len(df)}")

if "Year" in df.columns:
    before = len(df)
    df = df[(df["Year"] >= 2018) & (df["Year"] <= 2023)]
    print(f"Year filter(2018~2023): {before} -> {len(df)}")

if "GIS_ACRES" in df.columns:
    before = len(df)
    df = df[df["GIS_ACRES"].isna() | (df["GIS_ACRES"] >= 0)]
    print(f"GIS_ACRES minus removed: {before} -> {len(df)}")

if "zip" in df.columns:
    before = len(df)
    df = df.dropna(subset=["zip"])
    df["zip"] = df["zip"].astype(int)
    print(f"zip missing value removed: {before} -> {len(df)}")


# 7. Helper columns
if "ALARM_DATE" in df.columns:
    df["alarm_month"] = df["ALARM_DATE"].dt.month
    df["alarm_dayofyear"] = df["ALARM_DATE"].dt.dayofyear

if {"ALARM_DATE", "CONT_DATE"}.issubset(df.columns):
    df["fire_duration_days"] = (df["CONT_DATE"] - df["ALARM_DATE"]).dt.days

df["fire_event"] = 1

print("\ndone")


# 8. Save event-level output
df.to_csv(EVENT_OUTPUT, index=False)
print("\nevent-level saved:", EVENT_OUTPUT)
print("event-level shape:", df.shape)


# 9. ZIP-YEAR aggregation
agg_dict = {}

# weather / event numeric summaries
for col in ["avg_tmax_c", "avg_tmin_c", "tot_prcp_mm", "GIS_ACRES", "fire_duration_days"]:
    if col in df.columns:
        agg_dict[col] = "mean"

# fire context columns -> mode
for col in ["CAUSE", "C_METHOD", "OBJECTIVE", "station", "AGENCY_ID"]:
    if col in df.columns:
        agg_dict[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan

zip_year_fire = (
    df.groupby(["zip", "Year"], as_index=False)
      .agg(agg_dict)
)

fire_counts = (
    df.groupby(["zip", "Year"])
      .size()
      .reset_index(name="fire_count")
)

zip_year_fire = zip_year_fire.merge(fire_counts, on=["zip", "Year"], how="left")
zip_year_fire["fire_occurred"] = 1

print("\nfire occurred zip-year shape:", zip_year_fire.shape)


# 10. Negative sampling by full ZIP-YEAR grid
all_zips = sorted(df["zip"].dropna().unique())
all_years = sorted(df["Year"].dropna().unique())

full_pairs = pd.MultiIndex.from_product(
    [all_zips, all_years],
    names=["zip", "Year"]
).to_frame(index=False)

zip_year_full = full_pairs.merge(zip_year_fire, on=["zip", "Year"], how="left")

zip_year_full["fire_occurred"] = zip_year_full["fire_occurred"].fillna(0).astype(int)
zip_year_full["fire_count"] = zip_year_full["fire_count"].fillna(0).astype(int)

print("\nshape after negative sampling:", zip_year_full.shape)
print(zip_year_full["fire_occurred"].value_counts())


# 11. Fill missing values
# weather -> zip median -> global median
numeric_fill_cols = [c for c in ["avg_tmax_c", "avg_tmin_c", "tot_prcp_mm", "GIS_ACRES", "fire_duration_days"] if c in zip_year_full.columns]

for col in numeric_fill_cols:
    zip_year_full[col] = zip_year_full.groupby("zip")[col].transform(lambda x: x.fillna(x.median()))
    zip_year_full[col] = zip_year_full[col].fillna(zip_year_full[col].median())

# fire context -> zip mode -> global mode
mode_fill_cols = [c for c in ["CAUSE", "C_METHOD", "OBJECTIVE", "station", "AGENCY_ID"] if c in zip_year_full.columns]

for col in mode_fill_cols:
    zip_year_full[col] = zip_year_full.groupby("zip")[col].transform(
        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else np.nan)
    )
    if zip_year_full[col].notna().any():
        zip_year_full[col] = zip_year_full[col].fillna(zip_year_full[col].mode().iloc[0])

print("\ndone")


# 12. Save output
zip_year_full.to_csv(ZIP_YEAR_OUTPUT, index=False)

print("\nzip-year saved:", ZIP_YEAR_OUTPUT)
print("zip-year shape:", zip_year_full.shape)

print("\nfinal columns:")
print(zip_year_full.columns.tolist())

print("\nsamples:")
print(zip_year_full.head())

print("\nCaution")
print("- never use fire_count for feature")
print("CAUSE / OBJECTIVE / AGENCY_ID are context values, so attention needed for interpreting pure forecasting")