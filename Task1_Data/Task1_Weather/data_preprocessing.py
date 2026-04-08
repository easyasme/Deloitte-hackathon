import pandas as pd
import numpy as np

# STEP 1: Data Integration + Basic Cleaning
INPUT_FILE = "abfap7bci2UF6CTY_wildfire_weather.csv"

# 1. Load
df = pd.read_csv(INPUT_FILE, low_memory=False)

print("input data shape:", df.shape)
print("columns:")
print(df.columns.tolist())


# 2. Keep useful columns only
candidate_cols = [
    "OBJECTID",
    "Year",
    "ALARM_DATE",
    "CONT_DATE",
    "CAUSE",
    "GIS_ACRES",
    "latitude",
    "longitude",
    "zip",
    "avg_tmax_c",
    "avg_tmin_c",
    "tot_prcp_mm",
    "station"
]

keep_cols = [c for c in candidate_cols if c in df.columns]
df = df[keep_cols].copy()

print("\nafter shape:", df.shape)


# 3. Basic type cleaning
for col in ["ALARM_DATE", "CONT_DATE"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

numeric_cols = [
    "Year", "GIS_ACRES", "latitude", "longitude", "zip",
    "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\ntype conversion done")


# 4. Remove obvious duplicates
if "OBJECTID" in df.columns:
    before = len(df)
    df = df.drop_duplicates(subset=["OBJECTID"])
    print(f"\nOBJECTID duplicate removed: {before} -> {len(df)}")
else:
    before = len(df)
    df = df.drop_duplicates()
    print(f"\nrow duplicate removed: {before} -> {len(df)}")


# 5. Fill missing ZIP using (latitude, longitude)
def fill_zip_by_latlon_mode(group):
    if group["zip"].notna().any():
        zip_mode = group["zip"].dropna().mode()
        if not zip_mode.empty:
            fill_value = zip_mode.iloc[0]
            group["zip"] = group["zip"].fillna(fill_value)
    return group

if {"latitude", "longitude", "zip"}.issubset(df.columns):
    before_missing_zip = df["zip"].isna().sum()

    df = (
        df.groupby(["latitude", "longitude"], dropna=False, group_keys=False)
          .apply(fill_zip_by_latlon_mode)
    )

    after_missing_zip = df["zip"].isna().sum()

    print("ZIP before missing value :", before_missing_zip)
    print("ZIP after missing value :", after_missing_zip)
else:
    print("\nlatitude/longitude/zip columns lacked")


# 6. Basic row filtering
required_cols = [c for c in ["Year", "latitude", "longitude", "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm"] if c in df.columns]
before = len(df)
df = df.dropna(subset=required_cols)
print(f"\n{before} -> {len(df)}")

# Year range restrict
if "Year" in df.columns:
    before = len(df)
    df = df[(df["Year"] >= 2018) & (df["Year"] <= 2023)]
    print(f"Year range filter(2018~2023): {before} -> {len(df)}")

# GIS_ACRES minus removed
if "GIS_ACRES" in df.columns:
    before = len(df)
    df = df[df["GIS_ACRES"].isna() | (df["GIS_ACRES"] >= 0)]
    print(f"GIS_ACRES minus removed: {before} -> {len(df)}")


# 7. Create helper columns
if "ALARM_DATE" in df.columns:
    df["alarm_month"] = df["ALARM_DATE"].dt.month
    df["alarm_dayofyear"] = df["ALARM_DATE"].dt.dayofyear

if {"ALARM_DATE", "CONT_DATE"}.issubset(df.columns):
    df["fire_duration_days"] = (df["CONT_DATE"] - df["ALARM_DATE"]).dt.days

df["fire_event"] = 1


# 8. Final ZIP cleanup
if "zip" in df.columns:
    before = len(df)
    df = df.dropna(subset=["zip"])
    df["zip"] = df["zip"].astype(int)
    print(f"\nfinal missing value ZIP removed: {before} -> {len(df)}")


# 9. Save cleaned event-level data
event_output = "task1_step1_event_cleaned.csv"
df.to_csv(event_output, index=False)

print("\nevent-level saved:", event_output)
print("event-level final shape:", df.shape)


# 10. Build ZIP-YEAR aggregated dataset for later modeling

agg_dict = {}

for col in ["avg_tmax_c", "avg_tmin_c", "tot_prcp_mm", "GIS_ACRES"]:
    if col in df.columns:
        agg_dict[col] = "mean"

if "fire_duration_days" in df.columns:
    agg_dict["fire_duration_days"] = "mean"

if "station" in df.columns:
    agg_dict["station"] = lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan

zip_year_fire = (
    df.groupby(["zip", "Year"], as_index=False)
      .agg(agg_dict)
)

# fire_count added
fire_counts = (
    df.groupby(["zip", "Year"])
      .size()
      .reset_index(name="fire_count")
)

zip_year_fire = zip_year_fire.merge(fire_counts, on=["zip", "Year"], how="left")
zip_year_fire["fire_occurred"] = 1

print("\nfire occured zip-year shape:", zip_year_fire.shape)


# 11. Negative sampling by full ZIP-YEAR grid
all_zips = sorted(df["zip"].dropna().unique())
all_years = sorted(df["Year"].dropna().unique())

full_pairs = pd.MultiIndex.from_product(
    [all_zips, all_years],
    names=["zip", "Year"]
).to_frame(index=False)

zip_year_full = full_pairs.merge(zip_year_fire, on=["zip", "Year"], how="left")

zip_year_full["fire_occurred"] = zip_year_full["fire_occurred"].fillna(0).astype(int)
zip_year_full["fire_count"] = zip_year_full["fire_count"].fillna(0).astype(int)

feature_fill_cols = [c for c in ["avg_tmax_c", "avg_tmin_c", "tot_prcp_mm", "GIS_ACRES", "fire_duration_days"] if c in zip_year_full.columns]

for col in feature_fill_cols:
    zip_year_full[col] = zip_year_full.groupby("zip")[col].transform(lambda x: x.fillna(x.median()))
    zip_year_full[col] = zip_year_full[col].fillna(zip_year_full[col].median())

if "station" in zip_year_full.columns:
    zip_year_full["station"] = zip_year_full.groupby("zip")["station"].transform(
        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else np.nan)
    )
    if zip_year_full["station"].notna().any():
        zip_year_full["station"] = zip_year_full["station"].fillna(zip_year_full["station"].mode().iloc[0])

print("\nnegative sampling -> zip-year dataset shape:", zip_year_full.shape)
print("fire_occurred value counts:")
print(zip_year_full["fire_occurred"].value_counts(dropna=False))


# 12. Save modeling-ready step1 output
zip_year_output = "task1_step1_zip_year_ready.csv"
zip_year_full.to_csv(zip_year_output, index=False)

print("\nzip-year data saved:", zip_year_output)
print("zip-year final shape:", zip_year_full.shape)


# 13. Preview
print("\nevent-level sample")
print(df.head())

print("\nzip-year sample")
print(zip_year_full.head())