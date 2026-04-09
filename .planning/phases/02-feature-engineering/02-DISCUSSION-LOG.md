# Phase 2: Feature Engineering - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-09
**Phase:** 02-feature-engineering
**Areas discussed:** Target variable, Zero-exposure handling, Lag strategy, Rolling statistics, Category encoding, COVID-19 handling

---

## Target Variable

| Option | Description | Selected |
|--------|-------------|----------|
| Raw Earned Premium (Recommended) | Pure premium amount — simpler, more interpretable | ✓ |
| Premium ÷ Exposure ratio | Premium per unit exposure — normalizes for business volume | |
| Log-transformed premium | Log(1 + premium) — reduces skew for heteroscedastic data | |

**User's choice:** Raw Earned Premium (Recommended)
**Notes:** Skew from high-value policies is acceptable risk — Phase 3 models (LightGBM, Panel FE) handle skewed targets reasonably well

---

## Zero-Exposure Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Drop zero-exposure rows | Drop ~2,940 zero-exposure rows before modeling — simplifies but loses data | |
| Set ratio to 0 (Recommended) | Set ratio to 0 when exposure=0 — keeps rows, creates logical target | ✓ |
| Use epsilon | Add small epsilon to denominator — avoids division by zero but may distort low-exposure rows | |

**User's choice:** Set ratio to 0 (Recommended) — but noted that target is raw premium, not ratio

---

## Lag Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Single lag (t-1 only) (Recommended) | For target year Y, features come only from year Y-1. Simple, minimal leakage risk. | ✓ |
| Available lags (max available) | Use all available prior years — more data when available | |
| Fixed 2-year rolling window | Fixed 2-year window: for target year Y, use years Y-1 and Y-2 | |

**User's choice:** Single lag (t-1 only) (Recommended)
**Notes:** 2020 targets use 2019 features; 2019 targets use 2018 features; 2018 rows serve as lag-source only

---

## Rolling Statistics

| Option | Description | Selected |
|--------|-------------|----------|
| Expanding window (all history) (Recommended) | Expanding mean/std across all available years per ZIP — uses maximum data | ✓ |
| Fixed 2-year rolling | More responsive to recent changes but less data | |
| Skip rolling (lag-only) | Simpler pipeline — no rolling features | |

**User's choice:** Expanding window (all history) (Recommended)
**Notes:** Expanding mean/std of Avg Fire Risk Score and Earned Exposure per ZIP; expanding std of fire risk for volatility

---

## Category Encoding

| Option | Description | Selected |
|--------|-------------|----------|
| One-hot encode (6 columns) | One column per category. Works well with tree models. Binary (0/1). | ✓ |
| Ordinal encode (single column) | Single integer column. Tree models handle natively; simpler pipeline. | |
| Entity embeddings | Let the model learn embeddings — may help but adds complexity | |

**User's choice:** One-hot encode (6 columns)
**Notes:** 6 binary columns (Category_HO, Category_CO, Category_DT, Category_RT, Category_DO, Category_MH)

---

## COVID-19 Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Add COVID year indicator | Add binary flag: Year == 2020. Simple, interpretable. | ✓ |
| Downweight 2020 observations | Downweight 2020 samples (e.g., 0.5 weight). Reduces 2020 influence on fit. | |
| No special handling | Let the model fit 2020 naturally. May distort predictions. | |

**User's choice:** Add COVID year indicator
**Notes:** `is_covid_year` = 1 for Year == 2020, 0 otherwise; models can learn differential 2020 behavior

---

## Claude's Discretion

- **2018 lag handling**: 2018 rows have no lagged features (no 2017 data) — these rows serve as lag source for 2019 only, not as modeling targets with features
- **Zero-exposure row strategy**: Since target is raw premium (not ratio), zero-exposure rows are simply kept in dataset with no special handling at feature engineering stage

---

## Deferred Ideas

None — discussion stayed within Phase 2 scope.

