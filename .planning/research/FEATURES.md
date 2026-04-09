# Feature Landscape: Insurance Premium Time Series Forecasting

**Domain:** Insurance Premium Time Series Prediction (California ZIP codes, wildfire context)
**Researched:** 2026-04-09
**Data source:** California insurance-fire-census-weather dataset (47K+ rows, ZIP/Year/Category granularity)

---

## Data Structure Understanding

The dataset operates at **ZIP + Year + Category** granularity (6 policy categories: CO, DO, DT, HO, MH, RT). Years 2018-2021 are present. The dataset blends:
- **Aggregated insurance metrics** (premiums, exposures, losses) at ZIP-year-category level -- these are the main rows
- **Individual fire incident records** (FIRE_NAME, ALARM_DATE, GIS_ACRES, CAUSE) embedded as sparse additional rows -- populated for only a subset of records

The target is **Earned Premium** for 2021. Predictors come from 2018-2020 historical data.

---

## Table Stakes Features

Features users expect. Missing any of these and the model feels incomplete or unsound.

| Feature | Source Column(s) | Complexity | Why Expected | Notes |
|---------|------------------|-------------|--------------|-------|
| **Avg Fire Risk Score** | `Avg Fire Risk Score` | Low | The primary risk input; directly drives underwriting decisions | Range appears to be 0-2+ in data. Highest-priority predictor. |
| **Avg PPC (Public Protection Class)** | `Avg PPC` | Low | ISO fire department effectiveness rating (1=best, 10=worst) | Strong inverse correlation with fire protection capability |
| **Earned Exposure** | `Earned Exposure` | Low | Policy count / risk units; normalizes premium and loss metrics | Required for loss ratio calculation and premium normalization |
| **Policy Category** | `Category_CO/DO/DT/HO/MH/RT` | Low | Policy type determines risk profile and coverage scope | One-hot encoded; HO (homeowner) is typically highest volume |
| **Historical CAT Fire Losses** | `CAT Cov A Fire - Incurred Losses`, `CAT Cov C Fire - Incurred Losses` | Med | Past catastrophe losses are the strongest signal of future catastrophe losses | Lagged (prior-year) versions avoid data leakage |
| **Historical CAT Fire Claim Counts** | `CAT Cov A Fire - Number of Claims`, `CAT Cov C Fire - Number of Claims` | Med | Frequency of past claims predicts future frequency | Lagged; claims count is more stable than loss magnitude |
| **Cov A / Cov C Coverage Amounts** | `Cov A Amount Weighted Avg`, `Cov C Amount Weighted Avg` | Med | Insured dwelling/personal property values set the coverage floor | Weighted average by exposure; reflects property values in ZIP |
| **Non-CAT Fire Losses** | `Non-CAT Cov A Fire - Incurred Losses` etc. | Med | Routine fire risk independent of catastrophe events | Captures baseline fire exposure separate from wildfire events |
| **Non-CAT Smoke Losses** | `Non-CAT Cov A Smoke - Incurred Losses` etc. | Med | Smoke damage is a leading indicator of fire adjacency risk | Non-catastrophe smoke claims reflect local fire exposure |
| **Number of Fire Risk Exposure Units** | `Number of High/Low/Moderate/Very High/Negligible Fire Risk Exposure` | Med | Granular breakdown of risk distribution within a ZIP | Sum of these = total exposure; ratios capture risk mix |
| **Year** | `Year` | Low | Time series component; captures trend/inflation in premiums | Use as integer or derive temporal features (trend, COVID effect) |
| **ZIP Code** | `ZIP` | Low | Geographic unit of analysis; captures local market effects | Encode carefully -- raw ZIP is too granular; consider clustering |

---

## Differentiators

Features that set the model apart from a basic regression. Not expected, but valued when present.

| Feature | Source Column(s) | Complexity | Value Proposition | Notes |
|---------|------------------|-------------|-------------------|-------|
| **Lagged Loss Ratios (prior-year)** | Derived: CAT Loss[t-1] / Earned Premium[t-1] | Med | Actuarial best-practice: loss ratio is the core pricing lever | Avoids data leakage by using prior-year ratios only |
| **Rolling 2-3 Year Average of Fire Risk Score** | Derived: avg(Fire Risk Score)[t-1 to t-N] | Low | Smooths year-over-year volatility in fire risk assessment | Captures persistent risk elevation, not just one bad year |
| **Rolling 3-Year Average of CAT Losses** | Derived: avg(CAT Losses)[t-1 to t-3] | Med | Multi-year loss memory; a single bad year should not dominate | Particularly valuable for high-variance wildfire years |
| **Weather: Temperature Extremes** | `avg_tmax_c`, `avg_tmin_c` | Low | Heat and cold extremes correlate with fire ignition and spread |avg_tmax_c likely has strongest correlation with wildfire activity |
| **Weather: Precipitation** | `tot_prcp_mm` | Low | Drought conditions (low precipitation) elevate fire risk | Lagged precipitation (e.g., prior season) may be more predictive |
| **Year-over-Year Change in Fire Risk Score** | Derived: Fire Risk[t-1] - Fire Risk[t-2] | Low | Captures risk trend direction; rapidly increasing risk = premium shock | Trend features help with non-stationarity in time series |
| **Fire Risk Score x Earned Exposure Interaction** | `Avg Fire Risk Score` x `Earned Exposure` | Low | High-risk areas with high exposure = concentrated risk | Exposure-weighted risk is more actuarially sound |
| **Socioeconomic: Median Income** | `median_income` | Med | Wealthy areas may have higher property values and different risk profiles | Also correlates with ability to absorb losses / evacuate |
| **Socioeconomic: Housing Value** | `housing_value` | Med | Replacement cost proxy; higher-value homes = higher premiums | Correlates with Cov A weighted average but at census level |
| **Housing Age / Year Structure Built** | `year_structure_built` | Med | Older housing stock may have outdated electrical systems, less fire-resistant construction | Could be normalized by area or expressed as median year |
| **Occupancy Rate / Vacancy** | `housing_occupancy_number`, `housing_vacancy_number` | Med | Vacant properties may have higher fire risk (unmonitored) | Vacancy rate = housing_vacancy_number / total_housing_units |
| **Education Level** | `educational_attainment_bachelor_or_higher` | Low | Socioeconomic risk proxy; often correlates with loss mitigation behavior | Low complexity to include |
| **Geographic Coordinates** | `latitude`, `longitude` | Low | Spatial features capture location-specific risk (elevation, wind patterns, vegetation) | Key for identifying fire-prone corridors in California |
| **Spatial Clustering of Fire Risk** | Derived: aggregate neighboring ZIP fire risk scores | High | Adjacent ZIP codes share fire risk through terrain and weather patterns | Requires geospatial join; value depends on clustering method |
| **Premium per Exposure (prior-year)** | Derived: Earned Premium[t-1] / Earned Exposure[t-1] | Low | Pure rate indication independent of exposure volume | Clean actuarial rate-level feature, no target leakage |
| **Wildfire Season Indicator** | Derived from `year_month` or alarm dates | Med | California wildfire season (Oct-Nov) has elevated risk | Only applicable if fire incident data is joinable per ZIP |

---

## Anti-Features

Features to explicitly NOT build or include. These are common mistakes in this domain.

| Anti-Feature | Why Avoid | What to Do Instead |
|-------------|-----------|-------------------|
| **Current-year earned premium as a predictor** | This IS the target variable -- direct data leakage. The model will train to predict itself. | Use lagged premium (t-1) or derived rate metrics (premium/exposure). |
| **Current-year losses or claims as predictors** | Losses incurred in 2021 are not available at prediction time (2018-2020 training). | Use only prior-year (t-1) or rolling historical losses. |
| **Raw fire incident details (FIRE_NAME, GIS_ACRES, CAUSE) as primary features** | These are sparse -- populated only for a small subset of rows (individual fire events). Most rows have NA for these. Using them directly would throw away 90%+ of training data. | Fire Risk Score already summarizes this information at ZIP level. Use that instead. |
| **Unmodified ZIP code as a categorical feature in tree models** | Thousands of unique ZIPs in California. Direct encoding causes severe overfitting and memory issues. | Use ZIP clustering (e.g., group by fire risk tier, region, or use target encoding with regularization). |
| **Including both raw loss AND loss ratio for the same period** | Collinear. If you include CAT Fire Loss[t-1] AND CAT Fire Loss[t-1]/Earned Premium[t-1], the model has redundant information and unstable coefficients. | Choose one representation per metric. |
| **Year as a simple integer (2018, 2019, 2020) for linear models** | Linear models will impose a false linearity assumption (premium difference between 2018-2019 = 2019-2020). | Use one-hot year dummies or trend indicator. |
| **Aggregating away the Category dimension** | HO (homeowner), CO (commercial), RT (renter) have fundamentally different premium structures. Averaging across categories destroys signal. | Keep Category as a feature or train separate models per category. |
| **Ignoring zero-inflation in loss columns** | CAT loss columns are mostly zeros (most ZIP-year-category combos have no catastrophe). Naive regression will model the zeros and non-zeros with the same process. | Consider two-part models (logistic for P(loss>0) + gamma/Poisson for loss magnitude) or zero-inflated distributions. |
| **Treating fire risk score as exogenous (ignoring its own time series structure)** | Fire risk scores vary year-to-year. A static assumption ignores that the predictor itself is a forecast. | If fire risk for prediction year is not available, use rolling averages or trend extrapolation. |

---

## Feature Dependencies

```
Avg Fire Risk Score ──────┬──> Earned Premium (direct risk input)
                           │
Avg PPC ──────────────────┤
                           │
Historical CAT Losses ─────┤
Historical Non-CAT Losses ─┤
                           ├──> Loss Ratio (t-1) ──> Premium (t)
Earned Exposure ──────────┤
                           │
Cov A/C Amount Wtd Avg ────┘

Year ───────────────────> Trend / temporal features

Policy Category (CO/DO/DT/HO/MH/RT) ──> Category-specific premium floors

Weather (tmax/tmin/prcp) ──> Fire Risk Score ──> Premium

Demographics ──────────────> Cov A/C Amount Weighted Avg (surrogate)
                            ──> Risk behavior proxies
```

---

## MVP Recommendation

Prioritize in this order:

1. **Avg Fire Risk Score + Avg PPC + Earned Exposure** (Tier 1 table stakes)
   - These three alone explain the majority of premium variance.
   - Fire Risk Score is the single most important differentiator in wildfire context.

2. **Policy Category one-hot + Year** (Tier 1 table stakes)
   - Required to avoid aggregating across fundamentally different policy types.

3. **Lagged loss ratios and rolling loss averages** (Tier 2 table stakes)
   - Loss history is the actuarial foundation. Use prior-year loss ratio (loss/premium) not raw loss.

4. **Weather features (avg_tmax_c, avg_tmin_c, tot_prcp_mm)** (differentiator)
   - These capture the environmental driver that fire risk score is a proxy for. Together they improve model robustness.

5. **Demographics: median_income, housing_value, year_structure_built** (differentiator)
   - Census features are low-cost to include and capture socioeconomic risk variation.

**Defer:**
- Spatial clustering of neighboring ZIP risk (High complexity, requires geodata pipeline)
- Wildfire season indicators (requires incident-level join; fire risk score already captures this)
- Two-part zero-inflated loss models (valuable but complex; validate need from residual analysis first)

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Table stakes features | HIGH | All directly present in dataset; well-understood actuarial practice |
| Differentiators | MEDIUM | Based on actuarial literature and feature engineering principles; benefit magnitude not empirically validated here |
| Anti-features | HIGH | Data structure inspection confirms sparsity of fire incident columns; leakage risks are structural |
| Feature dependencies | MEDIUM | Actuarial domain knowledge applied to data structure; some interactions not empirically tested |

---

## Sources

- Dataset inspection: `abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv` (47,136 rows, ZIP/Year/Category granularity, 2018-2021)
- Feature descriptions: `abfaw7bci2UF6CTg_FeatureDescription_fire_insurance.csv`
- Actuarial pricing principles: ISO (Insurance Services Office) Public Protection Classification documentation
- Loss ratio as primary pricing lever: standard actuarial practice (verified through training data domain knowledge)
