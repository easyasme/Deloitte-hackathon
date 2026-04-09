# Project Research Summary

**Project:** Insurance Premium Time Series Prediction
**Domain:** Panel Time Series Forecasting (California ZIP codes x Years x Categories)
**Researched:** 2026-04-09
**Confidence:** MEDIUM

## Executive Summary

This is a panel time series forecasting problem: predict 2021 insurance premiums for ~1600 California ZIP codes using 2018-2020 historical data (3-year panel). The dataset operates at ZIP + Year + Category granularity (6 policy types: CO, DO, DT, HO, MH, RT). Experts in this domain use an ensemble of econometric panel regression (statsmodels PanelOLS Fixed Effects) for causal interpretability and gradient boosting (LightGBM with lag features) for predictive accuracy. Deep learning approaches are explicitly contraindicated -- only 3 observations per ZIP cannot support LSTM/Transformer architectures.

The recommended stack is Python 3.11+ with pandas, statsmodels, LightGBM, and scikit-learn. The core architectural decision is to use a global model (not per-ZIP models) that pools information across all ZIP codes. Key risks include feature leakage through current-year data, ignoring exposure as a volume normalizer, zero-inflated claims distributions, COVID-19 structural break in 2020 data, and overfitting to ZIP-level granularity. The MVP should prioritize Fire Risk Score + PPC + Exposure + Category + Year features with lagged loss ratios as the primary differentiator.

## Key Findings

### Recommended Stack

**Core technologies:**
- **Python 3.11+**: Runtime environment
- **pandas 2.2+**: Panel data manipulation, groupby operations, lag feature engineering
- **statsmodels 0.14+**: Panel Fixed Effects regression for interpretable coefficients and causal inference (required for regulatory filings)
- **LightGBM 4.5+**: Gradient boosting for tabular data; native categorical handling for ~1600 ZIP codes without one-hot explosion
- **scikit-learn 1.5+**: Preprocessing, cross-validation, evaluation metrics

**Approach:** Ensemble of Panel FE (statsmodels) + LightGBM. The econometric model provides interpretability; LightGBM captures non-linear relationships. Do NOT use Prophet, XGBoost (prefer LightGBM), or deep learning (insufficient data).

### Expected Features

**Must have (table stakes):**
- Avg Fire Risk Score -- primary wildfire risk input, directly drives underwriting
- Avg PPC (Public Protection Class) -- ISO fire department effectiveness rating
- Earned Exposure -- policy volume normalizer; required for loss ratio calculation
- Policy Category (one-hot CO/DO/DT/HO/MH/RT) -- fundamentally different pricing mechanics per type
- Historical CAT Fire Losses and Claim Counts -- strongest signal for future catastrophe losses (lagged by 1+ year)
- Cov A/C Coverage Amount Weighted Averages -- insured property values
- Year -- temporal trend capture; use one-hot dummies for linear models

**Should have (differentiators):**
- Lagged Loss Ratios (prior-year CAT losses / Earned Premium) -- actuarial best-practice pricing lever
- Rolling 2-3 Year Average of Fire Risk Score -- smooths volatility, captures persistent risk elevation
- Weather: avg_tmax_c, avg_tmin_c, tot_prcp_mm -- environmental drivers behind fire risk score
- Demographics: median_income, housing_value, year_structure_built -- socioeconomic risk variation
- Fire Risk Score x Earned Exposure interaction -- concentrated risk in high-exposure high-risk areas

**Defer (v2+):**
- Spatial clustering of neighboring ZIP fire risk -- high complexity, requires geodata pipeline
- Wildfire season indicators -- fire risk score already captures this
- Two-part zero-inflated loss models -- validate need from residual analysis first

### Architecture Approach

The system uses a layered architecture: Data Ingestion -> Feature Engineering -> Modeling -> Prediction. The recommended pattern is a **Global Model with Entity Features** -- train a single model on all ZIP codes rather than per-ZIP models. With only 3 time points per ZIP, pooling across ~1600 ZIPs is essential for statistical power.

**Major components:**
1. **Data Layer** (data/): CSV loading, joining on ZIP x Year, schema validation
2. **Feature Engineering Layer** (features/): Time-series lag features, cross-sectional ZIP aggregates, categorical encoding
3. **Modeling Layer** (models/): Panel models (statsmodels), tree models (LightGBM), ensemble stacking
4. **Validation Layer** (validation/): Time-series-aware splits (train 2018-2019, validate 2020)
5. **Prediction Layer** (prediction/): 2021 forecast generation, error analysis, output export

### Critical Pitfalls

1. **Deep learning on 3-year panel** -- LSTM/Transformer models require hundreds of observations; with 3 time points they fail to converge or memorize noise. Use Panel FE or LightGBM with lags instead.
2. **Ignoring exposure as volume normalizer** -- Premium = Exposure x Rate. A raw premium model tracks policy count changes, not pricing dynamics. Always include Earned Exposure as feature or model premium/exposure ratio.
3. **Feature leakage through future information** -- Using current-year losses or earned premium as predictors is direct target leakage. All concurrent features must be lagged by at least 1 year.
4. **Overfitting to ZIP granularity** -- Per-ZIP models or dummy variables for 1600 ZIPs causes memorization. Use global models with LightGBM native categorical support or target encoding with regularization.
5. **COVID-19 structural break in 2020** -- 2020 was an outlier: exceptional wildfire activity, economic disruption. Model will learn a non-repeating signal. Flag 2020 explicitly; run sensitivity analysis with and without it.

## Implications for Roadmap

### Suggested Phase Structure

**Phase 1: Data Foundation**
**Rationale:** Must establish clean, joined dataset before any feature work. All downstream depends on correct data loading and schema validation.
**Delivers:** Clean panel dataset (insurance + fire + census + weather joined on ZIP x Year)
**Addresses:** Table stakes features availability check; identifies any missing columns
**Avoids:** Data quality issues propagating to model (from PITFALLS analysis)

**Phase 2: Exploratory Analysis + Feature Engineering**
**Rationale:** EDA informs feature engineering decisions; feature pipeline must be built before modeling can begin.
**Delivers:** EDA report; feature matrix with lags, rolling stats, encodings
**Addresses:** Fire Risk Score, PPC, Exposure, Category, Year, lagged losses, demographics, weather
**Avoids:** Zero-inflation issues (identified in EDA), COVID outlier flagging

**Phase 3: Baseline Models + Temporal Validation**
**Rationale:** Establishes baseline metrics before advanced modeling; temporal split (train 2018-2019, validate 2020) is critical to avoid leakage.
**Delivers:** Baseline Panel FE model; baseline LightGBM; validation metrics
**Addresses:** Must-have features with simple models
**Avoids:** Per-ZIP ARIMA (anti-pattern), random train/test split (leakage), in-sample-only metrics

**Phase 4: Model Development + Ensemble**
**Rationale:** Core modeling phase where competitive advantage is built. Ensemble of Panel FE + LightGBM provides interpretability + accuracy.
**Delivers:** Tuned ensemble model; feature importance analysis
**Uses:** statsmodels PanelOLS, LightGBM, sklearn
**Implements:** Global Model with Entity Features pattern; hierarchical shrinkage if needed
**Avoids:** Deep learning (critical pitfall), overfitting to ZIP granularity

**Phase 5: 2021 Prediction + Error Analysis**
**Rationale:** Final deliverables. Error analysis identifies underperforming segments for potential iteration.
**Delivers:** 2021 premium predictions by ZIP x Category; error report by segment
**Addresses:** Business requirement: premium predictions for 2021
**Avoids:** Feature leakage in prediction features (only use 2020-end-of-year available data)

### Phase Ordering Rationale

- **Phases 1-2 must come first** because all modeling depends on clean data and proper features
- **Phase 3 establishes validation protocol** -- temporal splits prevent the critical in-sample-metrics pitfall
- **Phase 4 builds the actual models** -- ensemble approach informed by stack and architecture research
- **Phase 5 delivers output** -- error analysis informs whether Phase 4 needs iteration
- **Why not parallelize:** Data foundation must be complete before feature engineering; features must be complete before modeling

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (Feature Engineering):** Zero-inflated claims handling -- may need `/gsd-research-phase` for Tweedie GLM or two-part model implementation if residual analysis shows systematic underprediction at zero
- **Phase 4 (Model Development):** Ensemble weighting strategy -- needs empirical validation to determine optimal Panel FE vs LightGBM blend

Phases with standard patterns (skip research-phase):
- **Phase 1 (Data Foundation):** Standard pandas CSV loading; well-documented
- **Phase 3 (Baseline Models):** TimeSeriesSplit from scikit-learn; established pattern
- **Phase 5 (Prediction):** Standard model.predict() pipeline; no novel components

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Based on training data + established patterns; limited Context7 verification due to tool unavailability |
| Features | HIGH | Directly from dataset inspection; table stakes confirmed by domain knowledge |
| Architecture | MEDIUM | Standard ML architecture patterns; limited search access during research |
| Pitfalls | MEDIUM | Domain knowledge + actuarial principles; web search encountered API errors |

**Overall confidence:** MEDIUM

### Gaps to Address

- **Exact library versions:** Not verified via Context7 (tool unavailable). Recommendation: run `pip index versions <package>` before installation to confirm latest stable versions.
- **COVID-19 flagging approach:** 2020 anomaly is identified but specific mitigation strategy (down-weighting, exclusion, indicator variable) needs empirical validation during Phase 3.
- **Zero-inflated loss models:** Anti-pattern clearly documented but implementation approach (Tweedie GLM vs two-part model) not decided -- depends on residual analysis from Phase 3.
- **Ensemble weighting:** Optimal blend of Panel FE vs LightGBM depends on validation metrics; no single recommendation from research alone.

## Sources

### Primary (HIGH confidence)
- Dataset: `abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv` (47,136 rows, ZIP/Year/Category granularity, 2018-2021)
- Feature descriptions: `abfaw7bci2UF6CTg_FeatureDescription_fire_insurance.csv`
- statsmodels panel documentation: https://www.statsmodels.org/stable/panel.html
- LightGBM documentation: https://lightgbm.readthedocs.io/

### Secondary (MEDIUM confidence)
- sklearn TimeSeriesSplit documentation: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
- Actuarial pricing principles: ISO Public Protection Classification documentation
- Kaggle tabular competition trends (2022-2025): LightGBM/XGBoost dominant for structured tabular data

### Tertiary (LOW confidence)
- Pitfalls findings based on domain knowledge only; web search errors prevented corroboration -- should be validated against actuarial literature during Phase 2 EDA

---
*Research completed: 2026-04-09*
*Ready for roadmap: yes*
