# Deloitte Hackathon - Task 2: Insurance Premium Prediction

## What This Is

Time series model to predict insurance premiums in 2021 for California zip codes, using historical data from 2018-2020. Integrates wildfire risk (from Task 1 quantum model or existing fire risk score in dataset) to enhance predictions.

## Core Value

Predict 2021 insurance premiums accurately by zip code, using historical trends, risk features, and wildfire risk as a key predictor.

## Current State (v1.0 Shipped)

**Last updated:** 2026-04-10 after v1.0 milestone completion

### What Was Built
- Clean panel dataset: 31,343 training rows (2018-2020) + 12,708 holdout (2021)
- 67-column feature matrix with temporal lags, expanding stats, category encoding
- Ensemble model: 80% Panel FE (Pooled OLS) + 20% LightGBM
- 2021 predictions: RMSE 439k, MAE 180k on holdout

### Key Findings
- Fire risk does NOT improve 2021 premium prediction (ablation: RMSE delta = +1,460 with fire risk)
- 90% of top-20 highest-error ZIPs have above-median fire risk scores
- Panel FE (Pooled OLS) dominates ensemble weight (80%) — structural approach captures most signal
- Zero-inflated premium distribution (12.8% zeros) — Tweedie GLM deferred to v2

### Technical Debt
- 40.7% of predictions are negative (invalid for insurance premium — needs clipping)
- LightGBM predictions not persisted to disk (regenerated inline)
- Phases 1-3 VERIFICATION.md missing (pre-GSD enforcement)

## Requirements

### Validated (v1.0)

- ✅ TS-01: Load and explore insurance dataset (2018-2020) — v1.0
- ✅ TS-02: Preprocess features (fire risk, exposure, claims, demographics) — v1.0
- ✅ TS-03: Build time series model — v1.0 (Panel FE + LightGBM ensemble)
- ✅ TS-04: Integrate wildfire risk — v1.0 (Avg Fire Risk Score as input feature)
- ✅ TS-05: Evaluate model performance (RMSE, MAE, MAPE) — v1.0
- ✅ TS-06: Compare with classical approaches — v1.0 (vs linear regression, random forest)
- ✅ TS-07: Generate 2021 predictions by zip code — v1.0

### Active (v1.1)

Requirements for v1.1 to be defined via `/gsd-new-milestone`.

### Out of Scope

- Quantum/hybrid approaches (Task 1 scope)
- Task 1 wildfire model development (already completed)
- Policy-level predictions (zip code is finest granularity)

## Context

- **Existing work:** Task 1A (wildfire quantum prediction) and Task 1B (evaluation) completed
- **Data:** Task2_Data contains insurance data with fire risk scores per zip/year
- **Target:** Earned Premium for 2021 prediction
- **Key predictor:** Avg Fire Risk Score (from Task 1 quantum model OR dataset's existing score)
- **Training years:** 2018, 2019, 2020
- **Prediction year:** 2021
- **Stack:** Python, pandas, linearmodels (PanelOLS), LightGBM, scikit-learn

## Constraints

- **Data:** Only 3 years of historical data (2018-2020) — limited for deep time series
- **Features:** Many categorical and numerical features — selection needed
- **Wildfire risk:** Can use dataset's `Avg Fire Risk Score` directly or integrate Task 1 outputs

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Time series approach | Short horizon (1 year), multiple zip codes | Panel FE + LightGBM ensemble |
| Wildfire risk source | Task 1 model may not be available yet | Use dataset's Avg Fire Risk Score |
| Baseline comparison | Standard evaluation practice | Linear regression + random forest |
| Validation strategy | Temporal integrity critical | 2019 train / 2020 validation / 2021 holdout |
| COVID-19 handling | 2020 structural break | 0.5 sample weighting at evaluation |

---
*Last updated: 2026-04-10 after v1.0 milestone completion*
