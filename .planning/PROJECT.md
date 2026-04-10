# Deloitte Hackathon - Task 2: Insurance Premium Prediction

## What This Is

Time series model to predict insurance premiums in 2021 for California zip codes, using historical data from 2018-2020. Integrates wildfire risk (from Task 1 quantum model or existing fire risk score in dataset) to enhance predictions.

## Core Value

Predict 2021 insurance premiums accurately by zip code, using historical trends, risk features, and wildfire risk as a key predictor.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] TS-01: Load and explore insurance dataset (2018-2020)
- [ ] TS-02: Preprocess features (fire risk, exposure, claims, demographics)
- [ ] TS-03: Build time series model (e.g., ARIMA, Prophet, or LSTM) to predict 2021 premiums
- [ ] TS-04: Integrate wildfire risk from Task 1 or use dataset's Avg Fire Risk Score
- [ ] TS-05: Evaluate model performance (RMSE, MAE, MAPE)
- [ ] TS-06: Compare with classical approaches (baseline: linear regression, random forest)
- [ ] TS-07: Generate 2021 predictions by zip code

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

## Constraints

- **Data:** Only 3 years of historical data (2018-2020) — limited for deep time series
- **Features:** Many categorical and numerical features — selection needed
- **Wildfire risk:** Can use dataset's `Avg Fire Risk Score` directly or integrate Task 1 outputs

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Time series approach | Short horizon (1 year), multiple zip codes | Prophet or lightweight model |
| Wildfire risk source | Task 1 model may not be available yet | Use dataset's Avg Fire Risk Score |
| Baseline comparison | Standard evaluation practice | Linear regression + random forest |

---
*Last updated: 2026-04-10 after Phase 5 completion*
