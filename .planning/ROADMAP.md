# Roadmap: Insurance Premium Time Series Prediction

**Project:** Insurance Premium Time Series Prediction
**Granularity:** Standard
**Created:** 2026-04-09
**Coverage:** 24/24 v1 requirements mapped

## Executive Summary

Predict 2021 insurance premiums for ~1600 California ZIP codes using 2018-2020 historical panel data. Stack: Python + pandas + statsmodels (Panel FE) + LightGBM. Core pattern: Global Model with Entity Features (pool across ZIPs, not per-ZIP models).

## Phases

- [ ] **Phase 1: Data Foundation** — Load, clean, and validate panel dataset (insurance + fire + census + weather)
- [ ] **Phase 2: Feature Engineering** — Build lag features, rolling stats, encodings, exposure normalization
- [ ] **Phase 3: Baseline Models + Temporal Validation** — Establish metrics with linear regression and random forest baselines
- [ ] **Phase 4: Model Development + Ensemble** — Panel FE + LightGBM ensemble with COVID adjustment
- [ ] **Phase 5: 2021 Prediction + Error Analysis** — Generate predictions, analyze errors, document

## Phase Details

### Phase 1: Data Foundation

**Goal:** Users have a clean, validated panel dataset ready for feature engineering

**Depends on:** Nothing (first phase)

**Requirements:** DATA-01, DATA-02, DATA-03

**Success Criteria** (what must be TRUE):
1. Insurance dataset (2018-2020) loads with correct structure: ZIP x Year x Category granularity (~47k rows)
2. Missing values handled, data types validated, zero critical-field dropouts
3. Temporal split integrity confirmed: no data from 2021 leaks into training features
4. Fire Risk Score, Earned Exposure, and Category columns confirmed present and populated

**Plans:** 1 plan

Plans:
- [ ] 01-01-PLAN.md — Load insurance + fire + census + weather CSV, deduplicate on (Year, ZIP, Category), split 2021 holdout, produce validation report

---

### Phase 2: Feature Engineering

**Goal:** Users have a feature matrix with temporal lags, rolling statistics, and encodings ready for modeling

**Depends on:** Phase 1

**Requirements:** FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05, FEAT-06

**Success Criteria** (what must be TRUE):
1. Temporal lag features created: 2019→2020 and 2018→2019 for fire risk, exposure, losses
2. Rolling statistics per ZIP code: mean and std of fire risk and exposure across available years
3. Category one-hot encoded (HO, CO, DT, RT, DO, MH, NA as columns)
4. Premium normalized by earned exposure (premium/exposure ratio) as target option
5. Zero-inflated claims distribution documented; Tweedie vs two-part decision deferred to Phase 3
6. Avg Fire Risk Score from dataset used as primary wildfire risk predictor

**Plans:** TBD

---

### Phase 3: Baseline Models + Temporal Validation

**Goal:** Users have validated temporal split and baseline metrics to compare against advanced models

**Depends on:** Phase 2

**Requirements:** BASE-01, BASE-02, BASE-03, BASE-04

**Success Criteria** (what must be TRUE):
1. Temporal train/validation split enforced: 2018-2019 train, 2020 validation (no random split)
2. Linear regression baseline with lag features + fire risk score trained and evaluated
3. Random Forest baseline with same features trained and evaluated
4. RMSE, MAE, MAPE reported for both baselines on 2020 validation set
5. 2020 COVID-19 anomaly documented with explicit flag in dataset

**Plans:** TBD

---

### Phase 4: Model Development + Ensemble

**Goal:** Users have an interpretable + accurate ensemble model combining econometric and gradient boosting approaches

**Depends on:** Phase 3

**Requirements:** MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05

**Success Criteria** (what must be TRUE):
1. Panel Fixed Effects model (statsmodels PanelOLS) with ZIP fixed effects trained
2. LightGBM model with lag features and ZIP-level aggregations trained
3. Ensemble prediction created as weighted blend of Panel FE + LightGBM
4. Ensemble RMSE/MAE/MAPE shows improvement over baseline models
5. COVID-19 2020 structural break addressed via downweighting, masking, or indicator variable
6. Feature importance analysis available for both models

**Plans:** TBD

---

### Phase 5: 2021 Prediction + Error Analysis

**Goal:** Users have 2021 premium predictions by ZIP x Category with error analysis and documentation

**Depends on:** Phase 4

**Requirements:** PRED-01, PRED-02, PRED-03, PRED-04, DOC-01, DOC-02, DOC-03

**Success Criteria** (what must be TRUE):
1. 2021 Earned Premium predictions generated for all ZIP codes (or ZIP x Category combinations)
2. Predictions evaluated against holdout if 2021 actuals available in dataset
3. Error analysis identifies high-error ZIP codes and validates against fire risk distribution
4. Value added by fire risk features documented (compare with/without fire risk)
5. Model approach, hyperparameters, and assumptions documented
6. Performance metrics (RMSE, MAE, MAPE) reported for all models
7. Comparison with classical approaches discussed (vs quantum/Task 1)

**Plans:** TBD

---

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Foundation | 0/1 | Not started | - |
| 2. Feature Engineering | 0/1 | Not started | - |
| 3. Baseline Models + Temporal Validation | 0/1 | Not started | - |
| 4. Model Development + Ensemble | 0/1 | Not started | - |
| 5. 2021 Prediction + Error Analysis | 0/1 | Not started | - |

## Coverage Map

```
DATA-01, DATA-02, DATA-03         → Phase 1
FEAT-01, FEAT-02, FEAT-03,        → Phase 2
FEAT-04, FEAT-05, FEAT-06
BASE-01, BASE-02, BASE-03, BASE-04 → Phase 3
MODEL-01, MODEL-02, MODEL-03,     → Phase 4
MODEL-04, MODEL-05
PRED-01, PRED-02, PRED-03, PRED-04 → Phase 5
DOC-01, DOC-02, DOC-03
```

**Coverage:** 24/24 v1 requirements mapped ✓
**No orphaned requirements**
