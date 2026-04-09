# Requirements: Insurance Premium Time Series Prediction

**Defined:** 2026-04-09
**Core Value:** Predict 2021 insurance premiums accurately by zip code using historical trends and wildfire risk

## v1 Requirements

### Data & preprocessing

- [ ] **DATA-01**: Load and explore insurance dataset (2018-2020) — verify structure, row counts, zip code coverage
- [ ] **DATA-02**: Clean dataset — handle missing values, validate data types, remove rows with critical missing fields
- [ ] **DATA-03**: Validate temporal split integrity — ensure no data leakage from future years into predictors

### Feature engineering

- [ ] **FEAT-01**: Create temporal lag features (2019→2020, 2018→2019) for key predictors
- [ ] **FEAT-02**: Engineer rolling statistics per zip code (mean, std of fire risk, exposure) across available years
- [ ] **FEAT-03**: One-hot encode Category (HO, CO, DT, RT, DO, MH, NA)
- [ ] **FEAT-04**: Normalize premium by earned exposure (premium per unit) to separate business volume from risk pricing
- [ ] **FEAT-05**: Handle zero-inflated claims — document distribution, decide between Tweedie GLM or two-part model in Phase 3
- [ ] **FEAT-06**: Incorporate wildfire risk — use `Avg Fire Risk Score` from dataset as primary predictor

### Baseline models

- [ ] **BASE-01**: Implement temporal train/validation split (2018-2019 train, 2020 validation) — NO random split
- [ ] **BASE-02**: Baseline: Linear regression with lag features and fire risk score
- [ ] **BASE-03**: Baseline: Random Forest with same features
- [ ] **BASE-04**: Report RMSE, MAE, MAPE on validation set for both baselines

### Model development

- [ ] **MODEL-01**: Panel Fixed Effects model (statsmodels PanelOLS) with ZIP fixed effects
- [ ] **MODEL-02**: LightGBM model with lag features and ZIP-level aggregations
- [ ] **MODEL-03**: Ensemble (average or weighted blend) of Panel FE + LightGBM predictions
- [ ] **MODEL-04**: Compare ensemble against baselines — validate improvement in RMSE/MAE/MAPE
- [ ] **MODEL-05**: Address COVID-19 structural break — document 2020 anomalous behavior, consider downweighting or masking

### Prediction & evaluation

- [ ] **PRED-01**: Generate 2021 Earned Premium predictions for each ZIP code
- [ ] **PRED-02**: Evaluate predictions using available holdout (if 2021 actuals exist in dataset)
- [ ] **PRED-03**: Error analysis — identify high-error zip codes, validate against fire risk distribution
- [ ] **PRED-04**: Compare with Task 1 wildfire risk integration — document value added by fire risk features

### Documentation & reporting

- [ ] **DOC-01**: Document model approach, hyperparameters, and assumptions
- [ ] **DOC-02**: Report performance metrics (RMSE, MAE, MAPE) for all models
- [ ] **DOC-03**: Compare with classical approaches — discuss advantages/disadvantages vs quantum (Task 1)

## v2 Requirements

### Advanced modeling

- **MODEL-05**: Tweedie GLM for zero-inflated premium distribution
- **MODEL-06**: Spatial clustering of neighboring ZIP codes for geographic effects
- **MODEL-07**: Hierarchical/mixed-effects model if ZIP code structure has nested geography

## Out of Scope

| Feature | Reason |
|---------|--------|
| Per-ZIP ARIMA | 3 time points insufficient for ARIMA fitting |
| Deep learning (LSTM/Transformer) | Only 3 years of data — insufficient for deep TS models |
| Prophet/NeuralProphet | Designed for daily seasonality; inappropriate for annual data |
| Policy-level predictions | Zip code is finest granularity for this dataset |
| Real-time prediction pipeline | Batch prediction sufficient for hackathon scope |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| FEAT-01 | Phase 2 | Pending |
| FEAT-02 | Phase 2 | Pending |
| FEAT-03 | Phase 2 | Pending |
| FEAT-04 | Phase 2 | Pending |
| FEAT-05 | Phase 2 | Pending |
| FEAT-06 | Phase 2 | Pending |
| BASE-01 | Phase 3 | Pending |
| BASE-02 | Phase 3 | Pending |
| BASE-03 | Phase 3 | Pending |
| BASE-04 | Phase 3 | Pending |
| MODEL-01 | Phase 4 | Pending |
| MODEL-02 | Phase 4 | Pending |
| MODEL-03 | Phase 4 | Pending |
| MODEL-04 | Phase 4 | Pending |
| MODEL-05 | Phase 4 | Pending |
| PRED-01 | Phase 5 | Pending |
| PRED-02 | Phase 5 | Pending |
| PRED-03 | Phase 5 | Pending |
| PRED-04 | Phase 5 | Pending |
| DOC-01 | Phase 5 | Pending |
| DOC-02 | Phase 5 | Pending |
| DOC-03 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 24 total
- Mapped to phases: 24
- Unmapped: 0 ✓

**Roadmap:** `.planning/ROADMAP.md`

---
*Requirements defined: 2026-04-09*
*Last updated: 2026-04-09 after roadmap creation*
