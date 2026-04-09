# External Integrations

**Analysis Date:** 2026-04-09

## APIs & External Services

**No External APIs Detected:**
- This codebase consists of standalone Python scripts
- No REST API calls, SDK imports, or external service clients found

## Data Storage

**Local CSV Files (Input):**
- `Task1_Data/abfap7bci2UF6CTY_wildfire_weather.csv` - Wildfire and weather combined dataset
- `Task1_Data/abfakLbci2UF6CTU_Feature_Descsription_FireHistory_Census.csv` - Fire history and census feature descriptions
- `Task2_Data/abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv` - California insurance data with fire/census/weather

**Local CSV Files (Generated/Output):**
- `task1_step1_event_cleaned.csv` - Cleaned event-level data
- `task1_step1_zip_year_ready.csv` - ZIP-year aggregated modeling dataset
- `task1_step2_classical_ready.csv` - Feature-engineered dataset for classical ML
- `task1_step2_v2_classical_ready.csv` - V2 feature-engineered dataset with fire context
- `task1_step3_model_comparison.csv` - Classical ML model evaluation metrics
- `task1_step3_rf_feature_importance.csv` - Random Forest feature importance
- `task1_step3_test_predictions.csv` - Model predictions on test set
- `task1_step4_quantum_train.csv` - Training data prepared for quantum ML
- `task1_step4_quantum_test.csv` - Test data prepared for quantum ML
- `task1_step4_quantum_config.json` - Quantum ML configuration
- `task1_step5_vqc_metrics.csv` - VQC model evaluation metrics
- `task1_step5_vqc_predictions.csv` - VQC predictions on test set
- `task1_step5_vqc_training_log.csv` - VQC training iteration log

**File Storage:**
- Local filesystem only - All data stored in and read from the repository directories

**Database:**
- None - No database connections detected

## Authentication & Identity

**Auth Provider:**
- None - Not applicable for this batch processing pipeline

## Monitoring & Observability

**Error Tracking:**
- None - No error tracking service integrated

**Logs:**
- Print statements to stdout - Scripts use Python print() for progress and results
- No structured logging framework detected

## CI/CD & Deployment

**Hosting:**
- None - Not a deployed application

**CI Pipeline:**
- None - No CI/CD configuration detected

## Environment Configuration

**Required env vars:**
- None explicitly defined - Scripts rely on relative file paths and working directory

**Secrets location:**
- None - No secrets management detected

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

---

*Integration audit: 2026-04-09*
