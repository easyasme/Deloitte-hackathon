# Codebase Structure

**Analysis Date:** 2026-04-09

## Directory Layout

```
/home/dwk/code/Deloitte-hackathon/
├── Task 1B Report.docx              # Documentation
├── Task1_Data/                       # Task 1 datasets and scripts
│   ├── abfap7bci2UF6CTY_wildfire_weather.csv  # Raw input data
│   ├── Task1_Weather/               # Weather-only pipeline
│   └── Task1_Weather_Fire_Context/  # Weather + fire context pipeline
├── Task2_Data/                       # Task 2 datasets
│   ├── abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv
│   └── abfaw7bci2UF6CTg_FeatureDescription_fire_insurance.csv
└── .planning/codebase/               # Generated documentation
```

## Directory Purposes

**Task1_Data/Task1_Weather:**
- Purpose: Weather-only feature pipeline for wildfire prediction
- Contains: 5 Python scripts, 10+ intermediate CSV files
- Key files: `data_preprocessing.py` (step 1), `classical_ml_feature_engineering.py` (step 2), `classical_ml.py` (step 3), `quantum_ml_data_processing.py` (step 4), `quantum_ml.py` (step 5)

**Task1_Data/Task1_Weather_Fire_Context:**
- Purpose: Extended pipeline with fire context features (CAUSE, OBJECTIVE, C_METHOD, AGENCY_ID)
- Contains: 3 Python scripts, 6+ intermediate CSV files
- Key files: `data_processing_v2.py` (step 1 v2), `classical_ml_feature_engineering.py` (step 2 v2), `classical_ml.py` (step 3 v2)

**Task2_Data:**
- Purpose: Insurance and census data for Task 2
- Contains: 2 CSV files (insurance/fire/census data, feature descriptions)
- Not yet processed by pipeline scripts

**.planning/codebase:**
- Purpose: Generated architecture and structure documentation
- Contains: ARCHITECTURE.md, STRUCTURE.md, and other focus area documents

## Key File Locations

**Entry Points:**
- No formal entry points - scripts run as standalone Python files
- Pipeline executed by running scripts in sequence within each branch

**Raw Data:**
- `Task1_Data/abfap7bci2UF6CTY_wildfire_weather.csv` - Raw wildfire + weather data
- `Task2_Data/abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv` - Insurance/census data

**Pipeline Scripts (Task1_Weather):**
- `Task1_Data/Task1_Weather/data_preprocessing.py` - Step 1: data cleaning
- `Task1_Data/Task1_Weather/classical_ml_feature_engineering.py` - Step 2: feature engineering
- `Task1_Data/Task1_Weather/classical_ml.py` - Step 3: classical ML training
- `Task1_Data/Task1_Weather/quantum_ml_data_processing.py` - Step 4: quantum preprocessing
- `Task1_Data/Task1_Weather/quantum_ml.py` - Step 5: quantum ML training

**Pipeline Scripts (Task1_Weather_Fire_Context):**
- `Task1_Data/Task1_Weather_Fire_Context/data_processing_v2.py` - Step 1 v2
- `Task1_Data/Task1_Weather_Fire_Context/classical_ml_feature_engineering.py` - Step 2 v2
- `Task1_Data/Task1_Weather_Fire_Context/classical_ml.py` - Step 3 v2

**Output Files (intermediate CSVs):**
- `Task1_Data/Task1_Weather/task1_step1_event_cleaned.csv`
- `Task1_Data/Task1_Weather/task1_step1_zip_year_ready.csv`
- `Task1_Data/Task1_Weather/task1_step2_classical_ready.csv`
- `Task1_Data/Task1_Weather/task1_step3_model_comparison.csv`
- `Task1_Data/Task1_Weather/task1_step4_quantum_train.csv`
- `Task1_Data/Task1_Weather/task1_step4_quantum_test.csv`
- `Task1_Data/Task1_Weather/task1_step5_vqc_metrics.csv`

## Naming Conventions

**Files:**
- Pipeline scripts: `snake_case.py`
- Intermediate data: `task1_step{N}_{description}.csv`
- Config files: `task1_step{N}_{name}_config.json`
- Output metrics: `task1_step{N}_{name}_metrics.csv`

**Variables:**
- Columns: `snake_case` (e.g., `avg_tmax_c`, `fire_occurred`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `INPUT_FILE`, `RANDOM_STATE`)
- Feature names: `snake_case` with prefix for quantum features (e.g., `q_feature_0`)

**Step Naming:**
- Step 1: Data preprocessing and cleaning
- Step 2: Feature engineering
- Step 3: Classical ML training
- Step 4: Quantum ML data preparation
- Step 5: Quantum ML training

## Where to Add New Code

**New Feature Branch:**
- Create new directory under `Task1_Data/` (e.g., `Task1_Data/Task1_Weather_Fire_Context_Demographics/`)
- Copy existing pipeline structure as template
- Modify `data_processing*.py` to include new feature sources

**New Pipeline Step:**
- Add new script following step numbering (e.g., `classical_ml_v2.py` for improved classical ML)
- Use existing INPUT_FILE pattern
- Output to `task1_step{N}_{description}.csv`

**New ML Model:**
- Add to existing classical ML script or create new script
- Follow `evaluate_model` helper pattern
- Output metrics to CSV following naming convention

**Utilities:**
- No dedicated utils directory exists
- Helper functions defined inline within scripts (e.g., `fill_zip_by_latlon_mode`, `to_one_hot`)

## Special Directories

**Task2_Data:**
- Purpose: Contains raw data for Task 2 (insurance/census integration)
- Generated: No (external data)
- Committed: Yes (CSV files tracked in git)

**Intermediate Data Files:**
- Purpose: Pipeline stage outputs used as inputs to next stage
- Generated: Yes (by running pipeline scripts)
- Committed: Yes (stored in repository)

**Python Scripts:**
- Purpose: Standalone processing scripts
- Generated: No
- Committed: Yes

---

*Structure analysis: 2026-04-09*
