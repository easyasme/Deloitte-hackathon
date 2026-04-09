# Architecture

**Analysis Date:** 2026-04-09

## Pattern Overview

**Overall:** Pipeline-oriented data science project with staged processing

**Key Characteristics:**
- Standalone Python scripts organized as sequential pipeline stages
- No web framework or API layer
- Two parallel feature branches (weather-only vs weather+fire context)
- Hardcoded file paths for inter-stage data passing
- Time-based train/test split (train: 2018-2020, test: 2021)

## Layers

**Data Ingestion:**
- Purpose: Load raw CSV data and perform initial cleaning
- Location: `Task1_Data/Task1_Weather/data_preprocessing.py`, `Task1_Data/Task1_Weather_Fire_Context/data_processing_v2.py`
- Contains: Raw CSV loading, column selection, type conversion, deduplication
- Depends on: External CSV files (`abfap7bci2UF6CTY_wildfire_weather.csv`)
- Produces: Intermediate cleaned CSVs with event-level and zip-year aggregated data

**Feature Engineering:**
- Purpose: Transform raw data into model-ready features
- Location: `Task1_Data/Task1_Weather/classical_ml_feature_engineering.py`, `Task1_Data/Task1_Weather_Fire_Context/classical_ml_feature_engineering.py`
- Contains: Leakage removal, engineered features (temp_range_c, dryness_proxy, year_index), one-hot encoding for categorical variables
- Depends on: Step 1 output CSVs
- Produces: `task1_step2_*_classical_ready.csv`

**Classical ML:**
- Purpose: Train and evaluate traditional ML models
- Location: `Task1_Data/Task1_Weather/classical_ml.py`, `Task1_Data/Task1_Weather_Fire_Context/classical_ml.py`
- Contains: LogisticRegression (with StandardScaler pipeline), RandomForestClassifier
- Depends on: Step 2 feature-engineered data
- Produces: Metrics, predictions, feature importance CSVs

**Quantum ML Preprocessing:**
- Purpose: Prepare data for quantum computing (dimensionality reduction, scaling)
- Location: `Task1_Data/Task1_Weather/quantum_ml_data_processing.py`
- Contains: PCA (4 components), MinMax scaling to [0, pi] for angle encoding
- Depends on: Step 2 output
- Produces: `task1_step4_quantum_train.csv`, `task1_step4_quantum_test.csv`, config JSON

**Quantum ML:**
- Purpose: Train Variational Quantum Classifier (VQC)
- Location: `Task1_Data/Task1_Weather/quantum_ml.py`
- Contains: Custom AngleEncodingRY feature map, RealAmplitudes ansatz, COBYLA optimizer, StatevectorSampler
- Depends on: Step 4 quantum-ready data
- Produces: VQC metrics, predictions, training log

## Data Flow

**Weather-Only Branch:**
1. Raw CSV (`abfap7bci2UF6CTY_wildfire_weather.csv`) -/-> `data_preprocessing.py` -/-> `task1_step1_event_cleaned.csv`, `task1_step1_zip_year_ready.csv`
2. `task1_step1_zip_year_ready.csv` -/-> `classical_ml_feature_engineering.py` -/-> `task1_step2_classical_ready.csv`
3. `task1_step2_classical_ready.csv` -/-> `classical_ml.py` -/-> metrics/predictions
4. `task1_step2_classical_ready.csv` -/-> `quantum_ml_data_processing.py` -/-> `task1_step4_quantum_*.csv`
5. `task1_step4_quantum_*.csv` -/-> `quantum_ml.py` -/-> VQC metrics/predictions

**Weather+Fire Context Branch:**
1. Same raw CSV -/-> `data_processing_v2.py` -/-> `task1_step1_v2_event_cleaned.csv`, `task1_step1_v2_zip_year_ready.csv`
2. `task1_step1_v2_zip_year_ready.csv` -/-> `classical_ml_feature_engineering.py` (v2) -/-> `task1_step2_v2_classical_ready.csv`
3. `task1_step2_v2_classical_ready.csv` -/-> `classical_ml.py` (v2) -/-> metrics/predictions

**State Management:**
- No persistent state management system
- Inter-stage state passed via CSV files
- In-memory pandas DataFrames within each script

## Key Abstractions

**Pipeline Stage:**
- Purpose: Represents a processing step in the ML pipeline
- Examples: `data_preprocessing.py`, `classical_ml_feature_engineering.py`, `classical_ml.py`
- Pattern: Standalone script with INPUT_FILE/OUTPUT_FILE constants, sequential load/process/save

**Model Wrapper:**
- Purpose: Encapsulates model training and evaluation
- Examples: `classical_ml.py` (lines 116-137: `evaluate_model` helper)
- Pattern: fit -> predict -> evaluate -> save metrics

**Feature Engineering:**
- Purpose: Creates derived features from raw data
- Examples: `temp_range_c = avg_tmax_c - avg_tmin_c`, `dryness_proxy = avg_tmax_c / (tot_prcp_mm + 1)`
- Pattern: Column arithmetic, groupby aggregations, one-hot encoding

## Entry Points

**Pipeline Execution:**
- Location: Individual Python scripts run directly
- Triggers: `python data_preprocessing.py`, `python classical_ml.py`, etc.
- Responsibilities: Load input, process, write output

**No Web Entry Points:**
- No Flask, FastAPI, or similar web framework detected
- No API routes or endpoints

## Error Handling

**Strategy:** Basic validation with ValueError exceptions

**Patterns:**
- Column existence checks: `if missing_required: raise ValueError(f"no necessary columns: {missing_required}")`
- Empty dataframe checks: `if len(train_df) == 0: raise ValueError(...)`
- Feature presence validation before modeling

## Cross-Cutting Concerns

**Logging:** Print statements to stdout for progress tracking

**Validation:** Column presence checks, type conversions with `errors="coerce"`

**Randomization:** `random_state=42` used consistently for reproducibility

**Data Leakage Prevention:** Explicit removal of `fire_count`, `GIS_ACRES`, `fire_duration_days` in feature engineering steps

---

*Architecture analysis: 2026-04-09*
