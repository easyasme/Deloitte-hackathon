<!-- GSD:project-start source:PROJECT.md -->
## Project

**Deloitte Hackathon - Task 2: Insurance Premium Prediction**

Time series model to predict insurance premiums in 2021 for California zip codes, using historical data from 2018-2020. Integrates wildfire risk (from Task 1 quantum model or existing fire risk score in dataset) to enhance predictions.

**Core Value:** Predict 2021 insurance premiums accurately by zip code, using historical trends, risk features, and wildfire risk as a key predictor.

### Constraints

- **Data:** Only 3 years of historical data (2018-2020) — limited for deep time series
- **Features:** Many categorical and numerical features — selection needed
- **Wildfire risk:** Can use dataset's `Avg Fire Risk Score` directly or integrate Task 1 outputs
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.x - All code is written in Python; no version constraint file detected
- None detected
## Runtime
- Python interpreter (CPython)
- No virtual environment configuration detected (no requirements.txt, pyproject.toml, or environment.yml)
- Not explicitly defined - pip/conda usage inferred from import statements
- None present
## Frameworks
- None - This is a data science/ML pipeline codebase without a web framework
- pandas - Data manipulation and CSV I/O
- numpy - Numerical operations and array processing
- scikit-learn (sklearn) - Classical ML models and preprocessing
- qiskit - Quantum circuit construction and primitives
- qiskit_machine_learning - VQC (Variational Quantum Classifier) implementation
- Not detected - No test framework configuration or test files found
- None detected
## Key Dependencies
- pandas - Used in all scripts for data loading, preprocessing, and CSV output
- numpy - Universal dependency for numerical operations
- scikit-learn - Classical ML models (LogisticRegression, RandomForestClassifier), preprocessing (StandardScaler, MinMaxScaler, PCA), and metrics
- qiskit - Quantum circuit library (QuantumCircuit, ParameterVector, RealAmplitudes)
- qiskit_machine_learning.algorithms.VQC - Variational Quantum Classifier
- qiskit_machine_learning.optimizers.COBYLA - Classical optimizer for quantum training
- No external data providers - All data loaded from local CSV files in the repository
## Configuration
- No .env files or environment variable configuration detected
- All configuration is hardcoded in Python scripts (file paths, hyperparameters)
- No build configuration files detected (no setup.py, pyproject.toml, Makefile)
- `Task1_Data/Task1_Weather/task1_step4_quantum_config.json` - JSON config for quantum ML pipeline
## Platform Requirements
- Python 3.x with pip
- Required packages: pandas, numpy, scikit-learn, qiskit, qiskit_machine_learning
- Standalone Python scripts executed via command line
- No deployment platform detected (not a web service)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Python scripts use descriptive names: `data_preprocessing.py`, `classical_ml.py`, `quantum_ml.py`
- Lowercase with underscores (snake_case)
- Versioned files use suffix: `data_processing_v2.py`, `classical_ml_feature_engineering.py`
- snake_case naming
- Example: `to_one_hot(y)`, `fill_zip_by_latlon_mode(group)`, `evaluate_model(model_name, y_true, y_pred, y_prob)`
- No type hints present
- snake_case: `train_df`, `X_train`, `y_train_raw`, `feature_cols`
- Boolean prefixes not consistently used: `USE_PCA`, `USE_STANDARDIZE_BEFORE_PCA`
- UPPER_SNAKE_CASE: `RANDOM_STATE`, `MAXITER`, `TRAIN_FILE`, `INPUT_FILE`
- Module-level constants defined at top of files
- No type annotations or type hints detected
- Type checking done via runtime checks and pandas dtypes
## Code Style
- No automated formatter detected (no .prettierrc, .black, etc.)
- 4-space indentation observed
- Single blank lines between logical sections
- Numbered step comments: `# 1. Load`, `# 2. Define target`, etc.
- No ESLint or Pylint configuration detected
- No pre-commit hooks
- No enforced line length limit
- Lines can exceed 100 characters
## Import Organization
## Error Handling
- Broad `Exception` catches used (not specific exceptions)
- `warnings.filterwarnings("ignore")` used to suppress library warnings
## Logging
- Progress indicators: `print("train shape:", train_df.shape)`
- Section headers: `print("\n--- Model Training ---")`
- Results: `print(f"{model_name} evaluation result")`
- File outputs: `print("-", METRICS_OUT)`
## Comments
- Step markers for pipeline sections: `# STEP 3: Classical Baseline Model Training & Validation`
- Input/output documentation at top of files:
- Korean comments mixed with English (e.g., `# [3] leakage / post-event 컬럼 제거 완료`)
- Not used (Python codebase, no docstrings detected)
## Function Design
- Simple parameter lists, no *args or **kwargs observed
- Configuration passed via constants or JSON files
- Explicit returns for evaluation metrics
- DataFrames returned via `to_csv()`
- Helper functions return transformed data, not saving to disk
## Module Design
## Additional Patterns
- Scripts numbered sequentially: `task1_step1_...`, `task1_step2_...`, etc.
- Each step reads from previous step's output file
- Time-based train/test splits (train: 2018-2020, test: 2021)
- Engineer features in place: `df["temp_range_c"] = df["avg_tmax_c"] - df["avg_tmin_c"]`
- Column existence checked before operations: `if {"avg_tmax_c", "avg_tmin_c"}.issubset(df.columns):`
- Random state fixed: `random_state=42`
- Class balancing: `class_weight="balanced"`
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Standalone Python scripts organized as sequential pipeline stages
- No web framework or API layer
- Two parallel feature branches (weather-only vs weather+fire context)
- Hardcoded file paths for inter-stage data passing
- Time-based train/test split (train: 2018-2020, test: 2021)
## Layers
- Purpose: Load raw CSV data and perform initial cleaning
- Location: `Task1_Data/Task1_Weather/data_preprocessing.py`, `Task1_Data/Task1_Weather_Fire_Context/data_processing_v2.py`
- Contains: Raw CSV loading, column selection, type conversion, deduplication
- Depends on: External CSV files (`abfap7bci2UF6CTY_wildfire_weather.csv`)
- Produces: Intermediate cleaned CSVs with event-level and zip-year aggregated data
- Purpose: Transform raw data into model-ready features
- Location: `Task1_Data/Task1_Weather/classical_ml_feature_engineering.py`, `Task1_Data/Task1_Weather_Fire_Context/classical_ml_feature_engineering.py`
- Contains: Leakage removal, engineered features (temp_range_c, dryness_proxy, year_index), one-hot encoding for categorical variables
- Depends on: Step 1 output CSVs
- Produces: `task1_step2_*_classical_ready.csv`
- Purpose: Train and evaluate traditional ML models
- Location: `Task1_Data/Task1_Weather/classical_ml.py`, `Task1_Data/Task1_Weather_Fire_Context/classical_ml.py`
- Contains: LogisticRegression (with StandardScaler pipeline), RandomForestClassifier
- Depends on: Step 2 feature-engineered data
- Produces: Metrics, predictions, feature importance CSVs
- Purpose: Prepare data for quantum computing (dimensionality reduction, scaling)
- Location: `Task1_Data/Task1_Weather/quantum_ml_data_processing.py`
- Contains: PCA (4 components), MinMax scaling to [0, pi] for angle encoding
- Depends on: Step 2 output
- Produces: `task1_step4_quantum_train.csv`, `task1_step4_quantum_test.csv`, config JSON
- Purpose: Train Variational Quantum Classifier (VQC)
- Location: `Task1_Data/Task1_Weather/quantum_ml.py`
- Contains: Custom AngleEncodingRY feature map, RealAmplitudes ansatz, COBYLA optimizer, StatevectorSampler
- Depends on: Step 4 quantum-ready data
- Produces: VQC metrics, predictions, training log
## Data Flow
- No persistent state management system
- Inter-stage state passed via CSV files
- In-memory pandas DataFrames within each script
## Key Abstractions
- Purpose: Represents a processing step in the ML pipeline
- Examples: `data_preprocessing.py`, `classical_ml_feature_engineering.py`, `classical_ml.py`
- Pattern: Standalone script with INPUT_FILE/OUTPUT_FILE constants, sequential load/process/save
- Purpose: Encapsulates model training and evaluation
- Examples: `classical_ml.py` (lines 116-137: `evaluate_model` helper)
- Pattern: fit -> predict -> evaluate -> save metrics
- Purpose: Creates derived features from raw data
- Examples: `temp_range_c = avg_tmax_c - avg_tmin_c`, `dryness_proxy = avg_tmax_c / (tot_prcp_mm + 1)`
- Pattern: Column arithmetic, groupby aggregations, one-hot encoding
## Entry Points
- Location: Individual Python scripts run directly
- Triggers: `python data_preprocessing.py`, `python classical_ml.py`, etc.
- Responsibilities: Load input, process, write output
- No Flask, FastAPI, or similar web framework detected
- No API routes or endpoints
## Error Handling
- Column existence checks: `if missing_required: raise ValueError(f"no necessary columns: {missing_required}")`
- Empty dataframe checks: `if len(train_df) == 0: raise ValueError(...)`
- Feature presence validation before modeling
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
