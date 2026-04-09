# Codebase Concerns

**Analysis Date:** 2026-04-09

## Tech Debt

**Duplicate Pipelines:**
- Issue: Two parallel ML pipelines exist with significant code duplication
- Files: `Task1_Data/Task1_Weather/` and `Task1_Data/Task1_Weather_Fire_Context/`
- Impact: Maintenance burden, divergent behavior, harder to track which is "canonical"
- Fix approach: Consolidate into single parameterized pipeline

**Hardcoded Configuration Values:**
- Issue: Hyperparameters scattered across individual scripts with no central config
- Files:
  - `Task1_Data/Task1_Weather/classical_ml.py` (lines 99-106): `n_estimators=300, max_iter=2000`
  - `Task1_Data/Task1_Weather_Fire_Context/classical_ml.py` (lines 105-113): `n_estimators=400, max_iter=3000`
  - `Task1_Data/Task1_Weather/quantum_ml.py` (line 18): `N_QUBITS=4`
  - `Task1_Data/Task1_Weather/quantum_ml.py` (line 49): `MAXITER=200`
- Impact: Inconsistent model configurations, no way to experiment with hyperparameters systematically
- Fix approach: Create shared config file (JSON/YAML) for all hyperparameters

**Hardcoded File Paths:**
- Issue: All input/output file paths hardcoded at top of each script
- Files: All Python scripts (e.g., `data_preprocessing.py` line 5, `quantum_ml.py` line 15)
- Impact: Scripts must be run from specific working directories; moving data breaks everything
- Fix approach: Use relative paths from project root or environment variables

**Hardcoded Year Ranges:**
- Issue: Year filter `2018-2023` hardcoded in multiple places
- Files:
  - `Task1_Data/Task1_Weather/data_preprocessing.py` (line 100)
  - `Task1_Data/Task1_Weather_Fire_Context/data_processing_v2.py` (line 133)
- Impact: Cannot easily test different time periods without modifying multiple files
- Fix approach: Centralize temporal boundaries in config

**Magic Number: AGENCY_ID=9:**
- Issue: `AGENCY_ID=9` used as sentinel for missing values without explanation
- Files:
  - `Task1_Data/Task1_Weather_Fire_Context/data_processing_v2.py` (line 81)
- Impact: Business logic hidden in code; other developers may not understand this mapping
- Fix approach: Define as named constant with comment explaining source

## Known Bugs

**Silent Mode Failure in Aggregation:**
- Issue: Mode calculation returns empty Series when group has no mode, silently defaulting to NaN
- Files:
  - `Task1_Data/Task1_Weather_Fire_Context/data_processing_v2.py` (lines 178, 228-229)
  - `Task1_Data/Task1_Weather/data_preprocessing.py` (lines 149, 190-194)
- Trigger: Groups where all values differ or all are NaN
- Impact: Silent data loss in aggregation; wrongfill with NaN may propagate
- Workaround: Verify mode is non-empty before using

**Quantum Training Non-Deterministic:**
- Issue: No `random_state` set for quantum training (COBYLA optimizer)
- Files:
  - `Task1_Data/Task1_Weather/quantum_ml.py` (line 149)
- Trigger: Running quantum training multiple times produces different results
- Impact: Results not reproducible; cannot trust model comparisons
- Workaround: Set optimizer seed if supported

**Task 2 Data Has No Processing Scripts:**
- Issue: `Task2_Data/` contains raw CSV files but no Python scripts to analyze them
- Files: `Task2_Data/` directory
- Impact: Task 2 analysis cannot be reproduced or extended
- Workaround: Scripts must be created from scratch

## Security Considerations

**No Input Validation:**
- Risk: Scripts fail with unclear errors if input files are missing or have unexpected format
- Files: All Python scripts
- Current mitigation: None
- Recommendations: Add explicit file existence checks with helpful error messages

**No Environment Variable Handling:**
- Risk: Sensitive paths or config values hardcoded
- Files: All Python scripts
- Current mitigation: None (no secrets observed, but pattern is risky)
- Recommendations: Use environment variables for file paths and optional overrides

**Hardcoded Column Selection:**
- Risk: Scripts select columns by name with no validation that they exist
- Files: All data processing scripts
- Current mitigation: Partial - some use `if c in df.columns` checks
- Recommendations: Validate required columns exist before processing

## Performance Bottlenecks

**Large GroupBy Operations:**
- Problem: `groupby(["latitude", "longitude"]).apply()` on potentially large dataframes
- Files:
  - `Task1_Data/Task1_Weather/data_preprocessing.py` (lines 78-81)
  - `Task1_Data/Task1_Weather_Fire_Context/data_processing_v2.py` (lines 114-117)
- Cause: Collects all rows for each lat/lon pair before applying function
- Improvement path: Use `transform()` instead of `apply()` where possible

**No Index Optimization:**
- Problem: ZIP-year full grid created via `MultiIndex.from_product` without sorting
- Files:
  - `Task1_Data/Task1_Weather_Fire_Context/data_processing_v2.py` (lines 201-204)
- Improvement path: Sort indices before merge to improve join performance

**Quantum Statevector Simulation:**
- Problem: Uses `StatevectorSampler` which simulates quantum on classical hardware
- Files: `Task1_Data/Task1_Weather/quantum_ml.py` (line 21)
- Cause: Real quantum hardware access not available
- Impact: Results may not reflect actual quantum performance/scalability
- Improvement path: Document limitation; consider using Qiskit Aer simulator with noise models

## Fragile Areas

**Column Name Dependencies:**
- Why fragile: Scripts assume exact column names; renaming breaks everything
- Files: All scripts have column lists (e.g., `data_preprocessing.py` lines 16-29)
- Safe modification: Add column existence validation before use
- Test coverage: None

**Fire Count Leakage Risk:**
- Why fragile: `fire_count` directly reveals target; using it as feature invalidates model
- Files:
  - `Task1_Data/Task1_Weather_Fire_Context/data_processing_v2.py` (line 248-250 comment only)
  - `Task1_Data/Task1_Weather/classical_ml_feature_engineering.py` (lines 34-40)
- Safe modification: Explicitly remove from feature sets; add validation test
- Test coverage: None

**Conditional Drop Logic:**
- Why fragile: `drop_first=True` in `get_dummies` can cause issues if categories change
- Files: `Task1_Data/Task1_Weather_Fire_Context/classical_ml_feature_engineering.py` (line 96)
- Safe modification: Document which category is dropped and why
- Test coverage: None

**No Graceful Degradation:**
- Why fragile: Pipeline stops on first error rather than reporting which step failed
- Files: All scripts
- Safe modification: Add try/except with step identification
- Test coverage: None

## Scaling Limits

**In-Memory Processing:**
- Current capacity: Entire CSV loaded via `pd.read_csv(INPUT_FILE, low_memory=False)`
- Limit: RAM-constrained; large datasets will cause OOM
- Scaling path: Use chunked reading with `pd.read_csv(chunksize=...)` or Dask

**No Parallelization:**
- Current capacity: Single-threaded execution
- Limit: Quantum training and model training cannot scale horizontally
- Scaling path: Use joblib for RF parallelization (already using `n_jobs=-1` but only within models)

## Dependencies at Risk

**Qiskit Version Drift:**
- Risk: No `requirements.txt` pinning Qiskit version
- Impact: API changes in Qiskit may break quantum_ml.py
- Migration plan: Create requirements.txt with known-working versions

**Scikit-learn Version Sensitivity:**
- Risk: No version pinning
- Impact: Changes in sklearn API or default behaviors may affect results
- Migration plan: Pin sklearn version

## Missing Critical Features

**No Test Suite:**
- Problem: Zero test files in codebase
- Blocks: Safe refactoring, regression detection, confidence in results
- Fix: Add pytest tests for data processing, model training, and evaluation

**No Data Validation:**
- Problem: No schema validation for input data
- Blocks: Early detection of data quality issues
- Fix: Add pandera or great_expectations validation

**No Model Versioning:**
- Problem: No tracking of which model produced which results
- Blocks: Reproducibility, A/B testing, model comparison
- Fix: Add MLflow or similar for experiment tracking

**No Experiment Tracking:**
- Problem: Metrics saved to CSV but not searchable/queriable
- Blocks: Historical model comparison
- Fix: Use MLflow or Weights & Biases

## Test Coverage Gaps

**Data Processing Not Tested:**
- What's not tested: `data_preprocessing.py`, `data_processing_v2.py`
- Files: `Task1_Data/Task1_Weather/data_preprocessing.py`, `Task1_Data/Task1_Weather_Fire_Context/data_processing_v2.py`
- Risk: Silent data corruption, incorrect filtering, aggregation errors
- Priority: High

**Feature Engineering Not Tested:**
- What's not tested: Input/output contract, leakage prevention
- Files: `classical_ml_feature_engineering.py` files
- Risk: Incorrect features, leakage features included
- Priority: High

**Model Training Not Tested:**
- What's not tested: Training convergence, prediction correctness
- Files: `classical_ml.py`, `quantum_ml.py`
- Risk: Wrong metrics, broken predictions
- Priority: High

**No Integration Tests:**
- What's not tested: End-to-end pipeline from raw data to predictions
- Risk: Step compatibility issues, output format mismatches
- Priority: Medium

---

*Concerns audit: 2026-04-09*
