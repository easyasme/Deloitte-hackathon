# Coding Conventions

**Analysis Date:** 2026-04-09

## Naming Patterns

**Files:**
- Python scripts use descriptive names: `data_preprocessing.py`, `classical_ml.py`, `quantum_ml.py`
- Lowercase with underscores (snake_case)
- Versioned files use suffix: `data_processing_v2.py`, `classical_ml_feature_engineering.py`

**Functions:**
- snake_case naming
- Example: `to_one_hot(y)`, `fill_zip_by_latlon_mode(group)`, `evaluate_model(model_name, y_true, y_pred, y_prob)`
- No type hints present

**Variables:**
- snake_case: `train_df`, `X_train`, `y_train_raw`, `feature_cols`
- Boolean prefixes not consistently used: `USE_PCA`, `USE_STANDARDIZE_BEFORE_PCA`

**Constants:**
- UPPER_SNAKE_CASE: `RANDOM_STATE`, `MAXITER`, `TRAIN_FILE`, `INPUT_FILE`
- Module-level constants defined at top of files

**Types:**
- No type annotations or type hints detected
- Type checking done via runtime checks and pandas dtypes

## Code Style

**Formatting:**
- No automated formatter detected (no .prettierrc, .black, etc.)
- 4-space indentation observed
- Single blank lines between logical sections
- Numbered step comments: `# 1. Load`, `# 2. Define target`, etc.

**Linting:**
- No ESLint or Pylint configuration detected
- No pre-commit hooks

**Line Length:**
- No enforced line length limit
- Lines can exceed 100 characters

## Import Organization

**Order:**
1. Standard library: `import json`, `import warnings`, `import time`
2. Third-party: `import numpy as np`, `import pandas as pd`
3. scikit-learn: `from sklearn.metrics import (...)`, `from sklearn.pipeline import Pipeline`
4. Quantum: `from qiskit.circuit import QuantumCircuit`
5. Local scripts not imported (scripts are standalone executables)

**Style:**
```python
import json
import warnings
import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
```

## Error Handling

**Validation:**
```python
missing_features = [c for c in feature_cols if c not in df.columns]
if missing_features:
    raise ValueError(f"no necessary feature: {missing_features}")
```

**Pattern:** Check for missing columns/files early, raise `ValueError` with descriptive message

**Exception suppression:**
```python
try:
    y_prob_out = vqc.predict_proba(X_test)
except Exception:
    y_prob = None
```
- Broad `Exception` catches used (not specific exceptions)
- `warnings.filterwarnings("ignore")` used to suppress library warnings

**Data validation:**
```python
if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError("train/test no result. Check year value")
```

## Logging

**Framework:** Built-in `print()` only

**Patterns:**
- Progress indicators: `print("train shape:", train_df.shape)`
- Section headers: `print("\n--- Model Training ---")`
- Results: `print(f"{model_name} evaluation result")`
- File outputs: `print("-", METRICS_OUT)`

**No structured logging framework detected**

## Comments

**When to Comment:**
- Step markers for pipeline sections: `# STEP 3: Classical Baseline Model Training & Validation`
- Input/output documentation at top of files:
```python
# STEP 5: Quantum ML Model (VQC) Implementation & Comparison
# Inputs:
#   - task1_step4_quantum_train.csv
#   - task1_step4_quantum_test.csv
# Outputs:
#   - task1_step5_vqc_metrics.csv
```
- Korean comments mixed with English (e.g., `# [3] leakage / post-event 컬럼 제거 완료`)

**JSDoc/TSDoc:**
- Not used (Python codebase, no docstrings detected)

## Function Design

**Size:** Functions tend to be moderate length; scripts are structured as linear pipelines with helper functions

**Parameters:**
- Simple parameter lists, no *args or **kwargs observed
- Configuration passed via constants or JSON files

**Return Values:**
- Explicit returns for evaluation metrics
- DataFrames returned via `to_csv()`
- Helper functions return transformed data, not saving to disk

## Module Design

**Exports:** Not applicable (standalone scripts, no `__init__.py`)

**Barrel Files:** Not used

**Pattern:** Each step is a standalone script that reads input, transforms data, and writes output to CSV/JSON

## Additional Patterns

**Pipeline Steps:**
- Scripts numbered sequentially: `task1_step1_...`, `task1_step2_...`, etc.
- Each step reads from previous step's output file
- Time-based train/test splits (train: 2018-2020, test: 2021)

**Feature Engineering:**
- Engineer features in place: `df["temp_range_c"] = df["avg_tmax_c"] - df["avg_tmin_c"]`
- Column existence checked before operations: `if {"avg_tmax_c", "avg_tmin_c"}.issubset(df.columns):`

**Model Training:**
- Random state fixed: `random_state=42`
- Class balancing: `class_weight="balanced"`

---

*Convention analysis: 2026-04-09*
