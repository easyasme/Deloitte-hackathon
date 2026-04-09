# Technology Stack

**Analysis Date:** 2026-04-09

## Languages

**Primary:**
- Python 3.x - All code is written in Python; no version constraint file detected

**Secondary:**
- None detected

## Runtime

**Environment:**
- Python interpreter (CPython)
- No virtual environment configuration detected (no requirements.txt, pyproject.toml, or environment.yml)

**Package Manager:**
- Not explicitly defined - pip/conda usage inferred from import statements

**Lockfile:**
- None present

## Frameworks

**Core:**
- None - This is a data science/ML pipeline codebase without a web framework

**Data Processing & ML:**
- pandas - Data manipulation and CSV I/O
- numpy - Numerical operations and array processing
- scikit-learn (sklearn) - Classical ML models and preprocessing

**Quantum Computing:**
- qiskit - Quantum circuit construction and primitives
- qiskit_machine_learning - VQC (Variational Quantum Classifier) implementation

**Testing:**
- Not detected - No test framework configuration or test files found

**Build/Dev:**
- None detected

## Key Dependencies

**Critical:**
- pandas - Used in all scripts for data loading, preprocessing, and CSV output
- numpy - Universal dependency for numerical operations
- scikit-learn - Classical ML models (LogisticRegression, RandomForestClassifier), preprocessing (StandardScaler, MinMaxScaler, PCA), and metrics

**Quantum:**
- qiskit - Quantum circuit library (QuantumCircuit, ParameterVector, RealAmplitudes)
- qiskit_machine_learning.algorithms.VQC - Variational Quantum Classifier
- qiskit_machine_learning.optimizers.COBYLA - Classical optimizer for quantum training

**Data:**
- No external data providers - All data loaded from local CSV files in the repository

## Configuration

**Environment:**
- No .env files or environment variable configuration detected
- All configuration is hardcoded in Python scripts (file paths, hyperparameters)

**Build:**
- No build configuration files detected (no setup.py, pyproject.toml, Makefile)

**Hyperparameter Configuration:**
- `Task1_Data/Task1_Weather/task1_step4_quantum_config.json` - JSON config for quantum ML pipeline

## Platform Requirements

**Development:**
- Python 3.x with pip
- Required packages: pandas, numpy, scikit-learn, qiskit, qiskit_machine_learning

**Production:**
- Standalone Python scripts executed via command line
- No deployment platform detected (not a web service)

---

*Stack analysis: 2026-04-09*
