import pandas as pd
import numpy as np
import json

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# STEP 4: Quantum ML용 데이터 변환 (Scaling + Dimension Reduction)
# Input : task1_step2_classical_ready.csv
# Output:
#   - task1_step4_quantum_train.csv
#   - task1_step4_quantum_test.csv
#   - task1_step4_quantum_config.json

INPUT_FILE = "task1_step2_classical_ready.csv"

# 0. User-configurable settings
N_QUBITS = 4

# angle encoding range
ANGLE_MIN = 0.0
ANGLE_MAX = np.pi

# split
TRAIN_MAX_YEAR = 2020
TEST_YEAR = 2021

USE_PCA = True

# standardization before PCA
USE_STANDARDIZE_BEFORE_PCA = True

TRAIN_OUTPUT = "task1_step4_quantum_train.csv"
TEST_OUTPUT = "task1_step4_quantum_test.csv"
CONFIG_OUTPUT = "task1_step4_quantum_config.json"


# 1. Load
df = pd.read_csv(INPUT_FILE)

print("input data shape:", df.shape)
print("columns:")
print(df.columns.tolist())


# 2. Define raw classical features
target_col = "fire_occurred"

raw_feature_cols = [
    "avg_tmax_c",
    "avg_tmin_c",
    "tot_prcp_mm",
    "temp_range_c",
    "dryness_proxy",
    "year_index"
]

missing_features = [c for c in raw_feature_cols if c not in df.columns]
if missing_features:
    raise ValueError(f"no necessary raw features: {missing_features}")

if target_col not in df.columns:
    raise ValueError(f"no target columns: {target_col}")

if "Year" not in df.columns:
    raise ValueError("Year column needed")

print("\nraw classical feature:")
print(raw_feature_cols)
print("target:", target_col)


# 3. Time-based split
train_df = df[df["Year"] <= TRAIN_MAX_YEAR].copy()
test_df = df[df["Year"] == TEST_YEAR].copy()

if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError("check Year value")

X_train_raw = train_df[raw_feature_cols].copy()
y_train = train_df[target_col].copy().astype(int)

X_test_raw = test_df[raw_feature_cols].copy()
y_test = test_df[target_col].copy().astype(int)

print("train shape:", X_train_raw.shape)
print("test shape :", X_test_raw.shape)

print("\ntrain target distribution:")
print(y_train.value_counts())

print("\ntest target distribution:")
print(y_test.value_counts())


# 4. Optional standardization before PCA
if USE_STANDARDIZE_BEFORE_PCA:
    pre_pca_scaler = StandardScaler()
    X_train_for_pca = pre_pca_scaler.fit_transform(X_train_raw)
    X_test_for_pca = pre_pca_scaler.transform(X_test_raw)
    print("\nStandardScaler applied")
else:
    pre_pca_scaler = None
    X_train_for_pca = X_train_raw.values
    X_test_for_pca = X_test_raw.values
    print("\nStandardScaler skipped")


# 5. Dimension reduction
if USE_PCA:
    if N_QUBITS > X_train_for_pca.shape[1]:
        raise ValueError(
            f"N_QUBITS={N_QUBITS} number of raw feature larger than {X_train_for_pca.shape[1]}"
        )

    pca = PCA(n_components=N_QUBITS, random_state=42)
    X_train_reduced = pca.fit_transform(X_train_for_pca)
    X_test_reduced = pca.transform(X_test_for_pca)

    explained_variance = pca.explained_variance_ratio_.tolist()
    reduced_feature_names = [f"q_feature_{i}" for i in range(N_QUBITS)]

    print("train shape after PCA:", X_train_reduced.shape)
    print("test shape after PCA:", X_test_reduced.shape)
    print("explained variance ratio:", explained_variance)
    print("total explained variance:", sum(explained_variance))
else:
    pca = None
    X_train_reduced = X_train_for_pca
    X_test_reduced = X_test_for_pca
    reduced_feature_names = [f"q_feature_{i}" for i in range(X_train_reduced.shape[1])]
    explained_variance = None

    print("reduced shape:", X_train_reduced.shape)


# 6. Quantum scaling
angle_scaler = MinMaxScaler(feature_range=(ANGLE_MIN, ANGLE_MAX))
X_train_quantum = angle_scaler.fit_transform(X_train_reduced)
X_test_quantum = angle_scaler.transform(X_test_reduced)

X_train_quantum = np.clip(X_train_quantum, ANGLE_MIN, ANGLE_MAX)
X_test_quantum = np.clip(X_test_quantum, ANGLE_MIN, ANGLE_MAX)

print(f"range: [{ANGLE_MIN}, {ANGLE_MAX}]")
print("train min:", np.min(X_train_quantum))
print("train max:", np.max(X_train_quantum))
print("test min :", np.min(X_test_quantum))
print("test max :", np.max(X_test_quantum))


# 7. Build output DataFrames
train_out = pd.DataFrame(X_train_quantum, columns=reduced_feature_names)
train_out["fire_occurred"] = y_train.values
train_out["Year"] = train_df["Year"].values

test_out = pd.DataFrame(X_test_quantum, columns=reduced_feature_names)
test_out["fire_occurred"] = y_test.values
test_out["Year"] = test_df["Year"].values

print("train_out shape:", train_out.shape)
print("test_out shape :", test_out.shape)


# 8. Feature map / ansatz design config
quantum_config = {
    "step": 4,
    "input_file": INPUT_FILE,
    "train_output": TRAIN_OUTPUT,
    "test_output": TEST_OUTPUT,

    "time_split": {
        "train_max_year": TRAIN_MAX_YEAR,
        "test_year": TEST_YEAR
    },

    "raw_feature_cols": raw_feature_cols,
    "target_col": target_col,

    "preprocessing": {
        "use_standardize_before_pca": USE_STANDARDIZE_BEFORE_PCA,
        "use_pca": USE_PCA,
        "n_qubits": N_QUBITS,
        "angle_range": [float(ANGLE_MIN), float(ANGLE_MAX)],
        "angle_clipping_applied": True
    },

    "pca": {
        "used": USE_PCA,
        "n_components": N_QUBITS if USE_PCA else X_train_reduced.shape[1],
        "explained_variance_ratio": explained_variance,
        "total_explained_variance": float(sum(explained_variance)) if explained_variance is not None else None
    },

    "feature_map": {
        "name": "AngleEncoding",
        "description": "Each reduced feature is scaled and clipped to [0, pi], then mapped to one qubit rotation angle.",
        "n_features": X_train_quantum.shape[1],
        "n_qubits": X_train_quantum.shape[1],
        "rotation_gate": "RY"
    },

    "ansatz": {
        "name": "RealAmplitudes",
        "description": "Parameterized variational ansatz with single-qubit rotations and entanglement.",
        "reps": 2,
        "entanglement": "linear"
    },

    "resource_estimate": {
        "n_qubits": X_train_quantum.shape[1],
        "feature_dimension_after_reduction": X_train_quantum.shape[1],
        "train_samples": int(len(train_out)),
        "test_samples": int(len(test_out)),
        "note": "Final trainable parameter count depends on the ansatz implementation in Step 5."
    }
}

# 9. Save outputs
train_out.to_csv(TRAIN_OUTPUT, index=False)
test_out.to_csv(TEST_OUTPUT, index=False)

with open(CONFIG_OUTPUT, "w") as f:
    json.dump(quantum_config, f, indent=2)

print("-", TRAIN_OUTPUT)
print("-", TEST_OUTPUT)
print("-", CONFIG_OUTPUT)


# 10. Preview
print("\ntrain sample")
print(train_out.head())

print("\ntest sample")
print(test_out.head())

print("\nconfig summary")
print(json.dumps(quantum_config, indent=2))
