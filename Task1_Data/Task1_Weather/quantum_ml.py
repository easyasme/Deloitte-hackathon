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

# Qiskit
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA

warnings.filterwarnings("ignore")

# STEP 5: Quantum ML Model (VQC) Implementation & Comparison
# Inputs:
#   - task1_step4_quantum_train.csv
#   - task1_step4_quantum_test.csv
#   - task1_step4_quantum_config.json
#
# Outputs:
#   - task1_step5_vqc_metrics.csv
#   - task1_step5_vqc_predictions.csv
#   - task1_step5_vqc_training_log.csv

TRAIN_FILE = "task1_step4_quantum_train.csv"
TEST_FILE = "task1_step4_quantum_test.csv"
CONFIG_FILE = "task1_step4_quantum_config.json"

METRICS_OUT = "task1_step5_vqc_metrics.csv"
PRED_OUT = "task1_step5_vqc_predictions.csv"
LOG_OUT = "task1_step5_vqc_training_log.csv"

RANDOM_STATE = 42

# User-tunable settings
MAXITER = 200

OVERRIDE_ANSATZ_REPS = 3          
OVERRIDE_ENTANGLEMENT = "full"        

USE_ONE_HOT_LABELS = True


# 1. Load data
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

print("train shape:", train_df.shape)
print("test shape :", test_df.shape)

target_col = "fire_occurred"
feature_cols = [c for c in train_df.columns if c.startswith("q_feature_")]

if len(feature_cols) == 0:
    raise ValueError("q_feature_* no column")

if target_col not in train_df.columns or target_col not in test_df.columns:
    raise ValueError("fire_occurred no column")

X_train = train_df[feature_cols].values
y_train_raw = train_df[target_col].astype(int).values

X_test = test_df[feature_cols].values
y_test_raw = test_df[target_col].astype(int).values

n_qubits = len(feature_cols)

print("\nfeature cols:")
print(feature_cols)
print("n_qubits:", n_qubits)

print("\ntrain target distribution:")
print(pd.Series(y_train_raw).value_counts())

print("\ntest target distribution:")
print(pd.Series(y_test_raw).value_counts())


# 2. Optional one-hot labels
def to_one_hot(y):
    out = np.zeros((len(y), 2))
    out[np.arange(len(y)), y] = 1
    return out

if USE_ONE_HOT_LABELS:
    y_train = to_one_hot(y_train_raw)
    y_test = to_one_hot(y_test_raw)
else:
    y_train = y_train_raw
    y_test = y_test_raw


# 3. Build custom feature map
# Instead of ZZFeatureMap use RY angle encoding
x = ParameterVector("x", n_qubits)

feature_map = QuantumCircuit(n_qubits, name="AngleEncodingRY")
for i in range(n_qubits):
    feature_map.ry(x[i], i)

print("\ncustom feature map:")
print(feature_map)

# 4. Build ansatz
config_reps = config.get("ansatz", {}).get("reps", 2)
config_entanglement = config.get("ansatz", {}).get("entanglement", "linear")

ansatz_reps = OVERRIDE_ANSATZ_REPS if OVERRIDE_ANSATZ_REPS is not None else config_reps
ansatz_entanglement = OVERRIDE_ENTANGLEMENT if OVERRIDE_ENTANGLEMENT is not None else config_entanglement

ansatz = RealAmplitudes(
    num_qubits=n_qubits,
    reps=ansatz_reps,
    entanglement=ansatz_entanglement
)

num_trainable_params = ansatz.num_parameters

print("\nansatz:")
print(ansatz)
print("\nansatz trainable parameters:", num_trainable_params)


# 5. Optimizer / sampler / callback
objective_values = []

def callback_graph(weights, objective_value):
    objective_values.append({
        "iteration": len(objective_values) + 1,
        "objective_value": float(objective_value)
    })

optimizer = COBYLA(maxiter=MAXITER)

sampler = StatevectorSampler()

print("\noptimizer:", optimizer)
print("sampler: StatevectorSampler")
print("MAXITER:", MAXITER)


# 6. Create VQC
# VQC -> feature_map, ansatz, optimizer, sampler
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    loss="cross_entropy",
    callback=callback_graph,
    sampler=sampler
)

print("\nVQC generated")


# 7. Train
start_time = time.time()
vqc.fit(X_train, y_train)
train_time_sec = time.time() - start_time

print("\ndone training")
print(f"training time(sec): {train_time_sec:.2f}")
print(f"callbacks: {len(objective_values)}")


# 8. Predict
y_pred_out = vqc.predict(X_test)

# one-hot
if isinstance(y_pred_out, np.ndarray) and y_pred_out.ndim == 2:
    y_pred = np.argmax(y_pred_out, axis=1)
else:
    y_pred = np.array(y_pred_out).astype(int).reshape(-1)

# predict_proba
try:
    y_prob_out = vqc.predict_proba(X_test)
    if y_prob_out.ndim == 2 and y_prob_out.shape[1] >= 2:
        y_prob = y_prob_out[:, 1]
    else:
        y_prob = None
except Exception:
    y_prob = None

print("\ndone predicting")


# 9. Evaluate
metrics = {
    "model": "VQC",
    "feature_map": "AngleEncodingRY",
    "n_qubits": n_qubits,
    "ansatz_reps": ansatz_reps,
    "ansatz_entanglement": ansatz_entanglement,
    "num_trainable_params": int(num_trainable_params),
    "maxiter": MAXITER,
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "train_time_sec": float(train_time_sec),
    "accuracy": accuracy_score(y_test_raw, y_pred),
    "precision": precision_score(y_test_raw, y_pred, zero_division=0),
    "recall": recall_score(y_test_raw, y_pred, zero_division=0),
    "f1": f1_score(y_test_raw, y_pred, zero_division=0)
}

if y_prob is not None:
    try:
        metrics["roc_auc"] = roc_auc_score(y_test_raw, y_prob)
    except Exception:
        metrics["roc_auc"] = np.nan
else:
    metrics["roc_auc"] = np.nan

print("VQC evaluation result")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_raw, y_pred))

print("\nClassification Report:")
print(classification_report(y_test_raw, y_pred, zero_division=0))


# 10. Save outputs
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(METRICS_OUT, index=False)

pred_df = test_df[["Year"]].copy() if "Year" in test_df.columns else pd.DataFrame(index=np.arange(len(y_test_raw)))
pred_df["y_true"] = y_test_raw
pred_df["y_pred"] = y_pred
if y_prob is not None:
    pred_df["y_prob"] = y_prob
pred_df.to_csv(PRED_OUT, index=False)

log_df = pd.DataFrame(objective_values)
log_df.to_csv(LOG_OUT, index=False)

print("-", METRICS_OUT)
print("-", PRED_OUT)
print("-", LOG_OUT)


# 11. Resource summary
print("Resource summary")
print(f"Qubits used: {n_qubits}")
print(f"Trainable parameters: {num_trainable_params}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Training time (sec): {train_time_sec:.2f}")
print("Feature map: custom AngleEncodingRY")
print(f"Ansatz: RealAmplitudes(reps={ansatz_reps}, entanglement='{ansatz_entanglement}')")
print(f"Optimizer: COBYLA(maxiter={MAXITER})")