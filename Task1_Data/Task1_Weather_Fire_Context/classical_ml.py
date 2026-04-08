import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# STEP 3 v2: Classical Baseline Model Training & Validation
#            (Weather + Fire Context)
# Input : task1_step2_v2_classical_ready.csv
# Output:
#   - task1_step3_v2_model_comparison.csv
#   - task1_step3_v2_rf_feature_importance.csv
#   - task1_step3_v2_test_predictions.csv

INPUT_FILE = "task1_step2_v2_classical_ready.csv"

MODEL_COMPARISON_OUT = "task1_step3_v2_model_comparison.csv"
RF_IMPORTANCE_OUT = "task1_step3_v2_rf_feature_importance.csv"
TEST_PRED_OUT = "task1_step3_v2_test_predictions.csv"

print("STEP 3 v2")

# 1. Load
df = pd.read_csv(INPUT_FILE)

print("input data shape:", df.shape)
print("columns:")
print(df.columns.tolist())

# 2. Define target and feature set
target_col = "fire_occurred"

if target_col not in df.columns:
    raise ValueError(f"no target column: {target_col}")

if "Year" not in df.columns:
    raise ValueError("need year column for Time-based split")

# exclude zip, Year, target
exclude_cols = ["zip", "Year", target_col]
feature_cols = [c for c in df.columns if c not in exclude_cols]

if len(feature_cols) == 0:
    raise ValueError("no available features")

print("\nnumber of features to use:", len(feature_cols))
print("features:")
print(feature_cols)

# 3. Convert boolean dummies to int
for col in feature_cols:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

print("\nboolean dummy -> int converted")

# 4. Time-based split
# train: 2018 ~ 2020
# test : 2021
train_df = df[df["Year"] <= 2020].copy()
test_df = df[df["Year"] == 2021].copy()

if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError("train/test no result. Check year value")

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy().astype(int)

X_test = test_df[feature_cols].copy()
y_test = test_df[target_col].copy().astype(int)

print("\nTime series split done")
print("train shape:", X_train.shape)
print("test shape :", X_test.shape)

print("\n[4] train target distribution:")
print(y_train.value_counts())

print("\n[4] test target distribution:")
print(y_test.value_counts())

# 5. Define models
# Logistic Regression: scaling needed
logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ))
])

# Random Forest: scaling not needed
rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\ndefined model")


# 6. Train
logreg_pipeline.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

print("\ndone fitting model")


# 7. Evaluation helper
def evaluate_model(model_name, y_true, y_pred, y_prob):
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

    print(f"{model_name} evaluation result")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return metrics


# 8. Logistic Regression predict/eval
logreg_pred = logreg_pipeline.predict(X_test)
logreg_prob = logreg_pipeline.predict_proba(X_test)[:, 1]

logreg_metrics = evaluate_model(
    "Logistic Regression v2",
    y_test,
    logreg_pred,
    logreg_prob
)


# 9. Random Forest predict/eval
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_metrics = evaluate_model(
    "Random Forest v2",
    y_test,
    rf_pred,
    rf_prob
)


# 10. Compare results
results_df = pd.DataFrame([logreg_metrics, rf_metrics])

print("model comparison")
print(results_df)

results_df.to_csv(MODEL_COMPARISON_OUT, index=False)


# 11. Random Forest feature importance
rf_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("Random Forest Feature Importance (Top 20)")
print(rf_importance_df.head(20))

rf_importance_df.to_csv(RF_IMPORTANCE_OUT, index=False)


# 12. Save test predictions
predictions_df = test_df[["Year"]].copy()
predictions_df["y_true"] = y_test.values
predictions_df["logreg_pred"] = logreg_pred
predictions_df["logreg_prob"] = logreg_prob
predictions_df["rf_pred"] = rf_pred
predictions_df["rf_prob"] = rf_prob

predictions_df.to_csv(TEST_PRED_OUT, index=False)

print("\nsaved")
print("-", MODEL_COMPARISON_OUT)
print("-", RF_IMPORTANCE_OUT)
print("-", TEST_PRED_OUT)


# 13. Summary
best_model = results_df.sort_values(by="f1", ascending=False).iloc[0]["model"]

print(f"F1 best model: {best_model}")
print("CAUSE / OBJECTIVE / AGENCY_ID are context values, so attention needed for interpreting pure forecasting")