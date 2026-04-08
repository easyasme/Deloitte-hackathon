import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

# STEP 3: Classical Baseline Model Training & Validation
# Input : task1_step2_classical_ready.csv
# Output: metrics + predictions + feature importance

INPUT_FILE = "task1_step2_classical_ready.csv"

# 1. Load
df = pd.read_csv(INPUT_FILE)

print("input data shape:", df.shape)
print("columns:")
print(df.columns.tolist())

# 2. Define target and feature set
# exclude zip, Year
# use year_index
target_col = "fire_occurred"

feature_cols = [
    "avg_tmax_c",
    "avg_tmin_c",
    "tot_prcp_mm",
    "temp_range_c",
    "dryness_proxy",
    "year_index"
]

missing_features = [c for c in feature_cols if c not in df.columns]
if missing_features:
    raise ValueError(f"no necessary feature: {missing_features}")

if target_col not in df.columns:
    raise ValueError(f"no target column: {target_col}")

print("\nfeatures to use:")
print(feature_cols)
print("target:", target_col)


# 3. Time-based split
# train: 2018~2020
# test : 2021
if "Year" not in df.columns:
    raise ValueError("Year column needed for Time-based split")

train_df = df[df["Year"] <= 2020].copy()
test_df = df[df["Year"] == 2021].copy()

print("\ntime-based split")
print("train shape:", train_df.shape)
print("test shape :", test_df.shape)

if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError("Check Year")

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df[target_col].copy()

print("\ntrain target distribution:")
print(y_train.value_counts())

print("\ntest target distribution:")
print(y_test.value_counts())


# 4. Define models
# Logistic Regression: scaling needed
logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# Random Forest: scaling not needed
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

# 5. Train models
logreg_pipeline.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

print("\ntraining done")


# 6. Evaluation helper
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


# 7. Predict: Logistic Regression
logreg_pred = logreg_pipeline.predict(X_test)
logreg_prob = logreg_pipeline.predict_proba(X_test)[:, 1]

logreg_metrics = evaluate_model(
    "Logistic Regression",
    y_test,
    logreg_pred,
    logreg_prob
)


# 8. Predict: Random Forest
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_metrics = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_prob
)


# 9. Compare results
results_df = pd.DataFrame([logreg_metrics, rf_metrics])
print("model comparison table")
print(results_df)


# 10. Random Forest feature importance
rf_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("Random Forest Feature Importance")
print(rf_importance_df)


# 11. Save outputs
results_df.to_csv("task1_step3_model_comparison.csv", index=False)
rf_importance_df.to_csv("task1_step3_rf_feature_importance.csv", index=False)

predictions_df = test_df[["Year"]].copy()
predictions_df["y_true"] = y_test.values
predictions_df["logreg_pred"] = logreg_pred
predictions_df["logreg_prob"] = logreg_prob
predictions_df["rf_pred"] = rf_pred
predictions_df["rf_prob"] = rf_prob
predictions_df.to_csv("task1_step3_test_predictions.csv", index=False)

print("\nsaved:")
print("- task1_step3_model_comparison.csv")
print("- task1_step3_rf_feature_importance.csv")
print("- task1_step3_test_predictions.csv")


# 12. Final note
best_model = results_df.sort_values(by="f1", ascending=False).iloc[0]["model"]

print(f"best baseline: {best_model}")