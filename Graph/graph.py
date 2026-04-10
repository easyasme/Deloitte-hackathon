import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# =========================================================
# FILE PATHS
# =========================================================
TASK1_CLASSICAL_PRED_FILE = "task1_step3_test_predictions.csv"
TASK1_VQC_PRED_FILE = "task1_step5_vqc_predictions.csv"
TASK2_IMPORTANCE_FILE = "task2_step4_lightgbm_importance.csv"

# =========================================================
# FIGURE 1: TASK 1 ROC CURVE (NO RANDOM BASELINE)
# =========================================================
df_classical = pd.read_csv(TASK1_CLASSICAL_PRED_FILE)
df_vqc = pd.read_csv(TASK1_VQC_PRED_FILE)

plt.figure(figsize=(8.5, 6.5))

y_true_classical = df_classical["y_true"]

# Logistic Regression
if "logreg_prob" in df_classical.columns:
    fpr_logreg, tpr_logreg, _ = roc_curve(y_true_classical, df_classical["logreg_prob"])
    auc_logreg = auc(fpr_logreg, tpr_logreg)
    plt.plot(fpr_logreg, tpr_logreg, linewidth=2.5,
             label=f"Logistic Regression (AUC = {auc_logreg:.3f})")

# Random Forest
if "rf_prob" in df_classical.columns:
    fpr_rf, tpr_rf, _ = roc_curve(y_true_classical, df_classical["rf_prob"])
    auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, linewidth=2.5,
             label=f"Random Forest (AUC = {auc_rf:.3f})")

# VQC
fpr_vqc, tpr_vqc, _ = roc_curve(df_vqc["y_true"], df_vqc["y_prob"])
auc_vqc = auc(fpr_vqc, tpr_vqc)
plt.plot(fpr_vqc, tpr_vqc, linewidth=2.5,
         label=f"VQC Quantum (AUC = {auc_vqc:.3f})")

plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Task 1 Model Comparison: ROC Curve", fontsize=15)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figure1_task1_roc.png", dpi=300)
plt.show()

print("Saved: figure1_task1_roc.png")


# =========================================================
# FIGURE 2: TASK 2 FEATURE IMPORTANCE (NORMALIZED)
# =========================================================
df_imp = pd.read_csv(TASK2_IMPORTANCE_FILE)

df_imp["importance_pct"] = df_imp["importance"] / df_imp["importance"].sum() * 100
df_imp = df_imp.sort_values("importance_pct", ascending=False)

df_top = df_imp.head(10).copy()
df_top = df_top.iloc[::-1]

def get_color(feature):
    name = str(feature).lower()
    if "exposure" in name:
        return "#4C78A8"
    elif "fire" in name or "risk" in name:
        return "#86BC25"
    else:
        return "gray"

colors = [get_color(f) for f in df_top["feature"]]

plt.figure(figsize=(10, 6.5))
bars = plt.barh(df_top["feature"], df_top["importance_pct"], color=colors)

plt.xlabel("Normalized Importance (%)", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Task 2 Feature Importance (LightGBM, Normalized)", fontsize=15)

for bar, val in zip(bars, df_top["importance_pct"]):
    plt.text(bar.get_width() + 0.2,
             bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}%",
             va="center", fontsize=10)

plt.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig("figure2_feature_importance.png", dpi=300)
plt.show()

print("Saved: figure2_feature_importance.png")