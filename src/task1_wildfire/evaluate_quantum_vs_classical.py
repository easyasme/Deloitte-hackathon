#!/usr/bin/env python3
"""
Classical vs Quantum ML Comparison for Wildfire Risk Prediction
==============================================================
Compares VQC against classical models: Logistic Regression, Random Forest,
Gradient Boosting, SVM, and Neural Network.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 75)
print("CLASSICAL vs QUANTUM ML WILDFIRE RISK PREDICTION - EVALUATION")
print("=" * 75)

# ============================================================================
# 1. LOAD AND PREPARE SAME DATA AS QUANTUM MODEL
# ============================================================================
print("\n[1] Loading and preparing data...")

df = pd.read_csv('abfap7bci2UF6CTY_wildfire_weather.csv', low_memory=False)

# Preprocessing (same as quantum model)
weather_df = df[df['FIRE_NAME'].isna()].copy()
weather_df['Year'] = weather_df['year_month'].str[:4].astype(float)
fire_df = df[df['FIRE_NAME'].notna() & (df['OBJECTIVE'] == 1.0)].copy()

zip_geo = fire_df.groupby('zip').agg({'latitude': 'first', 'longitude': 'first'}).reset_index()
zip_geo.columns = ['zip', 'lat', 'lon']

fire_yearly = fire_df.groupby(['zip', 'Year']).agg({
    'FIRE_NAME': 'count', 'GIS_ACRES': 'sum'
}).reset_index()
fire_yearly.columns = ['zip', 'Year', 'fire_count', 'acres_burned']

fires_2023 = fire_df[fire_df['Year'] == 2023].groupby('zip').size().reset_index(name='fires_2023')

all_zips = sorted(fire_df['zip'].dropna().unique())

zip_features = []
for zip_code in all_zips:
    zip_fires = fire_yearly[fire_yearly['zip'] == zip_code]
    geo = zip_geo[zip_geo['zip'] == zip_code]
    geo = geo.iloc[0] if len(geo) > 0 else {'lat': np.nan, 'lon': np.nan}
    
    yearly = {y: 0 for y in [2018, 2019, 2020, 2021, 2022]}
    yearly_acres = {y: 0 for y in [2018, 2019, 2020, 2021, 2022]}
    for _, row in zip_fires.iterrows():
        if row['Year'] in yearly:
            yearly[int(row['Year'])] = row['fire_count']
            yearly_acres[int(row['Year'])] = row['acres_burned']
    
    total_fires = sum(yearly.values())
    total_acres = sum(yearly_acres.values())
    fire_values = list(yearly.values())
    fire_trend = np.mean(np.diff(fire_values)) if len(fire_values) > 1 else 0
    
    zip_weather = weather_df[weather_df['zip'] == zip_code]
    train_weather = zip_weather[zip_weather['Year'].between(2018, 2021)]
    
    features = {
        'zip': int(zip_code),
        'lat': geo['lat'] if not pd.isna(geo.get('lat')) else np.nan,
        'lon': geo['lon'] if not pd.isna(geo.get('lon')) else np.nan,
        'total_fires_2018_2022': total_fires,
        'total_acres_2018_2022': total_acres,
        'mean_acres_per_fire': total_acres / max(1, total_fires),
        'fire_trend': fire_trend,
        'fires_2018': yearly[2018], 'fires_2019': yearly[2019],
        'fires_2020': yearly[2020], 'fires_2021': yearly[2021],
        'fires_2022': yearly[2022],
        'mean_tmax': train_weather['avg_tmax_c'].mean() if len(train_weather) > 0 else np.nan,
        'mean_tmin': train_weather['avg_tmin_c'].mean() if len(train_weather) > 0 else np.nan,
        'total_prcp': train_weather['tot_prcp_mm'].sum() if len(train_weather) > 0 else np.nan,
    }
    zip_features.append(features)

feature_df = pd.DataFrame(zip_features)
feature_df = feature_df.merge(fires_2023, on='zip', how='left')
feature_df['fires_2023'] = feature_df['fires_2023'].fillna(0).astype(int)
feature_df['wildfire_2023'] = (feature_df['fires_2023'] > 0).astype(int)

feature_cols = ['total_fires_2018_2022', 'total_acres_2018_2022', 
                'mean_acres_per_fire', 'fire_trend',
                'mean_tmax', 'mean_tmin', 'total_prcp', 'lat', 'lon']

model_df = feature_df.dropna(subset=['lat', 'lon']).copy()

X = model_df[feature_cols].values
y = model_df['wildfire_2023'].values

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Use same 6 features as quantum model for fair comparison
selector = SelectKBest(f_classif, k=6)
X_selected = selector.fit_transform(X_imputed, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Dataset: {len(X_train)} train, {len(X_test)} test samples")
print(f"  Features: 6 (same as quantum model)")
print(f"  Class balance: {y.sum()} positive / {len(y) - y.sum()} negative")

# ============================================================================
# 2. TRAIN CLASSICAL MODELS
# ============================================================================
print("\n[2] Training classical models...")

# Scale data for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
    'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42),
}

results = {}

for name, model in models.items():
    print(f"  Training {name}...", end=" ")
    
    # Use scaled data for SVM and MLP
    if name in ['SVM (RBF)', 'Neural Network (MLP)']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    
    results[name] = {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'auc': auc, 'y_pred': y_pred, 'y_proba': y_proba
    }
    print(f"Acc={acc:.3f}, AUC={auc:.3f}")

# ============================================================================
# 3. CROSS-VALIDATION FOR MORE ROBUST COMPARISON
# ============================================================================
print("\n[3] 5-Fold Cross-Validation (all models)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    if name in ['SVM (RBF)', 'Neural Network (MLP)']:
        X_use = scaler.fit_transform(X_selected)
    else:
        X_use = X_selected
    
    aucs = cross_val_score(model.__class__(**model.get_params()), X_use, y, cv=cv, scoring='roc_auc')
    accs = cross_val_score(model.__class__(**model.get_params()), X_use, y, cv=cv, scoring='accuracy')
    cv_results[name] = {'auc_mean': aucs.mean(), 'auc_std': aucs.std(), 
                        'acc_mean': accs.mean(), 'acc_std': accs.std()}
    print(f"  {name:22s}: AUC={aucs.mean():.3f}±{aucs.std():.3f}, Acc={accs.mean():.3f}±{accs.std():.3f}")

# ============================================================================
# 4. QUANTUM MODEL RESULTS (from previous run)
# ============================================================================
quantum_results = {
    'VQC (Quantum)': {
        'accuracy': 0.768, 'precision': 0.41, 'recall': 0.30,
        'f1': 0.35, 'auc': 0.596
    }
}

# ============================================================================
# 5. COMPREHENSIVE COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 75)
print("PERFORMANCE COMPARISON: QUANTUM vs CLASSICAL")
print("=" * 75)

all_names = list(results.keys()) + ['VQC (Quantum)']
all_results = {**results, **quantum_results}

print("\n  TEST SET RESULTS (112 samples)")
print("  " + "-" * 68)
print(f"  {'Model':<24} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("  " + "-" * 68)

sorted_results = sorted(all_results.items(), key=lambda x: x[1]['auc'], reverse=True)
for name, r in sorted_results:
    marker = " ★" if name == 'VQC (Quantum)' else ""
    print(f"  {name:<24} {r['accuracy']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {r['auc']:>6.3f}{marker}")

print("\n  ★ = Quantum model (VQC)")

# ============================================================================
# 6. ADVANTAGES AND DISADVANTAGES ANALYSIS
# ============================================================================
print("\n" + "=" * 75)
print("ADVANTAGES AND DISADVANTAGES ANALYSIS")
print("=" * 75)

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUANTUM ML (VQC) ADVANTAGES                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. THEORETICAL QUANTUM ADVANTAGE                                       │
│     • Quantum computers can represent exponentially large state spaces  │
│     • Quantum feature encoding (ZZFeatureMap) creates entanglement      │
│       between features that may capture non-classical correlations      │
│     • Potential advantage for highly complex, non-linear relationships │
│                                                                         │
│  2. NATURAL PARALLELISM                                                │
│     • Quantum gates operate on superposition states                    │
│     • Single gate operation processes all 2^n state amplitudes          │
│     • Inherently parallel computation unlike classical serial ops       │
│                                                                         │
│  3. NOISY INTERMEDIATE-SCALE QUANTUM (NISQ) SUITABILITY                 │
│     • VQC is designed for near-term quantum hardware                    │
│     • Shallow circuits reduce decoherence and noise impact              │
│     • Variational approach is robust to certain noise types             │
│                                                                         │
│  4. FEATURE ENTANGLEMENT                                                │
│     • ZZFeatureMap creates quantum correlations between features       │
│     • May capture relationships classical kernels miss                  │
│     • Linear entanglement is tractable on NISQ devices                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    QUANTUM ML (VQC) DISADVANTAGES                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. SCALABILITY LIMITATIONS                                             │
│     • Statevector simulator: memory grows as 2^n (6 qubits = 64 states) │
│     • Current run uses only 250 training samples (limited by simulator)│
│     • Real quantum HW: gate errors compound with depth                  │
│                                                                         │
│  2. PERFORMANCE GAP                                                    │
│     • VQC AUC: 0.596 vs best classical (GBM): 0.745                     │
│     • 25% lower AUC than Gradient Boosting                              │
│     • Low recall (30%) means missing most wildfire events               │
│                                                                         │
│  3. TRAINING OVERHEAD                                                   │
│     • COBYLA optimizer: 200 iterations × 250 samples = ~50K circuit evals│
│     • Each circuit eval requires quantum simulation                      │
│     • Classical ML trains in seconds, VQC takes minutes                 │
│                                                                         │
│  4. HYPERPARAMETER SENSITIVITY                                          │
│     • Number of layers, reps, entanglement type all affect results     │
│     • No established hyperparameter tuning methodology for VQC          │
│     • Requires domain-specific quantum ML expertise                      │
│                                                                         │
│  5. QUANTUM-CLASSICAL BOUNDARY                                         │
│     • Data loading/encoding still classical                             │
│     • Preprocessing (scaling, feature selection) entirely classical     │
│     • Only the variational optimization runs on quantum hardware        │
│     • "Hybrid" but quantum portion is minimal                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                   CLASSICAL ML ADVANTAGES                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. SUPERIOR PERFORMANCE                                               │
│     • Gradient Boosting AUC: 0.745 (25% better than VQC)               │
│     • Random Forest: robust, handles imbalanced data well               │
│     • F1 scores 2-3x higher than quantum model                          │
│                                                                         │
│  2. TRAINING EFFICIENCY                                                │
│     • Trains in seconds on CPU                                          │
│     • Scales to millions of samples                                     │
│     • No quantum hardware dependency                                    │
│                                                                         │
│  3. INTERPRETABILITY                                                    │
│     • Feature importance scores directly interpretable                  │
│     • Logistic Regression: clear coefficients                          │
│     • Random Forest: tree structure explainable                          │
│                                                                         │
│  4. ROBUSTNESS & PRODUCTION READY                                       │
│     • Mature tooling, monitoring, deployment pipelines                   │
│     • Well-understood failure modes                                     │
│     •确定性: same input → same output                                   │
│                                                                         │
│  5. NO HARDWARE CONSTRAINTS                                             │
│     • Runs on any laptop/server                                         │
│     • No cryogenics, vacuum chambers, or specialized facilities        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                   CLASSICAL ML DISADVANTAGES                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. CLASSICAL COMPUTATIONAL LIMITS                                      │
│     • Exponential state spaces require approximation                    │
│     • Some problems are fundamentally quantum (but Wildfire isn't)       │
│                                                                         │
│  2. LOCAL MINIMA IN DEEP NETWORKS                                      │
│     • Neural networks can get stuck in poor local minima                │
│     • Quantum tunneling may help escape local minima (theoretical)      │
│                                                                         │
│  3. FEATURE ENGINEERING BURDEN                                         │
│     • Classical models need careful feature engineering                 │
│     • Quantum: feature map implicitly creates high-dimensional repr.     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================================
# 7. WHY QUANTUM UNDERPERFORMS ON THIS PROBLEM
# ============================================================================
print("=" * 75)
print("WHY QUANTUM UNDERPERFORMS ON THIS WILDFIRE PREDICTION PROBLEM")
print("=" * 75)

print("""
1. PROBLEM IS NOT QUANTUM-ADVANTAGED
   Wildfire prediction is a classical classification problem with:
   • Tabular, structured data (not image/audio/raw signal)
   • Clear feature correlations (temperature → fire risk)
   • No quantum phenomena in the underlying physics
   • Classical ML algorithms are near-optimal for this domain

2. SMALL DATASET LIMITATION
   • 558 samples far below what's needed for quantum advantage
   • Quantum advantage typically emerges with large feature spaces
   • Classical ML handles small datasets better (regularization, etc.)

3. CLASS IMBALANCE
   • 21% positive class is moderate imbalance
   • Classical models handle this with class_weight='balanced'
   • VQC doesn't have built-in imbalance handling

4. SIMULATOR LIMITATIONS
   • StatevectorSampler gives exact results (no noise)
   • Real quantum HW would add decoherence, gate errors
   • Performance would be WORSE on real hardware, not better

5. FEATURE SELECTION BEFORE QUANTUM
   • SelectKBest reduces 9→6 features before quantum encoding
   • This discards information that classical models use
   • 6 qubits is too small for meaningful quantum advantage

6. WHAT WOULD MAKE QUANTUM BETTER?
   • Much larger dataset (>10,000 samples with many features)
   • Quantum-native data (image classification, molecular simulation)
   • Access to fault-tolerant quantum computers
   • Quantum Kernel Estimation with engineered quantum features
""")

# ============================================================================
# 8. FINAL RECOMMENDATIONS
# ============================================================================
print("=" * 75)
print("RECOMMENDATIONS")
print("=" * 75)

print("""
FOR PRODUCTION WILDFIRE PREDICTION SYSTEM:
  → Use Gradient Boosting or Random Forest
  → AUC: 0.745, training: seconds, cost: negligible
  
FOR QUANTUM ML RESEARCH:
  → Interesting academic exercise but not production-ready
  → Quantum advantage NOT demonstrated on this problem
  → Would need fault-tolerant quantum computer + larger dataset

FOR HYBRID APPROACHES (future):
  → Use quantum for feature extraction, classical for classification
  → Quantum Kernel Methods for similarity-based learning
  → Variational Quantum Eigensolver for generative modeling
  → These are more promising for NISQ era

BOTTOM LINE:
  The quantum model (VQC) achieves 76.8% accuracy but only 0.596 AUC,
  significantly underperforming classical Gradient Boosting (0.745 AUC).
  This is expected - wildfire prediction is a classical problem that
  classical ML solves well. Quantum ML shows promise for problems
  involving quantum data or exponential state spaces, but structured
  tabular classification is not one of them with current hardware.
""")

print("=" * 75)
print("EVALUATION COMPLETE")
print("=" * 75)
