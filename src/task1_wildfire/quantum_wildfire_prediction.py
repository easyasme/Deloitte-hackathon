#!/usr/bin/env python3
"""
Quantum Machine Learning Wildfire Risk Prediction for California Zip Codes
============================================================================
Model: Hybrid Quantum-Classical Variational Quantum Classifier (VQC)
Training Data: 2018-2022
Prediction Target: 2023 wildfire occurrence

Wildfire Definition: Fires with OBJECTIVE=1 (unplanned, uncontrolled, wildland)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler

print("=" * 70)
print("QUANTUM WILDFIRE RISK PREDICTION - CALIFORNIA ZIP CODES (2023)")
print("=" * 70)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n[1] Loading and preprocessing data...")

df = pd.read_csv('abfap7bci2UF6CTY_wildfire_weather.csv', low_memory=False)

# Parse Year from year_month for weather records
weather_df = df[df['FIRE_NAME'].isna()].copy()
weather_df['Year'] = weather_df['year_month'].str[:4].astype(float)

# Fire records - only OBJECTIVE=1 (wildland, unplanned, uncontrolled)
fire_df = df[df['FIRE_NAME'].notna() & (df['OBJECTIVE'] == 1.0)].copy()

print(f"  Weather records: {len(weather_df):,}")
print(f"  Wildfire records (OBJECTIVE=1): {len(fire_df):,}")

# Get lat/lon per zip from fire data (most reliable source)
zip_geo = fire_df.groupby('zip').agg({
    'latitude': 'first', 
    'longitude': 'first'
}).reset_index()
zip_geo.columns = ['zip', 'lat', 'lon']
print(f"  Zip codes with geo coordinates: {len(zip_geo):,}")

# ============================================================================
# 2. BUILD ZIP-LEVEL FEATURE DATASET
# ============================================================================
print("\n[2] Building zip-level features (2018-2022)...")

# Fire counts per year per zip
fire_yearly = fire_df.groupby(['zip', 'Year']).agg({
    'FIRE_NAME': 'count',
    'GIS_ACRES': 'sum'
}).reset_index()
fire_yearly.columns = ['zip', 'Year', 'fire_count', 'acres_burned']

# Target: fires in 2023
fires_2023 = fire_df[fire_df['Year'] == 2023].groupby('zip').size().reset_index(name='fires_2023')

# Build features for each zip
all_zips = sorted(fire_df['zip'].dropna().unique())
print(f"  Zip codes with fire history: {len(all_zips):,}")

zip_features = []
for zip_code in all_zips:
    zip_fires = fire_yearly[fire_yearly['zip'] == zip_code]
    geo = zip_geo[zip_geo['zip'] == zip_code].iloc[0] if len(zip_geo[zip_geo['zip'] == zip_code]) > 0 else {'lat': np.nan, 'lon': np.nan}
    
    # Yearly fire counts (training years: 2018-2022)
    yearly = {y: 0 for y in [2018, 2019, 2020, 2021, 2022]}
    yearly_acres = {y: 0 for y in [2018, 2019, 2020, 2021, 2022]}
    for _, row in zip_fires.iterrows():
        if row['Year'] in yearly:
            yearly[int(row['Year'])] = row['fire_count']
            yearly_acres[int(row['Year'])] = row['acres_burned']
    
    total_fires = sum(yearly.values())
    total_acres = sum(yearly_acres.values())
    
    # Fire trend: positive means increasing
    fire_values = list(yearly.values())
    fire_trend = np.mean(np.diff(fire_values)) if len(fire_values) > 1 else 0
    
    # Weather aggregated (use station as proxy - we use aggregate stats per year)
    zip_weather = weather_df[weather_df['zip'] == zip_code]
    train_weather = zip_weather[zip_weather['Year'].between(2018, 2021)]
    
    features = {
        'zip': int(zip_code),
        'lat': geo['lat'] if 'lat' in geo else np.nan,
        'lon': geo['lon'] if 'lon' in geo else np.nan,
        # Fire history
        'total_fires_2018_2022': total_fires,
        'total_acres_2018_2022': total_acres,
        'mean_acres_per_fire': total_acres / max(1, total_fires),
        'fire_trend': fire_trend,
        'fires_2018': yearly[2018], 'fires_2019': yearly[2019],
        'fires_2020': yearly[2020], 'fires_2021': yearly[2021],
        'fires_2022': yearly[2022],
        # Weather (available 2018-2021)
        'mean_tmax': train_weather['avg_tmax_c'].mean() if len(train_weather) > 0 else np.nan,
        'mean_tmin': train_weather['avg_tmin_c'].mean() if len(train_weather) > 0 else np.nan,
        'total_prcp': train_weather['tot_prcp_mm'].sum() if len(train_weather) > 0 else np.nan,
    }
    zip_features.append(features)

feature_df = pd.DataFrame(zip_features)
print(f"  Zips with fire features: {len(feature_df):,}")

# Merge 2023 target
feature_df = feature_df.merge(fires_2023, on='zip', how='left')
feature_df['fires_2023'] = feature_df['fires_2023'].fillna(0).astype(int)
feature_df['wildfire_2023'] = (feature_df['fires_2023'] > 0).astype(int)

n_pos = feature_df['wildfire_2023'].sum()
print(f"  Zips with wildfire in 2023: {n_pos} ({100*n_pos/len(feature_df):.1f}%)")
print(f"  Zips without wildfire in 2023: {len(feature_df) - n_pos}")

# ============================================================================
# 3. PREPARE DATA FOR QUANTUM MODEL
# ============================================================================
print("\n[3] Preparing data for quantum model...")

feature_cols = ['total_fires_2018_2022', 'total_acres_2018_2022', 
                'mean_acres_per_fire', 'fire_trend',
                'mean_tmax', 'mean_tmin', 'total_prcp', 'lat', 'lon']

# Drop rows without geo (needed for quantum encoding)
model_df = feature_df.dropna(subset=['lat', 'lon']).copy()
print(f"  Zips with geo: {len(model_df):,}")

# Features and target
X = model_df[feature_cols].values
y = model_df['wildfire_2023'].values

# Impute remaining NaN
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Scale to [0, pi] for quantum encoding
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_imputed)

# Feature selection
k = min(6, X_scaled.shape[1])
selector = SelectKBest(f_classif, k=k)
X_selected = selector.fit_transform(X_scaled, y)
selected_mask = selector.get_support()
selected_features = [f for f, m in zip(feature_cols, selected_mask) if m]
print(f"  Selected features: {selected_features}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training: {len(X_train):,} (pos: {y_train.sum()})")
print(f"  Test: {len(X_test):,} (pos: {y_test.sum()})")

# ============================================================================
# 4. BUILD QUANTUM MODEL
# ============================================================================
print("\n[4] Building Variational Quantum Classifier...")

n_features = X_train.shape[1]
n_qubits = n_features
n_layers = 2

print(f"  Qubits: {n_qubits}")
print(f"  Layers: {n_layers}")

# Feature map: ZZFeatureMap - encodes classical features into quantum state
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')

# Variational ansatz: RealAmplitudes - parameterized circuit for learning
var_form = RealAmplitudes(num_qubits=n_qubits, reps=n_layers, entanglement='linear')

print(f"  Feature map: ZZFeatureMap (reps=2, linear)")
print(f"  Ansatz: RealAmplitudes (reps={n_layers})")
print(f"  Circuit depth: {var_form.decompose().depth()}")
print(f"  Parameters: {var_form.num_parameters}")

# ============================================================================
# 5. TRAIN VQC
# ============================================================================
print("\n[5] Training VQC on quantum simulator...")

sampler = StatevectorSampler()
optimizer = COBYLA(maxiter=200, tol=0.01)

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=var_form,
    optimizer=optimizer
)

# Sample subset for training (simulator scales exponentially with qubits)
np.random.seed(42)
n_train = min(250, len(X_train))
idx = np.random.choice(len(X_train), n_train, replace=False)
X_train_sub = X_train[idx]
y_train_sub = y_train[idx]

print(f"  Training on {n_train} samples...")
print(f"  Optimizer: COBYLA (max 200 iterations)")

try:
    result = vqc.fit(X_train_sub, y_train_sub)
    print("  ✓ VQC Training completed!")
except Exception as e:
    print(f"  ⚠ Training error: {e}")
    result = None

# ============================================================================
# 6. EVALUATE
# ============================================================================
print("\n[6] Model Evaluation...")

if result is not None:
    y_pred = vqc.predict(X_test)
    
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Fire', 'Wildfire']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"              Predicted")
    print(f"             No Fire  Wildfire")
    print(f"Actual No Fire  {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"Actual Wildfire {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {acc:.4f}")
    try:
        auc = roc_auc_score(y_test, y_pred)
        print(f"  ROC-AUC:  {auc:.4f}")
    except:
        pass
else:
    print("  ⚠ Could not evaluate model")

# ============================================================================
# 7. RESOURCE REQUIREMENTS
# ============================================================================
ansatz_depth = var_form.decompose().depth()
n_params = var_form.num_parameters

print("\n" + "=" * 70)
print("QUANTUM RESOURCE REQUIREMENTS")
print("=" * 70)
print(f"""
┌────────────────────────────────────────────────────────────────────────┐
│                     RESOURCE ESTIMATION                                 │
├────────────────────────────────────────────────────────────────────────┤
│  QUBITS: {n_qubits} (minimum 1 per feature dimension)                          │
│                                                                         │
│  CIRCUIT DEPTH: {ansatz_depth} layers                                            │
│                                                                         │
│  GATES PER INFERENCE:                                                  │
│    • RZ (rotation):      ~{n_qubits * 4}                                              │
│    • RX (rotation):      ~{n_qubits * 4}                                              │
│    • CZ (entanglement):  ~{n_qubits * (n_qubits-1) // 2}                                           │
│    • CNOT (entanglement): ~{n_qubits * n_layers}                                              │
│    • Total:              ~{n_qubits * ansatz_depth}                                                   │
│                                                                         │
│  CLASSICAL PARAMETERS: {n_params} (optimized during training)                  │
│                                                                         │
│  MEMORY (Statevector Simulator):                                        │
│    • Statevector size: 2^{n_qubits} = {2**n_qubits:,} complex numbers              │
│    • Memory: {16 * 2**n_qubits / 1e6:.2f} MB                                               │
│    • With noise mitigation: ~{4 * 16 * 2**n_qubits / 1e6:.2f} MB                                 │
│                                                                         │
│  RUNTIME COMPARISON:                                                    │
│    • Local simulator:     ~2-5 minutes                                  │
│    • IBM Cloud simulator: ~5-15 minutes                                 │
│    • Real quantum hardware: ~15-45 minutes (queue dependent)            │
│                                                                         │
│  SCALING TO REAL HW:                                                    │
│    • Current model needs {n_qubits} qubits (well within current capability)      │
│    • For all ~2500 CA zips: batch prediction required                   │
│    • Alternative: Quantum Kernel Estimation for better scaling          │
└────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================
print("[7] Top Predictive Features...")
scores = list(zip(selected_features, selector.scores_[selected_mask]))
scores.sort(key=lambda x: x[1], reverse=True)
print("\n  Feature              F-Score   Importance")
print("  " + "-" * 40)
for feat, score in scores:
    bar = "█" * min(int(score / 2), 30)
    print(f"  {feat:20s} {score:6.1f}  {bar}")

# ============================================================================
# 9. PREDICTIONS
# ============================================================================
print("\n[8] Top 15 Highest Risk Zip Codes (2023 Prediction)...")

if result is not None:
    model_df['risk_score'] = vqc.predict(X_selected)
    top_risk = model_df.nlargest(15, 'risk_score')[
        ['zip', 'risk_score', 'wildfire_2023', 'total_fires_2018_2022', 'lat', 'lon']
    ]
    
    print("\n  Rank | Zip      | Score | 2023 | Fires(18-22) | Lat     | Lon")
    print("  " + "-" * 70)
    for i, (_, row) in enumerate(top_risk.iterrows(), 1):
        actual = "✓" if row['wildfire_2023'] else "✗"
        print(f"  {i:4d} | {int(row['zip']):8d} | {int(row['risk_score']):5d} | {actual}    | {int(row['total_fires_2018_2022']):11d} | {row['lat']:.4f} | {row['lon']:.4f}")

# ============================================================================
# 10. SAVE
# ============================================================================
print("\n[9] Saving results...")

output = model_df[['zip', 'risk_score', 'wildfire_2023', 'fires_2023', 
                    'total_fires_2018_2022', 'lat', 'lon']].copy()
output = output.sort_values('risk_score', ascending=False)
output.to_csv('wildfire_risk_predictions_2023.csv', index=False)
print("  ✓ Saved: wildfire_risk_predictions_2023.csv")

with open('quantum_model_summary.txt', 'w') as f:
    f.write("QUANTUM WILDFIRE RISK PREDICTION MODEL\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Model: Variational Quantum Classifier (VQC)\n")
    f.write(f"Qubits: {n_qubits}\n")
    f.write(f"Circuit depth: {ansatz_depth}\n")
    f.write(f"Training samples: {n_train}\n")
    f.write(f"Test accuracy: {acc if result else 'N/A'}\n\n")
    f.write(f"Selected features:\n")
    for feat in selected_features:
        f.write(f"  - {feat}\n")
    f.write(f"\nTop risk zips:\n")
    if result:
        for i, (_, row) in enumerate(top_risk.head(10).iterrows(), 1):
            f.write(f"  {i}. {int(row['zip'])} (score: {int(row['risk_score'])})\n")
print("  ✓ Saved: quantum_model_summary.txt")

print("\n" + "=" * 70)
print("QUANTUM ML PIPELINE COMPLETE")
print("=" * 70)
