---
phase: "05-2021-prediction-error-analysis"
plan: "03"
subsystem: "documentation"
tags:
  - "documentation"
  - "fire-risk-ablation"
  - "model-comparison"
  - "task1-comparison"
dependency_graph:
  requires:
    - "05-01 (2021 predictions)"
    - "04-03 (ensemble model)"
  provides:
    - "DOC-01 (model approach documented)"
    - "DOC-02 (performance metrics reported)"
    - "DOC-03 (Task 1 comparison documented)"
    - "PRED-04 (fire risk value-add analyzed)"
tech_stack:
  added:
    - "linearmodels PanelOLS"
    - "LightGBM"
  patterns:
    - "Ablation study: train identical models with/without fire risk features"
    - "Weighted ensemble: 80% Panel FE + 20% LightGBM"
key_files:
  created:
    - "Task2_Data/task2_step5_fire_risk_ablation.py (ablation script)"
    - "Task2_Data/task2_step5_fire_risk_ablation.csv (ablation metrics)"
    - "Task2_Data/task2_step5_documentation.md (6-section comprehensive doc)"
    - "Task2_Data/task2_step5_predictions_no_fire.csv (2021 predictions without fire risk)"
decisions:
  - "Fire risk does NOT improve 2021 premium prediction (RMSE delta=+1,460 with fire risk)"
  - "Task 1 (quantum VQC) and Task 2 (classical ensemble) are complementary: Task 1 classifies fire risk, Task 2 uses it as feature for premium prediction"
metrics:
  duration_minutes: 5
  completed_date: "2026-04-10"
---

# Phase 05 Plan 03: Documentation, Fire Risk Ablation, and Task 1 Comparison Summary

## Objective

Document the model approach, hyperparameters, and assumptions; analyze fire risk value-add via ablation; compare with Task 1 quantum/classical approaches.

## What Was Built

### Fire Risk Ablation Analysis

Trained identical ensemble models (80% Panel FE + 20% LightGBM) with and without fire risk features on 2021 holdout.

**Ablation Results:**

| Metric | With Fire Risk | Without Fire Risk | Delta    |
|--------|----------------|-------------------|----------|
| RMSE   | 439,352        | 437,892           | +1,460   |
| MAE    | 180,125        | 177,595           | +2,530   |
| MAPE   | 10,511.04%     | 10,440.63%        | +70.41%  |

**Finding:** Fire risk features do NOT improve 2021 premium prediction. RMSE is slightly higher (worse) with fire risk features (+1,460 delta). This may be because fire risk is already captured indirectly through other features, or because 2021 was not a high-fire year.

### Comprehensive Documentation

Created `task2_step5_documentation.md` with 6 sections:

1. **Model Approach**: Ensemble architecture (80% Panel FE + 20% LightGBM), Pooled OLS with robust SE, LightGBM with early stopping
2. **Hyperparameters**: Full tables for Panel FE, LightGBM, and ensemble weights
3. **Key Assumptions**: Temporal lag integrity, COVID handling, expanding window validity, no 2021 leakage, ZIP-level pooling
4. **Performance Metrics**: 2020 validation metrics (from Phase 4) and 2021 holdout test metrics
5. **Fire Risk Value-Add**: Full ablation methodology and findings
6. **Comparison with Task 1**: Quantum VQC (classification) vs Classical Ensemble (regression), complementary value, computational cost comparison

### Task 1 Comparison

| Aspect | Task 1 (Quantum VQC) | Task 2 (Classical Ensemble) |
|--------|---------------------|----------------------------|
| Method | Quantum VQC | Panel FE + LightGBM |
| Problem | Classification | Regression |
| Target | Fire risk category | Earned Premium ($) |
| Fire risk role | Output | Input feature |
| Scalability | Limited by qubits | Unlimited |

Task 1 and Task 2 are complementary: Task 1 provides quantum-enhanced fire risk classification, Task 2 uses fire risk score as input for premium prediction.

## Key Files Created

| File | Description |
|------|-------------|
| `Task2_Data/task2_step5_fire_risk_ablation.py` | Ablation script (trains ensemble without fire risk) |
| `Task2_Data/task2_step5_fire_risk_ablation.csv` | Ablation metrics (3 rows: RMSE, MAE, MAPE) |
| `Task2_Data/task2_step5_documentation.md` | 6-section comprehensive model documentation |
| `Task2_Data/task2_step5_predictions_no_fire.csv` | 2021 predictions without fire risk (12,708 rows) |

## Deviations from Plan

None - plan executed as written.

## Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| PRED-04 (fire risk value-add) | DONE | Ablation delta=+1,460 RMSE |
| DOC-01 (model approach/hyperparams) | DONE | Section 1-2 of documentation |
| DOC-02 (performance metrics) | DONE | Section 4 of documentation |
| DOC-03 (Task 1 comparison) | DONE | Section 6 of documentation |

## Threat Flags

None - documentation and analysis only, no new network endpoints or auth paths.

## Self-Check

- [x] `task2_step5_fire_risk_ablation.csv` exists with 3 metric rows (RMSE, MAE, MAPE)
- [x] `task2_step5_documentation.md` exists with all 6 sections
- [x] Ablation finding printed: "Fire risk DOES NOT improve 2021 prediction (RMSE delta = +1,460)"
- [x] Task 1 comparison section is substantive (quantum vs classical tradeoffs documented)
- [x] Commits made for Task 1 and Task 2

## Self-Check: PASSED
