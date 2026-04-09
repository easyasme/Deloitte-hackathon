---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 5
status: "Phase 4 shipped — PR #1"
last_updated: "2026-04-09T23:17:10.156Z"
progress:
  total_phases: 5
  completed_phases: 3
  total_plans: 6
  completed_plans: 4
  percent: 67
---

# State: Insurance Premium Time Series Prediction

**Project:** Insurance Premium Time Series Prediction
**Core Value:** Predict 2021 insurance premiums accurately by ZIP code using historical trends and wildfire risk
**Current Phase:** 5

## Current Position

Phase: 04 (model-development-ensemble) — EXECUTING
Plan: Not started
| Field | Value |
|-------|-------|
| Current Phase | Roadmap created |
| Current Plan | None (phase not started) |
| Status | Not started |
| Progress | 0% |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Requirements mapped | 24/24 |
| Phases defined | 5 |
| Plans complete | 0/5 |

## Accumulated Context

### Key Decisions

- **Stack:** Python 3.11+, pandas, statsmodels, LightGBM, scikit-learn
- **Architecture:** Global Model with Entity Features (pool across ~1600 ZIPs, not per-ZIP models)
- **Approach:** Ensemble of Panel Fixed Effects (statsmodels) + LightGBM
- **Validation:** Temporal split (train 2018-2019, validate 2020) — NO random split
- **Target:** Earned Premium for 2021
- **Key predictor:** Avg Fire Risk Score from dataset
- **COVID-19:** 2020 flagged as structural break — needs explicit handling

### Critical Pitfalls (from research)

1. Deep learning on 3-year panel — insufficient data for LSTM/Transformer
2. Ignoring exposure as volume normalizer — always include Earned Exposure
3. Feature leakage through current-year data — all concurrent features must be lagged
4. Overfitting to ZIP granularity — use global models with categorical support
5. COVID-19 structural break in 2020 — flag explicitly, run sensitivity analysis

### Research Flags

- **Phase 2:** Zero-inflated claims handling — may need Tweedie GLM or two-part model
- **Phase 4:** Ensemble weighting strategy — needs empirical validation

### Deferred to v2

- Tweedie GLM for zero-inflated premium distribution
- Spatial clustering of neighboring ZIP codes
- Hierarchical/mixed-effects model

## Phase History

| Phase | Started | Completed | Plans |
|-------|---------|-----------|-------|
| 1. Data Foundation | 2026-04-09 | 2026-04-09 | 1 |
| 2. Feature Engineering | 2026-04-09 | 2026-04-09 | 1 |
| 3. Baseline Models + Temporal Validation | 2026-04-09 | 2026-04-09 | 1 |
| 4. Model Development + Ensemble | 2026-04-09 | - | 3 |

## Session Continuity

- Roadmap created: 2026-04-09
- All files in `.planning/` directory
- Next action: `/gsd-plan-phase 1` to start Phase 1 planning
