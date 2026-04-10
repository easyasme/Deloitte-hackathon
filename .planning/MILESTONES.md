# Milestones

## v1.0 — MVP

**Shipped:** 2026-04-10
**Phases:** 5 | **Plans:** 9 | **Requirements:** 24/24

### Accomplishments

1. Built clean panel dataset — 31,343 training rows + 12,708 holdout, temporal integrity verified
2. Engineered 67-column feature matrix with temporal lags, expanding stats, category encoding
3. Established temporal baselines — Linear Regression (RMSE 515k), Random Forest (RMSE 999k)
4. Developed Panel FE + LightGBM ensemble — 80/20 weighted blend, RMSE 493k on validation
5. Generated 2021 predictions with error analysis — RMSE 439k holdout, counterintuitive fire risk finding

### Key Findings

- Fire risk does NOT improve 2021 premium prediction (ablation: +1,460 RMSE delta)
- 90% of top-20 highest-error ZIPs have above-median fire risk
- Panel FE dominates ensemble (80% weight) — structural approach captures most signal
- 40.7% of predictions are negative (technical debt — needs clipping)

### Files

- Roadmap: `.planning/milestones/v1.0-ROADMAP.md`
- Requirements: `.planning/milestones/v1.0-REQUIREMENTS.md`
- Audit: `.planning/v1.0-v1.0-MILESTONE-AUDIT.md`

---
