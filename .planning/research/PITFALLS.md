# Domain Pitfalls: Insurance Premium Time Series Forecasting

**Domain:** Insurance Premium Time Series Prediction
**Project Context:** Predict 2021 insurance premiums for California zip codes using 2018-2020 data (3-year window)
**Researched:** 2026-04-09
**Confidence:** MEDIUM

## Pitfall 1: Insufficient Time Series Length for Deep Models

**What goes wrong:** Applying LSTM, deep RNN, or transformer-based forecasting models with only 3 time points per zip code (2018, 2019, 2020). These models require substantially longer sequences to learn meaningful temporal patterns and risk severe underfitting or complete failure to converge.

**Why it happens:** Deep time series models are designed for hundreds or thousands of observations. With 3 observations per zip code, there is no viable way to split into train/validation/test while retaining enough history for the model to initialize. The effective degrees of freedom far exceed the available data points.

**Consequences:**
- Model fails to learn any generalizable patterns
- Predictions degenerate to flat lines or random noise
- Internal validation metrics appear acceptable but out-of-sample performance is terrible
- Wasted development time on architecture that cannot work

**Prevention:** Recognize that 3 years of data constrains you to methods that can generalize from minimal temporal context. Viable approaches include:
- Gradient boosting with lag features (XGBoost, LightGBM) treating time as a supervised learning problem
- Classical GLM (Generalized Linear Models) with year as a feature
- Prophet with weak seasonality priors
- Simple AR-type models fitted per-zip with heavy regularization

**Warning Signs:**
- Model training loss does not converge or shows oscillation
- Validation loss increases while training loss decreases (classic overfitting signature)
- Predictions are identical across all zip codes or follow a random walk
- Cross-validation splits produce wildly different results

**Phase:** Model Selection / Architecture phase

---

## Pitfall 2: Ignoring Exposure as the Primary Volume Driver

**What goes wrong:** Building a model that predicts premium levels directly without accounting for Earned Exposure as a normalizing factor. This causes the model to learn the mechanics of policy count rather than pure premium rate dynamics, leading to poor predictions when exposure changes.

**Why it happens:** Earned Premium = Earned Exposure x Premium Rate. If exposure (policy volume) is increasing or decreasing across the three years, a raw premium model will track exposure changes rather than pricing dynamics. This conflates business growth with rate adequacy.

**Consequences:**
- Model predictions reflect exposure trends, not risk-based pricing
- Premium predictions are inflated/deflated based on policy count, not actual risk
- Cannot separate the effect of fire risk from the effect of market conditions (more policies written = more premium)

**Prevention:**
- Always include Earned Exposure as a feature or model the premium-to-exposure ratio
- Build separate models for frequency (claims/exposure) and severity (loss/exposure) and combine
- Log-transform premium and include exposure as offset term in GLM framework

**Warning Signs:**
- Premium predictions scale linearly with exposure changes
- Feature importance shows exposure dominating over risk scores
- Residual analysis shows heteroscedasticity correlated with exposure volume

**Phase:** Feature Engineering / Data Preprocessing

---

## Pitfall 3: Overlooking Zero-Inflation in Claims Data

**What goes wrong:** Treating CAT and Non-CAT claims columns as continuous distributions when they are mostly zeros with occasional large losses (wildfire events). Standard regression models assume a smooth distribution and will produce poor fits for zip codes with zero claims most years.

**Why it happens:** Most zip codes in any given year have zero catastrophe claims. The distribution is heavily zero-inflated with a long right tail of catastrophic losses. Gaussian-assumption models assign high probability to intermediate values that never occur and low probability to the actual zero outcome.

**Consequences:**
- Model predictions underestimate zero outcomes (insurance companies want to know when exposure has NO risk)
- Catastrophic loss predictions are pulled toward the mean
- Out-of-sample predictions for high-risk areas with zero historical claims are underconfident

**Prevention:**
- Use two-part hurdle models or zero-inflated distributions (Tweedie distribution is standard in insurance)
- Separate the classification problem (was there a claim?) from the severity problem (how large?)
- Use Tweedie GLM which natively handles the zero-inflated + continuous mixture

**Warning Signs:**
- Residuals show mass at zero that model cannot replicate
- Binned residual plots reveal systematic underprediction for low-claims segments
- Q-Q plots show severe deviation from normality with heavy right tail

**Phase:** Model Selection / Feature Engineering

---

## Pitfall 4: Treating Zip Codes as Independent Time Series

**What goes wrong:** Fitting separate models per zip code, or assuming zip code-level predictions are uncorrelated, when in reality wildfire risk and premium dynamics are highly spatially correlated across California.

**Why it happens:** Adjacent zip codes share weather patterns, fire risk, demographics, and insurer market presence. A wildfire that devastates one area typically affects neighboring zip codes. If one zip has sparse data (e.g., low exposure), its neighbors provide strong prior information about likely outcomes.

**Consequences:**
- Predictions for low-exposure zip codes are erratic and unreliable
- Model cannot borrow statistical strength across geographies
- High-variance estimates that do not smooth spatially
- Missing systemic risk: model may predict well for individual zips but be wrong overall

**Prevention:**
- Use hierarchical or mixed-effects models with region-level random effects
- Cluster zip codes into broader regions (e.g., by county or fire protection district) and fit region-level models
- Include spatial features (latitude, longitude, shared climate zone) to allow cross-zip correlation
- Post-process predictions with spatial smoothing or geostatistical constraints

**Warning Signs:**
- Predictions vary wildly between adjacent zip codes
- Low-exposure zip codes show high prediction variance
- Leave-one-zip-out cross-validation shows correlated errors with neighbor zip codes

**Phase:** Feature Engineering / Spatial Modeling

---

## Pitfall 5: Ignoring the COVID-19 Structural Break in 2020

**What goes wrong:** Treating 2020 as a normal year in the time series. COVID-19 fundamentally altered insurance dynamics through reduced driving (auto), reduced commercial activity, and altered housing markets. For wildfire insurance specifically, 2020 was an exceptional fire year in California (August Complex Fire, etc.).

**Why it happens:** 2020 is an outlier year. Premiums and claims were affected by:
- Stay-at-home orders reducing some exposures
- Exceptional wildfire activity in California (record burns)
- Economic downturn affecting coverage amounts and policy cancellations
- Insurer pullback from high-risk areas

**Consequences:**
- Model learns from a 2020 signal that will not repeat in 2021
- Overweights an anomalous year when fitting trend
- May predict continuation of unusual 2020 patterns into 2021
- Creates artificial confidence in predictions that is unjustified

**Prevention:**
- Explicitly flag 2020 as an anomaly year in feature engineering
- Include a binary COVID indicator variable
- Run sensitivity analysis with and without 2020 data
- Use weighted approaches that down-weight anomalous observations
- Consider using only 2018-2019 for trend estimation, then overlay 2020 risk signals separately

**Warning Signs:**
- 2020 data is statistically an outlier in EDA (large residuals, high leverage)
- Model predictions are heavily influenced by a single year's pattern
- Cross-validation shows 2020 holdout degrades predictions significantly

**Phase:** Data Preprocessing / Feature Engineering

---

## Pitfall 6: Using Avg Fire Risk Score Without Understanding Its Construction

**What goes wrong:** Treating the dataset's `Avg Fire Risk Score` as an exogenous variable without understanding what it measures and how it was constructed. Fire risk scores themselves may be endogenous to the premium/claims being modeled (risk classification feeds into pricing).

**Why it happens:** Insurance risk scores are often derived from the same historical loss data used as the target. If the score incorporates prior claims, it is not an independent feature -- it is a lagging indicator. Also, the score may reflect risk mitigation actions already taken (e.g., defensible space enforcement) that affect future outcomes.

**Consequences:**
- Data leakage: model uses information that would not be available at prediction time
- Circular reasoning: high fire risk causes high premium causes high fire risk
- Predictions for 2021 use a score that already reflects 2020's losses
- Model cannot be used prospectively (what if fire risk changes?)

**Prevention:**
- Investigate how `Avg Fire Risk Score` was constructed (from features description, it appears to be an ISO risk metric)
- Use the score as-is only if confirmed to be exogenous; otherwise treat it as a derived/lagged target
- If using Task 1 quantum wildfire predictions, ensure those are truly independent inputs
- Consider modeling claims/losses directly from underlying drivers rather than through precomputed risk scores

**Warning Signs:**
- Fire risk score is perfectly correlated with historical claims in training data
- Feature importance analysis shows fire risk score dominates but coefficients are counterintuitive
- Removing fire risk score changes predictions in ways that suggest it is a proxy for target

**Phase:** Feature Engineering / Data Validation

---

## Pitfall 7: Feature Leakage Through Future Information

**What goes wrong:** Including features in the 2021 prediction that would not be available at prediction time. This includes concurrent loss figures, final year exposure counts, or features that require end-of-year reporting.

**Why it happens:** In a real deployment, predictions for 2021 must be made before 2021 outcomes are known. However, the dataset may present data in a consolidated form where features like "CAT Cov A Fire - Incurred Losses" are available for all years simultaneously. Using these as direct features means the model has access to information that would only be known after the fact.

**Consequences:**
- In-sample performance looks excellent but out-of-sample is much worse
- Model cannot be deployed as a prospective tool
- Regulatory non-compliance if used for actual insurance pricing

**Prevention:**
- Create a strict temporal feature list: features available at the START of each year vs. features only known at year's end
- Lag all concurrent features by at least one year
- For 2021 prediction, only use features that could be known as of 2020-12-31
- Build separate "nowcast" vs. "forecast" feature sets

**Warning Signs:**
- Model performance degrades dramatically when evaluated with a proper temporal holdout
- Cross-validation results look too good to be true
- Features include "number of claims" for the prediction year itself

**Phase:** Feature Engineering / Validation

---

## Pitfall 8: Overfitting to Zip Code Granularity

**What goes wrong:** Building a model with enough degrees of freedom to fit per-zip coefficients, leading to a model that essentially memorizes historical premiums instead of learning generalizable relationships between risk factors and pricing.

**Why it happens:** With thousands of zip codes and only 3 years, naive approaches may try to fit zip-specific intercepts or slopes. This is effectively 3 data points per zip, which cannot support any per-zip parameters.

**Consequences:**
- Near-perfect in-sample fit that completely fails out-of-sample
- Zip code predictions that are just the historical average (memorization)
- No ability to predict for new zip codes or scenarios
- Model is non-transferable and non-auditable

**Prevention:**
- Constrain model complexity: prefer global models with zip code as a feature rather than per-zip models
- Use strong regularization (L1/L2 penalties, dropout equivalent in tree models)
- Hierarchical shrinkage: pool zip-level estimates toward regional/global means
- Reserve most degrees of freedom for the primary relationship (fire risk -> premium)

**Warning Signs:**
- Number of model parameters exceeds total data points
- Per-zip coefficients have extreme values (some very large/small)
- Prediction confidence intervals are implausibly narrow

**Phase:** Model Selection / Regularization

---

## Pitfall 9: Ignoring Premium Clustering by Category Type

**What goes wrong:** Aggregating all policy categories (Commercial Occupancy, Dwelling Owner, Homeowner, etc.) into one model when they have fundamentally different pricing mechanics, loss ratios, and exposure trends.

**Why it happens:** The dataset includes `Category` with values CO, DO, DT, HO, MH, RT. Each category has distinct:
- Premium per policy ranges (commercial vs. homeowner)
- Claims frequency and severity patterns
- Exposure trends (HO might grow while CO shrinks)
- Fire risk sensitivity

**Consequences:**
- One category dominates the premium aggregate due to scale, masking other patterns
- Model averages across categories, failing to capture distinct dynamics
- Commercial property pricing appears stable while homeowner pricing is volatile (or vice versa)
- Category-specific risk signals are diluted

**Prevention:**
- Build separate models per category or include category as a strong interaction feature
- If single model, use one-hot encoded category with interactions to fire risk
- Stratify evaluation metrics by category to ensure balanced performance
- Analyze premium per exposure (rate) rather than total premium to normalize for category mix

**Warning Signs:**
- One category dominates >80% of total premium
- Residuals show category-specific patterns
- Feature importance changes dramatically when certain categories are excluded

**Phase:** Model Selection / Stratified Evaluation

---

## Pitfall 10: Selecting Model Based on In-Sample Metrics

**What goes wrong:** Choosing the best-performing model based on training set RMSE/MAE/MAPE, without recognizing that these metrics are misleading with only 3 time points. Models with more parameters will always fit better in-sample.

**Why it happens:** With N=3 per series, in-sample metrics are meaningless for model comparison. A polynomial of degree 2 fits exactly through 3 points. A model with per-zip intercepts and slopes has more parameters than data points and still achieves zero training error.

**Consequences:**
- Selected model is the most overfit, not the best
- Out-of-sample performance is catastrophically worse than in-sample
- Model comparison appears to show one approach is superior when actually all are wrong

**Prevention:**
- Use external validation: hold out one year (2020) entirely and compare models on their 2020 predictions
- Use cross-validation adapted for time series (leave-one-year-out or rolling origin)
- Evaluate using business-relevant metrics: directional accuracy of premium change, not just magnitude
- Apply correction factors: penalized in-sample metrics that account for degrees of freedom
- Prefer simpler models: if ARIMA(1,0,0) and ARIMA(3,1,2) both fit, prefer the simpler unless significant improvement is demonstrated

**Warning Signs:**
- All candidate models show "good" in-sample metrics
- Adding more features always improves training metrics (sign of overfitting)
- Complexity is not reflected in test metrics degradation

**Phase:** Model Evaluation / Validation Strategy

---

## Summary Table

| Pitfall | Severity | Prevention Phase | Warning Signs |
|---------|----------|------------------|---------------|
| Insufficient time series length for deep models | Critical | Model Selection | No convergence, identical predictions |
| Ignoring exposure as primary volume driver | Critical | Feature Engineering | Exposure dominating predictions |
| Overlooking zero-inflation in claims | High | Model Selection | Residual mass at zero |
| Treating zip codes as independent | High | Feature Engineering | Adjacent zip variance |
| COVID-19 structural break in 2020 | High | Data Preprocessing | 2020 as outlier |
| Avg Fire Risk Score endogeneity | Medium | Data Validation | Perfect correlation with target |
| Feature leakage through future info | Critical | Feature Engineering | Perfect temporal holdout performance |
| Overfitting to zip code granularity | Critical | Model Selection | Parameters > data points |
| Ignoring premium clustering by category | Medium | Model Selection | Category dominance |
| Selecting model based on in-sample metrics | Critical | Model Evaluation | Metrics improve with complexity |

---

## Sources

- **Confidence: LOW** -- Web search encountered API errors. Findings based on domain knowledge of actuarial pricing, insurance GLM frameworks, and general time series forecasting principles.
- Findings should be validated against actuarial literature on insurance pricing with limited data.
