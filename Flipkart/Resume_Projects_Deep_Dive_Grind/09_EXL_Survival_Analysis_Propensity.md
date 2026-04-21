# 🔥 PROJECT GRIND: Survival Analysis & Propensity Modeling + A/B Testing
### Company: EXL (CVS Health/Aetna) | Role: Manager, Data Science & ML Engineering

> **Resume Bullets:**
> - Implemented Kaplan-Meier estimator and Cox proportional hazards model for patient outcomes and customer attrition analysis; developed custom feature importance method combining SHAP values and permutation importance.
> - Created interactive Streamlit application for healthcare providers to visualize survival curves and risk factors.
> - Designed and deployed automated marketing optimization strategies using advanced propensity modeling and A/B testing.

---

## 🏗️ 1. PRODUCTION ARCHITECTURE

```
┌──────────────────────────────────────────────────┐
│          SURVIVAL ANALYSIS PIPELINE              │
├──────────────────────────────────────────────────┤
│                                                  │
│  [Patient/Customer Event Data]                   │
│         │                                        │
│         ▼                                        │
│  ┌─────────────────────────┐                     │
│  │ Censoring Logic         │                     │
│  │ (Right-censored:        │                     │
│  │  alive patients / active│                     │
│  │  customers at cutoff)   │                     │
│  └──────────┬──────────────┘                     │
│             │                                    │
│    ┌────────┴────────┐                           │
│    │                 │                           │
│    ▼                 ▼                           │
│  Kaplan-Meier      Cox PH Model                  │
│  (Non-parametric   (Semi-parametric              │
│   survival curve)   w/ covariates)                │
│    │                 │                           │
│    │    ┌────────────┤                           │
│    │    │            │                           │
│    ▼    ▼            ▼                           │
│  Survival    Hazard    SHAP + Permutation        │
│  Curves      Ratios   Importance                 │
│    │           │           │                     │
│    └─────┬─────┘───────────┘                     │
│          │                                       │
│          ▼                                       │
│  [Streamlit Dashboard for Providers]             │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│        PROPENSITY + A/B TESTING PIPELINE         │
├──────────────────────────────────────────────────┤
│                                                  │
│  [Historical Campaign Data] → Feature Eng        │
│         │                                        │
│         ▼                                        │
│  Propensity Model (Logistic Reg / XGBoost)       │
│  P(responds to campaign | features)              │
│         │                                        │
│         ▼                                        │
│  Stratified Random Assignment (Treatment/Control)│
│         │                                        │
│         ▼                                        │
│  A/B Test Execution (4-6 weeks)                  │
│         │                                        │
│         ▼                                        │
│  Statistical Test (t-test / Mann-Whitney)        │
│  + Sequential Testing (Early stopping)           │
│         │                                        │
│         ▼                                        │
│  Incrementality Report → Business Decision       │
└──────────────────────────────────────────────────┘
```

---

## 🛠️ 2. PHASE-BY-PHASE DEEP DIVE & UNUSUAL EDGE CASES

### A. Survival Analysis Phase
*   **What you did:** Applied Kaplan-Meier for non-parametric survival curves (e.g., "What % of patients are readmission-free at 90 days?"). Applied Cox PH for identifying risk factors (age, comorbidity index, discharge disposition).
*   **The "Unusual" Issue:** **"Informative Censoring."** The Cox PH model assumes censoring is non-informative (a censored patient is no sicker than a non-censored one). But patients who voluntarily left the health plan were often sicker (they left because they were frustrated) — violating the assumption.
*   **The Fix:** Added a sensitivity analysis with Inverse Probability of Censoring Weighting (IPCW). Trained a secondary model to predict P(censored | X), and re-weighted the Cox PH contributions. Also ran a robustness check: compared Cox PH hazard ratios with and without IPCW — if they diverge significantly, the informative censoring bias is real.

*   **The "Unusual" Issue #2:** **"Proportional Hazards Violation."** The effect of "age" on readmission risk was NOT proportional over time — young patients' risk spiked in the first 7 days then dropped, while elderly patients' risk increased monotonically. Schoenfeld residuals showed a clear time-trend.
*   **The Fix:** Stratified Cox model: stratified by age group, allowing each stratum a different baseline hazard $h_0(t)$, while sharing covariate coefficients. Alternatively: added time-varying coefficients $\beta(t)$ for age.

### B. Propensity Modeling Phase
*   **What you did:** Built propensity models to predict P(customer responds to marketing campaign), used to target high-propensity customers.
*   **The "Unusual" Issue:** **"The Uplift vs. Propensity Confusion."** High propensity customers are those who are likely to convert WHETHER OR NOT they receive the campaign. Targeting them wastes budget on people who would have converted anyway. The business wanted *incremental* impact.
*   **The Fix:** Shifted from propensity modeling to **Uplift Modeling** (CATE estimation). Trained two models: $P(Y=1|T=1,X)$ and $P(Y=1|T=0,X)$. Uplift = difference. Targeted customers with HIGH uplift (the "persuadables"), not high propensity. This is the Conditional Average Treatment Effect (CATE) from causal inference.

### C. A/B Testing Phase
*   **What you did:** Ran A/B tests for marketing campaigns across multiple customer segments.
*   **The "Unusual" Issue:** **"Peeking Problem."** Stakeholders checked results daily. By day 3, one variant looked 5% better with p=0.04. They wanted to call the test. This is the multiple comparisons / peeking problem — the p-value is inflated by sequential testing.
*   **The Fix:** Implemented **Sequential Testing (Group Sequential Design)** with O'Brien-Fleming spending function. This adjusts the significance boundary at each interim analysis, requiring much stronger evidence early on (e.g., p<0.005 at midpoint, p<0.045 at endpoint). Alternatively, used a Bayesian framework: report P(variant B > variant A) at each checkpoint, with a pre-agreed "stop-if-P>0.99" rule.

---

## ⚔️ 3. FLIPKART ROUND-SPECIFIC GRINDING QUESTIONS

### 🔴 DDS (System Design)
1.  **"Flipkart wants to predict which sellers will churn (stop selling) in the next 90 days. Design this as a survival analysis problem instead of binary classification. Why is survival analysis better?"**
    *   *Ans:* Binary classification throws away time information (it only knows churned/not-churned as of today). Survival analysis models the *when*: "P(churn by day 30) = 0.2, P(churn by day 60) = 0.5, P(churn by day 90) = 0.7." This lets business prioritize outreach: intervene at day 20 for high-risk sellers, not after they've already left.
2.  **"Design a campaign targeting system for Flipkart that uses uplift modeling to identify which users to show a discount coupon to."**
    *   *Ans:* RCT on historical data → train two models ($T=1$ treated group, $T=0$ control) → compute uplift score per user → rank by uplift → target Top-K.

### 🔵 DMM (Mathematical Modeling)
1.  **"Derive the Kaplan-Meier estimator and prove why it handles right-censoring correctly."**
    *   *Ans:* $\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$ where $d_i$ = deaths at time $t_i$, $n_i$ = at-risk just before $t_i$. Censored observations reduce $n_i$ at their censoring time but don't contribute a death event — this is why the product limit correctly accounts for them.
2.  **"Write down the Cox PH partial likelihood and explain why it doesn't need to estimate the baseline hazard."**
    *   *Ans:* $L(\beta) = \prod_{i: \delta_i=1} \frac{\exp(\beta^T x_i)}{\sum_{j \in R(t_i)} \exp(\beta^T x_j)}$. The baseline hazard $h_0(t)$ appears in numerator and denominator identically and cancels out. This is the mathematical elegance of Cox — semi-parametric: estimates $\beta$ without assuming any distribution for survival times.
3.  **"You mentioned uplift modeling. Formulate the CATE mathematically and explain the two key assumptions needed."**
    *   *Ans:* CATE($x$) = $E[Y(1) - Y(0) | X = x]$. Assumptions: (1) Unconfoundedness: $(Y(0), Y(1)) \perp T | X$ (no hidden confounders). (2) Overlap/Positivity: $0 < P(T=1 | X=x) < 1$ for all $x$ (everyone has a chance of being treated or not).

### 🟢 HO (Hands-On)
1.  **"Using the `lifelines` library, write code to fit a Cox PH model, check the proportional hazards assumption via Schoenfeld residuals, and plot the survival curves for two patient cohorts."**
2.  **"Write a function that computes the required sample size for an A/B test given baseline conversion rate, minimum detectable effect, significance level, and power."**

### 🟡 HM (Hiring Manager)
1.  **"Tell me about a time when stakeholders wanted to stop an A/B test early because they 'already saw the results'. How did you handle it?"**
    *   *STAR:* **S:** Marketing VP saw p=0.04 on day 3 of a 4-week test. Wanted to roll out immediately. **T:** Prevent a false positive that could waste millions. **A:** Showed the "peeking inflation" chart — demonstrated that checking p-values daily inflates false positive rate to 26% even with alpha=0.05. Proposed the O'Brien-Fleming boundary as a compromise: "If p < 0.005 at midpoint, we stop early. Otherwise, we wait." **R:** Test completed at week 4, final p=0.12 — the early signal was a false alarm. Saved estimated $2M in misallocated campaign budget.
