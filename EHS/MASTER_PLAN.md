# Ensemble Health AI Team — Comprehensive Hiring Assessment Master Plan

> **Role:** Lead, AI — Ensemble Health Partners India Pvt. Ltd.  
> **Candidate:** Abhishek  
> **Prepared:** 2026-05-26 (synthesized from EDA, DeepSeek plan, Codex plan, own analysis + targeted web research)  
> **Purpose:** Definitive, unified, implementation-ready strategy document — supersedes all earlier drafts

---

## Table of Contents

1. [Strategic Context & Employer Intelligence](#1-strategic-context--employer-intelligence)
2. [Complete Problem Framing](#2-complete-problem-framing)
3. [Administrative & Confidentiality Obligations](#3-administrative--confidentiality-obligations)
4. [Data Audit — Every Finding That Matters](#4-data-audit--every-finding-that-matters)
5. [Target Leakage Prevention — Non-Negotiable](#5-target-leakage-prevention--non-negotiable)
6. [Metric Architecture](#6-metric-architecture)
7. [Feature Engineering — Complete Catalogue](#7-feature-engineering--complete-catalogue)
8. [Preprocessing Pipeline](#8-preprocessing-pipeline)
9. [Model Strategy — Baselines through Advanced](#9-model-strategy--baselines-through-advanced)
10. [Probability Calibration](#10-probability-calibration)
11. [Threshold & Risk Tiering Policy](#11-threshold--risk-tiering-policy)
12. [Explainability Strategy — SHAP + Attribution](#12-explainability-strategy--shap--attribution)
13. [GenAI Explanation Engine — Complete Design](#13-genai-explanation-engine--complete-design)
14. [Output CSV Specification](#14-output-csv-specification)
15. [PDF Write-Up Blueprint](#15-pdf-write-up-blueprint)
16. [Repository & Code Structure](#16-repository--code-structure)
17. [Testing & Quality Assurance Framework](#17-testing--quality-assurance-framework)
18. [Production Thinking — Monitoring, Drift & MLOps](#18-production-thinking--monitoring-drift--mlops)
19. [Dollar-Weighted Business Metrics](#19-dollar-weighted-business-metrics)
20. [Risks, Limitations & Mitigations](#20-risks-limitations--mitigations)
21. [Future Improvements Roadmap](#21-future-improvements-roadmap)
22. [5-Day Execution Schedule](#22-5-day-execution-schedule)
23. [Final Deliverables Checklist](#23-final-deliverables-checklist)
24. [References & Research Basis](#24-references--research-basis)

---

## 1. Strategic Context & Employer Intelligence

Before writing a single line of code, know who you are presenting to.

### 1.1 What Ensemble Health Partners Actually Does with AI

Ensemble Health Partners (EHP) is a revenue cycle management company that runs financial operations for hospital systems. In June 2025, EHP and Cohere launched the **first RCM-native agentic AI platform**, built on Cohere's "North" infrastructure. Key architectural decisions they have already made:

- **Agentic orchestration:** A network of AI agents that reason, learn, and act across the entire revenue cycle — from patient intake to account resolution.
- **RCM-native LLM (early 2026):** The healthcare industry's first LLM fine-tuned specifically on denial patterns, payer policies, documented RCM procedures, and real claim outcomes — trained on synthetic de-identified data to ensure HIPAA compliance.
- **Denial Prevention is a Core Pillar:** EHP's agents analyze thousands of denial audit letters and millions of claims to predict payer- and account-level risk, triggering preemptive authorizations or corrections *before submission*.
- **Measured outcomes:** 40% faster denial appeals, 15% better denial overturn rate.
- **Security model:** Sovereign AI, HIPAA-compliant, no identifiable client data in model training.

### 1.2 Implication for the Assessment

Your submission is being evaluated by engineers and leaders who are *building* agentic denial prevention right now. They expect:

1. **Correct problem framing** — pre-submission risk prioritization, not post-hoc analysis.
2. **Operational metric alignment** — the 25% capacity constraint is real, not theoretical.
3. **Grounded, safe GenAI usage** — they are building HIPAA-compliant LLM pipelines; yours must show awareness of hallucination risk, structured grounding, and human oversight.
4. **Production systems thinking** — they want to know you can think beyond the Jupyter notebook: drift monitoring, feedback loops, retraining triggers.
5. **Dollar-weighted perspective** — RCM is measured in dollars, not just claim counts.

---

## 2. Complete Problem Framing

### 2.1 The Business Problem

| Factor | Detail |
|---|---|
| **The event** | Insurance company denies a submitted claim |
| **The cost** | $25–$118 administrative overhead per denial; 35–65% of denied claims are never resubmitted → pure revenue loss |
| **Industry scale** | US hospitals lose ~$262 billion annually to denied claims |
| **The intervention point** | Before submission — a front-end billing review team can correct fixable issues |
| **The constraint** | Review team can inspect only the **top 25% of claims by risk score** |

### 2.2 What This Is NOT

- Not a denial determination system (the model doesn't decide if a claim IS denied — it predicts *risk*).
- Not an auto-approval system (low-risk claims still go through normal submission workflow).
- Not a retrospective analysis tool (the entire value is *before* submission).
- The LLM does NOT calculate risk, does NOT select claims, does NOT infer clinical facts.

### 2.3 Error Trade-Off Analysis

| Error Type | Operational Consequence | How We Handle It |
|---|---|---|
| **False Negative** (miss a true denial) | Claim leaves queue unreviewed → denial → rework or lost revenue | Maximize recall within top 25% |
| **False Positive** (flag a claim that pays) | Analyst spends scarce review time on a good claim | Track precision@25%; measure lift |
| **Miscalibrated probability** | Reviewers can't trust the score to prioritize within the queue | Calibrate with Platt Scaling; assess Brier score |

### 2.4 Operational Workflow (as designed for production)

```
1. Claims ready for pre-bill review are ingested
2. Feature engineering + frozen preprocessing applied
3. Denial risk model scores each claim → denial_probability
4. Claims ranked descending by probability
5. Top 25% (125 of 500) routed to review queue
6. Risk drivers and LLM explanation displayed for top 10
7. Analyst reviews, corrects issues, and submits or holds
8. Payer outcome logged → future retraining input
```

### 2.5 Formal Problem Definition

- **Task:** Supervised binary classification
- **Target:** `is_denied` (1 = denied, 0 = paid)
- **Available in scoring:** All columns in `current_claims.csv` (17 columns)
- **Absent in scoring:** `is_denied`, `denial_reason`, `split`
- **Split policy:** Use the pre-defined `split` column EXACTLY as given (temporal, chronological)
- **Output:** Calibrated `denial_probability` ∈ [0, 1] for all 500 current claims

---

## 3. Administrative & Confidentiality Obligations

> **This section was present in the Codex plan and is CRITICAL for the Lead, AI role — it demonstrates judgment.**

### 3.1 Conflicts to Resolve (Send Day 1)

Send a brief professional email resolving these conflicts BEFORE publishing anything:

1. **PDF states:** *"Confidential – For Candidate Use Only – Do Not Distribute."*  
   **Email states:** *"Upload your Assignment on your GitHub Account and share that link."*  
   → **Action:** Request permission for a **private GitHub repository** with reviewer access. Do not make a public repo until cleared.

2. **PDF states:** *"Within 5 business days of receiving this document."*  
   **Email states:** *"Deadline: 1 Week."*  
   → **Action:** Work to the **earlier 5-business-day deadline** until clarified.

### 3.2 Repository Safety Controls

- `.gitignore` must exclude: the assessment PDF, the email, raw CSVs (until confirmed acceptable), API keys, `.env` files, and generated model binaries if large.
- Store API keys in environment variables, never committed.
- Private repo → grant specific reviewer access.

---

## 4. Data Audit — Every Finding That Matters

### 4.1 Dataset Summary

| File | Rows | Columns | Purpose |
|---|---|---|---|
| `claims_history.csv` | 3,200 | 20 | Training + Validation + Test (pre-split) |
| `current_claims.csv` | 500 | 17 | Inference / Scoring only |

**Historical claims service period:** 2024-01 through 2024-12  
**Current claims service period:** 2025-01 through 2025-02  
⚠️ **Current data is temporally *after* all training/evaluation data** — temporal extrapolation limitation must be stated explicitly.

### 4.2 Temporal Split (Chronological — Must Not Be Randomized)

| Split | Months | Rows | Denied | Denial Rate |
|---|---|---|---|---|
| train | 2024-01 – 2024-08 | 2,122 | 448 | 21.11% |
| validation | 2024-09 – 2024-10 | 539 | 103 | 19.11% |
| **test** | **2024-11 – 2024-12** | **539** | **140** | **25.97%** |
| Total | Full year 2024 | 3,200 | 691 | 21.59% |

> **Critical note:** The test set has a meaningfully higher denial rate (26%) than train/validation (~20%). This is realistic temporal drift. State this honestly in the write-up; do NOT hide it in cross-validation averaging.

### 4.3 Target Distribution

```
Total claims:   3,200
Approved (0):   2,509 (78.4%)
Denied (1):       691 (21.6%)
Imbalance:      ~3.6:1 (moderate — not severe enough for SMOTE, but warrants class_weight='balanced')
```

A "predict all approved" naive model achieves 78.4% accuracy but captures **ZERO denials** — accuracy is not a valid metric.

### 4.4 Complete Column Reference

| Column | Type | Role | Notes |
|---|---|---|---|
| `claim_id` | text | **Drop — ID only** | Unique, no signal; 0 overlap between history & current |
| `payer_id` | categorical | Feature | 12 unique payers P001–P012. ⚠️ PDF says P001-P010 but P011 and P012 ARE present — document discrepancy, do NOT remove |
| `payer_type` | categorical | Feature | Commercial (14.9% denial), BCBS (21.3%), Medicare Advantage (25.4%), Medicaid MCO (30.6%) |
| `visit_type` | categorical | Feature | Outpatient (20%), Observation (18.1%), Emergency (18.3%), Inpatient (30.6%) |
| `total_billed` | numeric | Feature | Range ~$500–$95,000; right-skewed |
| `expected_payment` | numeric | Feature | Contractually expected receipt; range ~$200–$63,000 |
| `num_procedures` | numeric | Feature | Count of procedure codes (1–14); coding denials cluster at 4+ procedures |
| `num_diagnoses` | numeric | Feature | Count of diagnoses (1–17) |
| `prior_auth_required` | binary | Feature | Whether payer requires prior authorization |
| `has_prior_auth` | binary | Feature | Whether authorization is documented |
| `is_in_network` | binary | Feature | Provider in payer network |
| `days_to_submit` | numeric | Feature | Days from service to submission (1–78); timely filing threshold ~30 days |
| `missing_documentation_flag` | binary | Feature | Required docs appear missing |
| `eligibility_verified` | binary | Feature | Patient insurance verified pre-service |
| `referral_required` | binary | Feature | Payer requires referral |
| `referral_present` | binary | Feature | Referral documented |
| `service_month` | temporal | Feature (use with caution) | YYYY-MM; can become fragile temporal proxy |
| `split` | meta | **Drop — evaluation only** | train/validation/test |
| `is_denied` | binary | **TARGET** | 0=Paid, 1=Denied |
| `denial_reason` | text | **Post-outcome ONLY** | Only present for denied claims; NEVER use as feature |

### 4.5 Data Quality Checks

| Check | Result | Action |
|---|---|---|
| Missing values | Zero (except 2,509 blank `denial_reason` for paid claims — expected) | Still implement defensive imputers |
| Duplicate claim IDs | Zero in both files | Assert on ingest |
| ID overlap between files | Zero | Assert on ingest |
| `expected_payment > total_billed` | Zero violations | Assert as data integrity check |
| Binary field values | All 0/1 | Assert on ingest |
| `service_month` parse | All valid YYYY-MM | Assert on ingest |
| Payer ID discrepancy | P011, P012 not mentioned in PDF | Document; retain in model |
| Payment-to-billed ratio range | 0.18–0.82 in history; 0.25–0.77 in current | Safe overlap; note as feature in drift check |

### 4.6 Denial Rate by Actionable Condition (Pre-Submission Observable)

| Condition | Historical Occurrence | Denial Rate Present | Denial Rate Absent | Current Occurrence |
|---|---|---|---|---|
| Prior auth required but missing (`auth_gap`) | 310 (9.7%) | **46.5%** | 18.9% | 53 (10.6%) |
| Missing documentation flag | 604 (18.9%) | **38.6%** | 17.6% | 80 (16.0%) |
| Eligibility not verified | 420 (13.1%) | **36.0%** | 19.4% | 61 (12.2%) |
| Referral required but missing (`referral_gap`) | 261 (8.2%) | **33.0%** | 20.6% | 34 (6.8%) |
| Days to submit > 30 | 479 (15.0%) | **32.8%** | 19.6% | 61 (12.2%) |
| Out of network | 583 (18.2%) | **24.9%** | 20.9% | 77 (15.4%) |

**Compound effects (synergistic risks observed in data):**

| Combination | Denial Rate |
|---|---|
| Prior Auth Gap + Missing Documentation | **74.0%** |
| Out-of-Network + Prior Auth Gap | **52.0%** |
| Out-of-Network + Referral Gap | **46.9%** |

### 4.7 Denial Reason Distribution (Post-Hoc, Not Model Input)

| Denial Reason | Count | % of Denials | Expected Payment at Risk |
|---|---|---|---|
| Payer policy / medical necessity | 140 | 20.3% | $763,716.91 |
| Documentation incomplete or missing | 118 | 17.1% | $782,428.35 |
| Patient eligibility not verified | 92 | 13.3% | $580,477.74 |
| Claim submitted after timely filing | 83 | 12.0% | $491,880.12 |
| Authorization required but not on file | 73 | 10.6% | $668,638.03 |
| Procedure/diagnosis coding error | 69 | 10.0% | $655,742.99 |
| Provider not in network | 67 | 9.7% | $404,841.92 |
| Referral required but not present | 49 | 7.1% | $236,999.93 |

**Insight:** No single reason dominates. 8 distinct denial types require the model to capture multiple simultaneous signals — not just one simple rule.

### 4.8 Current Claims vs. History Distribution Drift

| Dimension | Historical Share | Current Share | Delta | Treatment |
|---|---|---|---|---|
| Inpatient visit type | 20.8% | 25.4% | +4.7 pp | Monitor — inpatient has highest denial rate |
| Payer P003 | baseline | +2.5 pp | Moderate | One-hot encoding handles this; note in write-up |
| Commercial payer type | baseline | +1.9 pp | Modest | Note |
| Out of network | 18.2% | 15.4% | -2.8 pp | Lower rate — could slightly suppress predictions |
| Auth gaps present | 9.7% | 10.6% | +0.9 pp | Modest increase in highest-risk condition |

→ **Current population is plausibly within the training distribution.** Temporal extrapolation (2025 vs 2024) is the bigger concern.

---

## 5. Target Leakage Prevention — Non-Negotiable

This deserves its own section because it is a common failure mode evaluated explicitly by the assessors.

### 5.1 Absolute Exclusion List (From ALL Model Inputs)

```python
LEAKAGE_COLUMNS = ['claim_id', 'split', 'is_denied', 'denial_reason']
```

- `denial_reason` is only populated AFTER a claim is denied — using it would be extreme target leakage that makes a fake 100% accurate model.
- `is_denied` is the target — cannot be a feature.
- `split` is a metadata field for evaluation routing only.
- `claim_id` is an identifier — no signal for new claims.

### 5.2 Programmatic Leakage Gate

Implement a `validate_features(X, y)` function that:
1. Asserts none of the leakage columns appear in `X`
2. Asserts `y` contains only the values in `{0, 1}`
3. Asserts `denial_reason` is not passed to any LLM input for current (unscored) claims
4. Fails loudly and immediately with a clear error message

### 5.3 Feature Eligibility Decision Table

| Column | Use as Model Input? | Reason |
|---|---|---|
| `service_month` | **Conditionally — run ablation** | Available pre-submission but may be a fragile temporal proxy that doesn't generalize to new months; ablate and document |
| All other non-leakage columns | **Yes** | Available before submission and independently observable |

---

## 6. Metric Architecture

### 6.1 The Capacity Constraint — Exact Definition

```text
review_capacity_k = floor(0.25 × number_of_claims_in_population)
```

| Population | Rows | Review Capacity k |
|---|---|---|
| Validation | 539 | **134** |
| Test | 539 | **134** |
| Current scoring | 500 | **125** |

Tie-breaking: sort descending by `denial_probability`, then ascending by `claim_id` (deterministic).

### 6.2 Primary Metric — Denial Capture Rate at Top 25%

```
Denial Capture Rate @ 25% =
    Number of actual denied claims in top-k ranked records
    ────────────────────────────────────────────────────────
    Total actual denied claims in the evaluation population
```

**Target:** ≥ 60% (random baseline = ~21.6%, perfect = 100%).

### 6.3 Supporting Metrics (Full Suite)

| Metric | Formula/Method | Purpose |
|---|---|---|
| **Precision @ Top 25%** | True denials in top-k / k | Queue efficiency — what fraction of reviewed claims were actually denied |
| **Lift @ Top 25%** | (Precision@25%) / (Overall denial rate) | How much better than random |
| **Expected Payment Capture @ 25%** | Sum `expected_payment` of denied claims in top-k / Sum all denied claims' `expected_payment` | Dollar-weighted denial capture — more aligned to business outcome |
| **ROC-AUC** | Standard | Overall ranking discrimination |
| **PR-AUC / Average Precision** | Preferred for imbalanced | Better than ROC-AUC for minority class |
| **Brier Score** | Mean squared error of probabilities vs. true labels | Probability calibration quality |
| **Reliability Diagram** | Visual | Calibration curve — is a 0.7 predicted probability actually 70% denial rate? |
| **Confusion Matrix** | At stated threshold | TP, FP, FN, TN breakdown |
| **F1 at stated threshold** | Harmonic mean of precision/recall | Supporting binary classification metric |
| **Log Loss** | Cross-entropy | Complementary probability quality metric |

### 6.4 Metric Hierarchy for Model Selection

1. **Validation Denial Capture @ 25%** ← Primary selection criterion
2. **Validation PR-AUC** ← Secondary: overall ranking quality for minority class
3. **Brier Score after calibration** ← Probability trustworthiness
4. **Logit interpretability / coefficient stability** ← For audit readiness in a regulated environment

### 6.5 Baselines for Context

| Model | Expected Denial Capture @ 25% |
|---|---|
| Random (uniformly sample 25%) | ~21.6% |
| "Always flag top 25% by billed amount" | ~25–28% |
| Rule-based heuristic (any gap flag) | ~35–45% |
| **Good ML model** | **≥ 55–65%** |

---

## 7. Feature Engineering — Complete Catalogue

### 7.1 Gap Features (Highest Business Value)

```python
# Prior Authorization Gap — 46.5% denial rate when present
df['auth_gap'] = ((df['prior_auth_required'] == 1) & (df['has_prior_auth'] == 0)).astype(int)

# Referral Gap — 33.0% denial rate when present
df['referral_gap'] = ((df['referral_required'] == 1) & (df['referral_present'] == 0)).astype(int)

# Total administrative gap count
df['total_admin_gaps'] = df['auth_gap'] + df['referral_gap'] + df['missing_documentation_flag'] + (1 - df['eligibility_verified']) + (1 - df['is_in_network'])
```

**Rationale:** These features directly encode the operational failure mode in a way that neither raw feature alone can. `prior_auth_required=1` with `has_prior_auth=1` is fine; the interaction is what matters.

### 7.2 Financial Features

```python
# Expected payment ratio (contractual coverage fraction)
df['payment_ratio'] = df['expected_payment'] / (df['total_billed'] + 1e-5)

# Raw dollar gap (billed but not expected)
df['payment_gap'] = df['total_billed'] - df['expected_payment']

# Log transforms for skewed financial features (essential for Logistic Regression)
df['log_total_billed'] = np.log1p(df['total_billed'])
df['log_expected_payment'] = np.log1p(df['expected_payment'])

# Complexity proxies
df['billed_per_procedure'] = df['total_billed'] / (df['num_procedures'] + 1e-5)
df['billed_per_diagnosis'] = df['total_billed'] / (df['num_diagnoses'] + 1e-5)
df['complexity_score'] = df['num_procedures'] * df['num_diagnoses']
```

### 7.3 Temporal Features

```python
# Extract from service_month string
df['service_month_num'] = pd.to_datetime(df['service_month']).dt.month
df['service_quarter'] = pd.to_datetime(df['service_month']).dt.quarter

# Timely filing buckets (common payer deadlines at 30, 60, 90 days)
df['submission_timeliness'] = pd.cut(df['days_to_submit'],
    bins=[0, 7, 14, 30, 60, 100],
    labels=['<1wk', '1-2wk', '2-4wk', '1-2mo', '>2mo'])

# Binary late filing flag (>30 days aligns with observed timely filing denial threshold)
df['late_filing'] = (df['days_to_submit'] > 30).astype(int)
```

### 7.4 Interaction Features

```python
# Out-of-Network compound risks
df['oon_auth_gap'] = ((df['is_in_network'] == 0) & (df['auth_gap'] == 1)).astype(int)
df['oon_referral_gap'] = ((df['is_in_network'] == 0) & (df['referral_gap'] == 1)).astype(int)

# High-risk payer + visit type combination
df['is_high_risk_combo'] = (
    (df['payer_type'] == 'Medicaid MCO') &
    (df['visit_type'].isin(['Inpatient', 'Emergency']))
).astype(int)

# Double documentation deficit
df['double_deficit'] = ((df['auth_gap'] == 1) & (df['missing_documentation_flag'] == 1)).astype(int)
```

### 7.5 Feature Summary Table

| Group | Count | Examples |
|---|---|---|
| Original categoricals | 4 | `payer_id`, `payer_type`, `visit_type`, `service_month` |
| Original binary | 8 | `prior_auth_required`, `has_prior_auth`, `is_in_network`, etc. |
| Original numeric | 5 | `total_billed`, `expected_payment`, `num_procedures`, `num_diagnoses`, `days_to_submit` |
| Engineered gap features | 4 | `auth_gap`, `referral_gap`, `total_admin_gaps`, `late_filing` |
| Engineered financial | 6 | `payment_ratio`, `payment_gap`, `log_total_billed`, `log_expected_payment`, `billed_per_procedure`, `complexity_score` |
| Engineered temporal | 3 | `service_month_num`, `service_quarter`, `submission_timeliness` |
| Engineered interaction | 4 | `oon_auth_gap`, `oon_referral_gap`, `is_high_risk_combo`, `double_deficit` |
| **Total candidate features** | **~34** | |

> **Ablation discipline:** Run an ablation study on validation to determine which feature groups add signal. Do not blindly include all 34. Document what was added and why.

---

## 8. Preprocessing Pipeline

Use `sklearn.pipeline.Pipeline` + `ColumnTransformer` — fit **only on training data**.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, RobustScaler
)
from sklearn.impute import SimpleImputer

# Categorical → OneHotEncoder (unknown categories handled with 'ignore')
categorical_features = ['payer_type', 'visit_type']

# High-cardinality → OrdinalEncoder (or target-mean encoding fitted on train only)
high_cardinality_features = ['payer_id']

# Temporal ordinal feature
temporal_ordinal_features = ['submission_timeliness']

# Binary flags → passthrough (already 0/1)
binary_features = [
    'prior_auth_required', 'has_prior_auth', 'is_in_network',
    'missing_documentation_flag', 'eligibility_verified',
    'referral_required', 'referral_present',
    'auth_gap', 'referral_gap', 'late_filing', 'oon_auth_gap',
    'oon_referral_gap', 'is_high_risk_combo', 'double_deficit'
]

# Numeric → StandardScaler (or RobustScaler for outlier robustness)
numeric_features = [
    'log_total_billed', 'log_expected_payment',
    'num_procedures', 'num_diagnoses', 'days_to_submit',
    'payment_ratio', 'payment_gap', 'billed_per_procedure',
    'complexity_score', 'total_admin_gaps', 'service_month_num'
]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('high_card', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), high_cardinality_features),
    ('ord', OrdinalEncoder(categories=[['<1wk', '1-2wk', '2-4wk', '1-2mo', '>2mo']]), temporal_ordinal_features),
    ('num', RobustScaler(), numeric_features),  # RobustScaler handles financial outliers better
    ('bin', 'passthrough', binary_features)
])
```

**Critical rules:**
- `.fit()` ONLY on training data
- `.transform()` on validation, test, and current claims separately
- Store the fitted `preprocessor` for inference; never refit it on scoring data

---

## 9. Model Strategy — Baselines through Advanced

### 9.1 Candidate Models (In Order of Complexity)

#### Tier 1: Baselines (Mandatory — Shows You Know What Random Looks Like)

**Baseline A: Prevalence/Stratified Dummy**
```python
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='stratified', random_state=42)
```
Expected Denial Capture@25%: ~21.6% (random)

**Baseline B: Rule-Based Heuristic (Operational Baseline)**
```python
def heuristic_score(df):
    return (
        df['auth_gap'] * 4 +        # Highest weight — 46.5% denial rate
        df['missing_documentation_flag'] * 3 +
        (1 - df['eligibility_verified']) * 3 +
        df['referral_gap'] * 2 +
        df['late_filing'] * 2 +
        (1 - df['is_in_network']) * 1
    )
```
Expected Denial Capture@25%: ~35–45%. Shows ML adds value beyond domain rules.

#### Tier 2: Interpretable Baseline

**Model C: Logistic Regression (L2, class-weighted)**
```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(
    C=10.0,          # Tune via validation: search [0.001, 0.01, 0.1, 1, 10, 100]
    class_weight='balanced',
    max_iter=1000,
    solver='lbfgs',
    random_state=42
)
```
- **Our EDA shows this achieves Test AUC: 0.6987 and Denial Capture@25%: 46.43% with engineered features.**
- Coefficients directly support feature attribution: $\text{contribution}_i = w_i \cdot x_i^{(transformed)}$
- Interpretable to business stakeholders without SHAP

#### Tier 3: Non-Linear Challengers (Capture Compound Risk Interactions)

**Model D: XGBoost (Production Industry Standard)**
```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3.6,     # ~imbalance ratio (2509/691)
    eval_metric='auc',
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1
)
# Fit with validation set for early stopping
xgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
```

XGBoost hyperparameter search space (RandomizedSearchCV, 50 iterations):

| Parameter | Search Range | Rationale |
|---|---|---|
| `n_estimators` | 100–500 | Convergence |
| `max_depth` | 3–8 | Control complexity |
| `learning_rate` | 0.01–0.2 | Speed vs. generalization |
| `subsample` | 0.6–1.0 | Stochastic boosting |
| `colsample_bytree` | 0.6–1.0 | Feature subsampling |
| `min_child_weight` | 1–10 | Regularization |
| `reg_alpha` | 0–1.0 | L1 regularization |
| `reg_lambda` | 1–10 | L2 regularization |
| `scale_pos_weight` | 2–5 | Class imbalance |

**Model E: LightGBM (Speed + Comparable Performance)**
```python
import lightgbm as lgb

lgbm_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```
- Faster training; leaf-wise growth; often similar performance to XGBoost

**Model F: CatBoost (Best for High-Cardinality Categoricals Like `payer_id`)**
```python
from catboost import CatBoostClassifier

catboost_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    auto_class_weights='Balanced',
    cat_features=['payer_id', 'payer_type', 'visit_type'],  # Native, no encoding needed
    random_seed=42,
    verbose=False
)
```
- Native categorical handling (no OHE needed)
- Ordered boosting reduces overfitting on structured data

#### Tier 4: Optional Ensemble (Time Permitting)

**Soft Voting Ensemble (Top 2-3 Models):**
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[('xgb', xgb_best), ('lgbm', lgbm_best), ('lr', lr_best)],
    voting='soft',
    weights=[0.5, 0.3, 0.2]   # Tune on validation
)
```
Often adds a 1–3% AUC boost with no additional feature work.

### 9.2 Key Insight From Our EDA

**Logistic Regression outperformed XGBoost and LightGBM in our experiments:**
- LR Test AUC: 0.6987 | LR Denial Capture@25%: 46.43%
- RF Test AUC: 0.6783 | RF Denial Capture@25%: 43.57%
- HistGB Test AUC: 0.6415 | HistGB Denial Capture@25%: 40.00%

**Why?** The synthetic data is generated by additive, independent rules — each gap feature contributes a fixed additive risk. This is exactly the data-generating process that Logistic Regression is designed to model. Tree-based models overfit on high-order interactions that don't genuinely exist in this specific dataset at this scale (2,122 training rows).

**Recommendation:** Use LR as primary, include XGBoost/LightGBM for completeness and to demonstrate you know when simpler models win. Discuss this explicitly in the write-up — it shows maturity.

---

## 10. Probability Calibration

### 10.1 Why Calibration Matters Here

The output column is named `denial_probability`. If the model says 0.70, it should mean roughly 70% of claims with that score are actually denied. Without calibration:
- Reviewers may over-trust or under-trust the scores
- The LLM explanation (e.g., "this claim has a 70% estimated denial risk") becomes misleading
- Business decisions about threshold selection become arbitrary

### 10.2 Assessment

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

# Check raw calibration on validation
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_val, y_proba_val, n_bins=10
)
brier = brier_score_loss(y_val, y_proba_val)
print(f"Brier Score (pre-calibration): {brier:.4f}")
```

### 10.3 Calibration Methods

| Method | When to Use | Risk |
|---|---|---|
| **Platt Scaling** | Sigmoid distortion; small validation set | More stable on smaller data |
| **Isotonic Regression** | Non-monotonic distortion; larger validation set | Overfits on small sets |

```python
from sklearn.calibration import CalibratedClassifierCV

# Platt scaling (sigmoid method)
calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_val, y_val)   # Fit calibrator on validation only

# Evaluate post-calibration
brier_post = brier_score_loss(y_val, calibrated_model.predict_proba(X_val)[:, 1])
```

### 10.4 Calibration Rules

- Fit calibrator ONLY on validation data
- Never use test data for calibration fitting
- Report Brier score and reliability diagram before and after calibration
- If calibration does not materially improve Brier score (< 0.01 improvement), use raw probabilities and document it

---

## 11. Threshold & Risk Tiering Policy

### 11.1 Two Distinct Concepts (Keep Separate in Documentation)

1. **Capacity-based queue ranking:** The top `floor(0.25 × n)` records go to review. This is rank-based, not threshold-based.
2. **Binary `predicted_denial` label:** A fixed probability threshold (frozen on validation) that determines the 0/1 column in the CSV.

### 11.2 Setting the Binary Threshold

```python
def select_threshold_for_top25(y_val, y_proba_val):
    """
    On validation set, find the probability at the 75th percentile cutoff.
    This is the natural threshold aligned to the 25% capacity constraint.
    """
    threshold = np.percentile(y_proba_val, 75)  # Top 25% = above 75th percentile
    return threshold

# Freeze threshold before test evaluation
FROZEN_THRESHOLD = select_threshold_for_top25(y_val, y_proba_val)
```

State this threshold explicitly in README and PDF. Apply without modification to test and current claims.

### 11.3 Risk Tier Assignment (Current Claims)

| Tier | Rule | Operational Meaning | Count |
|---|---|---|---|
| **High** | Top 25% of 500 claims by rank (positions 1–125) | Within available review capacity; flag for mandatory review | 125 |
| **Medium** | Next 25% (positions 126–250) | Next priority if review capacity expands | 125 |
| **Low** | Bottom 50% (positions 251–500) | Lower risk; standard submission workflow | 250 |

> **Important:** State in README that tier assignment is rank-based (not threshold-based) and therefore different from the frozen binary threshold. Both are reported in the CSV.

---

## 12. Explainability Strategy — SHAP + Attribution

### 12.1 Global Explainability (For Write-Up + Manager Communication)

```python
import shap

# For tree models
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test_preprocessed)

# Summary plot (global feature importance by mean |SHAP|)
shap.summary_plot(shap_values, X_test_preprocessed,
                  feature_names=feature_names, plot_type='bar')

# Beeswarm plot (direction + magnitude)
shap.summary_plot(shap_values, X_test_preprocessed, feature_names=feature_names)
```

**Output:** A ranked list of the most globally important features across all test claims — used in write-up Section 3 for the manager audience.

### 12.2 Local Explainability — Per-Claim Risk Factors

**For tree models (XGBoost/LightGBM/CatBoost):**
```python
# SHAP local values for a single claim
shap_single = explainer.shap_values(X_single_claim_transformed)

# Extract top positive drivers (features pushing toward denial)
contributions = pd.Series(shap_single, index=feature_names)
top_positive_drivers = contributions[contributions > 0].nlargest(3)
```

**For Logistic Regression (simpler, equally valid):**
```python
# Linear attribution: contribution_i = w_i * x_i
coefs = lr_model.coef_[0]
contributions = X_single_claim_transformed * coefs
top_positive_drivers = pd.Series(contributions, index=feature_names).nlargest(3)
```

### 12.3 Driver Rules (Critical for LLM Grounding)

- Report features that **increase** denial risk (positive contribution), not those that reduced it.
- Provide 2 drivers minimum; 3 drivers maximum. Do NOT manufacture a third if only 2 exist.
- Map transformed feature names to human-readable labels:

```python
FRIENDLY_LABELS = {
    'auth_gap': 'Missing Prior Authorization (required but not on file)',
    'referral_gap': 'Missing Referral (required but not on file)',
    'missing_documentation_flag': 'Missing Documentation',
    'eligibility_verified': 'Unverified Patient Eligibility',
    'is_in_network': 'Provider Not in Payer Network',
    'late_filing': 'Late Claim Submission (>30 days after service)',
    'days_to_submit': 'High Days to Submit',
    'log_total_billed': 'High Billed Amount',
    'payment_ratio': 'Low Expected Payment Ratio',
    'payer_type_Medicaid MCO': 'High-Risk Payer Type (Medicaid MCO)',
    'visit_type_Inpatient': 'Inpatient Visit (Higher Denial Risk)',
    'complexity_score': 'High Claim Complexity (Procedures × Diagnoses)',
    'double_deficit': 'Compound Deficit: Auth Gap + Missing Documentation',
}
```

- Drivers must trace directly to factual claim field values — validate that the driver label matches the actual record values.
- If a non-actionable feature (e.g., `payer_type`) is a top driver, include it as **context** but ensure the LLM recommended action targets an actionable item.

### 12.4 SHAP Interaction Values (Optional, Time Permitting)

```python
shap_interaction_values = explainer.shap_interaction_values(X_test_preprocessed)
# Reveals which pairs of features interact — e.g., missing docs matters MORE for OON claims
```

---

## 13. GenAI Explanation Engine — Complete Design

### 13.1 LLM Role Boundary (Non-Negotiable)

The LLM's ONLY job is **controlled language generation**. Specifically:
- ✅ Translate model-derived, validated risk factors into professional plain English
- ✅ Include exactly one actionable recommendation per supported operational action
- ✅ Include exactly one uncertainty qualifier
- ❌ NOT calculate risk scores
- ❌ NOT select or rank claims
- ❌ NOT create risk factors (SHAP/attribution does this)
- ❌ NOT infer payer policy or clinical information
- ❌ NOT reference historic denial reasons
- ❌ NOT fabricate missing patient or claim information

### 13.2 Structured Input Payload (JSON Format)

Pass structured, minimal, sanitized input to the LLM — not raw claim text:

```json
{
  "claim_id": "CCLM-00372",
  "denial_probability": 0.924,
  "risk_tier": "High",
  "top_risk_factors": [
    {
      "factor": "auth_gap",
      "fact": "Prior authorization is required by this payer and is not on file.",
      "permitted_action": "Confirm whether authorization can be obtained or documented before submission."
    },
    {
      "factor": "missing_documentation_flag",
      "fact": "Required supporting documentation appears to be missing.",
      "permitted_action": "Review and attach required clinical documentation before submission."
    }
  ]
}
```

Do NOT include in payload:
- The full historical `denial_reason`
- Patient PHI (there is none in synthetic data, but state this for production awareness)
- Unprovable policy assertions
- Raw model coefficients or SHAP values (these are internal, not analyst-facing)

### 13.3 Prompt Template (Version 1.0)

```
SYSTEM
You write pre-submission claim review notes for a hospital revenue-cycle billing analyst.
Use ONLY the facts supplied in INPUT. Do not infer payer decisions, denial reasons,
medical necessity, clinical facts, or policy requirements not present in INPUT.

Return exactly 2 or 3 plain-English sentences.
Each sentence must:
  1. In the first sentence: state that this claim has an ESTIMATED denial risk (not a guaranteed denial).
  2. In the second sentence: mention only the supplied risk factor facts from INPUT.
  3. In the third sentence (if present): give EXACTLY ONE concrete action drawn from the supplied permitted_action field.

Do NOT:
- Add ICD/CPT codes, dollar amounts, clinical conclusions, or patient details not in INPUT.
- Claim the recommended action guarantees payment.
- Use insurance or medical jargon that a non-specialist would not understand.
- Include more than one recommended action.

INPUT
{structured_claim_payload_json}
```

### 13.4 Example Explanations (Good vs. Bad)

**High-Risk Example A: Prior Auth Gap + Missing Documentation**

> This inpatient Medicaid MCO claim has an **estimated** denial risk of 92% — based on the model's assessment, not a guaranteed outcome. Two specific issues were identified: the required prior authorization is not on file, and supporting documentation appears incomplete. We recommend confirming with the payer whether authorization has been issued under any reference number and gathering any missing clinical notes before submission.

**High-Risk Example B: Referral Gap + Eligibility Not Verified**

> This outpatient claim carries an elevated estimated denial risk because the patient's eligibility has not been verified and a required referral is not on file. Please run an eligibility check with the payer and obtain the referral from the referring provider before submitting. Note: this is a statistical risk estimate and does not guarantee the claim will be denied.

**High-Risk Example C: Late Timely Filing**

> This claim was submitted 55 days after the date of service, which approaches or exceeds many payers' timely filing deadlines — making it an elevated denial risk. We recommend checking this payer's exact filing deadline before submission; if the deadline has passed, prepare a timely filing appeal with proof of original transmission. This is a risk estimate only and not a guaranteed denial.

**Low-Risk Example (Required Validation Test)**

> This outpatient Commercial claim shows no flagged gaps in authorization, documentation, eligibility, or network status, resulting in a low estimated denial risk. No pre-submission corrections appear to be needed based on available claim data. Routine submission is recommended.

### 13.5 Safety Guards

| Guard | Implementation |
|---|---|
| **Temperature** | Set to 0 or near 0 (≤ 0.1) for factual consistency |
| **Max tokens** | 150–200 tokens — enforce 2–3 sentence limit |
| **Structured JSON input** | All claim data passed as structured fields, not free text |
| **Programmatic validation** | Automated checks on EVERY generated output (see 13.6) |
| **Manual QA** | Manually inspect all 10 outputs — volume is small enough |
| **API key hygiene** | Read from `os.environ['GEMINI_API_KEY']`; never committed |
| **Versioned prompt** | Stored in `prompts/explanation_prompt_v1.txt` |
| **Fallback** | Deterministic template-based generation if API unavailable |

### 13.6 Programmatic Validation (Every Generated Output)

```python
def validate_explanation(text: str, claim_payload: dict) -> dict:
    """Returns validation result with specific failure reasons."""
    issues = []

    sentences = [s.strip() for s in text.replace('!', '.').split('.') if len(s.strip()) > 10]
    if len(sentences) < 2 or len(sentences) > 3:
        issues.append(f"Sentence count out of range: {len(sentences)}")

    UNCERTAINTY_KEYWORDS = ['estimated', 'estimate', 'risk', 'not a guarantee', 'not guaranteed', 'may']
    if not any(kw in text.lower() for kw in UNCERTAINTY_KEYWORDS):
        issues.append("Missing uncertainty/risk qualifier")

    ACTION_WORDS = ['recommend', 'verify', 'confirm', 'check', 'obtain', 'review', 'contact', 'gather']
    if not any(kw in text.lower() for kw in ACTION_WORDS):
        issues.append("Missing recommended action")

    HALLUCINATION_RISK = ['icd', 'cpt', 'hcpcs', 'diagnosis code', 'procedure code', 'clinical']
    for kw in HALLUCINATION_RISK:
        if kw in text.lower():
            issues.append(f"Potential hallucination: contains '{kw}' not in payload")

    return {'valid': len(issues) == 0, 'issues': issues, 'text': text}
```

### 13.7 LLM API Options

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| **Gemini 1.5 Flash / 2.0 Flash** | Free tier available; aligns with Google ecosystem | Slightly less reliable structure | ✅ Primary recommendation for this assessment |
| **OpenAI GPT-4o** | Best quality; very reliable instruction following | Paid API required | ✅ If API key available |
| **Claude 3.5 Sonnet** | Excellent at following structured instructions | Paid API required | ✅ Alternative |
| **Ollama + Llama 3.2 (local)** | Free, no API; no network needed | Slower; more prompt tuning | Use as fallback |
| **Manual/Template Fallback** | PDF explicitly allows documented examples | Not actually LLM-generated | Document clearly |

### 13.8 Offline Fallback Template Engine

For reviewers who cannot access an LLM API, implement a `generate_explanation_offline()` function that:
- Uses the `top_risk_factors` list to compose a deterministic, professional 2–3 sentence explanation
- Maps each factor code to a pre-written sentence template (e.g., `auth_gap` → "The required prior authorization is not on file.")
- Appends the uncertainty disclaimer
- Selects the highest-priority permitted action

---

## 14. Output CSV Specification

### 14.1 Required Schema (`predictions_current_claims.csv`)

| Column | Type | Definition | Quality Contract |
|---|---|---|---|
| `claim_id` | string | Original ID from `current_claims.csv` | All 500 IDs, each exactly once |
| `denial_probability` | float | Calibrated model probability ∈ [0, 1] | Non-null; sorted descending; full precision for ranking |
| `predicted_denial` | int | 0 or 1 based on frozen validation threshold | Only 0 or 1; threshold documented in README |
| `risk_tier` | string | 'High', 'Medium', or 'Low' | Exactly 125 High, 125 Medium, 250 Low |
| `top_risk_factors` | string | 2-3 positive risk driver labels, comma-separated | Traceable to model contribution; no leakage |
| `explanation` | string | LLM output for top 10; blank or 'N/A' for others | Top 10 validated; all others explicitly documented |

### 14.2 Sorting Policy

```python
df_scored = df_scored.sort_values(
    by=['denial_probability', 'claim_id'],
    ascending=[False, True]   # Descending probability, ascending ID for ties
).reset_index(drop=True)
```

**Critical:** Compute ranking from full-precision values. Rounded display values in the export must NOT alter the established sort order.

### 14.3 Final CSV Validation Checklist

```python
def validate_output_csv(filepath):
    df = pd.read_csv(filepath)
    assert len(df) == 500, f"Expected 500 rows, got {len(df)}"
    assert list(df.columns) == ['claim_id', 'denial_probability', 'predicted_denial', 'risk_tier', 'top_risk_factors', 'explanation']
    assert df['claim_id'].nunique() == 500
    assert df['denial_probability'].between(0, 1).all()
    assert df['denial_probability'].is_monotonic_decreasing
    assert df['predicted_denial'].isin([0, 1]).all()
    assert df['risk_tier'].isin(['High', 'Medium', 'Low']).all()
    assert (df['risk_tier'] == 'High').sum() == 125
    assert (df['risk_tier'] == 'Medium').sum() == 125
    assert (df['risk_tier'] == 'Low').sum() == 250
    assert df.head(10)['explanation'].notna().all()
    assert (df.head(10)['explanation'] != '').all()
    print("✅ All output validation checks passed")
```

---

## 15. PDF Write-Up Blueprint (2–3 Pages)

### Page 1: Problem, Data, and Approach (~0.8 pages)

**Section A: Problem & Why It Matters**
- Business context: pre-bill review teams prevent costly denials; only 25% of claims can be inspected
- Error trade-off table (False Negative vs. False Positive consequences)
- Why accuracy is the wrong metric; why Recall@25% is the right one
- Brief note on the $262B annual US healthcare denial burden

**Section B: 3 Non-Technical Data Findings for a Manager**
1. Medicaid MCO claims have a **30.6% denial rate** — more than twice the 14.9% rate for Commercial plans. Routing Medicaid MCO claims to the review team provides an easy early win.
2. When a claim is missing required prior authorization AND has incomplete documentation, **74% of such claims are denied**. Fixing either issue before submission directly prevents the most common denial combination.
3. The test period (Nov–Dec 2024) showed a **26% denial rate** — up from 21% in the training period — indicating that denial risk increases over time and a real-time model is essential.

**Section C: Synthetic Data & Data Dictionary Note**
- Brief mention that payer IDs P011 and P012 are present in the data but not listed in the PDF data dictionary; these records were retained and documented.

### Page 2: Modeling & Test Evaluation (~0.9 pages)

**Section D: Model Comparison**
Full table with all models tested:

| Model | Validation Capture@25% | Test AUC | Test Capture@25% | Brier Score |
|---|---|---|---|---|
| Random baseline | ~21.6% | ~0.50 | ~26.0% | — |
| Rule heuristic | ~38% | N/A | ~40% | N/A |
| Logistic Regression | [val result] | [value] | [value] | [value] |
| XGBoost | [val result] | [value] | [value] | [value] |
| LightGBM | [val result] | [value] | [value] | [value] |
| **Selected model** | **[best val]** | **[value]** | **[value]** | **[value]** |

**Section E: Final Model Details**
- Feature engineering decisions and rationale (gap features, log transforms, interaction terms)
- Threshold selection rationale (75th percentile of validation probabilities = top 25% capacity)
- Risk tier definitions and counts
- One chart: Denial Capture Curve (% claims reviewed vs % denials captured)
- Calibration: pre- and post-Platt scaling Brier score

**Section F: Leakage Controls**
- One sentence: `denial_reason` and `is_denied` excluded from all features; confirmed programmatically.

### Page 3: GenAI, Output & Future Work (~0.8 pages)

**Section G: LLM Explanation Design**
- Role boundary: LLM translates model-derived risk factors; it does NOT calculate risk
- Prompt design: structured JSON input, 2–3 sentence constraint, mandatory uncertainty qualifier, single actionable recommendation
- One full example (high-risk claim with explanation)
- Low-risk claim behavior test result
- Guardrails: zero temperature, output validation, human review of all 10

**Section H: Output Summary**
- Brief description of `predictions_current_claims.csv` (columns, sort order, tier counts)
- Top 5 highest-risk claims summary table

**Section I: Limitations (Be Honest)**
1. Synthetic data — performance may not transfer directly to real production claims which include ICD/CPT codes, clinical documentation, payer-specific remittance codes, and prior claim history.
2. Temporal extrapolation — model trained on 2024; current claims are from 2025. Drift monitoring should begin immediately.
3. Denial rate rising in test period (26% vs. 21% in train) — model needs periodic retraining.
4. `payer_type = "payer policy / medical necessity"` (20% of denials) is the hardest to predict from available features — coding review tools and clinical NLP would be needed to address this.

**Section J: Improvements with More Time**
(Concise — 4–5 bullets, covered fully in Section 21 of this plan)

---

## 16. Repository & Code Structure

```
ensemble-denial-risk-assessment/
├── README.md                                   # Full documentation
├── requirements.txt                            # Pinned dependencies
├── .gitignore                                  # Excludes .env, API keys, PDF, raw CSVs
├── .env.example                                # Template for environment variables
│
├── data/
│   └── README.md                               # Data schema & origin; raw CSVs omitted if restricted
│
├── notebooks/                                  # Optional — readable narratives
│   ├── 01_eda.ipynb                           # EDA with visualizations
│   ├── 02_feature_engineering.ipynb           # Feature creation + ablation study
│   ├── 03_model_training.ipynb                # All models + comparison + calibration
│   └── 04_explanations.ipynb                  # SHAP analysis + LLM prompt testing
│
├── src/
│   ├── __init__.py
│   ├── config.py                              # Paths, constants, RANDOM_STATE = 42
│   ├── validate_data.py                       # Data contract assertions (Section 6.3)
│   ├── feature_engineering.py                 # All feature creation functions
│   ├── preprocessing.py                       # ColumnTransformer pipeline builder
│   ├── models.py                              # All model training + hyperparameter search
│   ├── evaluation.py                          # All metric functions + threshold optimizer
│   ├── calibration.py                         # Platt Scaling + reliability diagram
│   ├── explainability.py                      # SHAP + LR attribution + friendly labels
│   ├── llm_explanation.py                     # Prompt template + API calls + offline fallback
│   ├── score_current.py                       # Complete inference pipeline
│   └── validate_outputs.py                    # Output CSV contract checks
│
├── prompts/
│   └── explanation_prompt_v1.txt              # Versioned prompt template
│
├── models/
│   └── final_pipeline.pkl                     # Serialized fitted Pipeline object
│
├── outputs/
│   ├── predictions_current_claims.csv         # 🎯 REQUIRED DELIVERABLE
│   ├── test_metrics.json                      # All test set metrics in machine-readable form
│   ├── model_comparison.csv                   # All models × all metrics
│   └── figures/
│       ├── denial_capture_curve.png           # Main chart for write-up
│       ├── precision_recall_curve.png
│       ├── calibration_curve.png
│       ├── shap_global_importance.png
│       └── shap_beeswarm.png
│
├── report/
│   └── Ensemble_AI_Assessment_Writeup.pdf     # 🎯 REQUIRED DELIVERABLE
│
└── tests/
    ├── test_data_contract.py                   # Unit tests for Section 5 leakage controls
    └── test_output_contract.py                 # Unit tests for Section 14 CSV spec
```

### 16.1 README.md Required Content

1. Assignment objective and operational review constraint (25% capacity)
2. Repository file map and how to run: validate → train → score → explain → validate outputs
3. Data use statement: synthetic dataset; excluded columns; split used as provided; P011/P012 discrepancy
4. Full feature list with engineered feature definitions
5. Model comparison and selection rationale
6. Metric definitions: exact `top 25%` calculation (floor(0.25×n)), tie-breaking rule
7. Frozen threshold value and risk tier assignment rules
8. Test set metric results (complete table)
9. GenAI provider/model, prompt version, validation steps, low-risk example
10. Deliverable links; limitations
11. Reproducibility: `pip install -r requirements.txt`, `python src/score_current.py`, `RANDOM_STATE=42`
12. Confidentiality handling statement

### 16.2 requirements.txt

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
catboost>=1.2
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
google-generativeai>=0.5   # OR openai>=1.0 OR anthropic>=0.21
python-dotenv>=1.0
```

---

## 17. Testing & Quality Assurance Framework

### 17.1 Data Contract Tests (`test_data_contract.py`)

```python
def test_unique_ids(): ...
def test_no_overlap_between_files(): ...
def test_target_only_binary(): ...
def test_leakage_columns_excluded_from_features(): ...
def test_binary_field_values(): ...
def test_monetary_validity(): ...
def test_split_chronological_order(): ...
def test_preprocessor_fitted_on_train_only(): ...
def test_unknown_categories_handled(): ...
def test_payer_id_discrepancy_documented(): ...
```

### 17.2 Metric Verification

| Check | Expected |
|---|---|
| Dummy classifier Denial Capture@25% | ~21.6% (random baseline) |
| Rule heuristic beats dummy | >25% |
| Selected model test Denial Capture@25% | ≥50% target |
| Independent recomputation of primary metric | Matches reported value ±0.001 |
| Dollar-weighted capture tracks count capture | Both reported; note divergence if any |

### 17.3 Output Contract Tests (`test_output_contract.py`)

All 9 assertions in Section 14.3, implemented as pytest unit tests.

### 17.4 GenAI QA (Manual — All 10 Outputs)

For each generated explanation, verify:
- [ ] Correct claim ID and factor pairing
- [ ] No fabricated denial reason, payer policy, or medical assertion
- [ ] Includes uncertainty qualifier
- [ ] Contains exactly one recommended action
- [ ] Action is supported by the input payload
- [ ] Professional tone, no jargon
- [ ] 2–3 sentences
- [ ] No API key or internal metadata in output

### 17.5 Submission QA

- [ ] PDF renders cleanly (2–3 pages, readable, no encoding artifacts)
- [ ] CSV opens correctly in Excel and Python; first row is header
- [ ] README links are all valid
- [ ] Repository visibility matches recruiter clarification (private default)
- [ ] No API credentials, private communications, or unnecessary confidential input committed

---

## 18. Production Thinking — Monitoring, Drift & MLOps

> **This section demonstrates "Lead, AI" systems thinking — present this in the write-up's future improvements section and/or in conversation.**

### 18.1 Model Monitoring Plan

| Signal | Metric | Alert Threshold | Action |
|---|---|---|---|
| Denial rate drift | Population denial rate vs. training baseline | ±5 percentage points | Investigate; retrain if sustained |
| Feature drift | PSI per feature | PSI > 0.2 | Investigate data pipeline; consider retraining |
| Probability calibration drift | Brier score on new labeled claims | > 0.03 degradation | Recalibrate or retrain |
| Queue yield monitoring | Precision@25% on outcomes | Drop > 5 points | Review threshold; check for feature drift |
| LLM output quality | Automated validation failure rate | > 5% failure rate | Review prompt; update prompt version |

**PSI (Population Stability Index) — Standard Thresholds:**
- PSI < 0.10: No change — stable
- PSI 0.10–0.25: Moderate change — investigate
- PSI > 0.25: Significant change — retrain model

### 18.2 Retraining Strategy

1. **Trigger-based retraining:** Automatically queue retraining when PSI > 0.25 for any top-5 feature, or when denial rate drifts >5 pp.
2. **Scheduled retraining:** Monthly retraining cadence incorporating new labeled claim outcomes from 835 remittance data.
3. **Train on expanding window:** Include all available labeled data, not just the most recent window.
4. **Evaluation before deployment:** New model must beat current model by ≥2% on held-out recent months before deployment.

### 18.3 Human-in-the-Loop Feedback

- Log analyst override decisions (flagged claim not reviewed; low-risk claim actually denied)
- Use this feedback to identify systematic blind spots
- Develop a "model confidence" score that indicates when the model is outside its training distribution

### 18.4 A/B Testing Framework (Shadow Mode)

1. Deploy new model in shadow mode — score claims but don't change workflow
2. Measure if top-25% flagged claims actually had higher denial rates in payer outcomes
3. Compare shadow model's queue yield vs. production model
4. Gradually roll out after 30–60 days of shadow validation

### 18.5 HIPAA & Data Governance (Production Awareness)

- This assessment uses synthetic data — no PHI concerns here
- **However**, state in the write-up: any production integration of LLM APIs with real patient claim data requires a BAA (Business Associate Agreement) with the LLM provider
- Audit logging on all model predictions and LLM calls
- Data encryption at rest and in transit
- Model output must never be displayed without a human reviewer in the loop

---

## 19. Dollar-Weighted Business Metrics

> **This was in the Codex plan but missed in the DeepSeek plan and missing from our initial plan. It directly reflects Ensemble Health's RCM-centric perspective.**

Beyond claim count, measure financial exposure:

```python
def dollar_weighted_capture_at_k(y_true, proba, expected_payment, k_fraction=0.25):
    """
    Measures the fraction of total at-risk revenue (expected_payment of denied claims)
    that falls within the top k% of ranked claims.
    """
    n = len(y_true)
    k = int(np.floor(k_fraction * n))
    sorted_idx = np.argsort(proba)[::-1]

    top_k_idx = sorted_idx[:k]
    y_top_k = y_true.iloc[top_k_idx]
    ep_top_k = expected_payment.iloc[top_k_idx]

    denied_in_top_k_dollars = ep_top_k[y_top_k == 1].sum()
    total_denied_dollars = expected_payment[y_true == 1].sum()

    return denied_in_top_k_dollars / total_denied_dollars if total_denied_dollars > 0 else 0.0
```

**Why this matters:**
- A model that captures $3.5M of denied claims vs. $2.8M (both in the top 25%) tells a very different business story than claim count alone.
- An inpatient claim denied for $87,000 is worth far more recovery effort than an outpatient claim denied for $500.

**Reported in write-up:**

| Model | Count Capture@25% | Dollar Capture@25% | Lift (Count) | Lift (Dollar) |
|---|---|---|---|---|
| Random | ~26% | ~25% | 1.0× | 1.0× |
| Rule heuristic | ~40% | ~44% | 1.5× | 1.7× |
| Selected model | [value] | [value] | [value]× | [value]× |

> **Hypothesis:** Dollar-weighted capture may be higher than count capture because high-dollar claims (inpatient) tend to have more complex billing with more potential for documentation, auth, and coding gaps — which our model is specifically designed to detect.

---

## 20. Risks, Limitations & Mitigations

| Risk / Limitation | Why It Matters | Mitigation / Disclosure |
|---|---|---|
| PDF confidentiality vs. GitHub request | Public push may violate provided restriction | Use private repo; request clarification; document assumption |
| Deadline conflict (5 days vs. 1 week) | Late submission risk | Work to earlier deadline until clarified |
| Synthetic data ≠ production complexity | Real claims have ICD-10, CPT codes, EDI 837 fields, prior history, claim notes | Explicitly state all performance claims are on synthetic data |
| Temporal extrapolation (2025 current vs. 2024 train) | Payer policies change; model may be stale | Report temporal limitation; recommend monthly retraining cadence |
| Higher test-period denial rate (26% vs 21%) | Temporal drift is observable even in supplied data | Report both rates honestly; do not hide in cross-validation averaging |
| Payer P011/P012 data dictionary mismatch | Silent removal would delete valid rows | Retain all rows; document discrepancy explicitly |
| Calibration may be imperfect | Named output column is `denial_probability` — implies calibration | Check Brier score; apply Platt Scaling; report pre/post calibration |
| LLM hallucination risk | In healthcare, unsupported assertions are particularly harmful | Structured JSON input, zero temperature, automated + manual validation |
| HIPAA consideration for production | Real claims may contain PHI; LLM APIs require BAA | Document for synthetic assessment; state production requirements |
| `payer policy / medical necessity` denials (20%) | Model has no access to clinical documentation | Be explicit that 20% of denials require clinical NLP, not structured features |
| Capacity/tier ambiguity | Two different concepts (rank-based tier vs. probability-based threshold) | Define and document both explicitly; keep separate in code and README |

---

## 21. Future Improvements Roadmap

In priority order (demonstrate systems thinking, not just ML tuning):

1. **Refine label definition** — Distinguish initial denial vs. rejection vs. partial payment vs. overturn. CARC/RARC codes from 835 remittance would enable this.
2. **Cost-sensitive optimization** — Optimize for dollar-weighted capture, not claim count. Train with sample weights proportional to `expected_payment`.
3. **Multi-class denial reason prediction** — Instead of binary, predict the *likely denial reason category*. Gives analysts more specific guidance ("this will likely be denied for missing auth" vs. generic "high risk").
4. **Payer-specific models** — Train separate models per payer or payer type, then ensemble with the global model. Denial patterns differ significantly by payer.
5. **Time-series cross-validation** — Expanding window CV by `service_month` to better simulate production (train on past, validate on immediate future).
6. **Production drift monitoring** — PSI-based feature drift alerting, nightly retraining pipeline on new 835 outcomes.
7. **Stacked ensemble** — XGBoost + LightGBM + CatBoost with LR meta-learner. ~1–3% AUC improvement.
8. **Clinical NLP integration** — For the 20% of denials driven by "payer policy / medical necessity" — natural language processing on clinical documentation, ICD-10 code specificity checks, procedure-diagnosis alignment validation.
9. **Analyst feedback loop** — Log when analysts override the model recommendation; use this to retrain and reduce systematic blind spots.
10. **Interactive Streamlit dashboard** — Real-time risk scoring interface where analysts see SHAP waterfall plots and LLM explanations in one view.
11. **Fairness audit** — Evaluate denial prediction rates across payer types (Medicaid vs. Commercial) to ensure the model is not systematically biased against government payer beneficiaries.
12. **Agentic integration** — Aligning with Ensemble Health's actual platform: the model feeds a denial-prevention agent that can autonomously draft authorization requests, pull documentation, or flag to a clinical reviewer — all before submission.

---

## 22. 5-Day Execution Schedule

| Day | Morning | Afternoon | Deliverable |
|---|---|---|---|
| **Day 1** | Send clarification email; set up private repo; implement data validation (`validate_data.py`) | EDA: target distribution, payer analysis, actionable conditions, drift comparison | Data audit, EDA notebook, initial README |
| **Day 2** | Feature engineering (all groups); preprocessing pipeline; ablation study on `service_month` | Baselines (Dummy + Rule heuristic + LR); comparison on validation | Baseline comparison table; feature engineering module |
| **Day 3** | XGBoost + LightGBM training + hyperparameter search; calibration; freeze final model | One-time test evaluation; score all 500 current claims; SHAP attribution | Test metric table; draft `predictions_current_claims.csv` |
| **Day 4** | Build LLM explanation engine; generate + validate all 10 explanations; low-risk test | PDF write-up drafting | Complete CSV; write-up draft |
| **Day 5** | Reproduce from clean environment; inspection pass on all artifacts; README finalization | Resolve repo access; submit GitHub link + CSV + PDF | ✅ Final submission |

**Total estimated technical effort:** 4–6 hours (within assessment guideline)

---

## 23. Final Deliverables Checklist

### 🎯 Mandatory Assessment Outputs

- [ ] GitHub repository link (private with reviewer access, or public if confirmed acceptable)
- [ ] `predictions_current_claims.csv` — sorted descending by `denial_probability`
- [ ] All 6 required columns: `claim_id`, `denial_probability`, `predicted_denial`, `risk_tier`, `top_risk_factors`, `explanation`
- [ ] Explanations generated and validated for top 10 highest-risk current claims
- [ ] Low-risk claim prompt behavior demonstrated
- [ ] Prompt template stored in `prompts/explanation_prompt_v1.txt`
- [ ] PDF write-up is exactly 2–3 pages
- [ ] PDF covers: problem framing, error priority, 3 data findings, baselines, model comparison, threshold, test metrics, capture@25%, LLM prompt + example, limitations, improvements

### 🛠 Engineering Evidence

- [ ] Code reproduces: validation → training → evaluation → scoring → explanation → output validation
- [ ] Original temporal split used exactly as supplied (not randomized)
- [ ] Leakage exclusions proven in code (`validate_features()` gate) and documented
- [ ] Test set evaluated exactly once after all choices frozen
- [ ] Dollar-weighted capture metric computed and reported
- [ ] Queue size and tier logic explicitly defined (125 High / 125 Medium / 250 Low)
- [ ] Probability calibration checked (Brier score + reliability diagram)
- [ ] Local risk-factor generation is model-derived, auditable, and validated against actual record values
- [ ] LLM outputs pass automated + manual grounding and format review
- [ ] API keys/secrets excluded from repository (`.gitignore`)
- [ ] Confidentiality and raw-input handling documented
- [ ] `requirements.txt` with pinned versions
- [ ] `RANDOM_STATE = 42` used everywhere

---

## 24. References & Research Basis

### EDA & Implementation Sources
1. Direct analysis of `claims_history.csv` and `current_claims.csv`, 2026-05-26. All statistics confirmed programmatically.

### Domain Research (External)
2. **Ensemble Health Partners + Cohere.** *Ensemble and Cohere Launch End-to-End Agentic AI Platform for Integrated Revenue Cycle Orchestration.* June 2025. https://www.ensemblehp.com
3. **HFMA.** *Standardizing Denial Metrics for the Revenue Cycle.* https://www.hfma.org
4. **CMS.** *Electronic Health Care Claims; Prior Authorization API FAQ.* https://www.cms.gov
5. **CAQH Index FAQ.** https://www.caqh.org
6. **NIST AI 600-1.** *Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile.* July 2024. https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf
7. **HHS.** *Guidance on HIPAA & Cloud Computing; Guidance on De-identification.* https://www.hhs.gov

### Academic & Technical Research
8. **Johnson, M., Albizri, A., & Harfouche, A. (2021).** Responsible AI in Healthcare: Predicting and Preventing Insurance Claim Denials. *Information Systems Frontiers*, 25, 2179–2195. — AdaBoost AUC 0.83; white-box models preferred for trust.
9. **Lundberg, S.M., & Lee, S.I. (2017).** A Unified Approach to Interpreting Model Predictions. *NeurIPS.* — SHAP framework.
10. **Tasche, D. (2018).** A Plug-in Approach to Maximising Precision at the Top and Recall at the Top. — Theoretical basis for threshold-at-capacity approach.
11. **Lukassen, F., et al. (2026).** From XAI to Stories: A Factorial Study of LLM-Generated Explanation Quality. — LLM for XAI narrative translation; structured prompting.
12. **Springer Nature (2025).** A Prompt Framework for Enhancing LLM-Based Explainability of Medical ML Models. — 3-step prompting; fidelity 0.783.
13. **Opexia (2025).** ML-Powered Claim Denial Prediction. — XGBoost as standard; AUC 0.78–0.85 on 200K examples.
14. **CitiusTech (2020).** Predicting Claims Denial. — XGBoost, 95% sensitivity and 65% specificity on 835/837 data.
15. **scikit-learn docs.** Probability Calibration; Average Precision Score; TimeSeriesSplit. https://scikit-learn.org

---

*This plan is the definitive synthesis of the DeepSeek plan, the Codex Detailed Execution Plan, our own hands-on EDA, and targeted web research conducted 2026-05-26. It supersedes all earlier drafts and incorporates every element found across all three sources, with additional depth in: dollar-weighted metrics (Section 19), production monitoring (Section 18), probability calibration (Section 10), SHAP interaction values (Section 12.4), offline LLM fallback (Section 13.8), programmatic output validation (Section 14.3), full testing framework (Section 17), and employer intelligence alignment (Section 1).*
