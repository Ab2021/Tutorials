# Ensemble Health -- Technical Analysis

> Pre-Bill Denial Prediction System: Model Performance, Experiment Results, and Technical Deep-Dive

---

## 1. Executive Summary

This document provides the technical analysis of the Ensemble Health pre-bill denial prediction system. The system predicts the probability that a healthcare claim will be denied by the payer before submission, ranks claims by risk for a 25% review capacity, and generates plain-English explanations for high-risk claims via GenAI.

**Production Model:** Calibrated Logistic Regression (Platt scaling, cv=5)

**Key Metrics (Test Set):**
| Metric | Value |
|---|---|
| Capture@25% | 45.7% |
| ROC-AUC | 0.691 |
| PR-AUC | 0.522 |
| Brier Score | 0.171 |
| Precision@25% | 47.8% |

---

## 2. Dataset Analysis

### 2.1 Overview

| Property | Historical Claims | Current Claims |
|---|---|---|
| Row count | 3,200 | 500 |
| Denial rate | 21.6% | Unknown |
| Time range | Jan-Dec 2024 | Jan 2025 |
| Features (raw) | 17 columns | 17 columns |

### 2.2 Administrative Gap Analysis

The single most important finding from EDA: denial probability is strongly additive with respect to administrative gaps.

| Gap Type | Denial Rate (Gap=1) | Denial Rate (Gap=0) | Lift |
|---|---|---|---|
| Missing Prior Authorization | 46.5% | 18.9% | +27.6 pp |
| Missing Documentation | 38.6% | 17.6% | +21.0 pp |
| Eligibility Not Verified | 36.0% | 19.4% | +16.6 pp |
| Late Filing (>30 days) | 32.8% | 19.6% | +13.2 pp |
| Missing Referral | 33.0% | 20.6% | +12.4 pp |
| Out-of-Network | 24.9% | 20.9% | +4.0 pp |

**Cumulative Effect:**
| Total Gaps | Denial Rate | Sample Size |
|---|---|---|
| 0 | 13.4% | 1,587 |
| 1 | 23.0% | 1,134 |
| 2 | 40.3% | 397 |
| 3 | 69.2% | 78 |
| 4 | 100.0% | 4 |

### 2.3 Payer-Type Analysis

| Payer Type | Denial Rate | vs. Baseline |
|---|---|---|
| Medicaid MCO | 28.6% | +7.0 pp |
| BCBS | 22.8% | +1.2 pp |
| Commercial | 21.2% | -0.4 pp |
| Medicare Advantage | 15.0% | -6.6 pp |

### 2.4 Temporal Drift

The test set (latest 15%, late 2024) has a 26.0% denial rate vs 21.1% in training -- a 4.9 pp gap confirming temporal pattern shifts. This drove the decision to retrain on full history for production scoring.

---

## 3. Feature Engineering

### 3.1 Final Feature Matrix

| Category | Count | Key Features |
|---|---|---|
| Administrative Gaps | 6 | auth_gap, referral_gap, doc_issue, elig_issue, network_issue, total_admin_gaps |
| Financial | 6 | payment_ratio, payment_gap, log_total_billed, log_expected_payment, billed_per_procedure, billed_per_diagnosis |
| Temporal | 4 | late_filing, submission_delay, service_month_num, service_quarter |
| Interactions | 4 | double_deficit, oon_auth_gap, oon_referral_gap, is_high_risk_combo |
| Complexity | 1 | complexity_score |
| One-Hot Encoded | 20 | payer_type (4), visit_type (4), payer_id (12) |
| Other Numeric | 5 | total_billed, expected_payment, num_procedures, num_diagnoses, days_to_submit |
| **Total** | **53** | |

### 3.2 Leakage Prevention

Columns programmatically excluded via `prepare_model_matrix()`:
- `claim_id`: Unique identifier
- `split`: Data partition label
- `is_denied`: Target variable
- `denial_reason`: Post-submission payer response
- `service_month`: Temporal identifier (converted to numeric)

### 3.3 Preprocessing Pipeline

```
Raw CSV -> build_features() (engineer 30+ features)
        -> prepare_model_matrix() (drop leakage columns)
        -> ColumnTransformer:
             OneHotEncoder(handle_unknown='ignore') -> categoricals
             StandardScaler() -> numerics
        -> 53 final features
```

---

## 4. Experiment Tracking & Active Learning

### 4.1 Experiment Leaderboard (10 Experiments)

| Rank | Experiment | Architecture | Capture@25 | ROC-AUC | Brier | PR-AUC |
|:---:|---|---:|---:|---:|---:|---:|
| 1 | exp1_lr_baseline | LR, C=0.1, balanced, lbfgs | **49.51%** | 0.711 | 0.209 | 0.430 |
| 1 | exp2_lr_calibrated | LR + Platt sigmoid, cv=5 | **49.51%** | 0.710 | 0.137 | 0.428 |
| 1 | exp5_lr_interactions | LR + interaction features | **49.51%** | 0.707 | 0.210 | 0.415 |
| 4 | exp6_gradient_boosting | GBM, depth=5, lr=0.05, subsample=0.8 | 48.54% | 0.661 | 0.151 | 0.375 |
| 5 | exp7_voting_ensemble | Soft Voting: LR(2x) + RF(1x) + GBM(1x) | 48.54% | 0.696 | 0.166 | 0.409 |
| 6 | exp8_stacking | Stacking: LR meta on LR+RF+GBM | 46.60% | **0.718** | **0.136** | **0.433** |
| 7 | exp10_svc_rbf | SVC RBF, C=1.0, Platt cv=3 | 45.63% | 0.641 | 0.148 | 0.327 |
| 8 | exp9_mlp_neural | MLP (64,32), ReLU, alpha=0.001 | 43.69% | 0.678 | 0.146 | 0.391 |
| 9 | exp4_rf_robust | RF, 200 trees, depth=12, balanced | 39.81% | 0.653 | 0.155 | 0.366 |
| 10 | exp3_xgb_tuned | XGB, 300 trees, depth=6, lr=0.05 | 35.92% | 0.615 | 0.180 | 0.341 |

### 4.2 Key Findings

1. **LR dominates due to data additivity.** The denial function is approximately linear in administrative gaps. LR captures this with O(d) parameters vs O(2^d) for trees.
2. **Stacking paradox:** Best ROC-AUC (0.718) and Brier (0.136) but worse Capture@25 (46.60%). The meta-learner compresses extreme-tail probabilities, hurting the top-25% ranking.
3. **XGBoost's catastrophic failure:** 35.92% capture, 13.6 pp below LR. L1/L2 leaf-weight regularization suppresses weak-but-critical gap signals needed for cumulative risk estimation.
4. **Neural networks data-starved:** 5,602 parameters on 2,240 samples (0.4:1 ratio) guarantees overfitting despite early stopping.
5. **Calibration as free lunch:** Platt scaling preserves ranking while improving Brier by 34% (0.209 to 0.137).

---

## 5. Model Selection: Calibrated Logistic Regression

### 5.1 Selection Criteria

| Criterion | exp2 (Calibrated LR) | exp8 (Stacking) | Winner |
|---|---|---|---|
| Capture@25 (validation) | 49.51% | 46.60% | LR |
| ROC-AUC | 0.710 | 0.718 | Stacking |
| Brier | 0.137 | 0.136 | Tie |
| Interpretability | High (coefficient attribution) | Low (three base models) | LR |
| Training time | ~0.3s | ~8s | LR |
| Deployment complexity | Single model | 4 models | LR |

**Decision:** Calibrated LR selected as production model. Equal capture with 34% better Brier than baseline, full interpretability, minimal deployment complexity.

### 5.2 Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| C (inverse regularization) | 0.1 | Optimal balance: suppresses noise, preserves gap signals |
| solver | lbfgs | Stable for small-to-medium datasets |
| class_weight | balanced | Compensates ~4:1 imbalance |
| Calibration | Platt, cv=5 | Monotonic probability rescaling without ranking impact |

### 5.3 Test Set Performance

| Metric | Value |
|---|---|
| Capture@25% | 45.7% |
| ROC-AUC | 0.691 |
| Brier (post-calibration) | 0.171 |
| Precision@25% | 47.8% |
| F1@25% | 46.9% |
| Validation Threshold (75th pctile) | 0.252 |

---

## 6. GenAI Explanation Engine

### 6.1 Architecture

- **Model:** gemma4:31b-cloud via `ollama.chat()`
- **Coverage:** 125 API calls (High-tier) + 375 deterministic templates (Medium/Low)
- **Prompt:** Structured JSON with role priming, risk factor inlining, and safety rules
- **Validation:** Two-layer Pydantic (ExplanationRequest input + ExplanationResponse output)
- **Fallback:** Three-tier: JSON parse -> regex extraction -> deterministic template

### 6.2 Pydantic Validation

**Input (ExplanationRequest):**
- `denial_probability`: Float, bounds [0, 1]
- `risk_estimate_label`: String, regex `^(High|Medium|Low)$`
- `top_risk_factors`: List[RiskFactorItem], max_length=5, each field min_length=1

**Output (ExplanationResponse):**
- `disclaimer`: Must contain "estimate", "statistical", or "not guaranteed"
- `claim_id`: Non-empty string
- `risk_description`: Min length enforced
- `recommended_action`: Min length enforced

### 6.3 Safety Rules (in prompt)

1. Do NOT invent new risks beyond those listed
2. Do NOT mention ICD-10, CPT, or HCPCS codes
3. Do NOT mention specific dollar amounts
4. Do NOT reference patient identifiers (names, MRNs, PHI)

### 6.4 Fallback Chain

| Tier | Method | Success Rate |
|---|---|---|
| 1 | Direct JSON parse | ~95% |
| 2 | Strip markdown fences, re-parse | ~4% |
| 3 | Regex extraction from malformed response | ~0.5% |
| Fallback | Deterministic template (build_fallback_response) | ~0.5% |

---

## 7. LLM Audit & Observability

### 7.1 Audit Infrastructure

The `LLMAuditLogger` (`src/llm_audit.py`) records every prediction:

```json
{
  "call_id": "unique-hash",
  "timestamp": "2026-05-27T11:32:44Z",
  "claim_id": "CLM01234",
  "tier": "High",
  "denial_probability": 0.62,
  "method": "api",
  "latency_ms": 1942,
  "prompt_tokens": 350,
  "completion_tokens": 95,
  "total_tokens": 445,
  "json_parse_method": "direct",
  "pydantic_valid": true,
  "disclaimer_valid": true,
  "has_uncertainty": true
}
```

### 7.2 Audit Coverage

| Category | Count |
|---|---|
| API calls (High-tier) | 125 |
| Template bypass (Medium/Low) | 375 |
| Total audit records | 500 (100% coverage) |
| JSON parse rate | 100% |
| Pydantic pass rate | 100% |

### 7.3 HIPAA Compliance Measures

- No external telemetry (self-contained local logging)
- No PII in audit logs (claim_id only, no patient data)
- Prompt rules prevent LLM from producing PHI
- All artifacts self-contained in `data/output/`

---

## 8. Experiment Tracker & Active Learning

### 8.1 Architecture

`ExperimentTracker` (`src/experiment_tracker.py`) provides MLflow-compatible artifact layout:

```
data/output/experiments/{exp_name}/{timestamp_hash}/
    params.json          -- Hyperparameters
    metrics.json          -- Validation metrics
    model.pkl             -- Serialized model
    predictions.csv       -- Validation predictions
    preprocessor.pkl      -- ColumnTransformer
    feature_names.json    -- Feature name mapping
    run_metadata.json     -- Timestamp, environment, git hash
```

### 8.2 Experiment Runner

`src/experiment_runner.py` executes all 10 experiments with:
- Consistent data splits (pre-defined split column)
- Identical preprocessing (shared ColumnTransformer for LR experiments)
- Automated metric computation (capture@25, ROC-AUC, Brier, PR-AUC)
- Comparison table generation

### 8.3 HIPAA Safety

- No external experiment tracking services (MLflow configured for local-only)
- No data leaves the execution environment
- All artifacts stored under `data/output/` (gitignored)

---

## 9. Risk Factor Extraction

### 9.1 Algorithm

```
For each feature f:
    contribution[f] = coefficient[f] * feature_value[f]
Sort descending by contribution magnitude (positive only)
Map raw feature names -> human-readable labels via NAME_MAP
Deduplicate labels, keep top 3 unique
If total_admin_gaps >= 2: promote "Multiple administrative gaps" as first factor
```

### 9.2 Feature Name Map (excerpt)

| Raw Feature Name | Human-Readable Label |
|---|---|
| auth_gap | Missing Required Prior Authorization |
| doc_issue | Missing Supporting Documentation |
| referral_gap | Missing Required Referral |
| elig_issue | Patient Eligibility Not Verified |
| network_issue | Provider Not in Payer Network |
| total_admin_gaps | Multiple administrative gaps |
| double_deficit | Compound deficit: missing auth + documentation |
| is_high_risk_combo | High-risk payer & visit combination |

---

## 10. Code Quality & Production Hardening

### 10.1 Best Practices Applied

| Practice | Implementation |
|---|---|
| Logging over print() | structured `logging` with timestamped format; 0 print() in pipeline |
| Validation over assert | `if/raise ValueError` for business-critical checks |
| DRY enforcement | Single source of truth for feature engineering (`build_features`) |
| Dynamic tiering | `n // 4` formula instead of hardcoded indices |
| Pre-flight validation | File existence checks before pipeline execution |
| Leakage prevention | Programmatic column exclusion in `prepare_model_matrix()` |
| ASCII-only source | No non-ASCII characters in Python source files |

### 10.2 Test Suite

| Module | Tests | Coverage |
|---|---|---|
| test_validate_data.py | 10 | CSV schema, value ranges, tier counts |
| test_feature_engineering.py | 16 | Feature creation, leakage exclusion, encoding |
| test_models.py | 14 | Training, evaluation, metrics |
| test_explainability.py | 10 | Coefficient extraction, factor mapping |
| test_explanations.py | 22 | Pydantic models, prompts, fallback |
| **Total** | **82** | **100% pass** |

---

## 11. Recommendations for Production Deployment

1. **Monthly retraining** with closed-claim outcomes to address temporal drift (4.9 pp gap)
2. **Per-payer sub-models** to reduce false positive rate for Medicaid MCO (35% of FPs vs 24% population)
3. **ICD/CPT code incorporation** to capture medical necessity and coding-related denials
4. **Online feedback loop** collecting biller corrections and claim outcomes for continuous improvement
5. **Performance monitoring** with automatic alerting on capture@25, Brier, and PSI degradation
6. **Model registry** with versioned artifacts and rollback capability
