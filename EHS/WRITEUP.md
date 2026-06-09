# Ensemble Health -- Project Writeup

> Pre-Bill Denial Prediction System: Approach, Methodology, and Results

---

## 1. Problem Statement

Healthcare revenue cycle management processes millions of claims annually. When a claim is denied after submission, the cost is substantial: $25-$118 in administrative rework per claim, 30-90 days of revenue delay, and potential permanent write-offs. The standard approach is reactive: wait for the denial, then fix it.

Ensemble Health wants a proactive approach: predict which claims are likely to be denied BEFORE submission, rank them by risk so billers can fix issues upfront, and explain why each high-risk claim is flagged.

---

## 2. Approach Overview

The solution follows a four-stage pipeline:

1. **Feature Engineering:** Transform 17 raw claim columns into 53 model-ready features capturing administrative gaps, financial patterns, temporal factors, and payer-visit interactions.

2. **Model Training:** Train 10 model architectures on 3,200 historical claims (with known denial outcomes), evaluate on a held-out validation set, and select the best performer.

3. **Risk Scoring:** Score 500 current claims using the selected model, assign High/Medium/Low risk tiers at the 25%/25%/50% splits, and extract top risk factors from model coefficients.

4. **Explanation Generation:** Generate plain-English explanations for all 500 claims: 125 High-tier via GenAI (gemma4:31b-cloud), 375 Medium/Low via deterministic templates.

---

## 3. Key Findings

### 3.1 The Data is Additive

The EDA revealed that denial probability follows an approximately linear, additive function of administrative gaps. Each missing authorization, documentation issue, or referral gap independently pushes denial risk up by a roughly constant amount. This is the central insight that drove every subsequent decision.

### 3.2 Linear Models Dominate

Logistic Regression achieved 49.51% Capture@25 on validation -- tied for best across 10 experiments. XGBoost, the industry-standard gradient boosting algorithm, scored only 35.92% -- a 13.6 percentage point gap. The additive data structure favors linear models fundamentally; tree-based models split on features one at a time and cannot efficiently represent additive relationships.

### 3.3 Calibration is a Free Lunch

Platt scaling (sigmoid calibration via 5-fold CV) preserved the capture rate while improving the Brier score by 34% (0.209 to 0.137). This means probability estimates became significantly more reliable without any degradation in the business metric.

### 3.4 The Stacking Paradox

The Stacking ensemble achieved the best ROC-AUC (0.718) and Brier (0.136) across all experiments -- but only ranked 6th on Capture@25 (46.60%). The meta-learner compressed extreme-tail probabilities, making the top 25% less distinguishable. This demonstrates that standard ML metrics can select suboptimal models for constrained business problems.

---

## 4. Active Learning Experiment Results

### 4.1 Experiment Design

10 experiments were executed covering 6 model families: linear (LR), tree ensembles (RF, XGB, GBM), neural networks (MLP), kernel methods (SVC), and ensemble methods (Voting, Stacking). The LR family included three variants: baseline, calibrated, and interaction-augmented.

### 4.2 Complete Results Table

| Exp | Model | Capture@25 | ROC-AUC | Brier | PR-AUC | Train(s) |
|:---:|:---|---:|---:|---:|---:|---:|
| 1 | LR Baseline | 49.51% | 0.711 | 0.209 | 0.430 | <0.1 |
| 2 | LR Calibrated | 49.51% | 0.710 | 0.137 | 0.428 | 0.3 |
| 5 | LR Interactions | 49.51% | 0.707 | 0.210 | 0.415 | <0.1 |
| 6 | GBM | 48.54% | 0.661 | 0.151 | 0.375 | 2 |
| 7 | Voting Ensemble | 48.54% | 0.696 | 0.166 | 0.409 | 3 |
| 8 | Stacking | 46.60% | 0.718 | 0.136 | 0.433 | 8 |
| 10 | SVC RBF | 45.63% | 0.641 | 0.148 | 0.327 | 10 |
| 9 | MLP Neural | 43.69% | 0.678 | 0.146 | 0.391 | 15 |
| 4 | RF Robust | 39.81% | 0.653 | 0.155 | 0.366 | 3 |
| 3 | XGB Tuned | 35.92% | 0.615 | 0.180 | 0.341 | 5 |

### 4.3 Artifact Structure

Each experiment run produces MLflow-compatible artifacts:
```
experiments/{exp_name}/{timestamp_hash}/
    params.json          -- Model hyperparameters
    metrics.json          -- Validation metrics (capture, AUC, Brier, PR-AUC)
    model.pkl             -- Serialized model
    predictions.csv       -- Validation predictions with probabilities
    preprocessor.pkl      -- ColumnTransformer for reproducibility
    feature_names.json    -- Feature names after encoding
    run_metadata.json     -- Timestamp, Python version, git hash
```

---

## 5. Production LLM Audit Infrastructure

### 5.1 Architecture

The audit system (`src/llm_audit.py`) provides HIPAA-compliant, self-contained logging for every prediction:

```python
class LLMAuditLogger:
    - Records token counts from ollama.ChatResponse (prompt_eval_count, eval_count)
    - Validates response quality (disclaimer, actionable language, min length)
    - Flags hallucination/PII markers (ICD codes, dollar amounts, patient identifiers)
    - Computes call hash, latency, and validation flags
    - Writes timestamped JSON logs to data/output/audit_logs/
```

### 5.2 Audit Coverage

| Aspect | Detail |
|---|---|
| API calls (High-tier) | 125 via gemma4:31b-cloud |
| Template bypass (Medium + Low) | 375 deterministic explanations |
| Total audit records | 500 (100% coverage) |
| Token tracking | prompt_eval_count + eval_count from ChatResponse |
| Quality validation | disclaimer, actionable content, min length, hallucination scan |
| PII markers checked | ICD codes, dollar amounts, patient identifiers in responses |

### 5.3 Quality Checks

| Check | Method | Pass Rate |
|---|---|---|
| JSON parse | Direct parse or markdown fence stripping | 100% |
| Pydantic validation | Field validators on ExplanationResponse | 100% |
| Disclaimer keyword | "estimate", "statistical", or "not guaranteed" | 100% |
| Hallucination markers | Regex for codes, amounts, identifiers | 0 hits |

### 5.4 Production Metrics

| Metric | Value |
|---|---|
| Total tokens consumed | ~56,000 |
| Average API latency | ~1,940 ms |
| Fallback activations | 0 (all API calls succeeded) |
| Template-generated explanations | 375 |

---

## 6. Production Recommendations

1. **Monthly retraining cycle** with closed-claim outcomes to address the 4.9 pp temporal drift between training and test periods.
2. **Per-payer sub-models** to reduce false positive rate for Medicaid MCO claims (35% of FPs vs 24% of population).
3. **ICD/CPT code integration** to capture medical necessity, coding error, and payer policy denials currently invisible to the model.
4. **Online feedback loop** collecting biller corrections, claim outcomes, and explanation quality ratings.
5. **Performance monitoring dashboard** with automatic alerting on capture@25, Brier score, and PSI degradation.
6. **Model registry with version control** for all experiment artifacts and production model rollback capability.

---

## 7. Deliverables Checklist

| # | Deliverable | Location | Status |
|---|---|---|---|
| 1 | predictions_current_claims.csv (500 rows, 6 columns) | data/output/ | Verified |
| 2 | metrics.json (key metrics, threshold) | data/output/ | Verified |
| 3 | Source code (modular, documented, tested) | src/ | Verified |
| 4 | Prompt files (Pydantic-validated) | src/prompts/ | Verified |
| 5 | Unit tests (82 passing, 6 modules) | tests/ | Verified |
| 6 | requirements.txt (exact versions) | root | Verified |
| 7 | .env.example (no secrets) | root | Verified |
| 8 | README.md (complete documentation) | root | Verified |
| 9 | ANALYSIS.md (technical deep-dive) | root | Verified |
| 10 | WRITEUP.md (methodology and results) | root | Verified |
| 11 | DETAILED_PROCESS_DOCUMENTATION.md | root | Verified |
| 12 | Project_documentation.docx (manager-facing) | root | Verified |
| 13 | INTERVIEW_QUESTIONS.md (150 questions) | root | Verified |
| 14 | INTERVIEW_ANSWERS (5 parts, all 150 answered) | root | Verified |
| 15 | Audit logs (timestamped JSON) | data/output/audit_logs/ | Verified |
| 16 | Experiment artifacts (10 experiments) | data/output/experiments/ | Verified |

---

## 8. Technical Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.12 |
| Data processing | pandas | 2.2.2 |
| ML framework | scikit-learn | 1.5.2 |
| Gradient boosting | xgboost | 2.1.0 |
| Data validation | Pydantic | >=2.0 |
| GenAI | ollama (gemma4:31b-cloud) | 0.4.7 |
| Configuration | python-dotenv | 1.0.1 |
| Testing | pytest, pytest-cov | >=8.0, >=5.0 |
| Manager deliverable | python-docx | (build-time) |
