# Ensemble Health Partners AI/ML Take-Home Assessment - Detailed Execution Plan

Prepared: 2026-05-26  
Purpose: implementation-ready plan for the supplied Classical ML + GenAI denial-risk assignment  
Source folder: `Hiring Assessment - AI Team - Classical ML + Gen AI problem/`

## 1. Executive Summary

This assessment is a pre-bill claims denial prioritization problem. The operational user is a front-end review team that can inspect only the highest-risk 25% of claims before submission. The submission must therefore do more than report generic classification accuracy: it must demonstrate that the chosen model concentrates true denials in a limited review queue and provides grounded, useful next actions for analysts.

The proposed submission will:

1. Build a leakage-safe binary classifier using `claims_history.csv` and its supplied temporal `split` column as-is.
2. Choose the model and operating logic using validation performance, with denial capture in the top 25% review queue as the primary business metric.
3. Evaluate exactly once on the held-out test period and report ranking, probability, classification, and dollar-exposure metrics.
4. Score all 500 records in `current_claims.csv`, producing the required `predictions_current_claims.csv`.
5. Generate constrained, field-grounded LLM explanations for only the top 10 current claims, with deterministic validation of the outputs.
6. Deliver a concise 2-3 page PDF write-up, reproducible code, a README, and the scored CSV.

Two administrative issues require action before publishing the final submission:

- The email requests upload to GitHub, but the PDF is marked `Confidential - For Candidate Use Only - Do Not Distribute`. Use a private repository or obtain written confirmation before making any assessment content public.
- The email says `Deadline: 1 Week`, while the PDF says `Within 5 business days of receiving this document`. Unless clarified, plan delivery to the earlier five-business-day requirement.

## 2. Supplied Artifact Review

### 2.1 Files received

| File | Purpose | Observed size | Use in solution |
| --- | --- | ---: | --- |
| `Hiring Assessment - ML + Gen AI problem.pdf` | Formal assessment instructions | 92,325 bytes, 3 pages | Requirements authority for technical outputs |
| `mail_recieved.txt` | Recruiter submission instruction and deadline note | 755 bytes | Administrative constraints and delivery channel |
| `claims_history.csv` | Labeled historic claim outcomes | 456,972 bytes, 3,200 rows | Model development and held-out evaluation |
| `current_claims.csv` | Current unlabeled claims | 59,681 bytes, 500 rows | Final scoring and LLM explanation population |

### 2.2 Requirements extracted from the PDF

| Requirement | Implementation response | Evidence to submit |
| --- | --- | --- |
| Build a binary classifier for denial prediction | Train candidate pipelines on supplied historical split | Code, model comparison table, test results |
| Use `claims_history.csv` for train/evaluation | Preserve `split=train/validation/test` exactly | Data-validation log and README |
| Score `current_claims.csv` | Apply frozen selected pipeline to all 500 current claims | `predictions_current_claims.csv` |
| Review team can inspect only top 25% by risk | Make capture at top 25% the primary selection metric | Test table and lift/capture chart |
| Generate top-10 claim explanations with an LLM | Pass only factual fields and deterministic drivers to LLM | Prompt template, outputs, validation evidence |
| LLM output must be grounded | Never send denial reasons or unobserved facts as claim evidence | Prompt constraints and post-generation checks |
| Explanation includes a recommended action | Map risk drivers to permitted operational actions | Top-10 explanation column |
| Explanation states risk is not a guarantee | Require qualifier in prompt and check output | Output validation |
| Explanation is 2-3 sentences, plain English | Enforce format and evaluate examples | CSV and PDF example |
| Show low-risk behavior | Run one low-risk prompt demonstration | README/PDF section |
| Output file named `predictions_current_claims.csv` | Generate exact filename | Repository output |
| Required CSV fields: `claim_id` | Retain source ID only as identifier, never feature | CSV column |
| Required CSV fields: `denial_probability` | Calibrated or verified probability score, `0.0-1.0` | CSV column and calibration evidence |
| Required CSV fields: `predicted_denial` | Apply disclosed validation-selected threshold | CSV and README |
| Required CSV fields: `risk_tier` | Define High/Medium/Low assignment explicitly | CSV and README |
| Required CSV fields: `top_risk_factors` | Produce two or three local model drivers | CSV and methodology |
| Required CSV fields: `explanation` | Populate top 10; blank/not-applicable for remaining records | CSV and README |
| Include prompt template | Versioned prompt in repo/report | `prompts/` file or README section |
| Submit PDF write-up of 2-3 pages | Produce focused executive/technical narrative | Final PDF |
| Submit GitHub link + Predictions + PDF | Package repository and delivery email | Submission checklist |

### 2.3 Requirements extracted from the email

| Email statement | Implication | Action |
| --- | --- | --- |
| Role: `Lead, AI` | Demonstrate judgment, not only model fitting | Include reproducibility, business metric, guardrails, limitation |
| Format shown as `[e.g., PDF, Word document]` | Email template is incomplete | Follow the PDF's explicit PDF write-up requirement |
| Deadline: `1 Week` | Conflicts with the PDF's five-business-day limit | Ask for confirmation; target earlier requirement meanwhile |
| Reply with assignment or link | A link-based delivery is acceptable | Send controlled-access repository link and attach/point to outputs |
| Upload assignment on GitHub and share link | Repository packaging is required | Use private repository unless confidentiality is clarified |

### 2.4 Conflicts and questions to resolve

Send one short clarification email early, without delaying implementation:

1. The assessment PDF states that it is confidential and must not be distributed, while the email asks for a GitHub link. May the repository be private with reviewer access, or is a public repository intended?
2. Should the submission deadline be interpreted as five business days (PDF) or one week (email)?

Until answered:

- Do not commit the PDF, source email, or raw supplied CSVs to a public repository.
- Keep implementation in a private/local workspace.
- Plan to submit by the earlier deadline.

## 3. Problem Framing

### 3.1 Business problem

A denied hospital claim causes avoidable rework, delayed payment, and potential lost revenue. The intervention occurs before claim submission: an analyst can review a limited queue and address correctable problems such as missing documentation, absent authorization, eligibility issues, referrals, or network concerns.

This is a prioritization system, not an automatic denial determination system:

- The model estimates denial risk from information available before submission.
- The review team decides what action to take.
- The LLM converts validated drivers into readable review guidance; it does not make coverage or clinical decisions.

### 3.2 Prediction target and prediction time

| Element | Definition |
| --- | --- |
| Unit of prediction | One claim |
| Target | `is_denied` (`1` denied, `0` not denied) |
| Prediction moment | Before claim submission/front-end review |
| Scoring population | Current claims awaiting review |
| Intervention capacity | At most 25% of scored claims |
| Required explanation population | Top 10 current claims by model probability |

### 3.3 Error trade-off

| Error | Operational consequence | Treatment in design |
| --- | --- | --- |
| False negative within review opportunity | A preventable denial may leave the queue without review, creating rework or lost revenue | Prioritize denial recall/capture within top 25% |
| False positive in review queue | Analyst spends scarce review time on a claim that may pay normally | Measure top-25% precision and lift; do not overfill queue |

Because review capacity is capped, globally maximizing recall without respecting queue size is not an appropriate objective. Ranking quality at the review cut is central.

### 3.4 Intended operational workflow

1. Ingest claims that are ready for pre-bill review.
2. Apply the same preprocessing and fitted denial-risk model.
3. Rank claims descending by `denial_probability`.
4. Route the top 25% to a review queue.
5. Display model-derived risk factors and, for the highest-priority subset, a short LLM explanation and recommended check.
6. Record analyst intervention and eventual payer outcome for future monitoring/retraining.

## 4. Domain Research and Design Implications

The following research was performed against external authoritative sources on 2026-05-26. Statements below are applied as design implications, not treated as additional labels in the supplied synthetic data.

| Source | Relevant finding | Implication for this assessment |
| --- | --- | --- |
| Ensemble Health Partners, AI/RCM announcement (2025) | Ensemble describes predictive models to prevent denials, front-end authorization activity, and a human-in-the-loop safety framework for AI across RCM. | Frame the submission as pre-bill decision support with human review and auditable explanations, aligned to the employer's stated operating model. |
| CMS, Electronic Health Care Claims | CMS distinguishes front-end edit/rejection steps from later payment-policy rejection or denial decisions, with returned reasons for failure. | State clearly that this supplied target models denials, not all clearinghouse/front-end claim rejections; a production label definition would need precision. |
| HFMA, Standardizing Denial Metrics | HFMA identifies initial denial rate by both claim volume and claim dollars as foundational measures for performance improvement. | Report count-based denial capture and dollar-weighted exposure capture, using available monetary fields. |
| CAQH Index FAQs | Administrative transaction measurement includes eligibility verification, prior authorization, claim submission, attachments, claim status, payment, and remittance advice. | Treat the supplied actionable flags as realistic administrative control points and recommend future integration with transaction outcomes. |
| CMS Prior Authorization API FAQ | CMS acknowledges automation in prior authorization while some decisions continue to need clinical reviewer evaluation. | Explanations recommend a review action, rather than stating the payer will deny or making an automatic authorization decision. |
| scikit-learn documentation | Time-ordered validation prevents training on future observations and evaluating on past observations; pipelines reduce preprocessing leakage; probability calibration must be verified. | Preserve the PDF-provided temporal split, fit transformations only on training data, and assess calibration before calling scores probabilities. |
| SHAP documentation | Tree SHAP can produce local feature contributions and can explain probability output under documented configurations. | Generate `top_risk_factors` from local model contributions, not an LLM guess. |
| NIST AI 600-1 GenAI Profile | GenAI may confidently generate false content; this is especially material in healthcare and consequential decision support; human over-reliance is a risk. | Constrain prompts to supplied facts/drivers, add deterministic validation and risk disclaimer, and retain a human reviewer. |
| HHS HIPAA cloud/de-identification guidance | A cloud provider handling ePHI for regulated work generally requires appropriate safeguards/BAA; properly de-identified information is treated differently under the HIPAA guidance. | The supplied synthetic files contain no real patient information per the brief; nevertheless, document that any production LLM integration with PHI requires approved handling and vendor governance. |

## 5. Data Audit Findings from Supplied CSVs

These findings were calculated directly from the supplied local CSV files without modifying them.

### 5.1 Dataset structure

| Item | Historical claims | Current claims |
| --- | ---: | ---: |
| Rows | 3,200 | 500 |
| Columns | 20 | 17 |
| Unique `claim_id` values | 3,200 | 500 |
| Duplicate claim IDs | 0 | 0 |
| Claim ID overlap between files | 0 | 0 |
| Service months | 2024-01 through 2024-12 | 2025-01 through 2025-02 |
| Outcomes present | Yes | No |
| Missing values | 2,509 expected blank `denial_reason` values on paid claims only | None |

### 5.2 Supplied temporal split

| Split | Months | Rows | Denied | Denial rate |
| --- | --- | ---: | ---: | ---: |
| Train | 2024-01 through 2024-08 | 2,122 | 448 | 21.11% |
| Validation | 2024-09 through 2024-10 | 539 | 103 | 19.11% |
| Test | 2024-11 through 2024-12 | 539 | 140 | 25.97% |
| Total | 2024-01 through 2024-12 | 3,200 | 691 | 21.59% |

Planning implications:

- The test period has a meaningfully higher denial rate than train/validation. The final write-up should treat this as temporal performance risk rather than hide it in random cross-validation.
- The split is already chronological and explicitly mandated by the brief. Do not randomize or re-split it.
- Current claims occur after test, making temporal generalization important.

### 5.3 Schema and integrity checks

| Check | Result | Implementation treatment |
| --- | --- | --- |
| Historical/current feature parity | Current contains all available-at-submission fields and omits `split`, target, and post-outcome reason | Build an explicit feature contract and assert equality before scoring |
| Target/reason consistency | 691 denied claims have reasons; 2,509 non-denied claims have blank reason; zero mismatches | Use `denial_reason` for post-hoc descriptive analysis only |
| Monetary validity | No record has `expected_payment > total_billed` | Retain check in validation script |
| Payment-to-billed ratio | Historical range `0.1800-0.8200`; current `0.2471-0.7712` | Optional derived pre-submission feature and drift check |
| Numeric missingness | None detected | Still implement missing-value handling for robustness |
| Binary values | Observed as `0/1` for all flag columns | Assert on ingest |
| Category parity | Payer types and visit types represented in both sets | Keep unknown-category handling in preprocessing |
| Data dictionary mismatch | PDF says payer IDs `P001-P010`, while both data files contain `P011` and `P012` | Document discrepancy; model observed categories and do not discard valid rows |

### 5.4 Volume and monetary exposure

| Population | Total billed | Total expected payment |
| --- | ---: | ---: |
| Historical claims | $38,926,101.51 | $19,302,921.46 |
| Historical denied claims | Not required for target framing | $4,584,725.99 |
| Current claims | $6,459,078.34 | $3,199,393.10 |

`expected_payment` is available pre-submission and represents the contractually expected receipt per the data dictionary. It should be used as a secondary business-impact weight, while the required prediction remains a probability of denial rather than a dollar-loss model.

### 5.5 Descriptive denial findings for manager-facing narrative

These are descriptive summaries of all historical labels for initial orientation. Model selection and threshold decisions must be based on the proper train/validation/test procedure, not on tuning against these full-data summaries.

| Group | Claims | Denied | Denial rate |
| --- | ---: | ---: | ---: |
| Medicaid MCO | 779 | 238 | 30.6% |
| Medicare Advantage | 558 | 142 | 25.4% |
| BCBS | 517 | 110 | 21.3% |
| Commercial | 1,346 | 201 | 14.9% |
| Inpatient | 664 | 203 | 30.6% |
| Outpatient | 1,477 | 295 | 20.0% |
| Emergency | 673 | 123 | 18.3% |
| Observation | 386 | 70 | 18.1% |

Candidate non-technical findings for the PDF:

1. Denial rate is 21.6% overall, and the latest two held-out months are higher at 26.0%, so the review team needs prioritization that remains useful as rates move over time.
2. Inpatient and Medicaid MCO claims each show about a 30.6% historical denial rate, substantially above commercial claims at 14.9%, suggesting useful routing context but not a sufficient standalone decision rule.
3. Several pre-submission administrative gaps are associated with substantially higher denial rates and are directly actionable by review staff.

### 5.6 Actionable condition findings

| Available-before-submission condition | Historical occurrence | Denial rate when present | Denial rate when absent | Current occurrence |
| --- | ---: | ---: | ---: | ---: |
| Prior authorization required but absent | 310 (9.7%) | 46.5% | 18.9% | 53 (10.6%) |
| Missing documentation flagged | 604 (18.9%) | 38.6% | 17.6% | 80 (16.0%) |
| Eligibility not verified | 420 (13.1%) | 36.0% | 19.4% | 61 (12.2%) |
| Referral required but absent | 261 (8.2%) | 33.0% | 20.6% | 34 (6.8%) |
| Days to submit over 30 | 479 (15.0%) | 32.8% | 19.6% | 61 (12.2%) |
| Out of network | 583 (18.2%) | 24.9% | 20.9% | 77 (15.4%) |

Use in submission:

- These conditions supply interpretable operational context and a defensible rule-based baseline.
- They also supply safe action mappings for the LLM, such as checking authorization evidence or resolving missing documentation.
- Do not claim causal prevention from descriptive association alone.

### 5.7 Denial reason distribution for post-outcome analysis only

| Historical denial reason | Count | Share of denied claims | Expected payment represented |
| --- | ---: | ---: | ---: |
| Payer policy or medical necessity issue | 140 | 20.3% | $763,716.91 |
| Documentation incomplete or missing | 118 | 17.1% | $782,428.35 |
| Patient eligibility could not be verified | 92 | 13.3% | $580,477.74 |
| Claim submitted after timely filing deadline | 83 | 12.0% | $491,880.12 |
| Authorization required but not on file | 73 | 10.6% | $668,638.03 |
| Procedure or diagnosis coding error | 69 | 10.0% | $655,742.99 |
| Provider not in network for this payer | 67 | 9.7% | $404,841.92 |
| Referral required but not present | 49 | 7.1% | $236,999.93 |

Critical leakage rule: `denial_reason` exists only after outcome and must never be used in preprocessing, features, probability calculation, risk-factor derivation, or an LLM input for a current claim. It may be used only to describe historic errors after predictions are frozen.

### 5.8 Initial current-versus-history drift observations

| Dimension | Largest observed current share movement vs history | Treatment |
| --- | --- | --- |
| Visit type | Inpatient increases from 20.8% to 25.4% (`+4.7` percentage points) | Include category drift table in artifacts; monitor impact because inpatient denial rate is elevated historically |
| Payer ID | `P003` increases `+2.5` percentage points; `P009` decreases `-2.0` | Do not over-interpret small sample shifts; support unknown categories |
| Payer type | Commercial increases `+1.9` percentage points; other shifts under `1.2` | Treat as modest shift |
| Timing | Current data is from 2025-01/02, later than all training/evaluation labels | Explicitly state temporal extrapolation limitation |

## 6. Data Contract and Leakage Prevention

### 6.1 Feature eligibility

| Column | Use as model input? | Reason |
| --- | --- | --- |
| `claim_id` | No | Identifier; no generalizable clinical/administrative signal |
| `payer_id` | Yes, categorical | Available before submission; data has 12 observed levels |
| `payer_type` | Yes, categorical | Available prior to outcome |
| `visit_type` | Yes, categorical | Available prior to outcome |
| `total_billed` | Yes, numeric | Known at scoring time |
| `expected_payment` | Yes, numeric | Known expected receipt; can also inform secondary weighted metric |
| `num_procedures` | Yes, numeric | Known before submission |
| `num_diagnoses` | Yes, numeric | Known before submission |
| Binary process/control flags | Yes | Known at review time and operationally interpretable |
| `days_to_submit` | Yes, numeric | Available at planned submission/review point |
| `service_month` | Primary model: exclude or use only after an ablation demonstrates benefit without temporal fragility | It is available, but can become a temporal proxy that does not generalize to new months |
| `split` | No | Evaluation assignment, not a feature |
| `is_denied` | Target only | Direct leakage if used as input |
| `denial_reason` | No | Post-outcome leakage |

### 6.2 Derived features permitted for candidate testing

Only derive features from information available before submission:

| Derived feature | Definition | Reason to test |
| --- | --- | --- |
| `payment_to_billed_ratio` | `expected_payment / total_billed` with zero guard | Contract/payment relationship may capture differing reimbursement contexts |
| `prior_auth_gap` | `prior_auth_required == 1 and has_prior_auth == 0` | Directly actionable missing-control pattern |
| `referral_gap` | `referral_required == 1 and referral_present == 0` | Directly actionable missing-control pattern |
| `admin_gap_count` | Sum of auth gap, referral gap, missing docs, unverified eligibility, out-of-network indicator | Simple rule-based burden/priority signal; test rather than assume |
| `submission_delay_band` | Transparent bands based on train-derived thresholds or unchanged numeric input | May improve interpretability; avoid payer-policy claims unsupported by data |

Do not derive:

- Any feature using outcome or denial reason.
- Encodings fitted on validation/test/current populations.
- Global target-mean encodings unless fitted within training-only folds and justified.

### 6.3 Programmatic validation gates

Implement a validation function that fails loudly before modeling or scoring if any condition is violated:

1. Input files load and required columns exist.
2. `claim_id` is non-null and unique within each file.
3. Historical and current IDs do not overlap.
4. `is_denied` contains only `0/1`; current has no target.
5. `denial_reason` is excluded from feature lists and is blank for non-denied records.
6. Numeric columns parse and are non-negative.
7. `expected_payment <= total_billed`.
8. Flag columns contain only binary values.
9. `service_month` parses as `YYYY-MM`.
10. Split values are exactly `train`, `validation`, and `test` with chronology preserved.
11. Current feature contract matches historical available-time columns.
12. Unknown categories at future scoring are handled without failure.
13. The observed payer-ID discrepancy (`P011`, `P012`) is reported, not silently removed.

## 7. Modeling Strategy

### 7.1 Principles

- Keep the work proportionate to a 4-6 hour assessment: rigorous, readable pipelines and focused comparisons are more valuable than a large tuning exercise.
- Prefer reproducibility and clear business alignment over an opaque performance gain.
- Preserve temporal validity and never tune on test.
- Evaluate probability quality because the required output calls the score `denial_probability`.
- Generate local risk drivers from the classifier, not from free-form LLM reasoning.

### 7.2 Development split discipline

| Stage | Data allowed | Purpose |
| --- | --- | --- |
| Initial EDA and validation | Descriptive view across files; label summaries clearly marked descriptive | Understand data contract and narrative |
| Fit candidate models | Train only (`2024-01` to `2024-08`) | Estimate parameters and preprocessing |
| Select candidate, feature set, calibration, threshold/tier policy | Validation only (`2024-09` to `2024-10`) | Make all choices before touching test metrics |
| Final unbiased evaluation | Test once (`2024-11` to `2024-12`) | Report generalization performance |
| Score current claims | Frozen tested pipeline and documented policy | Required submission output |

Recommended traceable approach: use the same frozen estimator evaluated on test to score current claims. If a later refit on train plus validation is attempted, label that output distinctly and do not imply its current scores are generated by the exact evaluated pipeline.

### 7.3 Candidate models

| Model | Role | Advantages | Required safeguards |
| --- | --- | --- | --- |
| Constant/prevalence baseline | Minimum benchmark | Shows expected performance without useful ranking | No threshold claim beyond baseline prevalence |
| Transparent rule score using actionable gaps | Operational baseline | Demonstrates value beyond obvious process flags | Define rules before evaluation; do not present as calibrated probability |
| Logistic regression pipeline | Core statistical baseline and likely production-friendly candidate | Interpretable, stable with small structured data, easy local contributions, often reasonable probability behavior | One-hot categories and scale numeric features in training-only pipeline; evaluate calibration |
| Gradient-boosted tree challenger (dependency permitting) | Nonlinear challenger | Captures interactions such as payer/visit/control combinations | Control complexity, calibrate if needed, derive SHAP drivers correctly |

Avoid a broad model zoo. The narrative should be: a clear baseline, a defensible interpretable model, and one nonlinear challenger.

### 7.4 Preprocessing pipeline

Implement a single reproducible pipeline per candidate:

1. Read CSVs with explicit schema checks.
2. Store identifiers outside model matrix.
3. Drop leakage fields (`claim_id`, `split`, `is_denied`, `denial_reason`) from `X`.
4. Add only permitted derived features inside a reusable transformer or deterministic function.
5. Impute numeric/categorical fields defensively even though supplied data has no relevant missing values.
6. For logistic regression:
   - One-hot encode categorical fields with unknown-category tolerance.
   - Scale numeric fields.
7. For tree challenger:
   - Apply encoding compatible with the selected library without fitting on non-training data.
8. Record package versions and random seeds.

### 7.5 Candidate comparison sequence

1. Fit the prevalence and rule-based baselines.
2. Fit logistic regression on training data.
3. Fit one nonlinear challenger with small, documented hyperparameter options.
4. Generate validation probabilities for all candidates.
5. Compare primary and supporting metrics on validation.
6. Test derived-feature and `service_month` ablations on validation only.
7. Examine calibration; calibrate only if it improves probability reliability without materially degrading review-queue capture.
8. Freeze the chosen model, features, calibration approach, risk tiers, and classification threshold.
9. Evaluate frozen pipeline exactly once on test.

### 7.6 Explainability strategy for `top_risk_factors`

The CSV requires two or three features most responsible for each current claim's score.

Preferred implementation by final model:

| Selected model | Driver method | Presentation form |
| --- | --- | --- |
| Logistic regression | Per-record transformed feature value multiplied by fitted coefficient; rank positive contributions | Human-readable labels such as `Missing documentation flagged` or `Required prior authorization is not on file` |
| Tree model | SHAP local contributions, configured and verified for the selected model output | Select top positive drivers, map back to plain labels |

Rules:

- Report drivers that increase estimated denial risk, not merely high absolute contributions that reduced risk.
- Provide two drivers if only two positive drivers exist; do not manufacture a third.
- Drivers must refer to factual claim fields or deterministic derived fields.
- Never call a feature a proven denial cause.
- Validate that displayed values match the scored input record.
- If a non-actionable factor is material (for example payer type), it may be displayed as context; the recommendation should still target a supported operational check.

## 8. Metrics and Decision Policy

### 8.1 Capacity definition

The team can review only the top 25% of claims by risk. To avoid exceeding stated capacity, define:

```text
review_capacity_k = floor(0.25 * number_of_claims)
```

Consequences:

| Population | Rows | Maximum review queue `k` |
| --- | ---: | ---: |
| Validation | 539 | 134 |
| Test | 539 | 134 |
| Current scoring file | 500 | 125 |

This policy should be stated in README and PDF. Sorting ties at the boundary must be deterministic, for example by descending probability then ascending `claim_id`.

### 8.2 Primary metric

**Denial capture at top 25%**:

```text
number of actual denied claims in top-k risk-ranked records
-----------------------------------------------------------
number of actual denied claims in evaluation population
```

Why primary: it answers the operational question directly - how many denials can be surfaced within the review team's fixed capacity?

### 8.3 Supporting metrics

| Metric | Purpose |
| --- | --- |
| Precision at top 25% | Indicates analyst queue yield: how frequently reviewed claims are actually denied |
| Lift at top 25% | Compares queue denial rate against population denial rate |
| Expected-payment-weighted denial capture at top 25% | Measures concentration of potential reimbursement exposure in the queue |
| Average precision / PR-AUC | Evaluates ranking with a minority denial class without relying only on a single cut |
| ROC-AUC | Supporting discrimination metric familiar to reviewers, not the decision driver |
| Brier score and/or log loss | Validates probability quality |
| Calibration curve/reliability summary | Checks whether values labeled probability are interpretable |
| Confusion matrix, precision, recall, F1 at declared threshold | Supports required `predicted_denial`, while not replacing queue metrics |

### 8.4 Threshold and risk-tier policy

Two concepts must be kept distinct:

- Queue routing is capacity-based ranking: top 125 current claims constitute the review queue.
- `predicted_denial` is a binary label required in the output CSV and needs a disclosed probability threshold.

Recommended decision procedure:

1. On validation only, choose a `predicted_denial` probability threshold based on the best documented trade-off between denied-claim recall and review burden, preferably aligned to the 25% operating capacity.
2. Freeze that threshold before test evaluation.
3. Apply it without change to test and current probabilities.
4. Report how many current records the frozen threshold flags; do not silently change it to force a desired count.

Recommended `risk_tier` scheme for operational clarity:

| Tier | Rule | Meaning |
| --- | --- | --- |
| High | Top 25% of current ranked claims (125 claims) | Within available manual review capacity |
| Medium | Next 25% of current ranked claims (125 claims) | Next priority if capacity expands |
| Low | Remaining 50% (250 claims) | Lower priority under current constraint |

This tiering is rank/capacity based and therefore separate from the fixed binary threshold. State that distinction clearly.

### 8.5 Test results table template

Populate this only after choices are frozen:

| Model | AP / PR-AUC | ROC-AUC | Brier | Capture @ top 25% | Precision @ top 25% | Lift @ top 25% | Expected payment capture @ top 25% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prevalence baseline |  |  |  |  |  |  |  |
| Rule baseline |  |  | N/A unless calibrated |  |  |  |  |
| Logistic regression |  |  |  |  |  |  |  |
| Tree challenger |  |  |  |  |  |  |  |
| Selected frozen model |  |  |  |  |  |  |  |

## 9. GenAI Explanation Plan

### 9.1 LLM role boundary

The LLM should perform constrained language generation only:

- Input: claim identifier, rounded risk probability, validated top local drivers, and only the actual relevant source field values needed to state an action.
- Output: a short analyst-facing explanation.
- It must not calculate risk, select top claims, create risk factors, predict denial reason, recommend a coverage decision, or infer missing clinical facts.

This separation is critical: classical ML supplies the ranking and deterministic explainer supplies the drivers; GenAI improves readability and actionability.

### 9.2 Input payload for each top-10 claim

Use a structured JSON object similar to:

```json
{
  "claim_id": "CCLM-00000",
  "denial_probability": 0.000,
  "risk_estimate_label": "High",
  "top_risk_factors": [
    {
      "factor": "prior_auth_gap",
      "fact": "Prior authorization is required and is not on file.",
      "permitted_action": "Confirm whether authorization can be obtained or documented before submission."
    },
    {
      "factor": "missing_documentation_flag",
      "fact": "Required supporting documentation appears to be missing.",
      "permitted_action": "Review and attach required documentation before submission."
    }
  ]
}
```

Do not provide:

- Historic `denial_reason`.
- Patient facts not included in the data.
- Unrestricted policy text or external clinical assertions.
- Raw records beyond necessary values.

### 9.3 Prompt template

Version a prompt file (for example, `prompts/explanation_prompt_v1.txt`) containing:

```text
SYSTEM
You write pre-submission claim review notes for a revenue-cycle analyst.
Use only the facts supplied in INPUT. Do not infer a payer decision, denial reason,
medical necessity, missing facts, or policy requirement beyond INPUT.

Return exactly 2 or 3 plain-English sentences.
Sentence content must:
1. Say that the claim has an estimated denial risk, not a guaranteed denial.
2. Mention only one or two supplied risk factors using their supplied fact text.
3. Give exactly one concrete action drawn from the supplied permitted actions.

Do not add codes, clinical conclusions, patient details, dollar amounts, or facts
that are not in INPUT. Do not claim the action guarantees payment.

INPUT
{structured_claim_payload}
```

### 9.4 Generation settings and audit record

| Control | Implementation |
| --- | --- |
| Temperature | Low/deterministic setting where supported (`0` preferred) |
| Model ID | Record exact provider/model/version in README and generation log |
| Prompt version | Store versioned prompt template in repository |
| API key | Read from environment variable; never commit secret |
| Population | Generate only for sorted top 10 records |
| Provenance | Store sanitized input payload, output text, prompt version, model name, and generation timestamp |
| Fallback | If no API access exists, include prompt and clearly label 2-3 manual example outputs as allowed by the PDF |

### 9.5 Programmatic and manual validation

Validate each generated output before populating the CSV:

| Validation | Failure handling |
| --- | --- |
| Output contains 2 or 3 sentences | Regenerate once with stricter reminder or manually correct and label correction |
| Mentions risk as estimate/not guarantee | Reject output that asserts a denial will happen |
| Includes exactly one supported recommended action | Reject unsupported action or multiple conflicting actions |
| Mentions only supplied driver facts | Reject hallucinated policy, missing information, or medical statements |
| Uses no real PHI | Only synthetic assessment input is permitted here |
| Plain-English review | Manually inspect all 10 outputs due to small volume |

### 9.6 Required low-risk behavior demonstration

Select the lowest-risk current claim after scoring, pass its actual low-risk drivers/protective facts through the same prompt, and show that:

- It does not claim the claim will be paid.
- It does not invent a required correction.
- It can state that no major supplied pre-submission risk flag is highlighted and recommend routine verification before submission.

Include one low-risk example in README or appendix without adding it to the required top-10 explanation rows unless desired and clearly identified.

### 9.7 Example explanation style (template only, not a scored result)

```text
This claim has an elevated estimated denial risk, but a denial is not certain, because
the supplied record indicates that required prior authorization is not on file. Before
submission, confirm whether the required authorization can be obtained and documented.
```

Final examples must be generated from actual selected claim facts and must contain one, not multiple, recommended actions in accordance with the validation rule.

## 10. Output CSV Design

### 10.1 Required output schema

Create `predictions_current_claims.csv`, sorted descending by `denial_probability`, with exact required columns:

| Column | Definition | Quality check |
| --- | --- | --- |
| `claim_id` | Original current claim ID | Every source ID appears exactly once |
| `denial_probability` | Selected model probability, ideally rounded to 4-6 decimals in exported display without altering ranking computation | Numeric in `[0, 1]`; descending order |
| `predicted_denial` | `1` if probability meets frozen validation-selected threshold, else `0` | Contains only `0/1`; threshold stated in README |
| `risk_tier` | Capacity/rank-defined `High`, `Medium`, `Low` | Exactly 125 High, 125 Medium, 250 Low under recommended scheme |
| `top_risk_factors` | Two or three model-derived positive risk drivers in readable form | Drivers trace to record and explainer; no leakage |
| `explanation` | LLM explanation for top 10 only | Top 10 non-blank and validated; other rows blank or clearly documented |

### 10.2 Sorting and tie behavior

Implement stable sorting:

```text
denial_probability descending, claim_id ascending
```

Calculate ranking from full precision probability values. Rounded export values must not change the established row order.

### 10.3 Final CSV validation checklist

1. Filename exactly matches requirement.
2. Row count equals 500.
3. Required columns appear in the required documented order.
4. All current claim IDs occur once and no unexpected ID exists.
5. Probabilities are non-null, numeric, within bounds, and sorted descending.
6. `predicted_denial` values match the stated threshold.
7. High tier contains the top 125 records under the recommended capacity rule.
8. Top 10 records contain validated explanation text.
9. Rows outside top 10 follow documented blank-explanation convention.
10. No historical target or denial reason appears in the file.

## 11. Recommended Repository and Artifact Structure

Do not initialize or publish until confidentiality handling is confirmed. The eventual private repository should be simple and reviewer-friendly:

```text
ensemble-denial-risk-assessment/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- data/
|   |-- README.md                       # data origin/schema; raw files omitted if required
|-- notebooks/
|   |-- 01_eda_and_model_review.ipynb   # optional readable analysis narrative
|-- src/
|   |-- validate_data.py
|   |-- features.py
|   |-- train_evaluate.py
|   |-- score_current.py
|   |-- generate_explanations.py
|   |-- validate_outputs.py
|-- prompts/
|   |-- explanation_prompt_v1.txt
|-- outputs/
|   |-- predictions_current_claims.csv
|   |-- test_metrics.json
|   |-- figures/
|       |-- precision_recall_curve.png
|       |-- calibration_curve.png
|       |-- top25_capture_summary.png
|-- report/
|   |-- Ensemble_AI_Assessment_Writeup.pdf
|-- tests/
|   |-- test_data_contract.py
|   |-- test_output_contract.py
```

Repository controls:

- Include `.env`, local API keys, secrets, generated model binaries if unnecessarily large, and any private communication in `.gitignore`.
- Determine whether source CSVs and the confidential PDF may be committed to a private repo; if not, document where reviewers can obtain the supplied inputs.
- Keep predictions and report reproducible by a single command or short documented sequence.

## 12. Implementation Work Plan

### Phase 0 - Clarify delivery constraints and secure workspace

Deliverables:

- Clarification email sent regarding confidential/public repository and deadline mismatch.
- Private local working directory and `.gitignore` prepared.
- Decision log capturing assumptions pending recruiter response.

Completion criteria:

- No confidential artifact is publicly shared.
- Target due date is tracked conservatively as the earlier stated constraint until clarified.

### Phase 1 - Build data validation and EDA

Tasks:

1. Implement loading and data-contract validations from Section 6.3.
2. Reproduce row counts, split/month alignment, label prevalence, and missingness results.
3. Create descriptive summaries by payer type, visit type, actionable flags, and denial reason.
4. Create current-versus-history distribution comparison.
5. Save a compact machine-readable data validation report and manager-friendly EDA tables/plots.

Required artifacts:

- Data validation script.
- EDA notebook or report-generating script.
- Table of the documentation mismatch (`P001-P010` described versus `P001-P012` present).

Completion criteria:

- Assertions pass on both supplied CSVs.
- No target/post-outcome leakage field exists in the training feature matrix.

### Phase 2 - Establish baselines and candidate model pipelines

Tasks:

1. Define capacity metric implementation with `floor(0.25*n)`.
2. Implement prevalence baseline.
3. Implement transparent actionable-gap/rule-ranking baseline.
4. Implement logistic regression pipeline.
5. Implement one tree challenger if dependency/environment permits.
6. Evaluate validation ranking, probability quality, and queue metrics.
7. Run a small feature ablation for deterministic derived features and optional exclusion/inclusion of `service_month`.

Completion criteria:

- All candidate comparisons use validation only for selection.
- At least one clear baseline and one learned model comparison exists.
- All preprocessing is fitted on training data only.

### Phase 3 - Freeze model and evaluate test period

Tasks:

1. Select candidate based primarily on validation denial capture at top 25%, with probability reliability and simplicity considered.
2. Select/freeze probability calibration method if needed.
3. Freeze classification threshold and tier policy.
4. Run one-time test evaluation.
5. Produce final metric table, PR curve, calibration summary, top-25% capture/lift and dollar exposure summary.
6. Conduct post-hoc error/denial-reason analysis only after predictions are fixed.

Completion criteria:

- Test results are not used for tuning.
- The final selected model and all decision rules are explicitly documented.
- Performance narrative honestly addresses the higher denial prevalence in the test months.

### Phase 4 - Score current claims and create risk drivers

Tasks:

1. Score all 500 current claims with the exact frozen evaluated model.
2. Sort deterministically and assign High/Medium/Low tiers.
3. Apply frozen binary threshold for `predicted_denial`.
4. Generate local positive-risk driver text for every claim, or at minimum all records if required for output completeness.
5. Validate ID coverage, ordering, and tier sizes.

Completion criteria:

- Intermediate predictions file passes output schema checks before any LLM calls.
- Top 10 records are fixed by model ranking and cannot be changed by LLM output.

### Phase 5 - Generate and validate GenAI explanations

Tasks:

1. Create versioned prompt template.
2. Build structured, minimal JSON payloads for the top 10 current claims.
3. Use approved LLM API or the PDF-permitted manually drafted fallback.
4. Validate every explanation for grounding, format, action, and uncertainty statement.
5. Run the same prompt on one actual low-risk claim and record behavior.
6. Insert only validated top-10 explanations into final output CSV.

Completion criteria:

- Every top-10 explanation is traceable to its model drivers and claim fields.
- No explanation introduces unsupported facts or claims guaranteed denial.
- Prompt, LLM method/fallback, and limitations are documented.

### Phase 6 - Produce write-up and README

Tasks:

1. Write the 2-3 page PDF using the outline in Section 13.
2. Write README with reproducibility, thresholds, artifacts, metrics, and GenAI method.
3. Ensure the README distinguishes descriptive historical findings from held-out test performance.
4. Cite relevant authoritative external sources without overstating real-world generalization from synthetic data.

Completion criteria:

- PDF meets every bullet required by the assessment.
- A reviewer can reproduce the outputs from README commands and locate the required CSV immediately.

### Phase 7 - Final verification and controlled submission

Tasks:

1. Run scripts from a clean environment with documented dependencies.
2. Re-run output contract checks.
3. Open and inspect PDF layout and final CSV columns.
4. Scan repository for secrets, local paths, confidential source material policy violations, and accidental generated clutter.
5. Push to the approved GitHub visibility level.
6. Submit link and requested deliverables by email.

Completion criteria:

- Controlled-access requirement resolved and honored.
- Link works for intended reviewer.
- Final submission contains the required predictions and PDF.

## 13. PDF Write-Up Blueprint (2-3 Pages)

The PDF needs to be short, so reserve detailed supporting evidence for README/notebook.

### Page 1 - Problem, data, and approach

- Business objective: focus limited pre-bill review capacity on likely denial risk.
- Error trade-off: missed denials versus wasted review effort.
- Dataset summary: historical/current counts, temporal split, overall denial rate.
- Two or three manager-facing data findings:
  - elevated test-period denial rate,
  - actionable gaps and their descriptive association,
  - high-risk payer/visit segments as context.
- Mention synthetic data and observed payer-ID dictionary discrepancy in a short note.

### Page 2 - Modeling and test evaluation

- Models compared: prevalence/rule baseline, logistic regression, tree challenger.
- Leakage controls and supplied temporal split.
- Selection metric: denial capture at top 25%; supporting metrics.
- Compact test metric table and one chart.
- Threshold and risk tier definitions.
- Clearly state limitation due to temporal shift and synthetic data.

### Page 3 - GenAI, output, and next improvements

- LLM role: translate model-derived risk factors into concise analyst guidance.
- Prompt template excerpt and one top-risk example.
- Low-risk behavior example.
- Guardrails: factual input only, human review, output validation, no PHI in supplied dataset.
- Improvement roadmap: richer claim-line/denial codes, feedback outcomes, drift monitoring, secure production LLM governance.

## 14. README Content Blueprint

The README should contain:

1. Assignment objective and operational review constraint.
2. Repository file map and how to run validation, training, scoring, explanation generation, and output validation.
3. Data use statement:
   - supplied synthetic dataset;
   - excluded leakage columns;
   - split used as provided;
   - payer-ID data dictionary mismatch.
4. Feature list and engineered-feature definitions.
5. Model comparison and selected model rationale.
6. Metrics definitions including exact `top 25%` calculation and tie breaking.
7. Frozen threshold and risk-tier definition.
8. Test set metric results.
9. GenAI provider/model or manual-output fallback, prompt version, validation steps, and low-risk example.
10. Deliverable links and limitations.
11. Reproducibility details: environment, dependency installation, random seed, and commands.
12. Confidentiality handling and whether raw inputs are intentionally omitted.

## 15. Testing and Quality Assurance

### 15.1 Unit/data tests

| Test | Expected result |
| --- | --- |
| Unique IDs and no history/current overlap | Pass |
| Exact expected schema | Pass or fail loudly on deviation |
| Binary field constraints | Pass |
| Split chronological ordering | Pass |
| Outcome/reason consistency | Pass |
| Leakage columns excluded from features | Pass |
| Feature transformer handles unseen category | Pass with no crash |
| Queue-size function does not exceed 25% | Validation/test `134`, current `125` |
| Probability bounds | Every scored probability in `[0,1]` |
| Output rows/ID set match current claims | Exactly 500 and identical ID set |
| Explanations only applied to correct top 10 | Pass |

### 15.2 Analytical QA

| Review | Question |
| --- | --- |
| Metric verification | Does an independent recomputation reproduce top-25% capture and weighted capture? |
| Leakage inspection | Is there any path through code/notebook where `denial_reason`, `is_denied`, or `split` enters features? |
| Calibration review | Is it defensible to label exported scores probabilities? |
| Explainability review | Do top factors correspond to positive local risk contributions? |
| Drift review | Does current population differ materially from train/test in observed inputs? |

### 15.3 GenAI QA

Manually inspect all ten high-risk explanations plus the low-risk demonstration:

- correct claim/risk/factor pairing;
- no fabricated denial cause;
- no fabricated payer policy;
- exactly one action;
- includes uncertainty qualifier;
- appropriate professional tone;
- no secret or source metadata disclosed.

### 15.4 Submission QA

- PDF renders cleanly and is 2-3 pages.
- CSV opens cleanly, has correct headers, and sorts correctly.
- README links are valid.
- Repository visibility matches recruiter clarification.
- No API credentials or unnecessary confidential inputs are committed.

## 16. Risks, Limitations, and Mitigations

| Risk or limitation | Why it matters | Mitigation / honest disclosure |
| --- | --- | --- |
| Confidential PDF versus GitHub request | Public push may violate provided restriction | Obtain confirmation; use private repository/access-controlled link by default |
| Deadline conflict | Late submission risk | Work to five-business-day interpretation while clarifying |
| Small synthetic dataset | Results may not represent production claim complexity | State limitation; avoid production performance claims |
| Outcome rate rises in test period | Temporal drift may reduce reliability | Use chronological evaluation, report period rates, propose monitoring |
| Data dictionary omits payer IDs present in data | Silent data cleaning would be unjustified | Use observed categories and document discrepancy |
| Denial reason is tempting but post-outcome | Leakage would invalidate performance | Exclude programmatically; use only for frozen-error analysis |
| Required probability score may be uncalibrated | Ranked scores can be misleading as absolute risks | Check calibration; calibrate based only on development data if necessary |
| A model can surface non-actionable segments | Analyst needs actionable work, not a label | Pair factors with supported administrative checks and keep human review |
| GenAI hallucination/confabulation | Unsupported actions can misdirect review | Restricted payload, constrained prompt, low temperature, manual check for all 10 |
| LLM API/data governance in real deployment | Real claims could contain PHI | For this synthetic assignment disclose no PHI; for production require approved HIPAA/security governance and contracts |
| Capacity/risk-tier ambiguity | Arbitrary tiers could misstate operational workload | Explicit rank-based tiering and separate frozen binary threshold |

## 17. Improvements With More Time or Production Data

Prioritize improvements in this order:

1. Refine label definition to distinguish initial denial, rejection, partial payment, overturned denial, and final write-off, aligned to revenue-cycle reporting.
2. Incorporate standardized remittance/denial reason categories (for example, appropriate CARC/RARC-derived outcomes) for post-outcome root-cause analysis, not as pre-submission leakage.
3. Add claim-line, payer policy, authorization workflow status, documentation completeness, eligibility transaction result, coding edits, and prior operational history features available before submission.
4. Optimize a cost-aware review policy using expected payment and measured cost of manual review or rework, while retaining claim-count capture.
5. Establish monitoring for prevalence drift, feature drift, probability calibration, queue yield, denial capture, expected-payment capture, analyst actions, and eventual denial outcomes.
6. Evaluate subgroup performance by payer type and visit type for operational fairness/reliability concerns, without asserting protected-class fairness from unavailable data.
7. Add secure, governed LLM integration with prompt/version logging, PHI controls, provider approval/BAA where applicable, red-team tests, and human acceptance tracking.
8. Conduct an analyst usability review of explanation text and action recommendations.

## 18. Suggested Five-Business-Day Schedule

The PDF states a 4-6 hour technical effort and a five-business-day submission window. The following schedule preserves buffer for clarification and packaging; core implementation can be compressed if needed.

| Day | Work package | Target output |
| --- | --- | --- |
| Day 1 | Send confidentiality/deadline clarification; set up private repo; implement data checks and EDA | Data audit, initial README, validation script |
| Day 2 | Build baselines and candidate pipelines; validation selection and ablations | Candidate comparison and selected policy |
| Day 3 | Freeze model; one-time test evaluation; score current records; derive factors | Test metric table and draft predictions |
| Day 4 | Run LLM/manual explanation workflow; validate outputs; draft PDF/README | Completed CSV and write-up draft |
| Day 5 | Reproduce clean run, inspect artifacts, resolve repository access, submit | Final GitHub link, CSV, PDF |

## 19. Final Deliverable Checklist

### Mandatory assessment outputs

- [ ] GitHub repository link shared under an approved visibility/access arrangement.
- [ ] `predictions_current_claims.csv`, sorted descending by probability.
- [ ] CSV columns: `claim_id`, `denial_probability`, `predicted_denial`, `risk_tier`, `top_risk_factors`, `explanation`.
- [ ] Explanations generated and validated for top 10 highest-risk current claims.
- [ ] Prompt template included.
- [ ] Low-risk claim prompt behavior demonstrated.
- [ ] PDF write-up is 2-3 pages.
- [ ] PDF covers problem framing, error priority, 2-3 data findings, model/baselines, threshold, test metrics, capture at top 25%, LLM prompt/example, limitation, and improvements.

### Engineering evidence

- [ ] Code reproduces validation, training, evaluation, scoring, and final CSV.
- [ ] Original split used exactly as supplied.
- [ ] Leakage exclusions proven in code and documentation.
- [ ] Test set evaluated after model/policy choices are frozen.
- [ ] Queue size and tier logic are explicitly defined.
- [ ] Probability/calibration evidence provided.
- [ ] Local risk-factor generation is model-derived and auditable.
- [ ] LLM outputs pass grounding and format review.
- [ ] API keys/secrets excluded from repository.
- [ ] Confidentiality and raw-input handling are documented.

## 20. Research Sources

### Supplied local sources

1. `Hiring Assessment - ML + Gen AI problem.pdf`, provided in the assessment folder, reviewed 2026-05-26.
2. `mail_recieved.txt`, provided in the assessment folder, reviewed 2026-05-26.
3. `claims_history.csv` and `current_claims.csv`, provided in the assessment folder, profiled 2026-05-26.

### External sources consulted

1. Ensemble Health Partners. *Ensemble and Cohere Launch End-to-End Agentic AI Platform for Integrated Revenue Cycle Orchestration.* June 10, 2025.  
   <https://www.ensemblehp.com/blog/ensemble-cohere-agentic-ai-rcm/>
2. Ensemble Health Partners. *Tech + Innovation / Revenue Cycle Management Technology.*  
   <https://www.ensemblehp.com/innovation/>
3. Centers for Medicare & Medicaid Services (CMS). *Electronic Health Care Claims.*  
   <https://www.cms.gov/medicare/coding-billing/electronic-billing/electronic-healthcare-claims>
4. CMS. *Prior Authorization API - Frequently Asked Questions.*  
   <https://www.cms.gov/priorities/burden-reduction/overview/interoperability/frequently-asked-questions/prior-authorization-api>
5. Healthcare Financial Management Association (HFMA). *Standardizing Denial Metrics for the Revenue Cycle.*  
   <https://www.hfma.org/guidance/standardizing-denial-metrics-revenue-cycle-benchmarking-process-improvement/>
6. CAQH. *CAQH Index Frequently Asked Questions.*  
   <https://www.caqh.org/insights/frequently-asked-questions>
7. scikit-learn. *TimeSeriesSplit documentation.*  
   <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>
8. scikit-learn. *Average Precision Score documentation.*  
   <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html>
9. scikit-learn. *Probability Calibration documentation.*  
   <https://scikit-learn.org/stable/modules/calibration.html>
10. SHAP. *TreeExplainer documentation.*  
    <https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html>
11. National Institute of Standards and Technology (NIST). *Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile (NIST AI 600-1).* July 2024.  
    <https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf>
12. U.S. Department of Health and Human Services (HHS), Office for Civil Rights. *Guidance on HIPAA & Cloud Computing.*  
    <https://www.hhs.gov/hipaa/for-professionals/special-topics/health-information-technology/cloud-computing/index.html>
13. HHS, Office for Civil Rights. *Guidance Regarding Methods for De-identification of Protected Health Information in Accordance with the HIPAA Privacy Rule.*  
    <https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html>
