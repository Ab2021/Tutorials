# 🏗️ Optum AI System Design — Healthcare Use Cases
### Domain Round System Design Playbook

> **Format:** For every system design question, spend 5 minutes clarifying, 35 minutes designing, 10 minutes on evaluation + monitoring + Responsible AI.
> Always end with: "The most critical thing I haven't mentioned yet is Responsible AI — here's how I'd handle HITL, auditability, and fairness..."

---

## THE 8-STEP SYSTEM DESIGN FRAMEWORK

```
Step 1: CLARIFY (5 min)
└── Who are the users? What is the latency requirement? What data exists? 
    What's the cost of a wrong decision? What regulatory constraints apply?

Step 2: SCOPE (2 min)
└── "For this conversation, I'll focus on X. I'll call out where I'm making 
    simplifying assumptions."

Step 3: HIGH-LEVEL ARCHITECTURE (5 min)
└── Draw the box diagram: Data Sources → Processing → Model → Output → Feedback Loop

Step 4: DATA LAYER (5 min)
└── Sources, format, preprocessing, labeling strategy, feature store

Step 5: MODEL LAYER (10 min)
└── Model selection + justification, training approach, alternatives considered

Step 6: SERVING LAYER (5 min)
└── Latency requirements, batching vs. real-time, feature computation at serve time

Step 7: EVALUATION + MONITORING (5 min)
└── Offline metrics, online A/B, drift detection, retraining triggers

Step 8: RESPONSIBLE AI (3 min)
└── HITL gates, fairness checks, explainability, HIPAA/audit trail
```

---

## DESIGN 1: Clinical Note Summarization System

**Prompt:** "Design an AI system that automatically summarizes clinical notes for physicians before patient visits."

### Clarification:
- Who reads summaries? Attending physicians (high expertise, time-constrained)
- Volume: ~500K notes/day across the network
- Latency: Pre-visit summary (batch overnight) vs. real-time (before physician enters room)
- Regulatory: PHI must stay in HIPAA-compliant environment; output is clinical decision support (CDS) not autonomous decision-making
- Accuracy requirement: Zero tolerance for fabricated clinical facts

### Architecture:

```
CLINICAL NOTE SUMMARIZATION PIPELINE

DATA SOURCES:
├── EHR System → Unstructured clinical notes (SOAP, discharge summaries, consult notes)
├── Claims Data → Structured diagnosis codes, procedure history
└── Lab/Imaging Reports → Semi-structured results

STAGE 1: DOCUMENT INGESTION & DE-IDENTIFICATION
├── NLP PII/PHI extractor (Presidio or AWS Comprehend Medical)
├── Replace PHI with tokens: [PATIENT_NAME], [DOB], [MEMBER_ID]
├── Retain de-identified version for LLM processing
└── PHI mapping table encrypted in KMS (for re-identification in final output only)

STAGE 2: PREPROCESSING
├── Document classification: Note type (progress note, discharge, consult, lab)
├── Temporal ordering: Sort notes chronologically per encounter
├── Relevance filtering: For visit-specific summary, filter to last 6 months
└── Chunk assembly: Aggregate relevant chunks within LLM context window

STAGE 3: STRUCTURED SUMMARIZATION (LLM)
├── Model: Claude 3 Sonnet via AWS Bedrock (within VPC, HIPAA-eligible)
├── Prompt structure:
│   - System: "You are a clinical documentation specialist. Summarize ONLY facts 
│              explicitly stated in the provided notes. Never infer or extrapolate.
│              Flag any inconsistencies across notes."
│   - Few-shot: 3 examples of gold-standard physician-approved summaries
│   - Output schema: {chief_complaint, active_conditions, medications, 
│                     recent_labs, allergies, visit_context, flags}
│   - Temperature: 0.1 (minimal creativity)
│
└── Faithfulness enforcement: Each summary section must cite source note + date

STAGE 4: QUALITY GATE
├── Faithfulness check: LLM-as-judge verifies each statement traced to source
├── Completeness check: Did all active medications get captured? (structured cross-check)
├── Hallucination flag: If faithfulness score <0.90, route to human review
└── PHI re-insertion: Token→PHI mapping applied after all quality checks

STAGE 5: DELIVERY
├── Physician EHR portal: Summary available 2 hours before scheduled appointment
├── Format: Structured, scannable (not a wall of text)
└── Feedback mechanism: Physician 1-click rating (accurate/inaccurate/missing info)

FEEDBACK LOOP:
├── Physician corrections → Weekly dataset for prompt refinement
├── 1-click ratings → Ongoing calibration of quality thresholds
└── Monthly: Sample of 100 summaries reviewed by clinical informaticists
```

### Evaluation:
| Metric | Target | How |
|---|---|---|
| Faithfulness | >0.92 | RAGAS faithfulness on 500-case gold standard |
| Completeness | >0.95 | % of active meds captured (structured cross-check) |
| Physician satisfaction | >4.0/5.0 | Weekly rating survey |
| Hallucination rate | <2% | Clinical informaticist spot check (10% sample) |
| Processing latency (batch) | <30 min for full overnight run | End-to-end pipeline timing |

### Responsible AI:
- **HITL:** Any note with faithfulness <0.90 routes to a human clinical documentation specialist
- **Explainability:** Every summary sentence is hyperlinked to its source note for physician verification
- **Fairness:** Monitor summary quality by patient demographic — ensure no systematic quality gap
- **Audit trail:** Full prompt + response + quality scores logged per summary, retained 6 years (HIPAA)
- **PHI handling:** PHI never leaves VPC; LLM is called within AWS Bedrock with BAA; de-identified before any LLM call

---

## DESIGN 2: Insurance Claims Fraud Detection (Healthcare Context)

**Prompt:** "Design an end-to-end fraud detection system for healthcare insurance claims."

*(Note: You've built this at Chubb — use that experience but adapt to healthcare/Optum scale)*

### Clarification:
- Claim types: Medical, pharmacy, durable medical equipment
- Volume: 1M+ claims/day (Optum scale)
- Latency: Real-time detection at submission (<200ms) OR batch (within 24h)?
- Fraud types: Provider billing fraud (upcoding, phantom services), member identity fraud, duplicate claims
- Ground truth: Delayed (confirmed fraud takes 90+ days from investigation)

### Architecture:

```
HEALTHCARE CLAIMS FRAUD DETECTION — TWO-SPEED SYSTEM

SPEED 1: REAL-TIME GATE (<200ms at claim submission)
├── Feature computation (pre-computed in feature store):
│   ├── Member: claim velocity, historical spend, recent address changes
│   ├── Provider: billing anomalies, procedure:diagnosis ratio, peer benchmarking
│   └── Claim: procedure code validity, date logic, duplicate fingerprint
├── Rule engine: Fast deterministic rules (known fraud codes, impossible combinations)
├── GBM model: LightGBM on 500+ tabular features (<20ms inference)
│   ├── Training: Point-in-time correct features (no temporal leakage)
│   ├── Class weight: scale_pos_weight = 99 (1% fraud base rate)
│   └── Threshold: Calibrated to Recall=80% / Precision=30% operating point
└── Output: Risk score 0-100; HIGH (>75) → pend claim; MEDIUM (50-75) → flag; LOW → auto-pay

SPEED 2: ASYNC DEEP ANALYSIS (within 24h for pended claims)
├── RAG pipeline: Retrieve similar past fraud cases from investigation KB
│   ├── Embedding: Domain fine-tuned on claims text pairs (fraud/legitimate)
│   └── Hybrid search: BM25 + dense for ICD code exact matching + semantic
├── NLP extraction: BERT-based IE from claim notes, physician letters, lab reports
├── Graph analysis: Provider network + member network for ring/collusion detection
│   ├── Node: member, provider, facility, address, phone number
│   ├── Edge: relationship (treated_by, billed_for, shares_address)
│   └── Signal: Unusually dense subgraph = potential fraud ring
├── LLM synthesis: GPT-4/Claude structured fraud assessment from all signals
│   ├── Output: {fraud_type, evidence, confidence, recommended_action}
│   └── Guardrails: Faithfulness check; no action taken without human review gate
└── Output: Comprehensive fraud assessment → SIU (Special Investigations Unit) queue

FEATURE STORE:
├── Online (Redis): Pre-computed real-time features per member/provider
├── Offline (BigQuery/Redshift): Historical features for batch training
└── Point-in-time joins: Critical — train features must not include post-claim information

FEEDBACK LOOP:
├── SIU investigator verdicts (90-day lag) → Label for model retraining
├── Appeals outcomes → Negative labels for false positive reduction
└── Quarterly retraining triggered by: PSI >0.25 OR F1 drop >5%
```

### Key Evaluation Metrics:
```
Offline:
├── PR-AUC (primary — class imbalance makes ROC-AUC misleading)
├── Recall@80%Precision as the operating point metric
└── Subgroup AUC by claim type, provider specialty (fairness)

Online (Shadow Mode → A/B):
├── Recall: % of confirmed fraud flagged (60-day lag validation)
├── False positive rate: % of legitimate claims flagged
├── SIU precision: % of flagged claims confirmed as fraud by investigators
└── Guardrail: Member false positive rate <5% (can't systematically deny legitimate care)
```

### Responsible AI:
- **HITL mandatory for all adverse actions:** No claim denied purely on model score
- **Explainability:** SHAP values for every flagged claim → SIU investigator can see top 5 reasons
- **Fairness auditing:** Flag if any protected class shows higher false positive rate >2% above baseline
- **Appeals process:** All flagged claims must have clear human review and appeals pathway
- **HIPAA:** All audit trails, member data handling strictly PHI-compliant

---

## DESIGN 3: Drug-Drug Interaction LLM Assistant

**Prompt:** "Design an AI assistant that helps pharmacists check for drug-drug interactions."

### Architecture (Abbreviated):

```
KNOWLEDGE BASE:
├── Structured: FDA DrugBank, interactions database (known DDIs)
├── Unstructured: Clinical literature on novel interactions
└── Update: Weekly ingestion of FDA safety communications

HYBRID RETRIEVAL:
├── Exact match: BM25 for drug name lookup (critical — 'metoprolol' must match exactly)
├── Semantic: Dense embeddings for symptom/mechanism queries
└── Re-ranking: By severity of interaction (fatal → serious → moderate → minor)

LLM GENERATION:
├── Claude/GPT-4 via Bedrock, grounded strictly in retrieved KB
├── Structured output: {interaction_severity, mechanism, clinical_management, citations}
├── Faithfulness: EVERY claim must cite a specific source in the KB
└── Uncertainty: If KB coverage is incomplete → EXPLICIT "No reliable data found; consult clinical pharmacist"

GUARDRAILS (ZERO TOLERANCE FOR ERRORS):
├── Hallucination: If faithfulness <0.95 → BLOCK output, escalate to pharmacist
├── Source verification: All cited papers verified to exist before output
├── Safety floor: If ANY interaction is rated CONTRAINDICATED → mandatory pharmacist review
└── Audit: Every query + response logged with pharmacist ID for malpractice protection

EVALUATION:
├── Recall on known DDI test set: Must be >0.99 (missing a fatal interaction = unacceptable)
├── Precision: >0.85 (some false positives acceptable vs. missing real interactions)
├── Faithfulness: >0.97 target (higher than general RAG due to safety stakes)
└── Latency: <3 seconds at point-of-care
```

---

## DESIGN 4: Member Health Risk Stratification

**Prompt:** "Design an ML system to stratify 10M+ members by health risk for proactive outreach."

```
PROBLEM FRAMING:
└── Predict: 12-month probability of high-cost event (hospitalization, ER visit)
    For: Chronic condition management outreach prioritization

DATA SOURCES:
├── Claims: Medical, pharmacy, lab (longitudinal, structured)
├── EHR (where available): Clinical notes, vitals, diagnoses
├── Social Determinants (SDOH): Zip-code level poverty, food access, transportation
└── Engagement: Prior program participation, call response rates

FEATURE ENGINEERING:
├── Comorbidity burden: Charlson Comorbidity Index (CCI) from diagnosis codes
├── Medication adherence: Proportion of Days Covered (PDC) from pharmacy fills
├── Utilization patterns: ER visit frequency, preventive care gaps
├── SDOH features: Food insecurity proxy, housing instability signals
└── Temporal features: Trend in utilization over 3/6/12 months (crucial!)

MODEL:
├── XGBoost (primary): Best tabular performance, interpretable via SHAP
├── Time-aware: Feature engineering captures temporal trends (not time-series model)
├── Calibration: Isotonic regression post-calibration — probability must be meaningful
│   (a 40% risk score must mean ~40% of those members will have an event)
└── Threshold by use case: Outreach budget determines operating point on PR curve

FAIRNESS (CRITICAL FOR OPTUM):
├── Train/evaluate separately by race, age group, gender, income proxy
├── Test: Equal calibration across groups (same predicted probability → same actual rate)
├── Mitigate: If model underperforms for a subgroup, add group-specific features or calibrate
└── Governance: Fairness report required before deployment (UHG RAI policy)

SCALE:
├── Training: PySpark on Databricks/EMR for 10M member dataset
├── Scoring: Batch weekly; store scores in feature store for downstream use
└── Update: Quarterly retraining (medical coding delays mean quarterly is most data-complete)
```

---

*End of System Design Playbook*
