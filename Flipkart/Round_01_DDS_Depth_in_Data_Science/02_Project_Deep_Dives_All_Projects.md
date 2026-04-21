# 🔴 Project Deep Dives — DDS Round
### All Projects: Training → Development → Production Architecture + Challenges

---

## PROJECT 1: Insurance Fraud Detection (RAG + LLMs + BERT) @ Chubb

### 🏗️ FULL PRODUCTION ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    INSURANCE FRAUD DETECTION SYSTEM                     │
│                          PRODUCTION ARCHITECTURE                        │
│                                                                         │
│  INGESTION LAYER                                                        │
│  ┌─────────────────────────────────────────────────────┐               │
│  │ Claims Sources: PDF, Email, Medical Notes, Adjuster │               │
│  │ Ingestion: Kafka topics (claims_stream, doc_upload) │               │
│  │ Storage: S3/GCS (raw) → Processed (Delta Lake)      │               │
│  └─────────────────────────┬───────────────────────────┘               │
│                             │                                           │
│  INFORMATION EXTRACTION                                                 │
│  ┌─────────────────────────▼───────────────────────────┐               │
│  │ ClinicalBERT / LegalBERT NER                        │               │
│  │ Extracts: dates, amounts, parties, medical codes    │               │
│  │ Weak supervision (Snorkel) for labeling functions   │               │
│  │ Output: Structured JSON entity map per claim        │               │
│  └─────────────────────────┬───────────────────────────┘               │
│                             │                                           │
│  EMBEDDING & VECTOR STORE                                               │
│  ┌─────────────────────────▼───────────────────────────┐               │
│  │ Domain fine-tuned embedding model (contrastive)     │               │
│  │ Vector DB: ChromaDB (dev) / Pinecone (prod)         │               │
│  │ Index: HNSW (Hierarchical Navigable Small World)    │               │
│  │ Collections: fraud_patterns, investigator_reports   │               │
│  └─────────────────────────┬───────────────────────────┘               │
│                             │                                           │
│  RAG + LLM RISK ASSESSMENT                                              │
│  ┌─────────────────────────▼───────────────────────────┐               │
│  │ Query: Structured claim JSON → embed → retrieve k=5 │               │
│  │ Context: top-k similar past fraud patterns          │               │
│  │ LLM: GPT-4 (temperature=0, structured JSON output)  │               │
│  │ Output schema: {risk_level, evidence[], confidence} │               │
│  └─────────────────────────┬───────────────────────────┘               │
│                             │                                           │
│  DUAL SERVING PATHS                                                     │
│  ┌────────────────┬────────▼────────────────────────────┐             │
│  │ Batch (Daily)  │  Real-time (New Claims)              │             │
│  │ Historical     │  Kafka Consumer → Feature Extract    │             │
│  │ backfill       │  → RAG Score → Redis cache           │             │
│  │ Airflow DAG    │  → Alert if score > threshold        │             │
│  └────────────────┴─────────────────────────────────────┘             │
│                                                                         │
│  MONITORING (Evidently AI + Custom)                                     │
│  ├── PSI on claim type distribution (weekly)                           │
│  ├── Embedding drift (centroid shift monitoring)                       │
│  ├── Faithfulness score trend (LLM-as-judge weekly sampling)           │
│  └── Investigator precision tracking (outcome feedback loop)           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Training Phase Challenges & Solutions:

| Challenge | Root Cause | Solution Applied |
|---|---|---|
| Label scarcity (< 2% fraud) | Insurance fraud is rare by definition | Weak supervision (Snorkel) — labeling functions from investigator rules |
| Domain vocabulary gap | General BERT doesn't know "subrogation", "NAIC codes", "loss runs" | Fine-tuned ClinicalBERT + LegalBERT on claims corpus |
| Hard negatives for contrastive learning | Random negatives too easy — model stops learning | FAISS-based hard negative mining: for each fraud claim, find top-20 nearest legitimate claims |
| Knowledge base freshness | Fraud patterns evolve quarterly | Automated KB refresh pipeline: new investigator-confirmed cases → embed → upsert to vector DB |
| LLM hallucination on rare fraud types | LLM invents evidence when retrieval returns weak context | Low-similarity threshold: if max cosine < 0.6, fallback to structured rules + human review |

### Development Phase Challenges:

| Challenge | Root Cause | Solution Applied |
|---|---|---|
| Evaluation without ground truth labels | Claims take months to mature → delayed labels | Shadow mode evaluation: compare to investigator verdicts 60 days after flagging |
| RAG latency too high (> 2s) | Serial: extract → embed → retrieve → generate | Parallelized extraction; async batch embedding; pre-computed top-k for common patterns |
| Context window overflow | Long claims > 4K tokens exceed LLM context | Hierarchical summarization: extract by section (injury, medical, financial), summarize each |
| Embedding model cold start | Initial knowledge base too small for meaningful retrieval | Started with rule-based system in parallel; KB grew from investigator feedback over first 90 days |

### Production Phase Challenges:

| Challenge | Root Cause | Solution Applied |
|---|---|---|
| Feedback loop delay | Fraud investigation takes weeks → delayed labels | Surrogate feedback: investigator opens case = positive signal even before verdict |
| Model-investigator trust gap | Investigators skeptical of "black box" AI flags | Added source citation (which past case was retrieved), SHAP for numerical features, human-readable evidence summary |
| Adversarial fraudsters adapting | Fraudsters learn model patterns over time | Added anomaly layer (isolation forest) orthogonal to supervised model — catches novel patterns |
| Regulatory audit requirements | Insurance regulators require explainable decisions | Every decision logged with: input features, retrieved evidence, LLM reasoning trace, final score |

---

## PROJECT 2: Agentic BI Tool @ Chubb (LangChain + Autonomous Agents)

### 🏗️ PRODUCTION ARCHITECTURE

```
User Query (Natural Language)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                  PLANNING LAYER                         │
│  LangChain ReAct Agent                                  │
│  System prompt: role + tool descriptions + constraints  │
│  Max iterations: 8 (loop prevention)                    │
│  Memory: ConversationBufferWindowMemory (last 5 turns)  │
└─────────────────────────────┬───────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐
│ SQL Query Tool  │  │ Python Exec Tool  │  │  Chart Gen Tool │
│ - Schema-aware  │  │ - Sandboxed       │  │ - Matplotlib    │
│ - Injection safe│  │ - Resource limits │  │ - Vega-Lite     │
│ - Read-only DB  │  │ - No file write   │  │ - PNG output    │
└────────┬────────┘  └────────┬─────────┘  └────────┬────────┘
         │                    │                      │
         └────────────────────┴──────────────────────┘
                               │
         ┌─────────────────────▼─────────────────────┐
         │              VALIDATION LAYER             │
         │  - Output schema validation               │
         │  - Numeric sanity check (no negatives)    │
         │  - Hallucination detection (cite sources) │
         │  - Safety filter (PII redaction)          │
         └─────────────────────┬─────────────────────┘
                               │
                        Final Report + Viz
```

### Training/Development Challenges:

| Challenge | Root Cause | Solution |
|---|---|---|
| Agent loops indefinitely on complex queries | No termination condition | Max iteration limit (8) + loop detection (same tool called 3× = abort) |
| SQL injection via user input | Malicious query embedded in NL | Parameterized queries only; NL → structured intent → safe SQL (never direct NL→SQL) |
| Token budget exhaustion | Multi-step conversation history grows unboundedly | Sliding window memory (last 5 turns) + hierarchical summary of older history |
| Hallucinated column names | LLM invents column names not in schema | Schema injection: full DDL in system prompt; validation: check all mentioned columns exist |
| Inconsistent answers to same query | LLM temperature > 0 | temperature=0 for all tool-calling steps; only final formatting step allows temp=0.1 |

### Production Challenges:

| Challenge | Root Cause | Solution |
|---|---|---|
| Out-of-scope queries | Users ask what's not in the DB | Scope classifier: embedding-based routing — OOD queries → graceful rejection |
| Data freshness confusion | User asks "current" but data is 24h stale | Inject data freshness timestamp in every response |
| Multi-turn context loss | Complex follow-up queries lose thread | RAG-based memory: store previous turns in vector DB, retrieve relevant context for follow-up |

---

## PROJECT 3: Marketing Mix Modeling (MMM) @ Axtria

### 🏗️ PRODUCTION ARCHITECTURE

```
Data Sources:
  ├── Sales data (weekly, by product/region)
  ├── Media spend (TV, print, digital, HCP calls)
  ├── External factors (competitor spend, formulary changes, holidays)
  └── Market research data

                    ┌──────────────────────────────────┐
                    │     DATA PIPELINE                │
                    │  Spark ETL → Feature Engineering │
                    │  Adstock & Saturation transforms │
                    └───────────────┬──────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
  Bayesian MMM             Ensemble MMM              Decomposition
  (Robyn/Meridian          (XGBoost + RF            (SHAP-based
  framework)               ensemble)                channel credit)
         │                          │
         └──────────┬───────────────┘
                    ▼
          Model Output:
          ├── Channel ROI ($ per $1 spent)
          ├── Marginal returns curve per channel
          ├── Saturation curves (diminishing returns)
          └── Optimal budget allocation

                    ┌──────────────────────────────────┐
                    │   OPTIMIZATION ENGINE            │
                    │  Genetic Algorithm (NSGA-II)     │
                    │  Objective: Maximize revenue     │
                    │  Constraints: Budget, channel    │
                    │  limits, risk concentration      │
                    └──────────────────────────────────┘
```

### Key Modeling Challenges:

| Challenge | Root Cause | Solution |
|---|---|---|
| Adstock (carryover) modeling | Marketing effects persist beyond spend period | Geometric adstock: $adstock_t = spend_t + \lambda \cdot adstock_{t-1}$, tune $\lambda$ per channel |
| Saturation curve estimation | Diminishing returns are non-linear | Hill transformation: $saturation = \frac{spend^\alpha}{K^\alpha + spend^\alpha}$ |
| Multicollinearity | TV and brand spend move together (both peak in season) | Ridge regression + VIF monitoring; Bayesian priors to inform coefficient direction |
| Attribution in launch phase | New drug has no history → model can't estimate TV lift | Bayesian prior from analogous drugs in same class |
| Uncertainty quantification | Single-point estimates mislead budget decisions | MCMC sampling (Bayesian MMM) → credible intervals per channel |

---

## PROJECT 4: Patient Readmission Risk @ EXL/CVS-Aetna

### 🏗️ PRODUCTION ARCHITECTURE

```
EHR Data Sources:
  ├── Structured: vitals, lab values, procedure codes (ICD-10), LOS
  ├── Unstructured: discharge summaries, clinical notes, nursing notes
  └── Administrative: insurance type, demographic data, prior admissions

                    ┌──────────────────────────┐
                    │     AIRFLOW DAG           │
                    │  Daily ingestion job      │
                    │  Data quality checks      │
                    │  Feature refresh          │
                    └────────────┬─────────────┘
                                 │
         ┌───────────────────────┼──────────────────────┐
         ▼                       ▼                      ▼
  Clinical Notes Text     Structured EHR         Administrative
  (Discharge Summaries)   (vitals, labs, Dx)     (insurance, demographics)
         │                       │                      │
         ▼                       ▼                      │
  ClinicalBERT            PySpark Feature Eng.          │
  (fine-tuned)            (Spark SQL transforms)        │
  [CLS] embedding         Comorbidity index             │
  768-dim vector          LOS, readmission count        │
         │                       │                      │
         └───────────────┬───────┘──────────────────────┘
                         │ Feature concatenation
                         ▼
                   XGBoost Classifier
                   (calibrated with Platt scaling)
                   
                         │
                         ▼
                   Risk Score (0-1, calibrated probability)
                   SHAP explanation (top 5 risk factors)
                   Decision support dashboard (Streamlit)
```

### Development Challenges:

| Challenge | Root Cause | Solution |
|---|---|---|
| Notes > 512 tokens (BERT limit) | Long discharge summaries exceed BERT context | Chunk notes by section (chief complaint, assessment, plan); embed each section; attention-pool |
| Readmission labels lag by 30 days | Can't retrain until readmission window closes | Use surrogate labels: ED visit within 7 days, urgent follow-up scheduled |
| Demographic bias discovered | ClinicalBERT pre-trained on younger adult notes | Age-stratified calibration; flag as limitation; added age feature explicitly |
| Compute cost of BERT at scale | Running BERT inference on 50K monthly patients is expensive | Cache BERT embeddings for patients with stable notes; only re-embed when notes are updated |

### Production Challenges:

| Challenge | Root Cause | Solution |
|---|---|---|
| HIPAA compliance | Patient data is PHI | On-premise deployment only; all models trained on de-identified data; audit logs |
| Model staleness | EHR coding practices change annually (ICD code updates) | Quarterly retrain with new ICD-10 codes; monitor feature distribution shift |
| Physician adoption | Model scores displayed but clinicians distrust AI | SHAP top-5 reasons shown in clinician language, not feature names |

---

## PROJECT 5: CLV Prediction @ EXL (PySpark at Scale)

### 🏗️ PRODUCTION ARCHITECTURE

```
Data: 2M+ insurance prospects in GCP BigQuery
      ├── Claims history (24 months)
      ├── Policy data (type, premium, channel)
      └── Demographic signals (age, location, channel acquired)

                    ┌──────────────────────────┐
                    │   SPARK SQL TRANSFORMS   │
                    │   GCP Dataproc cluster   │
                    │   5-15 nodes (autoscale) │
                    └────────────┬─────────────┘
                                 │
                    Feature Engineering:
                    ├── Recency, Frequency, Monetary (RFM)
                    ├── Claim frequency by type
                    ├── Premium-to-claim ratio
                    ├── Channel acquisition flags
                    └── Survival probability (Kaplan-Meier lookup)

                    ┌──────────────────────────┐
                    │  PySpark MLlib           │
                    │  Random Forest (200 trees)│
                    │  Distributed training    │
                    └────────────┬─────────────┘
                                 │
                    Evaluation:
                    ├── Spearman rank correlation (> 0.80 target)
                    ├── Decile lift chart (top 10% = 4.2x average)
                    └── 12-month revenue backtesting

                    ┌──────────────────────────┐
                    │  BATCH SCORING PIPELINE  │
                    │  Airflow DAG (monthly)   │
                    │  Output: CLV deciles      │
                    │  Export: BigQuery table  │
                    └──────────────────────────┘
```

### Scale Engineering Challenges:

| Challenge | Root Cause | Solution |
|---|---|---|
| Serialization overhead for 200-tree forest | Large model → slow broadcast to workers | Column pruning (keep only top-30 features), model compression |
| Data skew (some states have 100x more records) | Geographic imbalance in customer base | Repartition by composite key (state + policy_type) for even distribution |
| Training-serving skew | Feature computation had slight differences | Unified feature pipeline: same Spark SQL code for training and scoring |
| 70% time reduction from sklearn baseline | Single-node sklearn couldn't scale | GCP Dataproc: distributed inference; auto-scaling 5-15 nodes based on queue depth |

---

*This document covers all 5 key projects. Use companion: 05_Production_Architecture_Patterns.md for cross-cutting architecture patterns.*
