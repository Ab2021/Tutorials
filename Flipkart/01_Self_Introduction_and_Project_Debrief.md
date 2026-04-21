# 🎯 Self-Introduction & Deep Project Debrief
### Flipkart — Senior Data Scientist Interview

> **Candidate:** Alex Chen | **Interviewer:** Alex Rivera (Research Director, DS @ Flipkart | PhD, University of Copenhagen)

---

## 📌 PART 1: SELF-INTRODUCTION SCRIPT (2–3 Min Verbal Intro)

> *Memorize this — deliver it confidently, not robotically. Pause after each block.*

---

> "Hi Alex, I'm Alex Chen — a Senior Data Scientist and AI/ML Engineering Lead based in Bengaluru with **9+ years of end-to-end ML experience**, spanning healthcare, life sciences, and insurance domains.
>
> Currently I'm at **Chubb**, where I lead AI initiatives focused on **insurance fraud detection and risk modeling** — my most impactful work there has been architecting a production-grade **Agentic AI + RAG pipeline** using LLMs to identify potentially fraudulent claims across long-tail claim lifecycles. The system combines unstructured claims data, NLP transformers, and real-time risk scoring — it was recognized with multiple awards including the **Q1 2025 STAR Award**.
>
> Before Chubb, at **Axtria**, I led a team building **Marketing Mix Models** for global pharma clients like J&J, and built an end-to-end **GenAI system for personalized pharma communications** using GPT-4, Neo4j knowledge graphs, and Kubeflow — essentially an early production-grade agentic system before it became mainstream terminology.
>
> My earlier years at **EXL/CVS Health-Aetna** gave me deep roots in healthcare ML — I built Patient Readmission Risk models using BERT + PyTorch moving AUC from 0.82 to 0.89, distributed CLV models on PySpark at 2M+ scale, and NLP entity matching systems.
>
> What excites me about this opportunity at Flipkart is the **scale and complexity of the fraud and risk problem** — combining behavioral signals from 350M+ users, real-time inference, and the intersection of structured/unstructured data. I believe my RAG + Agentic AI + fraud domain experience is directly transferable and I'm keen to bring that to a platform at Flipkart's scale."

---

## 📌 PART 2: DEEP PROJECT DEBRIEF — AI, Agents & Fraud

---

### 🔴 PROJECT 1: Insurance Fraud Detection System (RAG + LLMs + Agentic AI)
**Company:** Chubb | **Duration:** Aug 2024 – Present | **Stack:** RAG, LLMs, BERT, NLP, Batch + Real-time pipelines

#### Problem Statement
Insurance fraud in long-tail claims (medical, liability) is inherently hard to detect because:
- Fraud signals are **buried in free-text** documents (medical notes, adjuster comments, legal filings)
- Claims lifecycle spans months — early-stage fraud is the most expensive to miss
- Traditional rule-based flags generate too many false positives, creating alert fatigue

#### Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   INSURANCE FRAUD DETECTION SYSTEM              │
│                                                                 │
│  Unstructured Claims Data (PDFs, notes, emails)                 │
│          │                                                      │
│          ▼                                                      │
│  ┌───────────────────┐    ┌─────────────────────────────┐      │
│  │ Information        │    │  RAG Pipeline               │      │
│  │ Extraction Layer   │───▶│  (Vector DB + Embeddings)   │      │
│  │ (BERT + NLP)       │    │  ChromaDB / FAISS           │      │
│  └───────────────────┘    └─────────────┬───────────────┘      │
│                                         │                       │
│                                         ▼                       │
│  ┌────────────────────────────────────────────────────────┐    │
│  │         LLM-powered Risk Assessment Agent              │    │
│  │  - Retrieves contextually similar fraud patterns       │    │
│  │  - Generates structured fraud risk reasoning           │    │
│  │  - Flags anomalies with explanations                   │    │
│  └────────────────────────┬───────────────────────────────┘    │
│                           │                                     │
│          ┌────────────────┴──────────────┐                     │
│          ▼                               ▼                      │
│  Batch Processing               Real-Time Scoring               │
│  (Historical Backfill)          (Claims ingestion)              │
│          │                               │                      │
│          └────────────────┬──────────────┘                     │
│                           ▼                                     │
│              Risk Score + Fraud Flag + Explanation Report       │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Technical Decisions (be ready to defend each)

| Decision | What | Why |
|---|---|---|
| **RAG over fine-tuning** | Retrieved fraud pattern library + LLM reasoning | Fraud patterns evolve — RAG allows dynamic knowledge updates without retraining |
| **BERT for IE** | Named entity recognition from claims text | Domain-specific extraction of parties, dates, amounts, medical codes |
| **Hybrid scoring** | Structured ML model + LLM narrative reasoning | LLM catches edge cases and provides explainability; ML ensures speed at scale |
| **Long-tail focus** | Focus on claims >90 days old | Where reserve manipulation and late-stage fraud concentrate |
| **Batch + Real-time** | Kafka + batch ETL | Real-time for new claims, batch for historical monitoring |

#### Metrics & Business Impact
- **Fraud detection accuracy** improved significantly with early-stage identification
- System processes claims within SLA, flagging high-risk cases for human review
- Multiple STAR awards: Q1 2025, Q3 2025 Advanced Analytics Spot Award
- Freed underwriter bandwidth through Claims Summarization (Q3 2025 North America Analytics Team Recognition)

#### Deep-Dive Questions You Must Nail
- *"How did you handle hallucination in the LLM fraud reasoning pipeline?"*
  - **Answer:** Implemented structured output (JSON schema forcing), confidence thresholds, retrieval validation (checking if source documents actually support the claim), and a human-in-the-loop review layer for high-stakes decisions. Grounded all LLM outputs in retrieved context, never bare generation.

- *"How did you evaluate the RAG pipeline?"*
  - **Answer:** Used "Ragas" evaluation framework — Faithfulness (generated answer grounded in context), Answer Relevance, Context Recall (did retrieval find the right docs). Also did offline evaluation against labeled fraud cases as a gold standard benchmark.

- *"What embedding model did you use and why?"*
  - **Answer:** Started with `text-embedding-ada-002` for baseline, moved to domain-fine-tuned embeddings using claims corpus with contrastive learning (similar claims pair positively, fraud vs. non-fraud as negatives). This improved retrieval precision measurably.

- *"How is this different from keyword/rule-based fraud detection?"*
  - **Answer:** Rules catch known patterns. Our system infers *intent* from context. A medical claim that mentions an injury date inconsistent with procedure codes — a rule misses it; our NLP extraction + LLM reasoning catches the semantic inconsistency.

---

### 🟠 PROJECT 2: Agentic BI Tool (LangChain + Autonomous Agents)
**Company:** Chubb (Innovation POC) | **Stack:** LangChain, GPT-4, SQL Agent, Python

#### Problem Statement
Data analysts spent 60–70% of time on repetitive report generation queries — pulling KPIs, comparing cohorts, building ad-hoc dashboards. Goal: allow natural language → automated analytical pipeline execution.

#### Architecture

```
User Natural Language Query
         │
         ▼
  LangChain ReAct Agent
  (Thought → Action → Observation loop)
         │
    ┌────┴─────────────────────┐
    │  Tools Available:        │
    │  - SQL Query Tool        │
    │  - Python Execution Tool │
    │  - Chart Generation Tool │
    │  - Data Validation Tool  │
    └────────────────┬─────────┘
                     │
                     ▼
         Final Report + Visualization
```

#### Key Technical Concepts to Discuss
- **ReAct pattern** (Reasoning + Acting): LLM reasons about what tool to call, observes result, re-reasons
- **Tool calling / Function calling**: Each tool defined with schema, LLM picks appropriate tool
- **Memory management**: Conversation history + context window management to avoid token overflow
- **Guardrails**: SQL injection prevention, output validation, sandboxed Python execution

#### Metrics
- Reduced report generation time significantly by automating complex analytical pipelines
- Enabled non-technical stakeholders to query data in natural language

---

### 🟡 PROJECT 3: Pharma Rep Communication System (GPT-4 + Knowledge Graph + Agentic AI)
**Company:** Axtria | **Duration:** 2022–2024 | **Stack:** GPT-4, Neo4j, NLP, Kubeflow, MLflow, Flask, Streamlit

#### Problem Statement
Pharmaceutical sales reps need highly personalized communication for each doctor — based on their specialty, prescribing history, drug affinity, and patient profile. Manual templates are generic and ineffective.

#### Architecture

```
┌─────────────────────────────────────────────────────┐
│           PHARMA REP COMMUNICATION SYSTEM           │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Neo4j Knowledge Graph                      │   │
│  │  Nodes: Doctors, Drugs, Conditions, Reps    │   │
│  │  Edges: Prescribed, Treats, Affiliated_With │   │
│  └──────────────────┬──────────────────────────┘   │
│                     │                               │
│          Graph Traversal (Cypher)                   │
│                     │                               │
│          ┌──────────▼──────────────────┐            │
│          │ Recommendation Engine       │            │
│          │ (collaborative filtering +  │            │
│          │  content-based hybrid)      │            │
│          └──────────┬──────────────────┘            │
│                     │                               │
│          ┌──────────▼──────────────────┐            │
│          │ GPT-4 Generation Layer      │            │
│          │ Prompt: Persona + Context + │            │
│          │ Recommended drug info       │            │
│          └──────────┬──────────────────┘            │
│                     │                               │
│          Personalized Message Output                │
└─────────────────────────────────────────────────────┘
```

#### Key Points to Articulate
- This is an **early Agentic system** — multi-step: Graph query → Recommendation → Generation
- Knowledge graph enabled **relationship-aware context** (e.g., doctor A and doctor B share patients → cross-recommendation intelligence)
- **Kubeflow** for ML workflow orchestration (DAG-based pipeline: data → features → model → serve)
- **MLflow** for experiment tracking — tracked 50+ experiments across hyperparameter configurations

---

### 🟢 PROJECT 4: Patient Readmission Risk — Deep Learning + BERT
**Company:** EXL/CVS Health-Aetna | **Stack:** PyTorch, BERT, PySpark, Airflow, MLflow

#### Problem Statement
Predict 30-day readmission risk for patients using both structured EHR data AND unstructured clinical notes — early intervention reduces cost and improves outcomes.

#### Key Technical Achievement
- **Baseline AUC: 0.82 → Final AUC: 0.89** (+8.5% relative improvement)
- Used **BERT** to extract rich embeddings from free-text clinical notes (diagnosis, discharge summaries)
- Concatenated BERT embeddings with structured features (vitals, lab values, demographics, comorbidities)
- **Airflow** for data ingestion pipeline orchestration
- **MLflow** for experiment tracking — tracked 30+ BERT fine-tuning runs

#### Architecture
```
Clinical Notes (Text)  +  Structured EHR Data
       │                          │
       ▼                          ▼
  BERT Fine-tuned            Feature Eng.
  (512 tokens)               (PySpark)
       │                          │
       └──────────┬───────────────┘
                  │ Concatenation
                  ▼
          Gradient Boosting / XGBoost
          (Tabular + Text Features)
                  │
                  ▼
          Readmission Risk Score (AUC: 0.89)
```

#### Defend Your Choices
- *"Why BERT over GPT for clinical notes?"*
  - BERT is bidirectional — captures both left and right context for classification tasks. Clinical note classification is a **discriminative** task, not generative. BERT + classification head is more compute-efficient.
- *"Why not use a transformer end-to-end?"*
  - Clinical notes were short (< 512 tokens), structured data dominated signal. Hybrid approach > pure transformer.

---

### 🔵 PROJECT 5: Customer Lifetime Value (CLV) at Scale — PySpark
**Company:** EXL/CVS Health-Aetna | **Scale:** 2M+ prospects | **Stack:** PySpark, GCP Dataproc, Random Forest

#### Key Metrics
- **70% reduction** in processing time using GCP Dataproc distributed inference
- Modeled CLV for 2M+ insurance prospects

#### Key Technical Points
- **Distributed Random Forest** — PySpark MLlib for training; ensemble of 200 trees
- **Feature Engineering on Spark SQL** — claims history, demographic signals, behavioral patterns
- **Deployment** — GCP Dataproc cluster (auto-scaling); batch scoring every 24 hours

---

## 📌 PART 3: CONNECTING YOUR EXPERIENCE TO FLIPKART CONTEXT

> Frame EVERY answer in terms of Flipkart's scale and problems

| Your Experience | Flipkart Equivalent |
|---|---|
| Insurance fraud detection (RAG + LLMs) | Seller/buyer fraud, return fraud detection at 350M+ user scale |
| Agentic BI Tool | Internal analytics automation for category managers, supply chain |
| Knowledge Graph (Neo4j) + Recommendation | Product knowledge graph, seller-buyer-product relationship graphs |
| CLV modeling at 2M scale | Customer propensity, churn prediction at e-commerce scale |
| Real-time risk scoring | Real-time fraud flagging at checkout, payment fraud |
| BERT for clinical notes | BERT for product desc, reviews, seller communication NLP |
| Marketing Mix Modeling | Marketing spend optimization across channels (social, search, affiliate) |

---

## 📌 PART 4: YOUR STORY ANGLES (USE THROUGHOUT INTERVIEW)

### Angle 1: "From Rules to Intelligence"
> "At every company, I've been brought in to replace brittle rule-based systems with intelligent, learning systems. At Chubb, I replaced threshold-based fraud flags with a reasoning LLM + RAG pipeline that understands context. That's the trajectory — and I want to bring that to Flipkart's fraud ecosystem."

### Angle 2: "Production Grade, Not Just Notebooks"
> "Everything I've built has gone to production. RAG pipeline serving real fraud flags. Pharma comm system deployed on Kubeflow serving reps. CLV scoring 2M+ customers on Dataproc. I care deeply about the ML engineering side — not just the model, but the pipeline, monitoring, drift detection, and business feedback loop."

### Angle 3: "End-to-End Ownership"
> "I've always owned the full lifecycle — from problem definition with business stakeholders to data engineering, modeling, evaluation, deployment, and monitoring. At Axtria, that meant building the entire MMM pipeline AND the GenAI layer on top. That's the Senior DS mindset."

### Angle 4: "Agentic AI Before It Was Cool"
> "The Pharma comm system I built in 2023 — knowledge graph → recommendation → GPT-4 generation — was essentially an agentic workflow before 'agents' became the buzzword. Now with LangChain, I've productionized this pattern. I understand both the theoretical foundations and the practical pitfalls."

---

*End of Self-Introduction & Project Debrief Document*
