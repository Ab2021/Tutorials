# 📖 Resume Deep Dive — Optum Context
### Mapping Your Experience to Optum's JD Requirements

> **Purpose:** Pre-built answers for every resume bullet an interviewer might probe.
> Each entry maps to the JD pillar it demonstrates. Lead with healthcare relevance always.

---

## 🔴 CHUBB — Senior Data Scientist II (Aug 2024 – Present)

### PROJECT 1: Insurance Fraud Detection (RAG + LLMs)

**JD Mapping:** GenAI & LLM Engineering ✅ | ML Foundations ✅ | LLM Security & Responsible AI ✅ | Cloud-Native ✅

**The 90-Second Structured Answer:**
> "Business problem: Insurance fraud in long-tail claims was going undetected for 30+ days, costing the business significantly. Traditional rule-based systems caught ~30% of fraud at ~45% precision — too low to be actionable.

> I built a 4-layer production system:
> - **Layer 1 (NLP/IE):** BERT-based information extraction from unstructured claims documents — extracted clinical entities, dates, parties, inconsistencies. Evaluated on 200 gold-standard labeled claims (F1 >0.85 per entity type).
> - **Layer 2 (Embedding):** Domain fine-tuned embedding model using contrastive learning on claims pairs (fraud/legitimate) — achieved +18% Precision@5 over OpenAI ada-002 baseline.
> - **Layer 3 (RAG):** Vector store for retrieving similar historical fraud cases as context for LLM assessment.
> - **Layer 4 (LLM Synthesis):** GPT-4 structured fraud risk scoring — JSON output schema enforced, faithfulness guardrail required all claims grounded in retrieved context.

> Evaluated via RAGAS framework on 100 investigator-labeled cases: Faithfulness 0.88, Context Recall 0.71. Shadow-mode validated for 60 days → 78% recall at 22% FPR → moved to production.

> **Optum relevance:** This is structurally identical to clinical claims fraud detection in healthcare. The same architecture — NLP IE → domain embeddings → RAG → structured LLM output → HITL gate — maps directly to Optum's claims intelligence use cases. The key difference is HIPAA compliance, which I've also worked with at EXL/Aetna."

**Counter-Questions and Answers:**

*"How did you handle the delayed ground truth problem in fraud?"*
> "Fraud labels arrive 60-90 days after the claim. I solved this with shadow mode: I ran the system in parallel with the existing process for 60 days, then validated its flags against what investigators confirmed during that period. It's not a perfect causal setup, but it mirrors actual production deployment cycles and gives you real-world distribution. I also built a synthetic evaluation set using investigator rule-books — generating labeled examples for scheme types that were rare in the training set."

*"What was the biggest failure mode in the RAG system?"*
> "Context recall at 0.71 — meaning our knowledge base missed some relevant fraud case patterns on retrieval. The root cause was sparse coverage of niche fraud schemes. I addressed it by generating synthetic case descriptions from investigator rule-books (essentially RAG-augmented knowledge base expansion), which improved context recall by ~8 percentage points."

*"How did you ensure the LLM didn't hallucinate fraud evidence that doesn't exist?"*
> "Three mechanisms: (1) Structured output schema — the model must output JSON with specific fields; free text hallucination is constrained. (2) Faithfulness check as a post-generation gate — any output with faithfulness <0.85 was flagged for human review, not acted upon. (3) System prompt pinning: explicit instruction 'Only cite evidence present in the retrieved context. If insufficient evidence, output: INSUFFICIENT_EVIDENCE.' That explicit output option reduces confabulation significantly."

---

### PROJECT 2: Agentic BI Tool

**JD Mapping:** Multi-Agentic Workflows ✅ | LangChain ✅ | GenAI & LLM Engineering ✅

**The 60-Second Answer:**
> "Problem: Data analysts were spending 40% of time on repetitive ad-hoc reporting requests from business stakeholders. I built a LangChain-based agent with three custom tools: a SQL query generator with schema validation, a Python executor for statistical analysis, and a chart generator. The key challenges were safety (prevent destructive SQL), reliability (graceful failure on ambiguous queries), and scope enforcement (block out-of-domain questions).

> I evaluated it using a 4-dimension framework: task completion (87%), tool call accuracy, answer correctness (exact match for factual queries, LLM-as-judge for summaries), and safety (0 failures on adversarial inputs). This cut recurring report generation time by ~60%.

> **Optum relevance:** The same agent architecture applies to clinical analytics — imagine a physician querying 'Show me my panel's patients overdue for A1C testing sorted by last visit date.' That's exactly this pattern applied to healthcare data."

---

### PROJECT 3: Underwriter Assistance Tool

**JD Mapping:** NLP ✅ | Information Retrieval ✅ | LLM Engineering ✅

**The 45-Second Answer:**
> "Built two NLP systems for underwriters: (1) Coverage Match — semantic search over policy documents against claim submissions to identify relevant coverage clauses. Used BERT embeddings + BM25 hybrid search, enabling underwriters to see relevant policy language instantly instead of manual search. (2) Claims Summarization — extractive + abstractive LLM pipeline producing structured summaries from multi-page claim files. Freed underwriter bandwidth significantly and was recognized in Q3 2025 North America Analytics Team award.

> **Optum relevance:** Clinical documentation summarization (Chart reviews, SOAP notes, pre-visit summaries) is the exact same problem type — structured summarization of long-form clinical documents with fidelity requirements."

---

## 🟠 AXTRIA — Decision Science & Engineering Manager (Sep 2022 – Jun 2024)

### PROJECT 4: Pharma Rep Communication System (GenAI + Knowledge Graph)

**JD Mapping:** GenAI & LLM Engineering ✅ | Collaboration & Mentorship ✅ | Cloud-Native (Flask + GH Actions + Kubeflow) ✅

**The 60-Second Answer:**
> "Built an end-to-end GenAI system for generating personalized pharma rep-to-physician communications. Architecture: Neo4j knowledge graph (drugs, physicians, prescribing patterns, patient profiles) → recommendation engine (which drug/topic to discuss) → GPT-4 generation (personalized communication). Deployed with Flask + Streamlit, CI/CD in GitHub Actions, ML orchestration via Kubeflow.

> Evaluation: A/B test against generic templates — Treatment (GPT-4 personalized) vs. Control (templates). Results: +23% email open rate, +17% meeting acceptance.

> **Optum relevance:** Member engagement and outreach personalization — generating personalized health coaching messages, care gap notifications, or medication adherence reminders — is the same pattern. Physician ↔ Drug ↔ Patient graph becomes Member ↔ Condition ↔ Care Plan graph."

---

### PROJECT 5: Marketing Mix Modeling (MMM)

**JD Mapping:** ML Foundations ✅ | Data Engineering at scale ✅

**Counter-Questions:**

*"Why did you use Bayesian optimization for hyperparameter tuning rather than grid search?"*
> "Grid search is O(n^d) in the number of hyperparameters — it becomes intractable with 10+ parameters. Bayesian optimization (using Gaussian Process as surrogate model) learns from each evaluation to propose the next hyperparameter combination most likely to improve performance. It achieved the same or better results with 60-70% fewer model evaluations. At Axtria's scale, that translated to significantly reduced Databricks compute costs."

*"How did you handle the attribution problem in multi-touch marketing?"*
> "We used a Markov Chain attribution model — model the customer journey as a Markov chain where each channel is a state, compute the transition probabilities, and then measure the 'removal effect' of each channel (how much conversion probability drops if you remove that channel from the journey). This gives a principled, data-driven attribution that accounts for channel order and interaction effects, unlike simple first-touch or last-touch heuristics. We validated it against a holdout test where we actually turned off one channel — the removal effect estimate was within 8% of the observed change."

---

## 🟡 EXL / CVS HEALTH / AETNA (Jul 2016 – Sep 2022)

### PROJECT 6: Patient Readmission Risk (BERT + PyTorch)

**JD Mapping:** ML Foundations ✅ | NLP ✅ | Healthcare Domain ✅

**The Key Numbers Answer:**
> "Built a clinical readmission risk prediction pipeline incorporating unstructured clinical notes. The AUC journey: Logistic Regression (structured only) → 0.76; XGBoost + TF-IDF text features → 0.82; XGBoost + ClinicalBERT embeddings → 0.89. The key insight was that clinical notes captured semantic nuance that structured codes couldn't — specifically negation ('no signs of infection' vs. 'signs of infection') and co-reference across a multi-day clinical encounter.

> Beyond AUC, I measured Brier score (0.08 — well calibrated) and ran subgroup analysis by demographic group — discovered minor AUC gap for age >80 (0.84 vs. 0.89 overall) which I flagged as a limitation and addressed with age-specific calibration.

> **Optum relevance:** This is a direct Optum use case. Aetna ↔ Optum is the same data universe — insurance claims + clinical notes + readmission risk. I've already built exactly this system."

---

### PROJECT 7: CLV Prediction at Scale (PySpark)

**JD Mapping:** ML Foundations ✅ | Data Engineering + Big Data ✅ | Cloud (GCP Dataproc) ✅

**The Scale Story:**
> "2M+ prospects, distributed Random Forest on PySpark, GCP Dataproc. Reduced processing time by 70% (6h → 1.8h) through three optimizations: geographic partitioning (by state) for even data distribution, broadcast joins for small lookup tables to avoid shuffle, and caching intermediate feature matrices reused across multiple scoring runs.

> Business evaluation: Top decile lift of 4.2x — the model's top 10% predicted CLV customers showed 4.2x higher actual 12-month value than average. That's what the marketing team used to allocate 60% of acquisition budget to the top 30% predicted segment.

> **Optum relevance:** Member risk stratification at 10M+ scale uses the same PySpark + distributed ML infrastructure. The decile lift evaluation framework translates directly to outreach prioritization."

---

### PROJECT 8: NLP Entity Matching (TF-IDF + Cosine Similarity)

**The Technical Answer:**
> "Matched 20K plan sponsors against 40K company name records — the challenge was fuzzy string matching at scale (typos, abbreviations, legal entity suffixes). Used TF-IDF vectorization on character n-grams (not word n-grams — better for fuzzy matching) + cosine similarity on Spark MLlib for distributed computation. Achieved 75% accuracy, saving hundreds of hours of manual matching.

> What I'd do differently today: Use a BERT-based semantic similarity model or a dedicated entity matching model (DeepMatcher) — character n-gram TF-IDF is fast but misses semantic equivalences like 'IBM' ↔ 'International Business Machines.'"

---

## 📋 AWARDS NARRATIVE (Use These Strategically)

| Award | Story in 2 Sentences |
|---|---|
| **Chubb Q1 2025 STAR** | For the RAG fraud detection system. First time the team had a production LLM system for claims analysis. |
| **Chubb Q3 2025 North America Analytics** | For Claims Summarization tool adopted across the NA underwriting team. Recognition came from US leadership for a system built in India. |
| **Chubb Q3 2025 Advanced Analytics Spot** | For the Agentic BI prototype that reduced analyst reporting time by ~60%. |
| **EXL 5/5 SLA × 3 Quarters** | Led DS offshore team of 3 FTEs. Owned end-to-end delivery for CVS/Aetna — not just modeling, but delivery, documentation, stakeholder management. |
| **EXL Q3 2019 STAR** | Underwriting automation — first ML system to automate underwriting screening at the organization. |

---

*End of Resume Deep Dive — Use these narratives to answer "Tell me about [project X]" questions*
