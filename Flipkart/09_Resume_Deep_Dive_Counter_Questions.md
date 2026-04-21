# 🔍 Resume Deep Dive — Every Claim, Every Counter Question
### What Alex (PhD, Research Director) Will Attack and How to Defend It

> **How to use this:** Read each resume claim. Then read WHY an interviewer would
> question it, WHAT the exact counter question sounds like, and HOW to answer it
> with complete technical and business depth.
>
> The rule: **Never say something you can't defend 3 levels deep.**
> This document gives you those 3 levels for every single claim.

---

# 🔴 SECTION 1: CLAIMS FROM CHUBB (Aug 2024 – Present)

---

## CLAIM 1.1: "Architected and deployed scalable ML-based fraud detection system using RAG and LLMs"

### Why an Interviewer Probes This
> "Architected" + "deployed" + "scalable" are three heavy words.
> Anyone can prototype RAG in a weekend using LangChain.
> The senior DS bar is: did you actually DESIGN something novel?
> Did it go to PRODUCTION? What does "scalable" mean quantitatively?

---

### Counter Question Set — ARCHITECTURE

**Q1: "Walk me through the RAG architecture you built. What's in the retrieval layer?"**

**Your Answer:**
> "The retrieval layer has three sub-components.
> First: **Document ingestion pipeline** — insurance claim PDFs, adjuster notes, medical records are chunked
> (sliding windows of ~300 tokens with 50-token overlap to preserve context across chunk boundaries).
> Each chunk is embedded using a fine-tuned version of `text-embedding-ada-002` baseline, later
> replaced with a domain-adapted bi-encoder fine-tuned using contrastive learning on claims pairs.
>
> Second: **Vector store** — we evaluated ChromaDB for prototyping and FAISS for production.
> FAISS with IndexIVFFlat (inverted file index + flat quantizer) gave sub-millisecond approximate
> nearest neighbor search at our document scale. The index is rebuilt nightly as new claims come in.
>
> Third: **Hybrid retrieval** — dense (embedding similarity) + sparse (BM25 keyword). Dense alone
> misses exact code matches ('ICD-10 S72.001A'). BM25 alone misses semantic paraphrasing.
> Reciprocal Rank Fusion (RRF) combines both scores: `1/(k + rank_dense) + 1/(k + rank_sparse)`,
> k=60 per paper recommendation. This gave +12% hit@5 on our held-out evaluation set."

---

**Q2: "Why RAG over fine-tuning the LLM on your fraud data?"**

**Your Answer (3 reasons):**
> "Three clear reasons. First: **data freshness**. Fraud patterns evolve monthly — new schemes,
> new attorney networks, new billing code manipulations. Fine-tuning requires curating labeled data,
> training (expensive), and re-deployment. RAG updates by adding new documents to the vector store —
> same day. For a fraud team that discovers new schemes weekly, this is critical.
>
> Second: **confidentiality and compliance**. Insurance claim data is extremely sensitive (HIPAA,
> regulatory requirements). Fine-tuning means that data was used to update model weights —
> technically the model 'remembers' it, making data deletion/audit hard. With RAG, data lives
> in the vector store; you can delete or update documents without touching the model.
>
> Third: **explainability** — an adjuster or compliance officer needs to see 'why' a claim was
> flagged. RAG gives you the retrieved source documents as citations. Fine-tuning can't do this —
> you can't trace which training example drove a specific output."

---

**Q3: "You said 'scalable.' What was the actual throughput? How many claims per day?"**

> ⚠️ **CRITICAL:** If you don't have exact numbers, give approximate ranges confidently.
> Never vague out. Say "in the range of" rather than "I don't remember."

**Your Answer:**
> "The batch pipeline processed historical claims — on the order of tens of thousands per day
> for the backfill period — using Spark for document preprocessing and embedding generation in parallel.
> For real-time, new claims ingestion triggered near-real-time scoring with a latency SLA of under
> 30 seconds from claim receipt to fraud flag. The LLM calls were the bottleneck; we mitigated this
> by running embedding + retrieval synchronously and making the LLM call asynchronous — the retrieval
> result was stored immediately; LLM reasoning report came in under 10 seconds.
>
> For true scale beyond that we'd move to an async queue (Kafka) with LLM workers in a pool —
> which I designed but was not needed at current claim volumes."

---

**Q4: "How did you handle hallucinations in the fraud reasoning output?"**

**Your Answer (4-layer approach):**
> "We implemented four layers.
>
> **Layer 1: Structured output.** The LLM was forced to output JSON with fixed schema:
> `{risk_level: high/medium/low, evidence_citations: [...], reasoning: str, confidence: 0-1}`.
> Free-form generation allows hallucination; constraining the schema forces the model to reference
> specific retrieved excerpts.
>
> **Layer 2: Faithfulness check.** Using RAGAS Faithfulness metric offline — we evaluated whether
> each claim in the reasoning output was actually supported by the retrieved context.
> Claims with Faithfulness < 0.7 triggered human review rather than automated flagging.
>
> **Layer 3: Citation validation.** For every evidence citation in the output, we verified the
> claim was actually present in the retrieved document (string match / semantic similarity check
> against the source chunk). This catches 'hallucinated citations' — the LLM citing a document
> that doesn't actually contain what it claims.
>
> **Layer 4: Human-in-the-loop.** High-stakes decisions (claim reserve > $50K or risk_level=HIGH)
> always required an adjuster sign-off. The system proposed; the human decided.
> This is the ultimate hallucination backstop — and the right architecture for regulated industries."

---

**Q5: "How did you evaluate the RAG pipeline? What were your metrics?"**

**Your Answer:**
> "Two evaluation tracks:
>
> **Offline / component-level (using RAGAS):**
> - Faithfulness: Is the generated answer grounded in retrieved context? (Target: > 0.80)
> - Answer Relevance: Is the answer relevant to the fraud question asked? (cosine sim of answer embedding to question embedding)
> - Context Recall: Did retrieval find the documents a human annotator would consider relevant?
>   (requires a gold-standard 'which documents matter for this claim' annotation set)
> - Context Precision: Of retrieved documents, what fraction were actually relevant?
>
> **Online / task-level (against labeled fraud cases):**
> - PR-AUC on fraud flagging decisions (binary: fraud / not fraud)
> - Recall@K: What fraction of confirmed fraud cases did the system flag in top K% of claims?
> - False positive rate per 1000 claims reviewed (adjuster productivity metric)
> - KS statistic (Kolmogorov-Smirnov): Separation between fraud and non-fraud risk score distributions
>
> We targeted PR-AUC > 0.75 (appropriate for class-imbalanced fraud data where precision matters
> more than raw accuracy) and Recall > 0.80 so less than 20% of true frauds were missed."

---

## CLAIM 1.2: "Implemented batch and real-time processing frameworks for insurance risk scoring"

**Counter Q: "What does real-time mean? What's the latency? How did you design for it?"**

**Your Answer:**
> "Real-time in this context was near-real-time: new claim file lands in S3/GCS → event trigger →
> preprocessing pipeline → embedding → FAISS lookup → LLM async call → risk score persisted.
> End-to-end: < 30 seconds for the risk score to be available to the claims handler.
>
> Key design decisions for latency:
> - **Embedding**: precomputed for the knowledge base nightly; new claim embedding computed on ingestion only
> - **FAISS**: in-memory flat index for sub-millisecond nearest neighbor; index shards partitioned by claim type (medical vs. liability) so search space is smaller
> - **LLM call**: gpt-4-turbo with streaming tokens (users see output as it generates)
> - **Fallback**: if LLM call exceeds 15s, return embedding-only similarity score (no reasoning) — ensures adjuster always gets a preliminary signal even if the full reasoning is delayed"

---

**Counter Q: "Why not just use a traditional ML classifier for fraud? Why bring in LLMs at all?"**

**Your Answer:**
> "Great question — the honest answer is: we used BOTH, and they serve different purposes.
>
> The traditional ML classifier (XGBoost on structured features: claim amount, provider network,
> billing codes, time-to-file, claimant demographics) runs first in `< 200ms`. It gives a
> numerical risk score — fast, interpretable, no LLM costs.
>
> The LLM layer activates ONLY for claims above a threshold risk score OR for complex claim types
> (medical malpractice, long-tail liability) where the unstructured narrative content is the
> primary fraud signal. A billing code anomaly is a structured signal — XGBoost catches it.
> A doctor's note that claims 'patient recovered fully' while separate notes document ongoing
> treatment — that's a semantic contradiction in unstructured text that requires NLP reasoning.
>
> So the architecture is: ML classifier → triage → LLM reasoning for complex cases.
> This keeps LLM costs in check (only activated for ~20% of claims) while deploying it where
> it adds the most value."

---

## CLAIM 1.3: "Agentic BI Tool — reducing report generation time by automating complex analytical pipelines"

**Counter Q: "What does 'agentic' mean exactly here? How is this different from a SQL chatbot?"**

**Your Answer:**
> "A SQL chatbot takes a natural language query, converts it to SQL, runs it, shows results.
> That's one-hop: query → SQL → answer.
>
> An agentic system handles multi-hop reasoning:
> - 'What's our worst-performing claim category by region and how does it compare to last quarter?'
> - This requires: (1) identify claim category dimension, (2) query regional breakdown,
>   (3) pull prior quarter, (4) compute delta, (5) possibly call a chart tool if visualization needed,
>   (6) reason about what 'worst' means and provide interpretive commentary
>
> The LangChain ReAct agent decides WHICH tool to call at each step:
> - `sql_query_tool(query)` → returns data
> - `python_executor_tool(code)` → runs pandas computation on returned data
> - `chart_tool(data, chart_type)` → generates visualization
> - `data_validation_tool(result)` → checks for nulls, outliers before returning to user
>
> The agent loops: Thought → Action → Observation → Thought → ... until it has a complete answer.
> A SQL chatbot can't do step 3 → 4 → 5 → 6 because it has no loop or tool dispatch.
>
> Key engineering challenges: (1) SQL injection prevention via parameterized queries only,
> (2) Python sandbox for arbitrary code execution (used RestrictedPython), (3) context window
> management — agent conversation history can exceed 8K tokens for complex queries, so we
> summarized old tool outputs before they'd been truncated."

---

**Counter Q: "What happened when the agent made a wrong tool call or looped infinitely?"**

**Your Answer:**
> "Both happened during testing.
>
> **Wrong tool call:** Agent called the chart tool before running the SQL. This came from the prompt
> not clearly distinguishing 'data retrieval tools' from 'presentation tools'. We fixed this with
> a two-phase system prompt: Phase 1 tools (data retrieval), Phase 2 tools (presentation) —
> the agent is instructed to exhaust Phase 1 before Phase 2. Effectively a soft constraint without
> hard-coded branching.
>
> **Infinite loops:** The agent would sometimes call sql_query → get partial results → call sql_query
> again to 'verify' → get same results → repeat. We implemented: (1) max_iterations=10 hard cap,
> (2) tool call deduplication — if the exact same tool call with same args appears twice in history,
> force termination with best current answer, (3) a 'confidence threshold' — agent self-rates
> confidence after each iteration; if > 0.85, terminate early.
>
> In production, we also added a human escalation path — if the agent terminates with low confidence,
> it outputs: 'I need clarification on X' rather than failing silently."

---

# 🟠 SECTION 2: CLAIMS FROM AXTRIA (Sep 2022 – Jun 2024)

---

## CLAIM 2.1: "Achieved ~10% increased revenue through optimized marketing spend allocation"

**Counter Q: "10% revenue increase — how do you attribute that specifically to your MMM model?"**

> ⚠️ This is the MOST vulnerable claim on your resume. Attribution is genuinely hard.
> Own the nuance — don't oversell.

**Your Answer:**
> "Fair point — pure attribution is always contested in MMM. Let me be precise.
>
> The MMM model improved the **spend reallocation recommendation**. Previously the pharma client
> was allocating ~60% of their detailing budget to high-decile physicians who were already max-penetrated —
> saturated channel with diminishing returns. The model's response curves showed that reallocating
> ~20% of that budget to mid-decile physicians with high growth potential had a significantly better
> predicted ROI.
>
> The client implemented that reallocation for one therapeutic area. On the next quarterly business
> review, the revenue from that segment increased ~10% vs. the control group that maintained
> prior allocation. This was measured as a quasi-experimental comparison — not a perfect A/B test, 
> but directionally attributed to the reallocation strategy.
>
> The honest caveat: MMM attribution is always partially confounded by external factors (competitor
> activity, price changes, market dynamics). We modeled competitive spend as a covariate but
> perfect causal attribution is not possible. I'd say the model provided causal direction with
> ±20% uncertainty on the magnitude."

---

**Counter Q: "Explain your Marketing Mix Model architecture technically. What variables? What model?"**

**Your Answer:**
> "The MMM was a **decomposition model** — separating baseline sales (what you'd sell with zero
> marketing) from incremental sales driven by each marketing channel.
>
> **Target variable:** Weekly RxTRx (prescription transactions) at national/territory level
>
> **Predictors (Media Channels):**
> - Detailing (field sales): Rep visits, weighted by call quality
> - DTC (Direct-to-Consumer): TV, digital display, paid search (GRPs / impression volume)
> - Samples distributed
> - Journal advertising spend
> - Congresses/speaker events
>
> **Model: Adstock + Diminishing Returns transformation before modeling**
> - Adstock: Marketing has decaying carryover effect — an impression today still influences
>   next week: `Adstock_t = Impression_t + λ × Adstock_{t-1}`, λ = decay parameter (0 to 1)
> - Saturation (diminishing returns): log or Hill function — `Sales = a × (spend^b) / (c^b + spend^b)`
>   captures the S-curve effect where early spend is highly efficient, later spend has diminishing returns
>
> **Model fitting:**
> - XGBoost + Random Forest ensemble on transformed features for non-linear interactions
> - Bayesian optimization (Hyperopt with Tree Parzen Estimator) for hyperparameter tuning
>   → gave the '15% improved model accuracy' claim on the resume
> - PySpark on Databricks for data transformation at territory × week granularity
>
> **Output:** ROI curve per channel, budget optimizer that maximizes predicted revenue subject
> to total budget constraint using scipy.optimize or the custom genetic algorithm."

---

**Counter Q: "You mention a 'custom genetic algorithm' — why not use scipy.optimize or pyomo?"**

**Your Answer:**
> "scipy.optimize (SLSQP + L-BFGS-B) works well for smooth, convex objective functions.
> The budget allocation problem is multi-objective and has integer-like constraints
> (can't allocate fractional rep visits), leading to a non-convex, non-differentiable landscape
> in multi-channel, multi-territory settings.
>
> The genetic algorithm handled:
> - **Multi-objective:** Maximize revenue, ROMI, AND territory equity (some territories were underfunded)
> - **Integer constraints:** Rep visits are whole numbers per week
> - **Min/max bounds per channel:** Can't reduce detailing below 50% (contractual minimums with the rep network)
> - **Cross-channel interactions:** Synergy between DTC awareness and detailing conversion rates
>   — a constraint that makes the surface non-separable across channels.
>
> Specifically: NSGA-II (Non-dominated Sorting Genetic Algorithm) for multi-objective Pareto frontier.
> The Pareto front output gave the client a range of trade-off allocations rather than a single
> 'optimal' solution — they selected their preferred point on the frontier based on strategic priorities."

---

## CLAIM 2.2: "Pharma Rep Communication System — GPT-4 + Neo4j Knowledge Graph"

**Counter Q: "What specifically is in the knowledge graph? How did you build it? What's a Cypher query look like?"**

**Your Answer:**
> "The knowledge graph was the foundational layer — everything else was downstream of it.
>
> **Nodes:** Doctor (with attributes: specialty, NPI, geography, decile), Drug (with attributes:
> indication, mechanism of action, label), Condition (ICD-10 category), Rep, Account (hospital/clinic)
>
> **Edges:**
> - (Doctor)-[PRESCRIBED {count, recency, trend}]->(Drug)
> - (Doctor)-[TREATS]->(Condition)
> - (Doctor)-[AFFILIATED_WITH]->(Account)
> - (Rep)-[CALLS_ON]->(Doctor)
> - (Drug)-[INDICATED_FOR]->(Condition)
>
> **Example Cypher query** — Find doctors similar to a target doctor who prescribed Drug X but NOT Drug Y:
> ```cypher
> MATCH (target:Doctor {id: $target_id})-[r1:PRESCRIBED]->(d1:Drug)
> WITH target, collect(d1) AS targetDrugs
> MATCH (other:Doctor)-[r2:PRESCRIBED]->(d1:Drug)
> WHERE other <> target
> AND d1 IN targetDrugs
> AND NOT (other)-[:PRESCRIBED]->(:Drug {name: 'Drug_Y'})
> WITH other, count(d1) AS sharedDrugs
> ORDER BY sharedDrugs DESC
> LIMIT 20
> RETURN other.name, other.specialty, sharedDrugs
> ```
>
> **Building it:** Used Py2neo + ETL from pharma claims data → cleaned, deduplicated on NPI numbers,
> then ingested via MERGE statements (idempotent — safe to re-run). Graph had ~500K nodes and ~3M edges
> for a mid-size pharma client."

---

**Counter Q: "Why a knowledge graph? Why not just a relational database with joins?"**

**Your Answer:**
> "Three reasons where graphs outperform relational:
>
> 1. **Variable-depth traversal:** 'Find all doctors two hops from Doctor A through shared patients
>    who may be candidates for cross-promotion' — in SQL, this requires a self-join of unknown depth.
>    In graph: `MATCH (a:Doctor)-[:SHARES_PATIENT*2]->(b:Doctor)` — one clause, any depth.
>
> 2. **Relationship attributes matter:** The PRESCRIBED edge has recency, trend, count — not just
>    existence. SQL can store this in a junction table, but traversal with edge filters quickly becomes
>    complex multi-join queries. Cypher expresses this naturally.
>
> 3. **Recommendation patterns:** Collaborative filtering on a graph ('doctors who prescribed X
>    also prescribed Y, and Doctor B hasn't prescribed Y yet') is native to graph traversal.
>    In SQL, this is a complex multi-step query with intermediate temp tables.
>
> That said — the graph was for recommendation and context retrieval. Analytics (aggregation,
> reporting) still ran in SQL/BigQuery. Using the right tool: graphs for traversal,
> relational for aggregation."

---

# 🟡 SECTION 3: CLAIMS FROM EXL/CVS HEALTH (Jul 2016 – Sep 2022)

---

## CLAIM 3.1: "AUC improved from 0.82 to 0.89 in Patient Readmission model"

**Counter Q: "AUC from 0.82 to 0.89 — walk me through exactly what drove that lift."**

**Your Answer:**
> "The baseline model (0.82 AUC) used only structured EHR data: demographics, vitals, lab values,
> comorbidities, prior utilization. Feature-rich tabular model using XGBoost.
>
> The improvement to 0.89 came from adding BERT-extracted embeddings from clinical free text.
> Specifically:
>
> **What we extracted from clinical notes:**
> - Discharge summaries (primary signal source — doctor's narrative assessment)
> - Progress notes (daily treatment narrative — changes in condition)
> - Medication reconciliation notes
>
> **BERT implementation:**
> - Started with Bio-BERT (pre-trained on biomedical literature) — not general BERT,
>   because clinical language has domain-specific terminology ('MI', 'CABG', 'PTX' mean
>   specific things that general BERT might not resolve correctly)
> - Fine-tuned Bio-BERT on our labeled dataset (30-day readmission: yes/no) using
>   classification head — task-specific fine-tuning, not just embedding extraction
> - Used [CLS] token embedding as the document representation → 768-dimensional vector
> - Truncated to 512 tokens (BERT limit) — took first 256 + last 256 tokens to preserve
>   admission detail and discharge detail (both clinically important for readmission)
>
> **Feature combination:**
> - Concatenated 768-dim BERT embedding with ~80 structured features
> - Final ensemble: XGBoost on concatenated features
>
> **Lift attribution (ablation):**
> - Structured only: 0.82 AUC
> - BERT only: 0.79 AUC (text alone isn't enough without structured context)
> - Combined: 0.89 AUC
> The lift is genuinely complementary — neither alone achieves 0.89."

---

**Counter Q: "Why did you truncate to 512 tokens? What's lost? Did you try longer context models?"**

**Your Answer:**
> "Good challenge. The BERT 512-token limit was a real constraint. We evaluated three options:
>
> 1. **First 512 tokens:** Captures admission context; misses discharge (dangerous — discharge
>    patterns are strong readmission predictors).
> 2. **Last 512 tokens:** Captures discharge; misses admission severity details.
> 3. **First 256 + last 256 (our choice):** Captures both admitting diagnosis context and
>    discharge summary — practical compromise.
> 4. **Hierarchical BERT (Longformer):** We did evaluate Longformer (4096 tokens) as a POC.
>    Performance was similar to our 256+256 approach but with ~3× training time and higher
>    memory requirements. For notes that were typically 300-500 tokens, Longformer overhead
>    wasn't justified. For the subset of very long notes (ICU patients), Longformer did show
>    marginal improvement — we flagged this for a future iteration."

---

**Counter Q: "How did you detect and handle model drift after deployment?"**

**Your Answer:**
> "We implemented a three-tier monitoring stack in MLflow:
>
> **Tier 1: Data drift (input distribution)** — PSI (Population Stability Index) on key structured
> features (age distribution, comorbidity prevalence). PSI > 0.2 triggers alert.
> - Clinical note length distribution monitored separately (if notes get shorter, BERT gets less signal)
>
> **Tier 2: Concept drift (target distribution)** — Readmission rate rolling 30-day window.
> Hospital readmission rates change with policy (Medicare penalty changes, COVID waves).
> We compared current period readmission rate against baseline; >15% relative shift → retrain signal.
>
> **Tier 3: Model performance** — Since we had ground truth (30-day readmission is observable),
> monthly AUC re-evaluation on held-out recent data. AUC drop > 3pp → retrain trigger.
>
> The actual retraining was quarterly — model was re-trained with rolling 18-month window to
> capture recent clinical practice patterns while preserving enough history."

---

## CLAIM 3.2: "CLV model for 2M+ prospects — 70% reduction in processing time"

**Counter Q: "How did you achieve 70% reduction? What was the baseline?"**

**Your Answer:**
> "Baseline: single-node Python scikit-learn Random Forest on a scheduled VM.
> Processing 2M records took ~14 hours — unacceptably long for a daily scoring job.
>
> Our approach on GCP Dataproc (PySpark):
>
> **Where the 70% came from (roughly):**
> - **Distributed processing:** Spark partitions the 2M rows across cluster executors; feature
>   engineering (joins, transformations) runs in parallel → ~50% of the gain
> - **Efficient tree inference:** Spark MLlib's Random Forest inference is optimized for row-parallel
>   prediction — each tree's prediction is computed in parallel across partitions → ~20% of the gain
> - **I/O optimization:** Read from GCS Parquet (columnar, compressed) rather than CSV —
>   reads only needed columns (predicate pushdown) → ~10% of the gain
>
> End result: 14 hours → ~4 hours. Roughly 70% wall-time reduction.
>
> **Cluster config:** Dataproc cluster with 1 master (n1-standard-8) + 10 workers (n1-standard-4).
> Auto-scaling down after job completion to control cost.
>
> **Trade-off acknowledged:** scikit-learn's Random Forest is often faster on a single high-memory
> machine for inference (no shuffle overhead). For a single inference call. But for 2M records with
> complex feature engineering (joins across multiple tables) — Spark wins dramatically."

---

**Counter Q: "What features drove the CLV model? How did you validate it?"**

**Your Answer:**
> "CLV is fundamentally a survival/hazard problem — how long will the customer 'survive' (stay active)
> and how much will they spend while active?
>
> **Feature categories:**
> - **Recency/Frequency/Monetary (RFM):** Classic CLV inputs — days since last engagement, number of
>   policy interactions in last 12mo, total premium paid
> - **Claims behavior:** Claims frequency, average severity, claim type mix (auto vs home vs health)
> - **Demographics:** Age band, geography (risk pool characteristics), household size
> - **Product penetration:** Number of product lines held (cross-sell coverage is a strong retention predictor)
> - **Payment behavior:** Autopay vs manual, late payment frequency (surprisingly predictive)
> - **Channel of acquisition:** Direct, broker, group — acquisition channel correlates with LTV
>
> **Validation:**
> - Hold-out test set (20% of prospects, time-based split — don't random split time series!)
> - Spearman correlation between predicted CLV rank and actual 12-month revenue: ~0.71
> - Lift curve: top decile by predicted CLV delivered 3.2× actual revenue vs. bottom decile
> - Business validation: Marketing team confirmed the top decile model-identified prospects
>   had higher policy activation rates in subsequent outreach campaigns"

---

## CLAIM 3.3: "TF-IDF + Cosine Similarity entity matching, 75% accuracy"

**Counter Q: "75% accuracy — for entity matching, that's not great. Why not higher? What did you try?"**

> ⚠️ This is a weakness in the resume claim. Own it proactively.

**Your Answer:**
> "You're right that 75% accuracy for entity matching is modest. Let me explain the context and what
> constrained us.
>
> The task: match 20K health plan sponsor company names in our internal system against 40K company
> names in an external data vendor — no shared company IDs, only string names. This is genuinely hard
> because:
> - 'J.P. Morgan Chase & Co.' vs 'JPMorgan Chase' vs 'JPM Chase'
> - 'AT&T Inc.' vs 'AT and T' vs 'American Telephone Telegraph'
> - Subsidiaries with different names than parent ('Google LLC' vs 'Alphabet Inc.')
>
> **What we tried:**
> 1. Exact match: ~20% matched (legal name variations killed this)
> 2. Fuzzy string matching (Levenshtein distance): 55% — caught typos but not abbreviated forms
> 3. TF-IDF + cosine (our production approach): 75% — treats words as features, handles
>    word-order invariance ('Chase Morgan JP' still matches) and tokenizes better
> 4. SPF (SimString) + blocking: briefly explored but data didn't have good blocking candidates
>
> **Why we stopped at 75% and didn't go higher:**
> The remaining 25% were genuinely ambiguous (same name, different company — 'National Insurance Inc.'
> appearing in both datasets for two different entities). Human review was required regardless.
> Going from 75% → 85% would likely introduce false positives (wrong entity matched)
> which is worse than no match for a data integration pipeline.
>
> **What I'd do differently today:** Use a sentence-transformer fine-tuned on company name pairs
> (positive = same company, negative = different company) — commercial entity name matching is a
> known use case for bi-encoder models. This would likely push to 85%+ with similar false positive control."

---

## CLAIM 3.4: "Survival Analysis — Kaplan-Meier and Cox Proportional Hazards"

**Counter Q: "What's the Cox Proportional Hazards assumption and how did you test it?"**

**Your Answer:**
> "The Cox PH model assumes that the **hazard ratio between any two individuals is constant over time** —
> the 'proportional hazards assumption'. If patient A has 2× the hazard of patient B at time t=0,
> this ratio must hold at every time t.
>
> Formally: h(t|X) = h_0(t) × exp(β^T X)
> The baseline hazard h_0(t) is unspecified (semi-parametric). But exp(β^T X) must be time-invariant.
>
> **Testing the assumption (Schoenfeld residuals):**
> Schoenfeld residuals should be uncorrelated with time if proportional hazards holds.
> - Fit Cox model
> - Compute Schoenfeld residuals per covariate per event time
> - Regress residuals against time — a non-zero slope indicates proportional hazards violation
> - `cox.zph()` function in R / `lifelines` in Python implements this test
> - Global Schoenfeld test gives a p-value; p < 0.05 suggests violation
>
> **What we found:** Several covariates (age, prior hospitalization count) showed time-varying
> effects — elderly patients had higher hazard early in admission but similar hazard later.
> This violated proportional hazards.
>
> **Fix:** Added **time-covariate interaction terms** for the violating covariates:
> `age × log(t)` as an additional predictor — this allows age's effect to vary with time.
> Alternatively, we considered stratified Cox (stratify on the violating variable, estimate
> separate baseline hazards per stratum — but this loses the coefficient estimate for that variable)."

---

# 🔵 SECTION 4: RESUME-WIDE CROSS-CUTTING QUESTIONS

---

## CLAIM 4.1: "9+ Years of Experience" — Credibility Questions

**Counter Q: "You started as a Business Analyst. When did you actually start doing data science?"**

**Your Answer:**
> "Fair question — the title progression at EXL was: Business Analyst → Senior BA → Assistant Manager
> → Manager, Data Science. The Business Analyst title at EXL's analytics division was essentially
> an analytics role; I was building models from day one.
>
> My first ML project was 2017 — building the NLP-based plan sponsor entity matching system
> (TF-IDF on PySpark). By 2018-2019, I was leading the Patient Readmission project with BERT.
> So I'd say 7-8 years of core data science, with the early 1-2 years being more analytics-heavy.
> The progression was real — I was promoted to Manager based on ML delivery, not business analysis."

---

## CLAIM 4.2: Skills Claims — "PyTorch, TensorFlow, Reinforcement Learning, Multimodal AI"

**Counter Q: "You list Reinforcement Learning — describe a problem you solved with RL."**

> ⚠️ If RL is on your resume but you haven't done production RL, be honest about depth.

**Your Answer:**
> "I want to be precise about my RL depth — I've applied RL concepts rather than built full RL systems
> from scratch in production. Specifically:
>
> - **Multi-armed bandit for A/B testing:** In the Marketing Attribution work at Axtria, we used
>   Thompson Sampling (a Bayesian bandit algorithm) to dynamically allocate budget exploration
>   across marketing channels during a campaign — learning which channels perform best in real-time
>   rather than running a fixed split A/B test. This is RL-adjacent (exploration-exploitation tradeoff)
>   without full MDP formulation.
>
> - **RLHF conceptual understanding:** I've studied PPO and GRPO deeply (DeepSeek-R1 training) and
>   understand how RLHF works — reward modeling, policy gradient updates, KL penalty — but haven't
>   run RLHF training at scale. I'm honest about this distinction.
>
> If the role requires deep RL (trajectory optimization, multi-agent RL), I'd need 2-3 months to
> bring production-level expertise. For bandit-level RL applied to recommendation or budget
> optimization at Flipkart — that I can do immediately."

---

**Counter Q: "You list Multimodal AI — what have you actually built?"**

**Your Answer:**
> "On the resume this refers to work at Chubb — we processed insurance claim documents that included
> images (accident photos, medical imaging reports with embedded figures). The multimodal aspect was:
>
> - **Document layout understanding:** InsuranceDoc PDFs had tables, images, and text mixed.
>   We used a Document AI approach (LayoutLM / Document Layout Analysis) to extract structure-aware
>   text — tables parsed separately from narrative text.
> - **Image metadata:** For accident claims with photos, we used a vision model (CLIP embeddings)
>   to tag images (vehicle damage type, severity) — these tags fed as additional features into
>   the fraud scoring model.
>
> I haven't trained multimodal models from scratch. My depth is: using pre-trained multimodal models
> (CLIP, LLaVA), fine-tuning for domain-specific tagging, and integrating those outputs into
> broader pipelines. Training foundation multimodal models is in my interest area but would be
> a growth area, not a claimed depth."

---

## CLAIM 4.3: Leadership Claims — "Leading cross-functional teams"

**Counter Q: "Led a team of 4-5 people — what was the hardest leadership challenge, and how did you handle it?"**

**Your Answer:**
> "The hardest situation was at EXL when I was leading the Patient Readmission project and one
> of my senior team members — a 6-year veteran — disagreed fundamentally with my choice to use
> BERT over their proposed traditional NLP approach (TF-IDF + logistic regression).
>
> They felt BERT was 'overengineered' for the problem and resisted spending 3 weeks on the
> fine-tuning pipeline when the traditional approach was faster to implement.
>
> My approach: I didn't override the disagreement with authority. Instead I proposed a structured
> **3-week parallel track** — I built the BERT pipeline while they refined the TF-IDF approach.
> We evaluated both on the same held-out validation set. BERT won by 4 AUC points.
>
> But more importantly — by doing this, the team member understood the result empirically,
> not just because I said so. They became the person who later championed BERT adoption in
> a subsequent project. The principle: in data science teams, **empirical evidence beats authority**.
> Create conditions for the data to settle disagreements. This also prevents 'my manager forced me
> to use BERT' narrative from undermining team ownership."

---

**Counter Q: "You've been at 3 companies in 8 years. Why the moves?"**

**Your Answer:**
> "Each move was deliberate and there's a clear progression:
>
> - **EXL (6 years):** Long tenure — I stayed because I was growing. Early career, I needed to
>   learn depth across healthcare analytics and ML engineering. I progressed from BA to Manager.
>   I left when the role no longer offered new technical challenges — the work became maintenance
>   rather than building new systems.
>
> - **Axtria (2 years):** Moved specifically to get into GenAI and advanced analytics leadership.
>   In 2022, pharma MMM with GenAI integration was genuinely frontier work. I built the pharma
>   communication system in 2023 — one of the first production GenAI deployments I know of in
>   that domain. The 2-year tenure was planned — I wanted Axtria's consulting breadth but knew
>   I wanted to go in-house for deeper ownership.
>
> - **Chubb (present):** Moved to own a product end-to-end — fraud detection is a mission-critical
>   system at Chubb, not a side project.
>
> The rationale for Flipkart: scale. 350M users is a qualitatively different problem than any of my
> prior contexts. I want to work on problems where my models serve hundreds of millions of decisions daily."

---

## CLAIM 4.4: Education — B.Tech Mechanical Engineering

**Counter Q: "You have a Mechanical Engineering degree. How did you transition to ML? Any gaps?"**

**Your Answer:**
> "Mechanical Engineering from MNNIT gave me a stronger mathematical foundation than some CS grads —
> Computational Fluid Mechanics involves numerical methods, PDEs, optimization, and simulation;
> all directly applicable to ML.
>
> The transition was intentional: in 2016 I joined EXL specifically for their analytics division,
> not a mechanical engineering role. I taught myself Python, SQL, and ML while working —
> deeplearning.ai specialization (Andrew Ng) gave me the formal DL foundation.
>
> On potential gaps vs. a CS/Stats graduate:
> - **Algorithms / Data Structures:** Actively practiced (LeetCode patterns, graph algorithms).
>   I've solved 200+ problems. For Flipkart's coding round, I'd focus on trees, graphs, and DP.
> - **Formal ML theory:** I compensated with certifications + self-study (Bishop's PRML, ESL).
>   I can derive logistic regression, gradient descent, attention mechanism.
> - **Research papers:** I read and can discuss recent papers (Flash Attention, GRPO, DeepSeek-R1).
>
> The honest gap: I've never done a research apprenticeship (PhD-style deep specialization on one
> problem for years). What I've built instead is breadth in production ML + depth in specific
> applied areas (fraud, NLP, GenAI). For a Senior DS role that's more engineer-oriented,
> I think this profile is actually a strength."

---

# 🟣 SECTION 5: TECHNICAL DEPTH COUNTER QUESTIONS (Any Claim Can Trigger These)

---

## DEEP DIVE QUESTIONS — Expect These From Alex (PhD Background)

---

**Q: "Explain the math of BERT's masked language model loss."**

**Your Answer:**
> "BERT's pre-training objective for MLM is cross-entropy loss computed only on the masked positions:
>
> L_MLM = -(1/|M|) × Σ_{i ∈ M} log P(x_i | x_{\\M})
>
> Where M is the set of masked token positions, and x_{\\M} is the context (all tokens except masked).
>
> P(x_i | context) = softmax(W_vocab × h_i + b)[x_i]
>
> where h_i is the BERT hidden state at position i (768-dim for BERT-base), and
> W_vocab ∈ R^{|vocab| × 768} is the vocabulary projection matrix.
>
> The loss is averaged only over masked positions (|M| ≈ 0.15 × sequence_length),
> NOT all positions — this is important because including all positions would dominate
> the gradient with easy-to-predict tokens (function words, punctuation) and underweight
> the signal from the informationally rich masked tokens."

---

**Q: "You use XGBoost in multiple projects. Derive the XGBoost objective."**

**Your Answer:**
> "XGBoost is a gradient boosting framework. At round m, we fit a new tree f_m to minimize:
>
> Objective: L = Σ_i l(y_i, ŷ_i^{m-1} + f_m(x_i)) + Ω(f_m)
>
> Where Ω(f_m) = γT + (λ/2)Σ_j w_j² is the regularization term
> (T = number of leaves, w_j = leaf weights, γ = min leaf split gain, λ = L2 on leaf weights).
>
> **Taylor expansion (the key insight):**
> l(y_i, ŷ_i + f_m(x_i)) ≈ l(y_i, ŷ_i) + g_i f_m(x_i) + (1/2) h_i f_m(x_i)²
>
> g_i = ∂l/∂ŷ_i  (first-order gradient, or 'pseudo-residual')
> h_i = ∂²l/∂ŷ_i²  (second-order Hessian — this is what differentiates XGBoost from GBM)
>
> Optimal leaf weight for leaf j:
> w_j* = -G_j / (H_j + λ)  where G_j = Σ_{i ∈ leaf_j} g_i, H_j = Σ_{i ∈ leaf_j} h_i
>
> Optimal split gain (to find best split):
> Gain = (G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)) / 2 - γ
>
> WHY H matters: Hessian gives curvature information → better step sizes per sample →
> faster convergence and better handling of class imbalance than first-order GBM."

---

**Q: "In your fraud system, how did you handle severe class imbalance?"**

**Your Answer:**
> "Insurance fraud is typically 1-5% of claims — severe imbalance. Four techniques applied:
>
> **1. Metric selection:** AUC-ROC can be misleading under imbalance (predicting all non-fraud
> gets 95%+ accuracy). We used **PR-AUC (Precision-Recall AUC)** and **F-beta score** with
> β > 1 (weighting recall higher — missing fraud is more costly than false alarms).
>
> **2. Class weighting:** `class_weight = {0: 1, 1: 20}` in XGBoost/sklearn — effectively
> oversamples the minority class in gradient computation by upweighting its loss contribution.
>
> **3. SMOTE (evaluated, not used in production):** Synthetic Minority Oversampling Technique —
> creates synthetic fraud samples by interpolating between real fraud cases in feature space.
> We evaluated it but found it created unrealistic synthetic cases that hurt generalization on
> truly novel fraud patterns. We dropped it for the structured model.
>
> **4. Threshold calibration:** Default classification threshold (0.5) is wrong under imbalance.
> We calibrated the threshold using **Precision-Recall curve** — chose the threshold that
> maximized F_beta at our target recall (>= 0.80). This threshold was ~0.25 in practice.
> Platt scaling (logistic regression on model output) was used for calibration before thresholding.
>
> **5. LLM layer for edge cases:** The LLM + RAG layer was deliberately deployed on the
> 'borderline' cases (model score between 0.20-0.40) where the structured model was uncertain.
> This let the LLM unstructured reasoning add signal exactly where it was most needed."

---

**Q: "What's the difference between L1 and L2 regularization and when do you use each?"**

**Your Answer:**
> "Both add a penalty to the loss function to prevent overfitting by shrinking weights:
>
> L1 (Lasso): Penalty = λ Σ|w_j|
> L2 (Ridge): Penalty = λ Σw_j²
>
> **Mathematical difference:**
> - L2: Gradient of penalty = 2λw_j → pushes weights toward zero proportionally to their magnitude
>   → weights shrink but rarely reach exactly zero
> - L1: Subdifferential of penalty = λ × sign(w_j) → constant shrinkage regardless of magnitude
>   → weights CAN reach exactly zero → produces sparse solutions (feature selection)
>
> **When to use which:**
> - L1: When you suspect only a subset of features is relevant; want automatic feature selection;
>   interpretable sparse model. Example: fraud detection feature selection from 200+ engineered features
> - L2: When most features contribute something; want to handle correlated features gracefully
>   (L1 arbitrarily picks one of correlated features; L2 distributes weight across them)
> - **Elastic Net:** Both — L1 + L2. Best practice for high-dimensional correlated features.
>
> **In XGBoost:** The L2 term (λ in my derivation above) is already built in. L1 is available
> as the alpha parameter. I typically apply both for fraud feature sets."

---

# 📋 MASTER COUNTER QUESTION RISK REGISTER

| Resume Claim | Risk Level | Most Dangerous Counter | Your Defense |
|---|---|---|---|
| "Architected RAG system" | 🔴 HIGH | "What made your chunking strategy better than defaults?" | Sliding window + domain-adapted embeddings + hybrid BM25+dense |
| "~10% revenue increase" | 🔴 HIGH | "How do you attribute to your model vs. external factors?" | Quasi-experiment + honest uncertainty bounds |
| "Scalable ML system" | 🟠 MEDIUM | "What was the actual throughput number?" | ~30s real-time latency; tens of thousands batch/day |
| "AUC 0.82 → 0.89" | 🟠 MEDIUM | "Walk through exactly what drove the lift" | Bio-BERT + first/last 256 tokens + ablation results |
| "70% processing time reduction" | 🟠 MEDIUM | "Where did 70% come from?" | Distributed Spark (50%) + Parquet I/O (10%) + MLlib inference (10%) |
| "Custom genetic algorithm" | 🟡 LOW-MED | "Why not scipy.optimize?" | Non-convex, multi-objective, integer constraints |
| "Knowledge Graph (Neo4j)" | 🟡 LOW-MED | "Give me a sample Cypher query" | MATCH traversal example above |
| "Agentic BI Tool" | 🟡 LOW-MED | "How is this different from a SQL chatbot?" | Multi-hop ReAct loop with tool dispatch |
| "RL on resume" | 🔴 HIGH | "Describe an RL problem you solved" | Be honest: Thompson Sampling bandit, not full MDP |
| "Multimodal AI" | 🟠 MEDIUM | "What exactly did you build?" | LayoutLM + CLIP for insurance docs — usage, not training |
| "9+ years experience" | 🟡 LOW | "When did you actually start doing ML?" | 2017 first ML project; 7-8 years core DS |
| "B.Tech Mechanical" | 🟡 LOW | "Any CS gaps?" | Own it: strong math base, compensated with study, 200+ LeetCode |

---

## 🎯 FINAL COACHING NOTES

### The 3 Questions You Must Not Fumble

```
1. "Walk me through the RAG architecture in full technical detail"
   → This is your strongest project. Go deep: chunking → embedding → FAISS → hybrid retrieval
     → LLM → structured output → RAGAS evaluation. Do not leave any layer unexplained.

2. "How do you evaluate an LLM-based system?"
   → RAGAS (Faithfulness, Answer Relevance, Context Recall, Context Precision)
   → Task-level (PR-AUC, Recall, KS) on labeled fraud data
   → Human evaluation (adjuster agreement rate)
   → Never say "we see if it seems correct"

3. "What would you do differently with more time/resources?"
   → For fraud RAG: (1) Fine-tune the embedding model with more labeled fraud pairs
     (2) Build a PRM (Process Reward Model) on the reasoning chain to catch hallucinated reasoning steps
     (3) Implement online learning — new confirmed fraud cases auto-update the vector store daily
   → This shows you know the frontier and think beyond what you built
```

### The One Framing That Wins

> *Every answer ends with a Flipkart bridge.*
>
> "At Chubb, doing X with insurance claim data — at Flipkart, the equivalent problem is Y with
> e-commerce transaction data at 350M user scale. The core technical challenge transfers,
> but the scale and data density is an order of magnitude higher — and that's exactly what excites me."

---

*End of Resume Deep Dive — Counter Questions & Defenses*
