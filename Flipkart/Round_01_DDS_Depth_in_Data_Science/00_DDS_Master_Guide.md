# 🏗️ Round 1: Depth in Data Science (DDS) — Master Guide
### Flipkart Senior Data Scientist | ML System Design Round

> **Round Format:** 60 min | Typically with a Senior Data Scientist or Principal DS
> **What They Test:** End-to-end ML system design, production thinking, business problem translation, trade-off reasoning
> **Your Winning Strategy:** Always lead with business context → formulate ML problem → justify every decision → show production maturity

---

## 🎯 WHAT THIS ROUND IS REALLY ABOUT

DDS is NOT a theoretical round. It's about:
1. **Can you design systems that work in production at scale?**
2. **Do you understand the full lifecycle — data → features → model → deploy → monitor?**
3. **Can you make and defend hard choices under constraints?**
4. **Do you think like a business owner, not just a data scientist?**

The interviewer will constantly push back with: "Why this model?", "What if the data isn't available?", "How does this scale to 350M users?", "What happens when it drifts?"

---

## 🗺️ THE 7-LAYER DDS FRAMEWORK (Use This for Every System Design)

```
Layer 1: PROBLEM FRAMING
├── What is the business problem?
├── What is the success metric (business KPI)?
├── What are the constraints? (latency, data, fairness, regulatory)
└── How do you translate to ML problem type?

Layer 2: DATA STRATEGY
├── What data exists? What's missing?
├── How do you label it? (supervised/unsupervised/weak supervision)
├── Data quality issues? Class imbalance? Distribution drift?
└── Feature Store design (online vs. offline)

Layer 3: FEATURE ENGINEERING
├── Tabular features (aggregations, velocity, lag features)
├── Unstructured features (embeddings, text, image)
├── Graph features (network-based)
└── Feature selection, importance, leakage prevention

Layer 4: MODEL SELECTION
├── Which model type and WHY (with tradeoffs)?
├── What did you try and reject?
├── How do you handle cold start / sparse data?
└── Ensemble strategy?

Layer 5: EVALUATION STRATEGY
├── Offline evaluation (which metrics and why?)
├── Online validation (A/B test design)
├── Business KPI validation
└── Fairness and bias evaluation

Layer 6: DEPLOYMENT & SERVING
├── Real-time vs. batch (latency requirements)
├── Feature computation (streaming vs. batch)
├── Model serving infrastructure
└── Rollout strategy (shadow → canary → champion-challenger)

Layer 7: MONITORING & ITERATION
├── Input drift detection (PSI, KS test)
├── Output/performance drift
├── Model refresh triggers
└── Feedback loop design
```

---

## 🔥 TOP 25 DDS QUESTIONS YOU WILL BE ASKED

### BLOCK A: ML System Design (Core)

**Q1: Design an end-to-end fraud detection system for Flipkart at 10M daily transactions.**
> Cover: real-time (<100ms) + batch. Features: velocity, device fingerprint, behavioral, graph. Models: GBM for real-time, GNN for ring detection. Serving: online feature store (Redis) + model server. Monitoring: PSI + AUC drift.

**Q2: Design a credit risk model for Flipkart Pay Later (EMI) — including cold start.**
> Cold start: use behavioral data (browse, add-to-cart, purchase history) as proxy. Warm users: bureau features + Flipkart behavioral. Model: LightGBM scorecard. Monitoring: vintage analysis (default rates by cohort).

**Q3: Design a return fraud detection system.**
> Multi-modal: text (return reason NLP), image (product condition vision), behavioral (return velocity, return history), network (linked accounts). Cascade: rule engine → ML → human review.

**Q4: Design a recommendation system for Flipkart homepage — cold start users.**
> Cold users: popularity-based + editorial picks → collect implicit feedback (click, dwell time). Warm users: collaborative filtering + content-based hybrid. Matrix factorization (ALS) or two-tower neural model. Evaluation: offline A/B proxy (NDCG@10), online CTR lift.

**Q5: How would you design an LLM-powered seller insights tool at Flipkart?**
> RAG over seller performance data + NLP over seller support tickets. Structured output schema. Hallucination guardrails. Human review for high-stakes recommendations.

---

### BLOCK B: Your Project Deep Dives

**Q6: Walk me through your fraud detection system at Chubb — production architecture.**
> Start: business problem (long-tail claim fraud, late detection). Architecture: BERT IE → Vector DB → RAG → Risk Score. Batch + real-time. Evaluation: RAGAS + shadow mode. Key decision: RAG over fine-tuning (fraud patterns evolve).

**Q7: How did you handle the cold start problem in your pharma recommendation system?**
> New doctors (no prescribing history): used demographic similarity (specialty, location, hospital type) + KG traversal to find similar established doctors → borrow their prescription patterns as initial prior.

**Q8: Explain your Agentic BI tool architecture. What are its failure modes?**
> LangChain ReAct agent. Tools: SQL, Python, chart generation. Failure modes: infinite loops (max iteration guard), hallucination of data (validation tool), SQL injection (sandboxed execution), context overflow (hierarchical summarization).

**Q9: In your CLV model at EXL — how did you validate at 2M scale without ground truth yet?**
> Offline: Spearman rank correlation between predicted CLV tiers and actual 12-month revenue by decile. Lift chart: top 10% scored users should have 4x+ actual revenue vs. average. Online: cohort-based revenue tracking — did users in high-CLV prediction group actually generate higher revenue over 12 months?

**Q10: How did your entity matching system prevent false positives at scale?**
> TF-IDF + cosine similarity gave top-k candidates. Second-pass: exact substring match + token sort ratio (fuzzy string matching). Thresholding: tuned on precision-recall curve for business tolerance (75% precision target). Post-processing: human spot check of low-confidence matches.

---

### BLOCK C: Production Challenges

**Q11: Your fraud model AUC drops 4% in production. What do you do?**
> Step 1: Is it data drift or model decay? Compute PSI on input features — identify which features drifted. Step 2: Is ground truth shifting? Plot confusion matrix over time — is FPR increasing or recall dropping? Step 3: Trigger retraining on recent data window. Step 4: If structural concept drift (new fraud type), expand feature set + knowledge base.

**Q12: How do you handle feedback loops in your fraud model?**
> The "suppression bias" problem: model blocks fraud → fraudster adapts → data looks like fewer frauds → re-trained model becomes weaker. Mitigation: (1) Counterfactual logging — log what would have happened if we didn't block. (2) Explore-exploit: randomly allow 0.1% near-blocked transactions for ground truth. (3) External fraud labels (chargeback data, investigator verdicts).

**Q13: How would you scale your RAG pipeline from 200K claims/year to 10M claims/year?**
> Three bottlenecks: (1) Embedding computation — batch GPU inference, async processing. (2) Vector DB — move from ChromaDB to Pinecone/Weaviate with ANN index (HNSW). (3) LLM inference — switch to smaller fine-tuned model (LLaMA-3 8B) or use vLLM with speculative decoding + batching. Add caching for frequently queried patterns.

**Q14: Explain your feature store design for real-time fraud scoring.**
> Online store (Redis): pre-computed user features, updated on every transaction event via Kafka consumer. Offline store (BigQuery/Hive): historical aggregations for training. Point-in-time correct join: critical — avoid training-serving skew. Feature lag: some features (rolling 7-day velocity) need stream processing (Flink). Feature freshness SLAs.

**Q15: How do you detect and handle concept drift in a production fraud model?**
> Detection: (1) Data drift — PSI on input feature distributions, KS test. (2) Label drift — monitor rate of confirmed fraud in flagged population. (3) Performance drift — AUC on rolling weekly evaluation set. Response: soft alert → retrain on recent window. Hard alert (AUC drop >5%) → emergency retraining + human review surge.

---

### BLOCK D: Architecture Pattern Questions

**Q16: When would you use batch vs. real-time for fraud scoring?**
> Real-time (<100ms): payment authorization point — can stop transaction. Need: online feature store + lightweight model. Batch (hours): return fraud, seller fraud — decision can wait, models can be heavier. Always: separate the data pipeline from the serving pipeline.

**Q17: How do you design a training data pipeline for imbalanced fraud data?**
> Data collection: all transactions in rolling window (90 days). Label: confirmed fraud/non-fraud from outcome tracking. Imbalance handling: stratified sampling during training (not random). SMOTE for minority class where class ratio >1:100. Threshold calibration post-training. Separate model for ultra-rare fraud types (few-shot).

**Q18: What is training-serving skew and how did you prevent it in your projects?**
> Definition: features computed differently at training time vs. inference time → model sees different distribution at serving, degrading performance. Prevention: (1) Use SAME feature computation code for both — single feature store abstraction. (2) Log serving features alongside predictions → validate distribution matches training. (3) Point-in-time correct joins — don't use future information during training.

**Q19: How would you design A/B testing for a new fraud model at Flipkart?**
> Unit of randomization: user-level (not transaction — avoids cross-contamination within user). Traffic split: 90% current model, 10% new. Duration: minimum 2 weeks (capture weekend + weekday patterns). Primary metric: fraud catch rate at fixed FPR. Guardrail: good-user false positive rate. Test statistical significance: Z-test for proportion difference. Challenge: survivorship bias (already-blocked fraudsters not in experiment).

**Q20: Explain your deployment rollout strategy for the Chubb fraud system.**
> Phase 1 (4 weeks): Shadow mode — system runs but outputs are logged only, no action taken. Purpose: find failure modes safely, compare to investigator ground truth. Phase 2 (2 weeks): Canary — 5% of claims go through system, investigators review both. Phase 3: Champion-challenger — system fully drives flagging, human review on high-risk flagged claims. Rollback trigger: false positive rate on confirmed good claims >30%.

---

### BLOCK E: Unusual and Hard Questions

**Q21: Your RAG system retrieves relevant fraud patterns but the LLM still hallucinates. What's your debugging strategy?**
> (1) Check if hallucination is in retrieval or generation: enable verbose logging — are retrieved docs actually relevant to the claim? If yes, issue is in generation. (2) For generation hallucination: switch to structured output (JSON schema forcing) — LLM can only output fields that are in the schema. (3) Add citation requirement — every claim must cite a retrieved document. (4) Lower temperature to 0.0. (5) If still failing: use a smaller, fine-tuned model that is better calibrated on this domain.

**Q22: How do you design a fair ML model that doesn't discriminate by demographics in credit scoring?**
> Fairness metrics: (1) Demographic parity: same approval rate across groups. (2) Equal opportunity: same true positive rate across groups. (3) Equalized odds: same TPR and FPR across groups. Mitigation: pre-processing (reweighting training data), in-processing (adversarial debiasing), post-processing (threshold adjustment per group). Regulatory context: India's data protection laws, BASEL compliance.

**Q23: The insurance fraud model performs well on historical data but misses novel fraud schemes. How do you address it?**
> Novel fraud = out-of-distribution. Approach: (1) Anomaly detection layer (isolation forest, autoencoder reconstruction error) to flag unusual claims even if not predicted fraudulent. (2) Active learning: flag low-confidence predictions for investigator review → the uncertainty itself is a signal. (3) LLM layer in RAG can catch semantic oddities rules miss. (4) Adversarial fraud simulation: generate synthetic novel fraud variants → add to training.

**Q24: How would you build a multi-tenant fraud model that works for both buyers and sellers at Flipkart?**
> Buyer fraud: transaction-level signals (payment method, address mismatch, velocity). Seller fraud: listing-level signals (price anomaly, review manipulation, inventory inflation). Separate model heads on shared base features. Transfer learning: shared embedding layer for product/user representations, task-specific heads. Evaluation: evaluate each head independently + combined system-level metric.

**Q25: What happens to your embedding model when new fraud patterns emerge that are not in the knowledge base?**
> Out-of-distribution problem. Detection: (1) Monitor cosine similarity of incoming claims to existing knowledge base centroids — low similarity = potential novel pattern. (2) Cluster incoming low-similarity claims — is a new cluster forming? Response: (1) Flag cluster for investigator review. (2) Investigator labels cases → add to knowledge base. (3) Retrain embedding model with new positive/negative pairs. Proactive: synthetic augmentation of the knowledge base using LLM-generated fraud variant descriptions.

---

## 🚦 COMMON DDS MISTAKES TO AVOID

| ❌ Mistake | ✅ What to Do Instead |
|---|---|
| Jumping straight to model selection | Always start with business problem + data strategy |
| Saying "XGBoost" without justification | Articulate WHY: non-linear interactions + tabular data + interpretability |
| Forgetting monitoring | Every system answer must end with: "and here's how I'd monitor it in production" |
| Ignoring cold start | Explicitly address: "For new users/items, I handle cold start by..." |
| Treating false positives as free | Quantify business cost of FP vs. FN — they're asymmetric |
| Not knowing your own system's latency | Have exact numbers: "RAG pipeline added ~400ms; scoring model <20ms" |
| Batch-only thinking | Always ask: "What's the latency requirement?" before committing |

---

## ⏱️ TIME MANAGEMENT IN THE DDS ROUND

```
0–5 min:   Clarify scope (ask 3-4 targeted questions about constraints)
5–20 min:  Data strategy + feature engineering (the often-overlooked part)
20–35 min: Model selection + evaluation (your core answer)
35–50 min: Deployment + monitoring (shows production maturity)
50–60 min: Interviewer deep-dives + your questions to ask
```

> Never rush to the model. The interviews score you more on data thinking + deployment than on model selection.

---

*See companion files: 01_ML_System_Design_Framework.md, 02_Project_Deep_Dives.md, 05_Production_Architecture_Patterns.md*
