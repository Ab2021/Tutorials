# 🏗️ Round 1: Depth in Data Science (DDS) — Master Guide
### Flipkart Senior Data Scientist | ML System Design Round

> **Round Format:** 60 min | Senior DS or Principal DS interviewer
> **What They Test:** End-to-end ML system design, production thinking, business problem translation, trade-off reasoning
> **Your Winning Strategy:** Always lead with business context → formulate ML problem → justify every decision → show production maturity

---

## 🎯 WHAT THIS ROUND IS REALLY ABOUT

DDS is NOT a theoretical round. It's about:
1. **Can you design systems that work in production at scale?**
2. **Do you understand the full lifecycle — data → features → model → deploy → monitor?**
3. **Can you make and defend hard choices under constraints?**
4. **Do you think like a business owner, not just a data scientist?**

The interviewer will constantly push back: *"Why this model?", "What if the data isn't available?", "How does this scale to 350M users?", "What happens when it drifts?"*

---

## 🗺️ THE 7-LAYER DDS FRAMEWORK (Apply to Every Answer)

```
Layer 1: PROBLEM FRAMING
├── What is the business problem? What is the cost of failure?
├── What is the success metric (business KPI)?
├── What are the constraints? (latency, data availability, fairness, regulatory)
└── Translate: business problem → ML task type

Layer 2: DATA STRATEGY
├── What data exists? What's missing? How do you get it?
├── How do you create labels? (supervised / unsupervised / weak supervision)
├── Data quality issues? Class imbalance? Distribution drift? Leakage risks?
└── Feature Store design (online vs. offline)

Layer 3: FEATURE ENGINEERING
├── Tabular: aggregations, velocity, lag features, recency/frequency/monetary
├── Unstructured: embeddings (text, image), NER extraction
├── Graph: network-based risk signals
└── Feature selection, importance ranking, leakage prevention

Layer 4: MODEL SELECTION
├── Which model type and WHY (with explicit tradeoffs vs. alternatives)?
├── What alternatives did you consider and reject?
├── How do you handle cold start / sparse data?
└── Ensemble strategy? Cascade architecture?

Layer 5: EVALUATION STRATEGY
├── Offline evaluation (which metrics, why not accuracy/AUC alone?)
├── Online validation (A/B test design, unit of randomization)
├── Business KPI validation (fraud caught ₹, revenue uplift ₹)
└── Fairness and bias evaluation

Layer 6: DEPLOYMENT & SERVING
├── Real-time vs. batch (driven by latency requirement, NOT preference)
├── Feature computation: streaming (Flink/Spark Streaming) vs. batch
├── Model serving: REST API / gRPC / embedded
└── Rollout: shadow → 5% canary → champion-challenger

Layer 7: MONITORING & ITERATION
├── Input drift: PSI, KS test per feature, weekly
├── Output/performance drift: AUC/PR-AUC on rolling labeled window
├── Model refresh triggers: automated vs. manual
└── Feedback loop design: how do labels flow back?
```

---

## ⏱️ TIME MANAGEMENT

```
0–5 min:   Clarify scope (ask 3-4 targeted questions about constraints)
5–20 min:  Data strategy + feature engineering (the most-underrated part)
20–35 min: Model selection + evaluation (your core answer)
35–50 min: Deployment + monitoring (shows production maturity — WIN HERE)
50–60 min: Interviewer deep-dives + your questions to ask
```
> **Never rush to the model.** Interviewers score you more on data thinking + deployment than on model selection.

---

---

# 🔥 BLOCK A: ML SYSTEM DESIGN (Core Questions)

---

## Q1: Design an end-to-end fraud detection system for Flipkart at 10M daily transactions.

### 🎯 Layer 1 — Problem Framing
**Business problem:** Flipkart loses money through fraudulent payments (stolen cards, friendly fraud, bot attacks). At 10M transactions/day, even a 0.1% fraud rate = 10,000 fraudulent transactions.

**ML task:** Binary classification — fraud vs. legitimate — at transaction time.

**Key constraints:**
- **Latency:** <100ms at payment authorization (customer is waiting)
- **Scale:** 10M/day = ~115 TPS sustained, spikes to 5-10x on sale days
- **Label delay:** Fraudulent transactions confirmed only after chargeback (7-45 days)
- **Class imbalance:** ~0.05-0.2% fraud rate → severe imbalance

**Success metrics:**
- *Business:* Fraud loss rate (₹ lost to fraud / ₹ GMV)
- *Model:* Precision-Recall AUC (not AUC-ROC — imbalanced), Precision@K at fixed investigation capacity
- *Operational:* False positive rate on legitimate transactions (customer experience impact)

---

### 📊 Layer 2 — Data Strategy

**Available data sources:**
| Source | Features | Freshness |
|---|---|---|
| Transaction logs | Amount, merchant, timestamp, payment method | Real-time |
| Device fingerprinting | Device ID, IP, browser fingerprint, geo | Real-time |
| User history | Past purchase patterns, return history | Daily batch |
| Payment gateway | Card BIN, issuing bank, 3DS status | Real-time |
| Network graph | Shared devices, shared IPs across accounts | Weekly |

**Labeling strategy:**
- **Ground truth:** Chargebacks + SFIU (Special Fraud Investigation Unit) verdicts
- **Label delay:** Add 45-day delay between transaction and label → use only transactions with confirmed outcomes for training
- **Unlabeled positives problem:** Many frauds go undetected (no chargeback). Solution: use semi-supervised approach — confirmed fraud + "high suspicion unlabeled" cluster.

**Class imbalance handling:**
- `scale_pos_weight` in XGBoost/LightGBM = (n_negatives / n_positives)
- Evaluate on PR-AUC, NOT AUC-ROC (AUC-ROC insensitive to imbalance)
- Threshold calibration on a held-out validation set

---

### 🔧 Layer 3 — Feature Engineering

**Velocity features (most predictive for fraud):**
```python
# Point-in-time correct — only use info available at transaction time
user_txn_count_1h      # transactions in last 1 hour
user_txn_count_24h     # transactions in last 24 hours
user_amount_sum_1h     # total spend in last 1 hour
distinct_merchants_24h # distinct merchants in last 24 hours
device_txn_count_1h    # transactions from this device in last 1 hour
```

**Behavioral deviation features:**
```python
amount_vs_user_avg_ratio  # this txn amount / user's 90-day avg
is_new_merchant           # 1 if user has never bought from this merchant
time_since_last_txn_min   # minutes since last transaction
is_nighttime              # 1 if txn is 11pm-4am
```

**Network / graph features (batch, updated daily):**
```python
shared_device_count       # how many accounts share this device
shared_ip_fraudster_ratio # % of accounts sharing this IP that are flagged fraud
known_fraud_ring_score    # proximity to known fraud ring nodes (GNN embedding)
```

---

### 🤖 Layer 4 — Model Selection

**Proposed architecture: Dual-stage cascade**

```
Stage 1 (Real-time, <20ms): LightGBM Rule+ML Hybrid
├── Hard rules (instant block): known bad IPs, blacklisted cards
├── LightGBM scorer: tabular features, returns probability score
└── Decision: score < 0.3 → pass | score > 0.8 → block | 0.3-0.8 → Stage 2

Stage 2 (Async, <80ms): Deep Learning + Graph
├── GNN embedding: 2-hop seller/device network risk propagation
├── Combine GBM score + GNN embedding → final MLP
└── Feeds the investigation queue for human review
```

**Why LightGBM over XGBoost for Stage 1?**
- Leaf-wise tree growth → faster training on large datasets
- Native sparse feature handling
- Histogram-based splitting → lower memory footprint
- Real-world: 2-3x faster inference at same accuracy

**Why GNN for Stage 2?**
- Fraud rings operate as connected networks (shared devices, addresses, bank accounts)
- GNNs propagate the "fraud signal" across the graph — isolate doesn't catch ring membership
- GraphSAGE (inductive) → can embed NEW sellers not seen in training

**Why not a single deep neural network?**
- Tabular fraud data: XGBoost/LightGBM empirically outperforms DNNs in most benchmarks
- DNN interpretability is poor → regulation + investigator trust issues
- Latency: LightGBM inference is <5ms; a DNN with many layers can be 30-50ms

---

### 📏 Layer 5 — Evaluation Strategy

**Offline metrics (in priority order):**
1. **PR-AUC** — primary (imbalanced classification)
2. **KS Statistic** — measures model discrimination; industry standard for risk models
3. **Precision@K** — where K = daily investigation capacity (e.g., 500 cases/day)
4. **Calibration (ECE)** — probability scores mean something to investigators

**Business translation:**
```
If model catches 70% of fraud (Recall) at 80% Precision:
  → 70% × ₹50 Cr monthly fraud loss = ₹35 Cr recovered
  → 20% FP rate on 500 daily reviews = 100 wasted investigator-hours
  → Net savings = ₹35 Cr - investigator cost ≈ ₹33 Cr/month
```

**Online A/B test design:**
- Unit of randomization: **user-level** (not transaction — fraud adapts within a session)
- Split: 90% champion, 10% challenger
- Duration: minimum 2 weeks (captures Monday-Friday + weekend patterns)
- Primary metric: fraud catch rate at fixed FPR threshold
- Guardrail metric: false positive rate on confirmed-good users

---

### 🚀 Layer 6 — Deployment & Serving

```
Transaction Event (Kafka topic)
        │
        ▼
┌───────────────────────────────────────────┐
│  Online Feature Store (Redis / Feast)     │
│  - User velocity features (TTL: 1 hour)  │
│  - Device signals (TTL: 24 hours)        │
│  Pre-computed, updated via stream consumer│
└───────────────────┬───────────────────────┘
                    │
                    ▼
        LightGBM Model Server (gRPC)
        - Served via Triton Inference Server
        - Model loaded in memory: <5ms
                    │
                    ▼
         Score → Decision Logic
         └──────────────────────▶ Block/Pass/Review Queue
```

**Rollout strategy:**
1. **Shadow mode (2 weeks):** System runs in parallel, decisions logged but not enforced. Compare against investigator verdicts.
2. **Canary (1 week):** 5% of transactions scored by new model, 95% by rules engine.
3. **Champion-Challenger:** New model is live champion. Old rules engine is challenger on 20% traffic for comparison.

---

### 📡 Layer 7 — Monitoring & Iteration

**Data drift:** Weekly PSI on each feature. Alert if PSI > 0.25 on velocity features.

**Performance drift:** Daily PR-AUC on labeled window (transactions from 45 days ago with confirmed outcomes). Alert if drops >3%.

**Model refresh:** Automated weekly retraining on rolling 90-day window. Emergency retrain if hard alert triggers.

**Feedback loop:** Investigator verdicts (confirmed fraud / confirmed good) flow back to training data within 48 hours via Airflow pipeline.

**The suppression bias problem:** Model blocks fraud → fraudsters decrease → re-trained model gets fewer positive labels → degrades. Fix: counterfactual logging (log what the decision *would have been* for blocked transactions, if we had let them through based on a 1% random passthrough).

---

---

## Q2: Design a credit risk model for Flipkart Pay Later (EMI) — including cold start.

### 🎯 Layer 1 — Problem Framing
**Business:** Flipkart wants to offer Buy Now Pay Later (BNPL). Wrong credit decisions → NPAs (Non-Performing Assets) → capital losses. Too conservative → lost revenue.

**ML task:** Predict Probability of Default (PD) within 90 days.

**Constraints:**
- Millions of users have NO credit bureau history (thin file / no file)
- Regulatory: RBI guidelines on fair lending; cannot use sensitive demographics
- Business KPI: Gini coefficient (industry standard), vintage default rates

---

### 📊 Layer 2 — Data Strategy (Cold Start First)

**The cold start hierarchy:**
```
Tier 1: NO history at all (new Flipkart user)
  → Use only device signals, app engagement, location
  → Give micro-limit (₹500) with rapid behavioral observation

Tier 2: Flipkart behavioral history, NO bureau
  → Purchase frequency, average order value, return rate, payment consistency
  → Behavioral proxy score (not a credit score — a propensity-to-repay proxy)

Tier 3: Bureau + Flipkart behavioral (warm users)
  → Full scorecard: bureau features + Flipkart behavioral + velocity
  → Highest credit limits possible
```

**Labeling:** Binary — defaulted (≥90 days past due) vs. non-defaulted, per loan cohort.

---

### 🔧 Layer 3 — Feature Engineering

**Bureau features (Tier 3 users):**
- Credit utilization ratio, number of active loans, DPD (Days Past Due) history, enquiry count last 90 days

**Behavioral proxy features (Tier 2 users — NO bureau):**
```python
flipkart_tenure_days       # how long they've been a Flipkart customer
gmv_3m_total               # total spend in 3 months
payment_on_time_rate       # % of COD/card payments settled without dispute
order_return_rate          # high returns → financial stress signal?
category_distribution      # luxury vs. essential goods ratio
app_session_frequency      # engagement = stability proxy
```

---

### 🤖 Layer 4 — Model Selection

**Separate models by tier (not one-size-fits-all):**
- **Tier 1 (no history):** Conservative rules-based + gradient boosted model on device/app signals. Low limits.
- **Tier 2 (behavioral only):** LightGBM with Flipkart behavioral features. Calibrated output = PD score.
- **Tier 3 (bureau + behavioral):** Full scorecard. LightGBM ensemble → Platt scaling for probability calibration.

**Why not a single neural network?** Interpretability. RBI regulators expect scorecards with explainable risk factors (SHAP values map to compliance documentation).

**Scorecard output format (what business needs, not just a score):**
```
Score: 720 (out of 900)
Top 3 risk factors:
  1. No bureau history (reduces score by 45 points)
  2. High return rate in last 90 days (reduces score by 30 points)
  3. Multiple payment method changes (reduces score by 15 points)
```

---

### 📏 Layer 5 — Evaluation

**Primary metrics:**
- **Gini coefficient** (= 2 × AUC-ROC − 1): Industry standard for credit models. Target > 0.40.
- **KS Statistic:** Maximum separation between default and non-default distributions. KS > 0.40 is acceptable.
- **Vintage analysis:** Track actual default rates by first disbursement month cohort. If Cohort Jan'25 shows 5% default at 6 months vs. model's predicted 3% → model overconfident.

**Fairness check:** Ensure approval rates and defaults per demographics (age band, location tier) are within acceptable variance. No proxy discrimination.

---

### 🚀 Layer 6 — Deployment

**Batch scoring:** Nightly score all eligible users → store in Redis (TTL 24 hours). At checkout → read score from cache → decision in <5ms.

**Real-time re-scoring triggers:** Score invalidated if: user makes a large return, payment dispute occurs, new inquiry on bureau → triggers immediate rescore.

---

### 📡 Layer 7 — Monitoring

**Vintage analysis dashboard:** Default rates by 60/90/180-day vintage, by cohort, by credit tier, by geography.

**PSI on score distribution:** If distribution shifts (e.g., more users scoring 700+ suddenly) → investigate.

**Approval rate monitoring:** If approval rate drops below X% (business floor) → alert.

---

---

## Q3: Design a return fraud detection system for Flipkart.

### 🎯 Layer 1 — Problem Framing
**Business problem:** Return fraud costs Flipkart billions annually. Types:
- **Empty box returns:** Customer returns empty box or brick
- **Wardrobing:** Buy, use, return as unused
- **Switch fraud:** Return a different (cheaper/broken) item
- **Collusion fraud:** Seller + buyer collude to get refund + keep product

**ML task:** Multi-signal risk score at claim submission time. Output: auto-approve / auto-reject / manual review.

**Constraints:**
- Cannot block all returns (legitimate returns must be fast — customer experience)
- Legal: Cannot discriminate by demographics
- Cost asymmetry: FN (missed fraud) costs full item price; FP (blocking good return) costs customer churn lifetime value

---

### 🔧 Layer 3 — Feature Engineering

**Return behavior features:**
```python
user_return_count_30d           # returns in last 30 days
user_return_value_30d           # ₹ value returned in 30 days
return_rate_by_category         # user return rate in this specific category
days_until_return_after_delivery # how fast they returned after receiving
return_reason_encoded           # NLP embedding of return reason text
```

**Product-level features:**
```python
item_category_fraud_rate        # historical fraud rate for this category
price_percentile_in_category    # is this a high-value item (more fraud risk)?
seller_fraud_rate               # seller associated with this return's fraud history
```

**Multi-modal signals:**
- **Text:** NLP on return reason description — cluster reasons. Suspicious phrases: "empty box", "damaged on arrival" for expensive electronics
- **Image (if uploaded):** Vision model to check if returned item image matches original product image
- **Network:** Do multiple accounts all return from the same delivery address?

---

### 🤖 Layer 4 — Model Architecture

**Cascade design:**
```
Level 1: Hard Rules (instant, <1ms)
  → Known blacklisted users: auto-reject
  → Perfect account history: auto-approve
  → Everyone else → Level 2

Level 2: ML Risk Score (batch, within 5 min of return submission)
  → Multi-modal model: tabular + text embedding + image (optional)
  → Score < 0.2: auto-approve
  → Score > 0.8: auto-reject / escalate to SIU
  → 0.2-0.8: manual review queue

Level 3: Human Review (within 24 hours for flagged cases)
  → Investigator interface shows: SHAP explanations, similar past fraud cases
```

**Why cascade?** Auto-approve the obvious good cases fast (customer satisfaction) → focus human investigation on the ambiguous 20%.

---

### 📏 Layer 5 — Evaluation

**Key business metric:** Fraud recovery rate = ₹ of fraud prevented / ₹ of legitimate returns facilitated.

**Precision@K:** At K = daily investigation capacity, what Precision are we achieving?

**Latency SLA:** Auto-approve decision < 5 minutes (else customer experience degrades below threshold).

---

---

## Q4: Design a recommendation system for Flipkart homepage — including cold start users.

### 🎯 Layer 1 — Problem Framing
**Business:** Homepage recommendations drive 25-30% of Flipkart's total GMV. The goal is to show the right product to the right user at the right time.

**ML tasks:**
1. **Candidate retrieval:** From 100M products → top 1,000 candidates (milliseconds)
2. **Re-ranking:** From 1,000 → ranked list of 20-50 to display (10s of ms)

**Constraints:**
- **Latency:** Total < 200ms (retrieval + ranking combined)
- **Scale:** 350M users, 100M products, Big Billion Days peak: 10x normal traffic
- **Cold start:** New users (~20% of daily visits) need reasonable recommendations

---

### 📊 Layer 2 — Data Strategy

**Signals (implicit feedback — no explicit ratings):**
| Signal | Weight | Rationale |
|---|---|---|
| Purchase | High | Strongest signal of real preference |
| Add to cart | Medium-High | Clear purchase intent |
| Click + dwell >30s | Medium | Interest signal |
| Click + immediate back | Low/Negative | Mis-click, not interest |
| Impression without click | Negative | Potentially irrelevant |

**Cold start data strategy:**
- Collect: device type, app language, location (city/tier), session timestamp, first browsed category
- Use even partial signals within the SAME session (first 3 clicks this session)

---

### 🤖 Layer 4 — Model Architecture

**Two-Tower Neural Network (Candidate Generation):**
```
User Tower                          Item Tower
[User ID embed]                     [Item ID embed]
[Age, location]    →  Dense →       [Category embed]  →  Dense →
[Last 10 purchases]   128-d         [Title BERT embed]    128-d
[Last 5 searches]   vector          [Price, rating]     vector

               Cosine Similarity → Train with In-Batch Negatives
               ANN Index (FAISS HNSW) → Top-500 retrieval
```

**Re-ranking (Feature-rich model):**
- Input: user context + item features + context features (time of day, device, sale flag)
- Model: LightGBM (tabular) or Transformer-based sequential model (for session context)
- Output: Ranked list of 20-50 items

**Cold start strategy (layered):**
```
Session 0 (brand new user, 0 clicks):
  → Popularity-based: Top selling items in user's city
  → Trending by category (no personalization yet)

Session 0 (after 3 clicks this session):
  → Session-based collaborative filtering
  → Embed clicked items → find nearest users in embedding space → borrow their top items

Returning user (has history):
  → Full Two-Tower personalization
```

---

### 📏 Layer 5 — Evaluation

**Offline proxies (fast iteration):**
- NDCG@10 on held-out interaction data
- Catalog coverage: % of items ever recommended (detect echo chambers)
- Serendipity: % of recommended items not in user's historical category

**Online A/B test:**
- Unit: user-level randomization
- Primary metric: GMV per user session
- Guardrail: CTR (can't sacrifice engagement for revenue)
- Duration: 2 weeks minimum (captures pay cycle effects)

---

---

## Q5: How would you design an LLM-powered seller insights tool at Flipkart?

### 🎯 Layer 1 — Problem Framing
**Business:** Flipkart has 500K+ sellers. 80% struggle with poor product listings, wrong pricing, or low discoverability. Human intervention doesn't scale.

**Goal:** Build an AI tool that analyzes a seller's performance data and provides actionable, personalized insights in natural language.

**ML task:** RAG + Agents — retrieve seller context → LLM generates specific, grounded recommendations.

---

### 🔧 Layer 3 — Feature Engineering (for context injection)

```python
seller_context = {
    "gmv_30d": "₹2,45,000",
    "gmv_trend": "-12% vs last month",
    "top_categories": ["Electronics", "Accessories"],
    "return_rate": "8.2% (category avg: 5.1%)",
    "listing_quality_score": 62,  # out of 100
    "missing_attributes": ["color", "warranty", "material"],
    "price_vs_competitors": "+15% above market median",
    "fulfillment_sla_breach_rate": "3.2%"
}
```

---

### 🤖 Layer 4 — System Architecture

```
Seller Dashboard → Query
         │
         ▼
┌─────────────────────────────────┐
│  Retrieval Layer               │
│  - Seller metrics (BigQuery)   │
│  - Similar seller benchmarks   │
│  - Category best practices KB  │
│  - Past interventions + outcomes│
└──────────────┬──────────────────┘
               │
               ▼
    LLM Synthesis (GPT-4 / Gemini)
    Prompt: "Given seller data below,
    provide 3 specific, prioritized
    actions to improve GMV. Cite
    specific numbers from the data."
               │
               ▼
    Structured Output (JSON schema)
    + Citation verification
    + Confidence scores
               │
               ▼
    Action Tracking Database
    (Did seller take action? What was GMV impact?)
```

**Guardrails:**
- Every recommendation must cite a specific data point from the seller's data (no hallucination)
- Post-generation: LLM-as-judge verifies groundedness
- Sensitive advice (pricing strategy): routed to human Seller Success Manager

---

---

# 🔥 BLOCK B: YOUR PROJECT DEEP DIVES

---

## Q6: Walk me through your fraud detection system at Chubb — production architecture.

### Full Answer (STAR + Technical + Production Maturity)

**Situation:** Insurance fraud is a massive cost driver. Claims fraud can be filed months after the incident, making it a "long-tail" problem — you need to assess risk early in the claim lifecycle, not after full investigation.

**Task:** Build a system that flags suspicious claims within 48 hours of submission, not months later.

**Architecture I built:**

```
Claim Submitted (Unstructured: PDFs, Photos, Adjuster Notes)
                    │
                    ▼
       OCR + Layout-Aware Parser (PaddleOCR / AWS Textract)
                    │
                    ▼
     BERT-Based NER: Extract - Claimant, Incident Date, 
                   Injury Type, Medical Provider, Attorney
                    │
                    ├──────────────────────────────────────┐
                    │                                      │
                    ▼                                      ▼
         Text Chunking (by section:              Structured Data Extraction
         Incident / Medical / Legal)             (claim amount, date, code)
                    │                                      │
                    ▼                                      │
     Embedding Model (BERT fine-tuned            XGBoost Risk Scorer
     on insurance domain corpora)                (structured signals)
                    │                                      │
                    ▼                                      │
     Vector DB (Milvus) — retrieves                       │
     Top-K similar historical fraud claims                 │
                    │                                      │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                        LLM Synthesis (GPT-4)
                        Output: Risk Narrative + Score
                                       │
                                       ▼
                        SIU Investigation Dashboard
```

**Key decisions I made:**
1. **RAG over fine-tuning:** Fraud patterns evolve weekly. A fine-tuned model would be stale in 3 months. RAG stays current by updating the vector DB with newly confirmed fraud cases.
2. **Time-decay on retrieval:** Modified cosine similarity scoring with exponential decay on historical claim age. Recent fraud patterns weighted 3x vs. 2-year-old patterns.
3. **Provenance tagging:** Every chunk knows which document section it came from. This lets the LLM identify contradictions across incident description vs. medical report.

**Evaluation:**
- **RAGAS metrics:** Faithfulness (0.91), Answer Relevance (0.87), Context Recall (0.83)
- **Shadow mode for 4 weeks:** System's flags compared against SIU investigator outcomes → 74% overlap at top decile
- **Business impact:** Early detection increased intervention rate during claim maturation period significantly

**What I'd do differently:** Implement named entity linking to connect claimant names to an external fraud network graph — the missing layer that would catch organized rings.

---

## Q7: How did you handle the cold start problem in your pharma recommendation system?

**The problem:** New doctors have zero prescribing history. The KG has no edges from them. Any interaction-based system gives uniform recommendations.

**My solution — 3-tier cold start:**

**Tier 1 (Day 0, no visit history):**
- Doctor specialty + Hospital type + Geographic market → find the top-100 most similar doctors by these attributes
- Borrow their prescription patterns: "Doctors like you in similar hospitals in this region tend to start with Drug X for this indication"
- *This is empirical Bayes / beta-binomial prior — population average shrunk toward specialty cohort*

**Tier 2 (After 3 visits, partial data):**
- Bayesian update: posterior = prior (Tier 1 cohort) updated with actual prescribing observations
- If Dr. Y prescribed Drug A twice in first 3 visits → their drug affinity vector shifts toward Drug A

**Tier 3 (Established, 12+ months):**
- Full collaborative filtering on prescribing similarity
- KG traversal: "Doctors who prescribe Drug A for Condition X also respond to Drug B messaging"

**Why this matters for Flipkart:** Exact same pattern applies to new sellers, new buyers, new product listings. The framework is: population prior → sparse Bayesian update → full model.

---

## Q8: Explain your Agentic BI tool — architecture, failure modes, and mitigations.

**Architecture:**
```
User Query
     │
     ▼
Coordinator Agent (LLM)
  → Parse intent → select tools
     │
     ├──── SQL Coder Agent
     │         → Schema RAG (vector DB of data dictionary)
     │         → Writes SQL → Execution Sandbox (read-only)
     │         → If error: self-reflection loop (max 3 retries)
     │
     ├──── Python/Pandas Agent
     │         → Receives CSV path (NOT raw data in context)
     │         → Generates charts, stats
     │
     └──── Synthesizer Agent
               → Combines SQL results + chart
               → Writes natural language answer + citations
```

**Failure modes I encountered and mitigations:**

| Failure Mode | Root Cause | Mitigation |
|---|---|---|
| Schema hallucination death spiral | LLM invents column names → SQL errors → invents again | Schema RAG — must query data dictionary before writing SQL |
| Context window crash | `SELECT *` returns 5M rows | Execution wrapper: always applies LIMIT 1000, passes only `df.head(5)` + `df.describe()` to context |
| SQL injection / DROP TABLE | LLM wrote destructive SQL | Read-only DB credentials + AST parser blocking DDL/DML |
| Infinite loop | Repeated same failed tool call | Loop detection: if Action+Input hash matches previous 3 steps → force terminate, hand off to human |
| Sensitive data exposure | SQL result containing PII | Regex filter on query results before passing to LLM context |

---

## Q9: In your CLV model — how did you validate predictions at 2M-customer scale without immediate ground truth?

**The core challenge:** CLV is a future-looking metric. You can't validate predictions against actual 3-year CLV on day 1.

**My validation strategy (multi-horizon):**

**Offline validation (immediate):**
- Backtesting: Train on customers from 2018-2021. Predict their CLV as of 2021. Compare to actual 2021-2024 revenue. Calculate Spearman rank correlation between predicted CLV deciles and actual revenue deciles.
- Target: Top decile should have 4-6x actual revenue vs. bottom decile (lift chart).

**Proxy validation (1-month horizon):**
- Repurchase rate by predicted CLV tier: Did the "High CLV" predicted group actually repurchase at a higher rate in the next 30 days? This validates the model's short-term discriminative power.

**Business validation (6-month later):**
- Cohort tracking: Customers scored in January → track their actual spend through June → compare against predicted CLV. Report this vintage-style analysis to business as ongoing model health.

**Key insight I share:** "Validating CLV is like validating a weather forecast. You don't know if the 5-day forecast is right until 5 days later. The discipline is building the feedback system in advance so you're tracking it, not wing-ing it."

---

## Q10: How did your entity matching system prevent false positives at scale?

**My multi-stage approach:**

**Stage 1 — Blocking (reduce candidate pairs from O(n²) to manageable):**
- Block on first 3 characters of company name + ZIP code
- MinHashLSH to find pairs with Jaccard similarity > 0.3
- Reduces 800M pairs to ~2M candidate pairs

**Stage 2 — Feature engineering per candidate pair:**
```python
{
    'jaro_winkler': jaro_winkler("Jhonson Corp", "Johnson Corporation"),  # 0.91
    'token_set_ratio': fuzz.token_set_ratio(...),  # 87
    'acronym_match': 1 if "IBM" maps to "International Business Machines",
    'zip_match': 1 if same ZIP code,
    'phone_hash_match': 1 if same phone hash
}
```

**Stage 3 — Classifier with 3-tier output:**
- **P > 0.95:** Auto-merge (high confidence)
- **0.6 < P < 0.95:** Human review queue (show suggested match, 1-click approve/reject)
- **P < 0.6:** No match

**Stage 4 — Transitive closure:**
- If A=B and B=C → merge A,B,C into one canonical entity
- Use Union-Find (amortized O(α(n)) ≈ O(1) per operation)

**Quality monitoring:**
- Weekly: sample 100 auto-merged pairs, verify manually. Alert if spot-check precision drops below 97%.
- Business impact: Reduced analyst matching time from 3 weeks/quarter to 2 hours.

---

---

# 🔥 BLOCK C: PRODUCTION CHALLENGES

---

## Q11: Your fraud model AUC drops 4% in production. What's your exact debugging protocol?

**Step 1 — Diagnose WHERE the problem is (5 minutes):**
```
AUC drop
   ├── Is it data drift? → Compute PSI on all input features
   │         PSI < 0.1: stable | 0.1-0.25: moderate | >0.25: significant
   ├── Is it label distribution shift? → Check actual fraud rate in labeled window
   └── Is it model degradation? → Is the model still discriminating, just on different threshold?
```

**Step 2 — Identify which features drifted (forensics):**
- Sort features by PSI descending
- If `amount_vs_user_avg_ratio` drifted → shopping behavior changed (sale season?)
- If `device_txn_count_1h` drifted → bot attack pattern shifted

**Step 3 — Determine if it's recoverable without retraining:**
- Recalibrate threshold: Plot current PR curve → select new threshold for target precision
- If drift in stable features (e.g., time-of-day) → may self-correct after sale season

**Step 4 — Emergency response protocol:**
- **Soft alert (4% AUC drop, PSI 0.1-0.25):** Increase monitoring frequency. Prepare retraining run.
- **Hard alert (>5% AUC drop, PSI >0.25):** Trigger emergency retraining on most recent 30-day window. Simultaneously surge human review capacity. Escalate to SIU.

**Step 5 — Root cause documentation:**
- Was it a data pipeline bug? (feature computation changed)
- Was it fraud pattern evolution? (new fraud ring tactic)
- Was it business change? (new merchant category added)

---

## Q12: How do you handle feedback loops in your fraud model?

**The suppression bias problem explained:**
```
Model blocks fraudsters
    → Caught fraud rate appears to fall
    → Training data has fewer positive labels
    → Re-trained model has lower fraud recall
    → More fraud slips through
    → Feedback loop complete ↑
```

**Three mitigations I implement:**

**1. Counterfactual logging (most important):**
- For every blocked transaction, log: "IF we had allowed this, what was the transaction context?"
- A small % of near-blocked transactions (P_score 0.75-0.85) are allowed through with enhanced monitoring → generate ground truth labels
- This is the "explore" arm of an explore-exploit strategy

**2. External label injection:**
- Chargebacks from payment gateway (ground truth, delayed by 45 days)
- SFIU investigator verdicts (manually investigated cases)
- These bypass the suppression bias because they're external to the model's decisions

**3. Temporal label auditing:**
- Check: are confirmed fraud counts trending down because fraud is actually decreasing? Or because we stopped catching it?
- Leading indicator: fraud ring intelligence reports from financial industry sharing groups (CIFAS equivalents in India)

---

## Q13: How would you scale your RAG pipeline from 200K claims/year to 10M claims/year?

**The three bottlenecks at scale:**

**Bottleneck 1 — Embedding computation:**
- 200K claims: sequential embeddings fine
- 10M claims: need GPU batch inference
- Fix: Async embedding pipeline with a GPU fleet. Use `sentence-transformers` with CUDA, batch size 512. Throughput: ~50K embeddings/hour on 4xA100 → 10M over the weekend in batch mode.

**Bottleneck 2 — Vector database:**
- ChromaDB (dev tool): Can't handle 10M+ vectors at production query rates
- Production: Migrate to **Pinecone** (managed, serverless) or **Weaviate** (self-hosted with HNSW)
- Index type: HNSW (Hierarchical Navigable Small World) — O(log n) query time
- At 10M vectors, HNSW with ef=128 returns top-100 in ~15ms

**Bottleneck 3 — LLM inference:**
- 10M claims × 1 LLM call each = massive cost + latency
- Fix 1: **Semantic caching** — if two claims are nearly identical (cosine sim > 0.97), return cached LLM response
- Fix 2: **Model distillation** — fine-tune a smaller LLaMA-3 8B on GPT-4 outputs → 10x cheaper inference
- Fix 3: **vLLM with continuous batching** — group concurrent LLM requests into a single forward pass

**Cost estimation I'd give:**
```
GPT-4 at 10M claims: ~$500K/year
Fine-tuned LLaMA-3 8B on own GPU: ~$50K/year
→ 10x cost reduction with <5% quality degradation
```

---

## Q14: Explain your feature store design for real-time fraud scoring.

**The two-layer feature store architecture:**

```
┌─────────────────────────────────────────────────────┐
│                 OFFLINE STORE                       │
│              (BigQuery / Hive)                      │
│  - Historical aggregations (90-day, 365-day)       │
│  - User behavioral profiles                         │
│  - Slow-moving features: recomputed nightly        │
│  - Training data source (point-in-time correct)    │
└─────────────────────────────────────────────────────┘
                          │
                          │ Nightly ETL (Airflow)
                          ▼
┌─────────────────────────────────────────────────────┐
│                 ONLINE STORE                        │
│                   (Redis)                           │
│  - Fast-moving features: TTL 1-24 hours            │
│  - user_{id}_velocity_1h: txn count last hour      │
│  - device_{id}_txn_count_1h: device-level count    │
│  - Updated via Kafka consumer in real-time         │
│  - Lookup latency: <2ms                            │
└─────────────────────────────────────────────────────┘
```

**The training-serving skew problem (most critical):**
- At training time: you use historical aggregations from Hive → correct
- At serving time: you use Redis cached features → potentially different computation logic
- **Fix:** Single feature definition codebase shared between training and serving. Every feature has a unit test that runs at both training and serving time and asserts outputs match on the same input.

**Point-in-time correct join (for training):**
```sql
-- WRONG: Uses future data to compute features
SELECT user_id, txn_count_30d  -- if this is computed as of today, not as of txn date
FROM features JOIN transactions ON user_id

-- CORRECT: Feature value as of transaction timestamp
SELECT f.user_id, f.txn_count_30d
FROM features_snapshot f
JOIN transactions t ON f.user_id = t.user_id
  AND f.snapshot_date = DATE(t.timestamp)  -- feature as it existed at txn time
```

---

## Q15: How do you detect and handle concept drift in a production fraud model?

**Three types of drift, three detection methods:**

**1. Input (covariate) drift:**
- Monitor: PSI (Population Stability Index) weekly on each feature
- PSI formula: $\sum (P_{new} - P_{ref}) \times \ln(P_{new}/P_{ref})$
- Threshold: PSI > 0.25 = significant drift → investigation required

**2. Concept drift (P(Y|X) changes):**
- The relationship between features and fraud changes (new fraud method exploits different signals)
- Detect: Rolling window AUC on labeled holdout (45-day delayed labels)
- If AUC drops 3%+ vs. 6-week moving average → concept drift suspected

**3. Label drift:**
- The actual fraud rate changes in the population
- Detect: Compare model's predicted positive rate vs. confirmed positive rate in labeled window
- If model predicts 0.5% fraud but actual confirmed is 1.2% → underestimating

**My response protocol:**
```
Soft alert (PSI 0.1-0.25 OR AUC drop 1-3%):
  → Increase monitoring cadence from weekly to daily
  → Analyze which specific features drifted
  → No immediate action

Hard alert (PSI >0.25 OR AUC drop >5%):
  → Emergency retrain on last 30-day window
  → Temporary: fallback to rule-based engine for highest-risk category
  → Surge human review capacity
  → Post-mortem within 48 hours
```

---

---

# 🔥 BLOCK D: ARCHITECTURE PATTERN QUESTIONS

---

## Q16: When would you use batch vs. real-time for fraud scoring — and how do you decide?

**The decision framework:**

| Question | Answer → | Architecture |
|---|---|---|
| Can we stop the transaction? | Yes (payment authorization) | **Real-time** (<100ms) |
| Can we stop the transaction? | No (return already submitted) | **Batch** (hours okay) |
| How fresh must features be? | <1 minute | **Streaming features** (Kafka + Flink) |
| How fresh must features be? | <1 day | **Pre-computed batch** (Airflow nightly) |
| Transaction volume? | >1K TPS | **Async real-time** |
| Transaction volume? | <100 TPS | **Sync real-time** fine |

**My production decision at Chubb:**
- **Real-time component:** Initial risk score at claim submission (lightweight BERT NER + XGBoost)
- **Batch component:** Deep forensic analysis (full RAG pipeline) triggered asynchronously post-submission. Results ready in <1 hour.
- **Why hybrid?** "Stopping" a claim instantly is operationally complex (legal implications). But the SIU needs the full analysis within 4-8 hours to prioritize their workload.

---

## Q17: How do you design a training data pipeline for severe class imbalance (fraud)?

**Step 1 — Label definition (most critical, often botched):**
- Don't use "flagged by rule engine" as positive label → selection bias
- Use: confirmed fraud verdicts from investigators OR chargebacks. Be explicit about label definition.
- Add a label delay buffer: only use transactions from > 45 days ago (label stabilization period).

**Step 2 — Data collection window:**
- Don't use all-time historical data: fraud patterns from 2019 are irrelevant to 2025 patterns
- Rolling 90-day window: captures recent patterns, avoids stale fraud schemes

**Step 3 — Imbalance handling (in order of preference):**
1. **`scale_pos_weight` in LightGBM:** Equivalent to upsampling positives. Computationally free.
2. **Stratified K-fold cross-validation:** Ensures each fold has same fraud rate.
3. **SMOTE (Synthetic Minority Oversampling):** Only if imbalance ratio >1:300 AND tabular features only. Never use SMOTE on sequence/graph features.
4. **Separate anomaly detection model:** For ultra-rare new fraud types (<50 examples) — use isolation forest or autoencoder reconstruction error as a complementary signal.

**Step 4 — Threshold calibration:**
- Never use default 0.5 threshold for imbalanced classification
- Use Platt scaling or isotonic regression on a separate calibration set
- Set threshold to match business constraint: "Flag top 500 per day" → threshold at 99.995th percentile

---

## Q18: What is training-serving skew? How did you prevent it in your projects?

**Definition:** Features computed differently at training time vs. inference time → model performance degrades silently in production despite good offline metrics.

**Common causes in my projects:**

| Cause | Example | Impact |
|---|---|---|
| Different aggregation logic | Training: rolling 7-day sum. Serving: rolling calendar week sum | Different numeric values → wrong predictions |
| Time zone handling | Training data in UTC. Serving features in IST | Off-by-5.5h errors in time-based features |
| NULL handling difference | Training: fillna(0). Serving: NULL passed as-is | Model sees unseen null values |
| Feature lag | Training: feature computed at t=0 using future data. Serving: must use t=0 only | Temporal leakage |

**Prevention in my CLV/fraud work:**

1. **Shared feature computation library:** Single Python module used by both `train.py` and `serve.py`. Feature logic only defined ONCE.
2. **Feature validation tests:** Unit tests run on both training pipeline and serving pipeline using same synthetic inputs → assert outputs match within floating point tolerance.
3. **Serving feature logging:** Log every feature vector served alongside prediction → weekly distribution comparison against training distribution via Jensen-Shannon divergence.
4. **Point-in-time correct joins (most important for time-series):** Feature values are joined at the timestamp of the training event, not at the time of training run.

---

## Q19: How would you design A/B testing for a new fraud model at Flipkart?

**Design decisions:**

**Unit of randomization:** User-level, NOT transaction-level.
- Why: If you randomize at transaction level, the same user might see old model 5x and new model 3x during a session. Fraud adapts intra-session.

**Traffic split:** 90% champion (current model), 10% challenger (new model).

**Duration:** Minimum 2 weeks.
- Why: Need to capture: weekly seasonality (Mon-Fri different from weekends), pay cycle effects (fraud spikes around month-end), potential novelty effects.

**Primary success metric:** Fraud catch rate at fixed FPR (False Positive Rate).
- Not absolute recall: FPR on good users is a guardrail metric. We can't harm 99.9% of users to catch more fraud.

**Statistical test:** Two-proportion z-test with Bonferroni correction if testing multiple thresholds.
- Power calculation: If current fraud catch rate = 0.65, we want to detect 0.70 (+5 pp) with 80% power → need ~N users per arm (compute via sample size formula).

**The survivorship bias problem:** Fraudsters already blocked by the champion model are NOT in the experiment population → challenger model is always tested on "easier" cases. Solution: Include the champion model's block decisions in the evaluation — does the challenger agree with champion's blocks? How often does it over-ride them?

---

## Q20: Explain your deployment rollout strategy for the Chubb fraud system.

**My 4-phase rollout with specific gates at each stage:**

**Phase 0 — Offline validation (2 weeks before any production traffic):**
- Backtest against 3 months of historical claims with investigator verdicts
- Gate: PR-AUC > 0.70, False Positive Rate on confirmed good claims < 25%
- Shadow simulation: replicate production traffic from logs

**Phase 1 — Shadow mode (4 weeks):**
- System generates recommendations but NO action is taken
- Investigators work normally, don't see system output
- After 4 weeks: compare system flags vs. investigator dispositions
- Gate: Top decile overlap with investigator-confirmed fraud > 65%

**Phase 2 — Assisted mode (2 weeks):**
- System output shown to investigators as a "second opinion" — not primary driver
- Investigators still make independent decisions
- Gate: Investigator satisfaction survey > 70% positive on system recommendations

**Phase 3 — Champion-Challenger (ongoing):**
- System drives flagging for 80% of claims
- 20% random sample still goes through full manual review (ground truth collection)
- Rollback trigger: False positive rate on CL (confirmed legitimate) claims > 30%

**Phase 4 — Full production:**
- System drives all flagging
- Human review reserved for high-value claims + low-confidence system flags
- Weekly performance reviews with SIU leadership

---

---

# 🔥 BLOCK E: UNUSUAL & HARD QUESTIONS

---

## Q21: Your RAG system retrieves relevant context but the LLM still hallucinates. Debugging strategy?

**Step 1 — Isolate: Is it a retrieval problem or a generation problem?**
```python
# Enable verbose logging to inspect retrieved chunks
retrieved_docs = vector_db.query(claim_embedding, top_k=5, return_text=True)

# Manual inspection: Are retrieved docs actually relevant?
# If retrieved docs are irrelevant → retrieval problem
# If retrieved docs are relevant but answer is wrong → generation problem
```

**If retrieval problem:**
- Re-rank with a cross-encoder (DeBERTa-based) to improve relevance
- Check chunk size: too small = context lost; too large = noisy
- Check time-decay: are old irrelevant claims being retrieved?

**If generation problem (the interesting case):**

1. **Structured output (JSON schema):** Constrain LLM to ONLY output fields defined in schema. Can't hallucinate new facts in structured fields.

2. **Citation requirement:** Every statement in the LLM output must cite a retrieved document. Add: `"You must include [Doc: <document_id>] citation after every factual claim."`

3. **Chain-of-Verification (CoVe):** 
   - Prompt 1: "List all facts you plan to include in your answer"
   - Prompt 2: "For each fact, verify it exists in the retrieved context. If not — remove it."
   - Prompt 3: "Now write the final answer using only verified facts."

4. **Temperature = 0:** Deterministic sampling reduces creative hallucination.

5. **Post-hoc NLI check:** Use a DeBERTa-v3-NLI model to classify: Does the generated answer *entail* from the retrieved context? If entailment score < 0.8 → trigger retry or return "INSUFFICIENT CONTEXT" response.

---

## Q22: How do you design a fair ML model for credit scoring that doesn't discriminate by demographics?

**Fairness metrics (define what fairness means FIRST with business):**

| Fairness Criterion | Definition | Math |
|---|---|---|
| Demographic parity | Same approval rate across groups | $P(\hat{Y}=1 \| A=0) = P(\hat{Y}=1 \| A=1)$ |
| Equal opportunity | Same TPR across groups | $TPR_{group A} = TPR_{group B}$ |
| Equalized odds | Same TPR AND FPR across groups | Both above |
| Individual fairness | Similar individuals get similar scores | $d(x_i, x_j) < \epsilon \Rightarrow |f(x_i) - f(x_j)| < \delta$ |

**Note: These definitions are mathematically incompatible (Impossibility Theorem).** You must choose which one aligns with regulatory and business requirements.

**Mitigation approaches:**

**Pre-processing (my preferred):**
- Reweighting training data: Upweight under-represented groups in training
- Feature removal: Remove protected attributes AND their proxies (ZIP code is a proxy for ethnicity)

**In-processing:**
- Adversarial debiasing: Train a main model WHILE training an adversary that tries to predict protected attribute from the main model's representation. Maximize main model performance while minimizing adversary accuracy.

**Post-processing:**
- Threshold adjustment by group: Set different decision thresholds such that TPR is equalized across groups

**My approach at EXL:**
- Excluded ZIP code (proxy for demographics) from features
- Added geographic region as a higher-level feature (less granular → less discriminatory)
- Monthly fairness audit: Compute approval rate by income band and geography → alert if any group diverges by >5 percentage points from model-wide rate

---

## Q23: The fraud model performs well on historical data but misses novel fraud schemes. How do you address it?

**The root cause:** Supervised models are interpolators — they learn patterns in training data. Novel fraud = extrapolation problem.

**Three-layer defense:**

**Layer 1 — Anomaly detection as a complementary signal:**
- Isolation Forest or Autoencoder reconstruction error on ALL transactions
- High anomaly score = unusual transaction, even if not predicted as fraud
- Rule: "Flag for review if anomaly score > threshold AND fraud model score > 0.4" — catches out-of-distribution cases that the fraud model alone would miss

**Layer 2 — Active learning (uncertainty sampling):**
- Monitor model confidence (predicted probability) distribution
- Transactions with P(fraud) between 0.35-0.65 = high model uncertainty = potentially novel patterns
- Route these to investigators proactively: "Model is unsure about these — your expertise needed"
- Investigator labels → immediate training data for the next model version

**Layer 3 — LLM narrative analysis (the catch-all):**
- The base fraud model scores tabular features
- If tabular score is <0.4 but the claim narrative reads semantically unusual → flag for secondary review
- The LLM can catch semantic oddities that no rule or tabular model can: "The claimant says they were in two different cities simultaneously"

**Proactive defense:**
- Red team exercise quarterly: Hire external fraud consultants to simulate novel fraud attempts. Use their attack vectors to generate synthetic training data (LLM-augmented fraud descriptions) BEFORE real fraud ring discovers the attack surface.

---

## Q24: How would you build a multi-tenant fraud model for both buyers and sellers at Flipkart?

**The challenge:** Buyer fraud (payment/return) and Seller fraud (listing manipulation, fake reviews) have fundamentally different feature spaces, fraud patterns, and label sources.

**Architecture: Shared Representation + Task-Specific Heads**

```
                    SHARED BASE
         ┌──────────────────────────────┐
         │  Entity Embedding Layer      │
         │  - User/Seller ID embeddings │
         │  - Product embeddings        │
         │  - Shared behavioral signals │
         └──────────────────────────────┘
                    │           │
        ┌──────────────┐   ┌──────────────┐
        │  BUYER HEAD  │   │  SELLER HEAD │
        │  - Txn amt   │   │  - GMV trend │
        │  - Return vel│   │  - Review rate│
        │  - Device sig│   │  - Listing Q  │
        │              │   │  - Price anom │
        └──────────────┘   └──────────────┘
             │                    │
        Buyer Fraud Score    Seller Fraud Score
```

**Why shared base?**
- Product embeddings are shared: same product, same risk context
- User-seller interaction history (did this buyer ONLY buy from this seller?) is a signal for collusion fraud — requires both models to see the same interaction space.

**Why separate heads?**
- Feature spaces are mutually exclusive (buyer has payment data; seller has inventory data)
- Label sources differ (buyer: chargebacks; seller: quality team audits)
- Different class imbalances and threshold requirements

**Training strategy:**
- Multi-task learning: Train shared base + both heads simultaneously with weighted loss
- $\mathcal{L}_{total} = \alpha \mathcal{L}_{buyer} + (1-\alpha) \mathcal{L}_{seller}$, tune $\alpha$ by validation performance

---

## Q25: What happens to your embedding model when new fraud patterns emerge not in the knowledge base?

**The out-of-distribution problem:**

**Detection (before it becomes a crisis):**
- Monitor cosine similarity distribution of incoming claims to their nearest neighbor in the knowledge base
- If new claims' max cosine similarity < 0.6 (i.e., nothing in the KB is close) → out-of-distribution signal
- Cluster weekly: "Are a group of new claims forming a new cluster that doesn't overlap with existing clusters?" (HDBSCAN on incoming embeddings)

**Response protocol:**

**Immediate (days 1-3):**
- Flag low-similarity claims for manual SIU review
- Log all low-similarity claims as a "novel pattern candidate" queue

**Short-term (week 1-2):**
- SIU investigates the novel candidate queue
- Investigator confirms: is this indeed a new fraud pattern OR legitimate claims?
- Labeled examples → immediately added to the RAG knowledge base (no retraining needed for RAG — just add new documents and re-index)

**Medium-term (month 1-2):**
- If N>50 confirmed novel fraud cases → retrain the embedding model with new positive/negative pairs
- The old model stays in production while new model is shadow-tested

**Proactive design:**
- LLM-assisted KB enrichment: Generate synthetic variants of each confirmed fraud pattern in the KB using GPT-4 ("Write 10 different ways a claimant might describe slipping and falling that could be staged"). Expands coverage of known patterns before real variants appear.

---

---

## 🚦 COMMON DDS MISTAKES TO AVOID

| ❌ Mistake | ✅ What to Do Instead |
|---|---|
| Jumping straight to model selection | Always start with business problem + data strategy |
| Saying "XGBoost" without justification | Articulate WHY: non-linear interactions + tabular data + interpretability needs |
| Forgetting monitoring | Every answer must end with monitoring strategy |
| Ignoring cold start | Explicitly address: "For new users/items, I handle cold start by..." |
| Treating false positives as free | Quantify business cost of FP vs. FN — they're always asymmetric |
| Not knowing your own system's latency | Have exact numbers ready: "RAG adds ~400ms; scoring model <20ms" |
| Batch-only thinking | Always ask: "What's the latency requirement?" before committing |
| Proposing one model for everything | Different user tiers / use cases often need different models |
| Ignoring fairness | Every credit/fraud model needs explicit fairness evaluation story |
| Skipping the feedback loop | Show you understand how labels flow back to model improvement |

---

*Cross-reference: `01_ML_System_Design_Grind_50Q.md` for 50 additional questions | `02_Project_Deep_Dives_All_Projects.md` | `03_Evaluation_All_Systems_Deep_Dive.md`*
