# SYSTEM DESIGN TEMPLATES — 7-Block Architecture Framework
> Critical gap: You answer system design with bullet lists. Use this structured template for every design question.

---

## THE 7-BLOCK FRAMEWORK

For ANY ML system design question, walk through these 7 blocks in order. Spend the most time on blocks 2-5.

```
Block 1: PROBLEM SCOPE         → What are we building? Scale? Latency? Constraints?
Block 2: DATA INGESTION        → How does data enter the system?
Block 3: FEATURE ENGINEERING   → How are features computed and stored?
Block 4: MODEL LAYER           → Training pipeline + inference
Block 5: SERVING LAYER         → How predictions reach users/downstream
Block 6: MONITORING            → How do we know it works?
Block 7: FEEDBACK LOOP         → How do we improve over time?
```

**How to use in an interview:**
1. Say "Let me scope this first" → ask 2-3 clarifying questions
2. Sketch the 7 blocks on whiteboard/paper
3. Walk through blocks 1-7, spending ~2 minutes per block
4. At each block: WHAT → WHY this choice → TRADEOFF vs alternative
5. Close with monitoring and feedback loop — this differentiates senior candidates

---

## DESIGN 1: REAL-TIME FRAUD DETECTION SYSTEM

### Block 1: Problem Scope

**Clarifying questions to ask:**
- What type of fraud? (Claims fraud, transaction fraud, account takeover?)
- What scale? (Claims per day? Peak load?)
- What latency? (Real-time decision at submission? Or batch overnight?)
- What's acceptable precision/recall? (Business SLA?)
- Regulatory constraints? (GDPR? Insurance regulations? Audit trail needed?)

**Scoped requirements:**
- **Scale:** 10,000 claims/day real-time + 1M claims/night batch
- **Latency:** < 200ms p99 for real-time, batch complete by 6am
- **Accuracy:** Precision > 70% (real-time), Recall > 80% (batch)
- **Compliance:** Audit trail for every flag, no PII in logs, SHAP explainability for regulation

---

### Block 2: Data Ingestion

**Real-time path:**
```
Claim Created (CRM system)
    → Kafka topic: 'claim-events'
    → Claim consumer service validates schema
    → Publishes to 'claim-valid' topic
    → Real-time scorer consumes 'claim-valid'
```

**Batch path:**
```
Overnight ETL (Airflow CronJob at 1am)
    → Extract from Snowflake: claims, policy, claimant tables
    → Extract from external: ISO watchlists, credit data
    → Extract from internal: SIU feedback, confirmed fraud labels
    → Load to Delta Lake on Databricks
    → Validate: null checks, row counts, schema validation
```

**Data sources:**
| Source | Data | Latency | Update Frequency |
|--------|------|---------|-----------------|
| CRM (Claim system) | Claim details, adjuster notes | Real-time | On claim creation |
| Policy DB | Policy terms, coverage | Real-time | On policy change |
| ISO Services | External claimant history | Real-time API | Per query |
| Internal history | Prior claims, fraud history | Redis (batch-refreshed) | Daily |
| Adjuster notes | Unstructured text | Async batch | As created |

---

### Block 3: Feature Engineering

**Online Feature Store (Redis) — for real-time:**
```
Key: claimant_id
Value: {
    "claims_last_30d": 3,
    "avg_claim_amount_90d": 12500.0,
    "days_since_last_claim": 45,
    "prior_fraud_flags": 0,
    "claimant_risk_score": 0.23
}
TTL: 24 hours (refreshed nightly by batch job)
```
These are pre-computed features that would be too slow to compute at request time.

**Offline Feature Store (Delta Tables) — for training:**
```
Delta table: fraud_features_daily
Columns: claim_id, date, [100+ engineered features]
Time travel: exact training set reproducible from any date
```

**Feature categories:**
1. **Claim features:** amount, type, description keywords, days since policy start
2. **Claimant behavioral:** claims frequency last 30/90/365 days, avg claim amount, time patterns
3. **Network features:** shared phone/address with known fraudsters, provider fraud history
4. **Geographic:** ZIP code fraud rate, county fraud rate, distance from home address
5. **Temporal:** claim submitted on weekday/weekend, holiday proximity, time of day
6. **NLP-derived:** BERT flags from claim notes (30-40 binary fraud indicators)
7. **External:** ISO CLUE report history, credit risk score

**Feature engineering code pattern (PySpark):**
```python
# Behavioral feature with rolling window
window = Window.partitionBy("claimant_id").orderBy("claim_date").rowsBetween(-90, 0)

features_df = claims_df \
    .withColumn("claims_90d", F.count("claim_id").over(window)) \
    .withColumn("avg_amount_90d", F.mean("claim_amount").over(window)) \
    .withColumn("max_amount_90d", F.max("claim_amount").over(window)) \
    .withColumn("same_day_claims", F.count("claim_id").over(
        Window.partitionBy("claimant_id", "claim_date"))) \
    .withColumn("provider_fraud_rate", provider_fraud_rates[F.col("provider_id")])
```

---

### Block 4: Model Layer

**Training Pipeline:**
```
Daily trigger (Airflow) → Feature Delta table snapshot
    → Train/validation/test split (temporal: train on 18mo, val on 3mo, test on 1mo)
    → Feature selection (IV > 0.1, VIF < 10)
    → Hyperparameter optimization (Optuna, 50 trials, optimize PR-AUC)
    → Cross-validation (StratifiedKFold, 5 folds)
    → Champion vs Challenger comparison
    → Register in MLflow if challenger wins
```

**Model choice by use case:**
| Model | Used For | Why |
|-------|----------|-----|
| XGBoost | Batch overnight scoring | Best calibrated probabilities for risk tiering |
| LightGBM | Real-time scoring | Sub-15ms inference, smaller memory footprint |
| Logistic Regression | Baseline, interpretability audit | Regulatory explainability when needed |
| BERT + RAG | NLP flag extraction from notes | Semantic understanding of unstructured text |

**Ensemble for batch:**
```
Final Score = 0.7 * XGBoost_score + 0.3 * LightGBM_score + NLP_flags_adjustment
```
LightGBM provides speed, XGBoost provides calibration, NLP flags add leading indicators.

**Inference optimization for real-time:**
- Model serialized with ONNX → 2-3x faster inference than native XGBoost/LightGBM
- Model loaded in memory at pod startup (not per-request)
- Feature extraction cached in Redis (no database calls during inference)

---

### Block 5: Serving Layer

**Real-time API:**
```
Client (CRM) → Load Balancer (ALB) → FastAPI Pod (K8s)
    ↓
    → Redis lookup (online features, <1ms)
    → LightGBM inference (<15ms)
    → SHAP top-3 factors (<5ms)
    → Response: {fraud_prob: 0.91, flag: true, factors: [...]}
    ↓
Total: < 50ms (well within 200ms SLA)
```

**Kubernetes configuration:**
```yaml
Deployment:
  replicas: 3 (min)  → 20 (max via HPA)
  resources.requests: 1 CPU, 512MB RAM
  resources.limits: 2 CPU, 2GB RAM
  
HPA:
  targetCPUUtilization: 70%
  scaleUpCooldown: 30s   (fast scale-up for traffic spikes)
  scaleDownCooldown: 300s (slow scale-down to avoid thrashing)
```

**Batch API:**
```
Airflow CronJob (2am)
    → Databricks job: reads 1M claims from Snowflake
    → Distributed XGBoost scoring (pandas_udf)
    → Write results to Snowflake: fraud_scores table
    → Trigger: SIU dashboard refresh, claim management alerts
    → Complete by 5:30am (SLA: 6am)
```

**API Response Schema:**
```json
{
    "claim_id": "CLM-2025-12345",
    "fraud_probability": 0.91,
    "fraud_flag": true,
    "risk_tier": "HIGH",
    "top_factors": [
        {"feature": "claims_90d", "value": 7, "impact": "+0.23"},
        {"feature": "amount_vs_avg", "value": 3.4, "impact": "+0.18"},
        {"feature": "provider_fraud_rate", "value": 0.15, "impact": "+0.12"}
    ],
    "model_version": "xgboost-fraud-v3.2",
    "timestamp": "2025-01-15T14:22:33Z"
}
```

---

### Block 6: Monitoring Layer

**Input monitoring (daily):**
```
For each of top 20 features:
    PSI(training_distribution, last_7_days_production) → alert if > 0.2
    KS-test p-value → alert if < 0.05
```

**Output monitoring (weekly):**
```
PSI(training_score_distribution, last_7_days_scores)
Predicted fraud rate vs historical average
Score distribution by risk tier
```

**Performance monitoring (on SIU feedback — 4-week lag):**
```
Every Monday: match last month's flags with SIU confirmed outcomes
    → Precision, Recall, PR-AUC on confirmed cases
    → Capture@25 (% fraud in top 25% risk tier)
    → Alert if PR-AUC drops > 3% from baseline
```

**Infrastructure monitoring (real-time):**
```
Prometheus → Grafana dashboards:
    - p50, p95, p99 latency (target: p99 < 200ms)
    - Requests per second
    - Error rate (target: < 0.1%)
    - Pod CPU/Memory
    - HPA scaling events
```

**Alerting strategy:**
| Alert | Threshold | Severity | Action |
|-------|-----------|----------|--------|
| PSI > 0.2 (feature) | Daily | High | Investigate root cause → retrain if needed |
| PR-AUC drop > 3% | Weekly | Critical | Immediate investigation, potential rollback |
| p99 > 500ms | Real-time | High | Check pod scaling, investigate bottleneck |
| Error rate > 1% | Real-time | Critical | PagerDuty, potential rollback |
| Pod count at max (20) | Real-time | Warning | Capacity planning |

---

### Block 7: Feedback Loop

**SIU Feedback Integration:**
```
SIU Investigators review flagged claims
    → 2-4 weeks later: confirm fraud or clear
    → Feedback written to labeled_claims table
    → Weekly batch: join predictions with confirmed outcomes
    → Compute performance metrics
    → If degraded: trigger retraining workflow
```

**Continuous Improvement:**
1. **Feature feedback:** SIU identifies new fraud pattern → data science adds new features → retrain
2. **Threshold optimization:** quarterly review of precision/recall tradeoff → adjust thresholds
3. **Model A/B testing:** challenger models always in 10% shadow mode
4. **NLP prompt refinement:** monthly review of RAG extraction accuracy → update prompts

**Interview answer (summary):**
> "The architecture has two paths: real-time (Kafka → FastAPI/LightGBM → Kubernetes) and batch (Airflow → Databricks/XGBoost → Snowflake). The feature store (Redis for online, Delta for offline) ensures consistency between training and serving. Monitoring is three-layered: input drift (PSI), output drift (score distribution), and business performance (SIU feedback). Feedback loops drive retraining every 4-6 weeks for fraud."

---

## DESIGN 2: RAG-BASED CLAIM NOTES ANALYSIS

### Block 1: Scope
- **Goal:** Extract 30-40 binary fraud indicators from unstructured claim notes
- **Scale:** 5,000 documents/day during business hours
- **Latency:** < 5 seconds per document
- **Output:** Structured fraud flags that feed into XGBoost model as features

---

### Block 2: Data Ingestion

**Document sources:**
1. Adjuster notes — plain text from claims management system
2. Emails — Microsoft Graph API → text extraction
3. PDFs — PyMuPDF/PDFplumber → text extraction

**Pre-processing pipeline:**
```
Raw document
    → Text extraction (OCR if scanned)
    → PII removal (Microsoft Presidio: NER-based masking)
    → Language detection (English only, flag others)
    → Length check (reject if < 50 words or > 50,000 words)
    → Cleaned text ready for chunking
```

---

### Block 3: Feature Engineering (Document Processing)

**Chunking strategy:**
```
Chunk size: 500 tokens (roughly 400 words)
Overlap: 100 tokens (preserves context at boundaries)
Method: RecursiveCharacterTextSplitter from LangChain
Reason: 500 tokens fits in context window, overlap prevents losing multi-sentence patterns
```

**Embedding:**
```python
# Option A: OpenAI (production choice for quality)
from openai import OpenAI
client = OpenAI()
embedding = client.embeddings.create(
    model="text-embedding-3-small",  # 1536 dimensions, faster
    input=chunk_text
).data[0].embedding

# Option B: Local Sentence Transformers (cost-sensitive)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
embedding = model.encode(chunk_text)
```

**Vector store:**
- FAISS for batch (fast, in-memory, no persistence needed for processing pipeline)
- Pinecone/ChromaDB for persistent historical pattern library
- Historical fraud case embeddings stored as "few-shot examples" for retrieval

**Ground truth dataset:**
- 200 confirmed fraud cases with manually annotated flags (underwriters)
- 300 confirmed genuine cases for contrast
- Used for: RAG retrieval examples AND weekly evaluation

---

### Block 4: Model Layer

**RAG Pipeline:**
```
Claim note text
    → Chunk into 500-token pieces
    → Embed each chunk (OpenAI ada-002)
    → Retrieve top-5 most similar historical fraud case chunks (FAISS)
    → Construct prompt:
        [System: fraud taxonomy + extraction instructions]
        [Retrieved examples: 2-3 similar fraud cases with their flags]
        [User: extract flags from this claim note]
    → GPT-4o with structured output (JSON schema)
    → Output: {flag_1: true, flag_2: false, ..., flag_40: true, confidence: {...}}
```

**LLM-as-Judge evaluation:**
```
Sample 5% of extracted flags daily
    → Judge prompt: "Given the claim text and extracted flag, 
                     is this extraction faithful and accurate?"
    → GPT-4.5 as judge (more capable than GPT-4o for evaluation)
    → Faithfulness score 1-5 per flag
    → Alert if average faithfulness < 4.0
```

**Prompt structure (simplified):**
```
SYSTEM: You are a fraud analyst. Extract the following 40 fraud indicators 
from the claim note. Return ONLY valid JSON with boolean values.
Fraud taxonomy: [detailed descriptions of each indicator]

RETRIEVED EXAMPLES:
Case 1 (confirmed fraud): [claim text] → [flags extracted]
Case 2 (genuine claim): [claim text] → [flags extracted]

CLAIM TO ANALYZE:
[current claim note]

Return JSON:
{
  "inconsistent_timeline": bool,
  "exaggerated_damages": bool,
  ...40 fields...
}
```

---

### Block 5: Serving Layer

**Async processing (not blocking real-time scoring):**
```
New claim created
    → Real-time: score immediately with structured features (no NLP yet)
    → Background: enqueue document processing job (Celery)
    → Celery worker: RAG pipeline extracts NLP flags (~3-5 seconds)
    → NLP flags written to claim_nlp_features table
    → Batch overnight: re-score claim with full feature set including NLP flags
```

Why async? NLP processing takes 3-5 seconds — too slow for real-time scoring SLA.
Solution: give immediate score without NLP, then enhance overnight with NLP features.

**API endpoint:**
```
POST /extract-flags
{
    "claim_id": "CLM-2025-12345",
    "document_text": "...",
    "document_type": "adjuster_notes"
}

Response:
{
    "flags": {"inconsistent_timeline": true, "exaggerated_damages": false, ...},
    "confidence": {"inconsistent_timeline": 0.92, ...},
    "retrieval_context": ["Case CLM-2023-789 matched with score 0.87"],
    "processing_time_ms": 3200
}
```

---

### Block 6: Monitoring

**RAG-specific metrics (Ragas framework):**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# Weekly evaluation on ground truth set
results = evaluate(
    dataset=ground_truth_qa_pairs,
    metrics=[faithfulness, answer_relevancy, context_recall]
)
# Alert if faithfulness < 0.8 or context_recall < 0.7
```

**Business metrics:**
- Precision/Recall of NLP flags vs human-annotated ground truth (weekly)
- Coverage: % of claims successfully processed (target > 99%)
- Processing time: p99 < 10 seconds (Celery worker)
- Cost: LLM API cost per document (target < $0.05/doc)

**Model version monitoring:**
- When OpenAI releases new model version: re-run ground truth evaluation before switching
- Prompt drift: compare outputs on 50 fixed test cases monthly to detect behavioral changes
- Known issue documented in transcripts: "reasoning models like GPT-o1 need different prompts — chain-of-thought prompting becomes unnecessary"

---

### Block 7: Feedback Loop

**Underwriter corrections:**
```
SIU reviews NLP-flagged claims
    → Confirms or corrects flag extractions
    → Corrections feed back to ground truth dataset
    → Ground truth dataset grows from 200 to 1000+ cases
    → Better RAG retrieval quality (more examples)
    → Periodic re-evaluation to measure improvement
```

**Prompt iteration cycle:**
- Monthly: analyze top 20 extraction errors → update prompt taxonomy
- Quarterly: consider fine-tuning if error rate > 10% on systematic patterns

---

## DESIGN 3: MARKETING MIX MODELING PIPELINE (AXTRIA)

### Block 1: Scope
- **Goal:** Attribute sales to marketing channels, optimize budget allocation
- **Scale:** Weekly data aggregation, 3 years of history per analysis
- **Output:** Channel contribution (ROI), optimized budget recommendations

### Block 2: Data Ingestion
- Sales data from company ERP (weekly by region, product)
- Channel spend data: TV (Nielsen), Digital (Google Analytics, Facebook Ads API), Print (vendor invoices)
- External data: seasonality index, competitor pricing, economic indicators
- All merged into weekly granularity time series

### Block 3: Feature Engineering
- Adstock transformation per channel: `Adstock(t) = Spend(t) + decay * Adstock(t-1)`
- Saturation transformation: `Saturation(t) = 1 - exp(-k * Spend(t))` (diminishing returns)
- Calendar features: holidays, seasonality indices, day-of-year
- Lag features: 1-week, 2-week, 4-week lags for slow-response channels
- Normalization: all features scaled to same range for model interpretability

### Block 4: Model Layer
- **Baseline:** Linear regression with adstock → establish interpretable baseline
- **Advanced:** XGBoost for non-linear saturation and channel interactions
- **Validation:** Holdout last 3 months, MAPE < 10%, R-squared > 0.85
- **Attribution:** SHAP values for channel contribution (XGBoost) or coefficients (linear)

### Block 5: Serving
- Offline analysis: Python scripts, Databricks notebooks, output Excel/PowerPoint for stakeholders
- Optimization module: SciPy constrained optimization given budget B, maximize predicted sales
- Scenario tool: simple web app (Streamlit) for business team to run what-if simulations

### Block 6: Monitoring
- Monthly: re-run model with latest data, compare new predictions vs actuals
- MAPE tracking: if MAPE degrades > 15%, investigate data quality or model drift
- Business validation: channel attribution vs business team priors (sanity check)

### Block 7: Feedback Loop
- Quarterly refresh: retrain with full updated dataset
- Business feedback: if attribution disagrees with business intuition → investigate
- A/B testing: limited (can't easily A/B test TV spend), rely on holdout periods

---

## HOW TO ANSWER SYSTEM DESIGN IN AN INTERVIEW

### The First 3 Minutes

> "Before I jump into the design, can I ask a few clarifying questions?"

**Always ask:**
1. "What scale are we talking about — how many claims/transactions per day?"
2. "What's the latency requirement — real-time in under 200ms, or can we do batch?"
3. "What does 'fraud detection' mean in this context — suspicious flagging or hard block?"
4. "Any compliance constraints I should design around?"

### The Structure (Use This Every Time)

```
"I'll walk through this in blocks.

First, let me scope it: [Block 1 - 1 minute]

For data ingestion, here's how data enters... [Block 2 - 1 minute]

Feature engineering is the key complexity here... [Block 3 - 2 minutes]

For the model layer... [Block 4 - 2 minutes]

Serving architecture... [Block 5 - 2 minutes]

Monitoring and observability... [Block 6 - 1 minute]

Feedback loop for continuous improvement... [Block 7 - 1 minute]

Any area you'd like me to go deeper on?"
```

### The Golden Rule for System Design

> **Never start with the most complex thing.** Start with the simplest architecture that meets the requirements, then add complexity with justification.

Example of WRONG start:
> "So I'd use Kafka with exactly-once semantics, FAISS vector store, LangGraph agentic orchestration..."

Example of RIGHT start:
> "For 10K claims/day, the simplest approach is a REST endpoint that takes claim features, scores with a pre-loaded LightGBM model, and returns fraud probability. Let me then explain where we add complexity and why..."

### At Every Block, Say: WHAT → WHY → TRADEOFF

> "For the serving layer, I use Kubernetes with HPA [WHAT]. This is because our traffic has 10x variability between midnight and 2pm peak [WHY]. The tradeoff vs serverless (Lambda) is: K8s has 50ms cold start vs Lambda's 500ms, which matters for our 200ms SLA [TRADEOFF]."

---

## SECTION 8: VECTOR DATABASE SELECTION AND ANN TRADE-OFFS

### When to Use a Vector Database

Use a vector database when the problem requires approximate nearest neighbor (ANN) search over dense embeddings. Do not use it if an inverted index or full-text search is sufficient.

**Use cases in your experience:**
- Retrieval of similar historical fraud claim notes
- Semantic search over adjuster notes
- Few-shot example retrieval for NLP flag extraction
- Entity and document similarity in knowledge graphs

### ANN Index Types

| Index Type | Mechanism | Strengths | Weaknesses | Best Use Case |
|---|---|---|---|---|
| Flat (brute-force) | Exact distance to every vector | Perfect recall | O(N) memory and query cost | Small datasets, prototyping, recall benchmarking |
| IVF (inverted file) | Partitions space into Voronoi cells, searches nearest cells | Fast, low memory overhead | Recall depends on nprobe setting; struggles with high-dimensional sparse data | Medium scale (millions), balanced recall vs latency |
| HNSW (hierarchical navigable small world) | Multi-layer proximity graph | Very fast, high recall at high throughput | High memory, slow construction, fragile to deletes/updates | Real-time high-recall search at scale |
| PQ (product quantization) | Compresses vectors into codes | Low memory, fast scanning | Lower recall, distance is approximate | Billions of vectors, memory-constrained deployments |
| IVFPQ | Combines IVF clustering with PQ compression | Low memory + faster search | Build time, recall trade-offs | Large-scale semantic search |

### Flat vs IVF vs HNSW Decision Framework

**Small dataset (< 100K vectors):**
- Flat index is enough
- Use Flat to measure baseline recall

**Medium dataset (100K to 10M vectors):**
- IVF with nprobe tuned between 10 and 100
- Good balance of latency, recall, and memory

**Large dataset (> 10M vectors) with strict latency:**
- HNSW if memory budget allows
- IVFPQ if memory is constrained

### Hosted vs Self-Hosted Vector Stores

| Option | Strengths | Weaknesses |
|---|---|---|
| FAISS (self-hosted) | Free, fast, flexible | No persistence, no multi-node, ops burden |
| Pinecone | Managed, scalable, metadata filtering | Cost, vendor lock-in, limited customization |
| ChromaDB | Easy local/dev, lightweight | Not production-grade for high scale |
| Weaviate / Qdrant | Open-source, managed options, hybrid search | Need Kubernetes/platform expertise |
| pgvector | SQL-native, ACID, familiar | Performance lower than specialized stores |

### Interview One-Liner

> "For claim note retrieval I would benchmark three FAISS configurations: Flat as a recall baseline, IVF for cost efficiency at medium scale, and HNSW if we need sub-100ms retrieval at millions of vectors. I would also evaluate Pinecone if operational overhead needs to be minimized."

### Metadata Filtering and Hybrid Search

Real systems rarely do pure vector search. They combine vector similarity with metadata filters:
- Claim type filter (auto vs property)
- Date range filter
- Policy line of business filter
- Fraud confirmation status

Hybrid search (vector + BM25/keyword) improves results when embeddings miss exact terminology.

---

## SECTION 9: AGENTIC AI AND LLM WORKFLOW SYSTEM DESIGN

### What Makes a System "Agentic"

An agentic system is one where an LLM is not just generating text but making decisions, choosing tools, iterating toward a goal, and maintaining state across turns.

**Core capabilities:**
- Tool use: call APIs, query databases, run code
- Planning: break a complex task into sub-tasks
- Memory: retain context across interactions
- Reflection: evaluate and correct its own outputs

### State Machine Approach

Use a state machine to manage agent lifecycle:
- START → GATHER_EVIDENCE → REASON → GENERATE → REVIEW → END
- Each state has defined inputs, outputs, and transitions
- Guards prevent infinite loops (max iterations, timeouts)

**Why state machines matter in production:**
- Deterministic observability
- Easier testing of each state
- Clear rollback points
- Compliance and auditability

### Context Management

**Three types of context:**
1. **System prompt:** persistent instructions, taxonomy, guardrails
2. **User prompt:** current task or question
3. **Working memory:** intermediate results, tool outputs, prior reasoning

**Context window pressure strategies:**
- Summarize long working memory before each LLM call
- Use retrieval to pull in only relevant documents
- Keep few-shot examples small and targeted
- Maintain separate short-term and long-term memory stores

### KV Cache and Efficiency

For autoregressive models, the KV cache stores key and value tensors from prior tokens to avoid recomputation. In agentic loops:
- Reuse KV cache across turns when context is mostly unchanged
- Watch for cache invalidation when system prompt or retrieved context changes
- Large context windows increase memory pressure

**Interview framing:**
> "In an agentic workflow I treat the LLM call as expensive state transitions. I use a state machine to control loops, a vector store for long-term memory, and I summarize working memory to stay within context limits."

### Tool Calling and Structured Output

**Tool calling contract:**
- Define function schemas with name, description, parameter types, and required fields
- LLM returns a JSON object matching the schema
- Executor invokes the function and returns result

**Structured output enforcement:**
- Use JSON mode or function calling
- Validate output against Pydantic schema
- Retry with stronger prompt or smaller model temperature on parse failure
- Fallback to deterministic rules if the LLM repeatedly fails

### Guardrails

- Input guardrails: reject PII, toxicity, out-of-scope requests
- Output guardrails: enforce schema, block disallowed content, hallucination checks
- Operational guardrails: max calls, cost limits, timeout, rate limiting

### Reflection Layers

Add a reflection step where a second LLM or the same LLM with a different prompt reviews the output:
- Does the answer match the evidence?
- Are there unsupported claims?
- Does the output follow the required schema?

Reflection reduces hallucination and improves reliability.

### Agentic AI Interview Scenario

**"Design an agentic system that helps fraud investigators review a claim."**

> "I would design a state machine with these states: PLAN, GATHER_EVIDENCE, REASON, DRAFT_SUMMARY, REVIEW. The agent first plans what evidence it needs, then calls tools to retrieve policy details, prior claims, network links from Neo4j, and similar fraud cases from the vector store. It reasons over the evidence, drafts a structured summary, and a reflection layer checks the summary against the evidence. All tool calls are logged for audit."

---

## SECTION 10: KNOWLEDGE GRAPH INTEGRATION IN FRAUD DETECTION

### Why a Knowledge Graph

Fraud rings are inherently relational. A knowledge graph captures relationships between claimants, addresses, phone numbers, providers, vehicles, policies, and repair shops.

**Graph advantages:**
- Detects connected fraud rings that tabular features miss
- Enables explainable network reasoning
- Supports dynamic entity resolution
- Provides features for ML models (e.g., shared-neighbor count, PageRank)

### Entity Types and Relationships

**Entities:**
- Claimant, Provider, Adjuster, Policy, Claim, Vehicle, Address, Phone, Email, BankAccount

**Relationships:**
- CLAIMANT_HAS_POLICY
- CLAIM_SUBMITTED_BY_CLAIMANT
- CLAIMANT_SHARES_ADDRESS_WITH
- CLAIMANT_SHARES_PHONE_WITH
- PROVIDER_WORKED_ON_CLAIM
- VEHICLE_LINKED_TO_CLAIM

### Graph Construction

- Source data: CRM, policy systems, prior claims, external watchlists
- Entity resolution: deterministic matching (exact phone) and probabilistic matching (fuzzy name + address)
- Updates: streaming via CDC into Neo4j or a graph store
- Storage: Neo4j for transactional graph queries, or property graph on Databricks for batch

### Graph Algorithms for Fraud

| Algorithm | Fraud Signal | Example |
|---|---|---|
| Connected components | Find clusters of shared identifiers | Same phone used by 20 claimants |
| PageRank / centrality | Identify influential suspicious entities | A provider connected to many fraud cases |
| Common neighbors | Measure relationship strength | Two claimants share provider and address |
| Community detection (Louvain) | Discover hidden fraud rings | A group of colliding claimants and clinics |
| Shortest path | Trace relationship chains | Distance between a new claimant and known fraudster |

### Combining Graph with ML

1. **Graph features into tabular model:** common neighbor count, Jaccard similarity, graph embedding
2. **Graph as post-processing filter:** flag claims linked to known fraud ring within 2 hops
3. **Graph-assisted investigation UI:** show fraud investigators the network context

**Interview one-liner:**
> "I would use Neo4j to store claimants, providers, addresses, and policies as nodes. Graph algorithms like connected components and PageRank would surface rings. These features would feed the LightGBM/XGBoost model, and the graph itself would be exposed to investigators for explainability."

---

## SECTION 11: ONLINE VS OFFLINE INFERENCE TRADE-OFFS

### Online Inference

**Definition:** A request arrives, features are fetched, model scores, and a response is returned synchronously.

**When required:**
- Fraud decision at claim submission
- Real-time recommendations
- Risk scoring during a user session

**Design priorities:**
- Latency: p99 must be under SLA
- Availability: 99.9% uptime or higher
- Consistency: same features used in training and serving
- Caching: pre-compute features and cache hot entities

### Offline Inference

**Definition:** Batch process all entities, store scores, downstream systems read from table.

**When used:**
- Overnight batch scoring for investigator queues
- Campaign targeting lists
- Daily risk tier refresh

**Design priorities:**
- Throughput: millions of rows per hour
- Cost: prefer spot instances / auto-terminating clusters
- Reproducibility: version data and model
- Partitioning and idempotency

### Trade-off Table

| Factor | Online | Offline |
|---|---|---|
| Latency | sub-second | hours OK |
| Complexity | high (serving, caching, auto-scaling) | lower |
| Cost | always-on baseline | pay per batch run |
| Feature freshness | real-time | refreshed periodically |
| Error impact | immediate customer impact | can be fixed before use |
| Best for | real-time decisions | periodic reports and queues |

### Mixed Design

Most production ML systems use both:
- Real-time lightweight model for immediate decision
- Offline comprehensive model for final risk tier and investigation prioritization
- Real-time gives speed, offline gives accuracy and completeness

---

## SECTION 12: MULTI-TENANT ML SYSTEMS

### Tenant Isolation Models

| Isolation Level | Description | Pros | Cons |
|---|---|---|---|
| Single shared model | One model serves all tenants | Simple, cost-effective | Cannot adapt to tenant-specific patterns |
| Tenant-specific fine-tuning | Shared backbone, tenant-specific head or weights | Balanced customization and cost | More deployment artifacts |
| Fully isolated models | Separate model per tenant | Maximum customization | High operational cost |

### Tenant-Aware Features

- Add tenant_id as a categorical feature if allowed by privacy
- Use per-tenant normalization or binning where distributions differ
- Be careful not to leak data across tenants

### Cost and Fairness Considerations

- Smaller tenants may not have enough data for a dedicated model
- Fairness metrics should be evaluated per tenant
- Resource quotas prevent one tenant from consuming all serving capacity

**Interview one-liner:**
> "For a multi-tenant fraud platform I would start with a shared model and add tenant-specific calibration layers. Only the largest tenants would get dedicated fine-tuned models because smaller tenants lack enough labeled fraud cases."

---

## SECTION 13: RESILIENCE AND FAILURE HANDLING

### Failure Modes in ML Serving

1. **Model loading failure** → container crash loop
2. **Feature store unavailable** → missing features
3. **Downstream API slow** → timeout or degraded response
4. **Model returns invalid probability** → schema violation
5. **Score distribution shifts** → concept drift

### Resilience Patterns

**Circuit breaker:**
- If feature store or downstream API fails repeatedly, stop calling it temporarily
- Return a default score or use cached features
- Prevent cascading failure

**Fallback scoring:**
- Simple rule-based score when model is unavailable
- Example: if no model, flag claims with amount > threshold and recent claim frequency

**Graceful degradation:**
- If NLP features are slow, return score without them
- Log degraded response and re-score later

**Retry and timeout:**
- Timeouts on every external call
- Exponential backoff with jitter
- Max retry limits

**Health checks:**
- Readiness probe: model loaded, feature store reachable
- Liveness probe: process is healthy
- Startup probe: allow slow model loading

### Interview One-Liner

> "I design serving with circuit breakers on external dependencies, a rule-based fallback if the model fails, and readiness probes that only mark pods live after the model is loaded. If the feature store is slow, I serve with cached values and flag for re-scoring."

---

## SECTION 14: COST, LATENCY, AND QUALITY TRIANGLE

### The Inevitable Trade-Off

You can optimize any two of: cost, latency, model quality. The third will suffer.

| Priority | Strategy | Trade-Off |
|---|---|---|
| Low latency + high quality | Heavy model, GPU serving, caching | High cost |
| Low cost + low latency | Small model, CPU only | Lower quality |
| Low cost + high quality | Large model, batch offline only | High latency |

### Practical Optimizations

**Latency optimization:**
- Model quantization (INT8)
- ONNX / TensorRT inference
- In-memory model loading
- Feature caching in Redis
- Batched GPU inference when possible

**Cost optimization:**
- Use spot/preemptible instances for batch training
- Right-size Kubernetes resources
- Use smaller embedding models when recall is sufficient
- Cache repeated queries

**Quality optimization:**
- Ensemble models
- Richer features (NLP, graph)
- More labeled data
- Longer inference time for complex cases

### Interview Framing

> "I start with the business SLA. If the requirement is 200ms p99, I choose LightGBM on CPU with Redis features. If the requirement is maximum recall on a nightly batch, I use XGBoost with graph and NLP features on a larger cluster. I do not over-engineer beyond the stated constraint."

---

## SECTION 15: SYSTEM DESIGN INTERVIEW SCENARIOS

### "Design a fraud detection system that uses both structured and unstructured data."

> "I would split the problem into two paths. Structured data goes through Kafka to a FastAPI/LightGBM endpoint for real-time scoring. Unstructured claim notes are processed asynchronously by a RAG pipeline that extracts binary fraud flags and writes them to the feature store. Both feature sets are combined in the nightly XGBoost batch model. A Neo4j knowledge graph provides ring-detection features and investigator explainability."

### "How do you decide between a batch and real-time ML system?"

> "It depends on the action latency. If the business must block a claim at submission, we need real-time. If we only need to prioritize an investigator queue, batch is sufficient. Batch is cheaper and more accurate because it can use richer features. Real-time is more complex and must optimize for latency."

### "Design an LLM-powered claim summarization system."

> "I would use an async pipeline. Documents are extracted, PII-masked, chunked, and embedded. A RAG module retrieves similar historical cases. A structured-output LLM call produces a JSON summary with confidence scores. A reflection layer checks the summary against the source text. The summary is stored and linked to the claim. All LLM calls are logged for audit."

### "How would you handle a 10x traffic spike in a real-time scoring system?"

> "Kubernetes HPA scales pods based on CPU and request queue depth. The model is pre-loaded so pods start serving immediately. Redis caching reduces database load. If autoscaling reaches max capacity, I would implement load shedding and serve only high-priority claims. The fallback rule-based system keeps core decisions flowing."

### "What is your approach to choosing a vector database?"

> "I define requirements: dataset size, recall target, latency SLA, metadata filtering needs, and budget. I benchmark FAISS Flat, IVF, and HNSW on a representative query set. If operational overhead is acceptable, FAISS on Kubernetes works. If not, I evaluate managed options like Pinecone or Weaviate. I also consider hybrid search if exact keyword matching matters."

---

## SECTION 16: LEAD-LEVEL SYSTEM DESIGN EXPECTATIONS

### What Interviewers Look for in Lead Candidates

1. **Scope before solution.** Ask clarifying questions about scale, latency, compliance, and success metrics before proposing architecture.
2. **Explain trade-offs.** Every design choice should have a reason and an alternative considered.
3. **End-to-end ownership.** Show you understand data ingestion, feature engineering, model training, serving, monitoring, and feedback loops.
4. **Operational rigor.** Mention rollback, fallback, observability, cost, and security without being asked.
5. **Business context.** Connect technical choices to business outcomes: investigator efficiency, customer satisfaction, cost, regulatory compliance.
6. **Simplicity first.** Reject unnecessary complexity. Add components only when justified.

### Phrases That Signal Seniority

- "Let me scope this first."
- "The simplest architecture that meets the requirements is..."
- "The trade-off here is..."
- "We monitor three layers: input, output, and business outcome."
- "My rollback plan is..."
- "We prevent training-serving skew by..."
- "For this use case I would benchmark Flat, IVF, and HNSW indexes."
- "In a lead role, I would set the standard for runbooks, post-mortems, and SLOs."

### How to Close a System Design Answer

> "To summarize, I would start with a simple REST scoring service on Kubernetes for real-time decisions, backed by Redis online features and an offline XGBoost batch model. I would add a RAG/NLP pipeline for unstructured data and a Neo4j knowledge graph for ring detection. Monitoring covers input drift, prediction drift, and business outcomes. Rollback and fallback are pre-defined. What area would you like me to go deeper on?"
