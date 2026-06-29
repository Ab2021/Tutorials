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

---

## SECTION 17: DESIGN 4 — RECOMMENDATION SYSTEM

### Block 1: Problem Scope

**Goal:** Recommend relevant items to users (products, movies, content).
**Scale:** 10M users, 1M items, 100M interactions/day.
**Latency:** < 100ms for real-time recommendations.
**Constraints:** personalization, freshness, diversity, business rules.

### Block 2: Data Ingestion

- **User interactions:** clicks, views, purchases, ratings — Kafka events
- **Item metadata:** categories, descriptions, prices — batch load to data warehouse
- **User profiles:** demographics, preferences — batch/refreshed daily
- **Context:** time of day, device, location — real-time at request time

### Block 3: Feature Engineering

- **User features:** recent interests, historical purchase categories, average price point
- **Item features:** popularity, category, price, text/image embeddings
- **Interaction features:** recency, frequency, implicit signals (dwell time, skip rate)
- **Two-tower embeddings:** separate user and item towers, dot product for scoring

### Block 4: Model Layer

**Candidate generation (fast, recall-focused):**
- Collaborative filtering: users who viewed X also viewed Y
- Approximate nearest neighbor on user/item embeddings
- Content-based filtering for cold-start items

**Ranking (slower, precision-focused):**
- Gradient-boosted ranker or neural ranker
- Features: user-item affinity, popularity, freshness, diversity, business constraints
- Loss: pairwise ranking loss or softmax cross-entropy

**Re-ranking:**
- Diversity rules
- Business rules (promote sponsored items within limits)
- Freshness boost for new items

### Block 5: Serving Layer

- Pre-compute candidate pools in batch
- Real-time API fetches user profile, runs ANN retrieval, scores top-K candidates with ranking model
- Cache popular user recommendations
- A/B test ranking models on recommendation click-through rate and downstream conversion

### Block 6: Monitoring

- Offline: recall@K, NDCG, MAP
- Online: click-through rate, conversion rate, dwell time, coverage (% items recommended)
- Guardrails: bad recommendations rate, inappropriate content rate

### Block 7: Feedback Loop

- Capture user clicks and skips
- Update embeddings and ranking model on a schedule
- Handle feedback loops: popular items get more exposure, reinforcing popularity — use exploration/exploitation balance

**Interview one-liner:**
> "A production recommender has three stages: candidate generation with collaborative filtering and ANN, ranking with a learned model, and re-ranking for diversity and business rules. I monitor both offline ranking metrics and online engagement, and I balance exploration so new items get a chance."

---

## SECTION 18: DESIGN 5 — SEARCH AND AD RANKING SYSTEM

### Block 1: Problem Scope

**Goal:** Rank search results or ads by relevance/expected value.
**Scale:** 100K queries/second, billions of documents/ads.
**Latency:** < 50ms p99.
**Primary metric:** click-through rate, conversion rate, or revenue.

### Block 2: Data Ingestion

- **Queries:** real-time search logs
- **Documents/ads:** batch index with metadata
- **Click/conversion feedback:** delayed, often hours to days
- **User context:** device, location, time, search history

### Block 3: Feature Engineering

- **Query features:** length, category intent, location
- **Document/ad features:** relevance score, quality score, bid, historical CTR
- **User features:** location, device, recent searches
- **Query-document features:** BM25, semantic similarity, click history for this query-document pair

### Block 4: Model Layer

- **Retrieval:** inverted index + ANN for semantic matches
- **Ranking:** LambdaMART, XGBoost, or neural ranker
- **Objective:** maximize expected utility (clicks, conversions, revenue)
- **Calibration:** predicted probabilities should match actual click rates for ad pricing

### Block 5: Serving Layer

- Query arrives → retrieval engine fetches candidate set
- Feature lookup from feature store
- Ranker scores candidates
- Apply business rules and filters
- Return top-K

### Block 6: Monitoring

- Online: CTR, conversion rate, revenue per query, latency, error rate
- Offline: NDCG, precision@K on labeled query-result pairs
- Calibration: reliability diagrams for predicted CTR

### Block 7: Feedback Loop

- Log every impression, click, and conversion
- Retrain ranker on delayed labels
- A/B test new rankers against production baseline
- Monitor for position bias and selection bias

**Interview one-liner:**> "Search ranking uses retrieval to fetch candidates, a learned ranker to score them using query-document and context features, and calibration so predicted click probabilities match reality for pricing. I log impressions and clicks, handle delayed feedback, and A/B test before rollout."

---

## SECTION 19: DESIGN 6 — ETA / DEMAND PREDICTION SYSTEM

### Block 1: Problem Scope

**Goal:** Predict delivery ETA or ride arrival time.
**Scale:** millions of trips/day.
**Latency:** < 50ms per request.
**Primary metric:** MAE or RMSE of predicted vs actual time.

### Block 2: Data Ingestion

- Historical trip records: start time, route, distance, weather, traffic, driver, vehicle type
- Real-time traffic: streaming traffic speed data
- Weather, events: batch/external APIs

### Block 3: Feature Engineering

- Route features: distance, number of turns, road types
- Temporal features: hour of day, day of week, holidays
- Traffic features: current segment speeds, historical averages
- Driver/restaurant features: historical prep/delivery times
- Weather features: rain, snow, temperature

### Block 4: Model Layer

- Baseline: heuristic based on distance and average speed
- Advanced: gradient boosting or neural network with route and traffic features
- For ETA, regression with MAE or custom loss penalizing late predictions more than early ones

### Block 5: Serving Layer

- Real-time API receives origin, destination, and context
- Route engine computes candidate path
- Model predicts ETA for each path segment
- Aggregate and return ETA with confidence interval

### Block 6: Monitoring

- MAE, RMSE, bias (are we systematically early or late?)
- Segment-level errors to identify bad routes or traffic data
- Customer complaints about late deliveries

### Block 7: Feedback Loop

- Compare predicted vs actual ETAs
- Retrain model weekly or daily depending on data volume
- Update traffic and weather features continuously

**Interview one-liner:**> "An ETA system combines route computation with a regression model that uses distance, traffic, weather, and historical segment speeds. I monitor MAE and bias, and I penalize late predictions more than early ones because customer impact is asymmetric."

---

## SECTION 20: DESIGN 7 — CONTENT MODERATION CLASSIFIER

### Block 1: Problem Scope

**Goal:** Detect toxic, harmful, or policy-violating user-generated content.
**Scale:** millions of posts/day.
**Latency:** < 100ms for synchronous decisions.
**Primary metric:** precision and recall per violation category.

### Block 2: Data Ingestion

- User posts, images, videos
- Human moderator labels
- User reports

### Block 3: Feature Engineering

- Text: embeddings, keyword lists, language detection
- Images: vision embeddings, object detection, OCR text
- User history: prior violations, report rate, account age

### Block 4: Model Layer

- Lightweight rules for obvious violations
- Text classifier (BERT or smaller transformer) for nuanced text
- Image classifier for visual content
- Ensemble for final decision

### Block 5: Serving Layer

- Synchronous API for live posts: allow, flag for review, or block
- Asynchronous pipeline for uploaded media
- Human review queue for borderline cases

### Block 6: Monitoring

- Per-category precision/recall
- False positive rate (legitimate content blocked)
- Appeal rate and overturn rate
- Throughput and latency

### Block 7: Feedback Loop

- Moderator labels feed back into training data
- Handle adversarial evasion by monitoring new slang, image perturbations
- Balance automation with human review for high-stakes categories

**Interview one-liner:**> "Content moderation uses rules for obvious cases, BERT-based classifiers for text, and vision models for images. Borderline cases go to human review. I monitor per-category precision/recall and false positive rate because blocking legitimate content is a serious user experience issue."

---

## SECTION 21: ETL AND DATA QUALITY SYSTEM DESIGN

### Why Data Quality Is a System Design Concern

Bad data silently breaks models. Data quality must be designed into the pipeline, not added later.

### Data Quality Checks

| Check | Purpose | Example |
|---|---|---|
| Schema validation | Columns exist with correct types | claim_amount is numeric |
| Null rate check | Missing values within expected range | provider_id null rate < 1% |
| Distribution check | Values within expected range | claim_amount between 0 and 10M |
| Freshness check | Data arrived on time | Claims table updated within 1 hour |
| Volume check | Row counts within expected range | Daily claims between 5K and 50K |
| Uniqueness check | Primary keys are unique | claim_id has no duplicates |
| Referential check | Foreign keys resolve | policy_id exists in policy table |

### ETL Pipeline Block

```
Source systems
    → Ingestion (batch or streaming)
    → Validation layer
    → Transformation
    → Feature store / data warehouse
    → Downstream consumers
```

### Handling Data Quality Failures

- **Warn:** minor deviation, pipeline continues, alert owner
- **Block:** severe deviation, pipeline halts, downstream models use last known good data
- **Quarantine:** suspect rows written to quarantine table for investigation

### Interview One-Liner

> "I design data quality into the ETL pipeline with schema, null, distribution, freshness, volume, uniqueness, and referential checks. Severe issues block the pipeline and alert the owner; minor issues warn and continue. Suspect rows are quarantined for review."

---

## SECTION 22: SCHEMA EVOLUTION IN ML SYSTEMS

### Why It Matters

Upstream producers change schemas. If the feature pipeline does not handle this, models receive wrong inputs and predictions fail silently.

### Schema Contract Approach

- Define expected schema per feature pipeline version
- Validate incoming data against the contract
- Version the contract alongside model versions

### Handling Different Change Types

- **New column:** warn and ignore for current model; may be useful for next model
- **Removed column:** fail if the model requires it; activate fallback
- **Type change:** fail and alert
- **New category value:** handle via unknown-category encoding or retraining plan

### Interview One-Liner

> "I enforce schema contracts at ingestion and inference. Breaking changes fail the pipeline. Additive changes trigger warnings. Each model version is tied to a specific schema contract so we can reproduce behavior and detect mismatches."

---

## SECTION 23: MULTI-TENANT ML PLATFORM — DEEPER ARCHITECTURE

### High-Level Architecture

```
Tenant requests
    → API Gateway (authentication, rate limiting, routing)
    → Tenant Router (selects model variant and features)
    → Feature Store (tenant-isolated or tenant-aware features)
    → Model Serving (shared or tenant-specific models)
    → Monitoring (per-tenant metrics)
    → Billing (cost attribution)
```

### Key Components

- **Tenant router:** maps tenant to correct model version and feature set
- **Feature isolation:** logical separation of tenant data in feature store
- **Model registry per tenant:** or shared registry with tenant-tagged versions
- **Per-tenant monitoring:** dashboards and alerts per tenant
- **Cost attribution:** track compute, storage, API calls per tenant

### Operational Patterns

- Shared infrastructure with tenant-specific weights/calibration
- Separate namespaces for high-value tenants
- Gradual rollout per tenant

### Interview One-Liner

> "A multi-tenant ML platform has a tenant router, isolated feature namespaces, shared or tenant-specific models, per-tenant monitoring, and cost attribution. This balances operational efficiency with customization and data isolation."

---

## SECTION 24: ADDITIONAL SYSTEM DESIGN SCENARIOS

### "Design a real-time feature platform."

> "I would use Kafka for streaming events, Flink or Spark Structured Streaming for real-time aggregations, Redis for online serving, and Delta Lake for offline historical storage. The same computation logic runs in both paths to prevent training-serving skew. A schema registry validates event schemas."

### "How do you design for low-latency inference at scale?"

> "I minimize request-time computation by pre-computing features and caching hot entities. I use small, quantized models or ONNX. I deploy on Kubernetes with HPA and keep minimum replicas warm. I avoid synchronous calls to slow dependencies and use circuit breakers."

### "Design a system for A/B testing ML models."

> "Randomize by the unit that receives treatment, not by request. Use a consistent hash on user/claim ID so the same entity always sees the same model. Log model version with every prediction. Wait for lagged labels before computing metrics. Monitor guardrails and have a rollback plan."

---

## SECTION 25: MULTI-MODAL CLINICAL PREDICTION SYSTEM (ICU OUTCOMES)

### Problem Statement

Predict ICU patient health outcomes using multiple data sources: medical devices (vital signs streams), EHR (diagnoses, medications, labs), and clinical notes.

### Design Walkthrough

**1. Data ingestion**
- Medical devices: Kafka or MQTT for high-frequency streams; downsample and validate ranges.
- EHR: batch extract via FHIR/HL7 into Delta Lake.
- Notes: NLP pipeline with de-identification before any LLM/BERT processing.

**2. Feature engineering**
- Time-series features: rolling mean, trend, alarms, device-derived severity scores.
- Structured features: comorbidities, lab values, medication history.
- Text features: entity extraction, clinical note embeddings, medication/diagnosis mentions.

**3. Modeling**
- Early warning: LightGBM/XGBoost on tabular features for interpretability and speed.
- Complex cases: small transformer or multimodal fusion if data supports it.
- Always compare against simple clinical scores (APACHE, SOFA) as baseline.

**4. Serving**
- Real-time scoring from streaming vitals via FastAPI with Redis for precomputed patient context.
- Asynchronous comprehensive scoring for new admissions using batch EHR + notes.
- Human-in-the-loop: predictions feed a dashboard, not a direct medical decision.

**5. Safety and governance**
- De-identify and restrict PHI access.
- Model cards record intended use, patient populations, and failure modes.
- Bias checks across demographics; continuous validation against outcomes.
- Regulatory: maintain audit trail, versioning, and rollback.

**6. Monitoring**
- Input drift on vital sign distributions and lab ordering patterns.
- Performance drift on mortality/length-of-stay labels after maturation.
- Alert clinical stakeholders when model confidence is low.

### Interview One-Liner

> "For ICU outcome prediction I fuse streaming vitals, structured EHR, and clinical notes. I use interpretable gradient boosting for early warnings, keep all PHI de-identified, require human-in-the-loop for clinical decisions, and validate against simple clinical baselines and outcome data before trusting any signal."

---

## DESIGN X: MULTI-TENANT ENTERPRISE GENAAI PLATFORM (AXTRIA — FROM PROD.TXT)

> This is a real system I built. Use this design to answer: "Design a production LLM platform," "Design a multi-tenant AI system," or "How would you build an agentic AI backend?"

### Block 1: Problem Scope

**Business goal:** Deliver 6 AI-powered product surfaces (Text-to-Agent, Text-to-SQL, RAG, Multi-Agent, Chat, Automation) to enterprise clients through a single, secure, observable backend.

**Key constraints:**
- Multi-tenancy: strict data isolation between clients — one client cannot see another's data
- Real-time UX: LLM responses must stream token-by-token; no blocking REST waits
- Observability: every LLM call, tool call, and agent state transition must be traced
- Security: secrets never stored in code; all endpoints authenticated
- Reliability: agent failures must be recovered gracefully, not silently dropped

**Scale:** 30+ REST endpoints, 6 AI surfaces, 8+ person engineering team, enterprise SLA

---

### Block 2: Data Ingestion and Document Processing

**Document pipeline (for RAG surface):**
```
Document Upload (PDF, DOCX, TXT)
    → Chunking with configurable overlap (e.g., 512 tokens, 64 token overlap)
    → Embedding generation (text-embedding-3-small)
    → Dual indexing:
        - Dense index: ChromaDB / pgvector (for semantic similarity search)
        - Sparse index: BM25 (for exact keyword and token matching)
    → tenant_id tagged on every chunk at write time
```

**Why dual indexing matters:**
- Dense search finds semantically similar content ("revenue" ≈ "income")
- BM25 finds exact matches (invoice numbers, product codes, proper nouns)
- Neither alone is sufficient; both together cover the full retrieval space

**Structured data ingestion (for Text-to-SQL surface):**
- Database schema introspection at session start
- Schema metadata stored per tenant for the LLM to use in query generation
- Query results validated before returning to user

---

### Block 3: Agent Orchestration Layer (LangGraph StateGraph)

**Why LangGraph StateGraph over a linear chain:**
- The platform serves 6 different AI surfaces — a linear chain cannot handle conditional routing
- StateGraph defines explicit nodes (states) and conditional edges (routing decisions)
- Each node is independently testable
- Built-in checkpointing enables pause/resume and debugging of any execution

**Routing Logic (conditional edges):**
```
User Request
    → Intent Classifier Node
    → Conditional Edge:
        - "data_query"     → Text-to-SQL Agent
        - "document_qa"    → RAG Agent
        - "complex_task"   → Multi-Agent Orchestrator
        - "conversation"   → Chat Agent with Redis memory
        - "automation"     → Async Background Agent
```

**Plan-and-Execute Framework:**
Instead of ReAct (one-step-at-a-time), the LLM emits a complete JSON execution plan upfront:
- All steps defined before any tool call is made
- Cross-step result chaining: step N can reference output of step M by variable name
- Dynamic module loading: executor loads the right tool module at runtime
- Plan can be logged and validated before execution begins

**Why plan-and-execute over ReAct for this use case:**
- ReAct is exploratory and flexible but makes one decision at a time — expensive and hard to validate
- Enterprise clients need predictable, auditable execution paths — plan-and-execute delivers this
- The plan is a first-class artifact: it can be reviewed, replayed, and compared across runs

---

### Block 4: Retrieval — Hybrid RAG with RRF Fusion

**Full Hybrid Retrieval Pipeline:**
```
User Query
    → Embed query (text-embedding-3-small)
    → Parallel search:
        - Dense: cosine similarity against pgvector/ChromaDB (top-K results)
        - Sparse: BM25 token matching against keyword index (top-K results)
    → Reciprocal Rank Fusion (RRF):
        - Score(doc) = Σ 1/(k + rank_i) for each ranking list
        - Documents appearing high in both lists score highest
    → Reranking (Cross-Encoder) for final top-5 selection
    → Inject into LLM prompt with source citations
    → Generate answer grounded in retrieved content
```

**Tenant isolation in retrieval:**
- Every vector chunk is tagged with `tenant_id` at write time
- Every retrieval query includes a mandatory `where tenant_id = X` filter
- This filter is enforced at the retrieval layer, not application code

---

### Block 5: Real-Time Serving — WebSocket Streaming and Redis Memory

**WebSocket Streaming Architecture:**
```
Client (browser / app)
    → WebSocket connection to FastAPI endpoint
    → Server begins LLM generation
    → Each token chunk → pushed immediately over socket
    → Async keepalive ping every 15s (prevents timeout during slow generation)
    → Final chunk signals completion
    → Socket remains open for follow-up turns
```

**Redis-Backed Conversation Memory:**
```
Turn 1: user message + assistant response → stored in Redis under session_id key
Turn 2: read prior history from Redis → prepend to new prompt → LLM generates
Turn N: Redis TTL expires automatically → memory cleaned up
```

**Why Redis for memory, not in-process:**
- Pod restarts (Kubernetes rolling deploys, crashes) lose in-process state
- Multiple API replicas can serve the same session without context loss
- TTL ensures automatic cleanup without a separate garbage collection job

**LLM-Powered Error Recovery Layer:**
```
Agent action fails
    → Classify failure: transient vs semantic
    → Transient: exponential backoff retry (3 attempts)
    → Semantic: LLM intent reformulation
        - Original intent + failure details → LLM rewrites query/params
        - Reformulated attempt retried
        - All state persisted to Redis (survives pod restart)
    → If N reformulations fail → escalate to human review
```

---

### Block 6: Security and Multi-Tenancy

**Authentication Stack:**
- JWT validation on every API request (stateless, scalable)
- OAuth2 flows for SSO integration (enterprise identity providers)
- Vault (HashiCorp) manages all secrets at runtime injection

**Why Vault over environment variables:**
- Environment variables are often logged, leaked in crash reports, or visible in container orchestration UIs
- Vault injects secrets at runtime with audit logging and rotation
- Secrets can be rotated without redeploying the application

**Row-Level Security for Tenant Data Isolation:**
```sql
-- PostgreSQL RLS policy (simplified)
CREATE POLICY tenant_isolation ON documents
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::uuid);

-- Application sets tenant context at request start
SET app.current_tenant = 'tenant_A_uuid';
-- Now ALL queries on documents automatically filtered to tenant_A's rows
```

**Why RLS over application-layer filtering:**
- Application filtering relies on developers always adding the WHERE clause — one mistake exposes all data
- RLS is enforced by the database engine — it cannot be bypassed by application bugs
- Audit-friendly: RLS policies are inspectable and versioned separately from application code
- Tenant onboarding: just insert a new tenant record — no new database instance needed

---

### Block 7: Observability — Langfuse Integration

**What is traced per LLM call:**
- Exact prompt (system + user messages)
- Model version and temperature
- Token count (prompt + completion)
- Cost in dollars
- Latency (time to first token, total generation time)
- All grouped under a session trace ID

**Automated Quality Scoring (after every generation):**

| Dimension | What It Checks |
|---|---|
| Completeness | Did the response address all parts of the question? |
| Helpfulness | Is the response actionable and accurate? |
| Trajectory | Did the agent use the optimal tool sequence? |
| Faithfulness | Are factual claims supported by retrieved evidence? |

**Regression Detection Workflow:**
1. Baseline: measure average quality scores over 7 days
2. Deploy prompt change or model update
3. Run automated evaluation on fixed test set
4. Compare new scores against baseline
5. If degradation > threshold: block deployment or alert team

**Human Annotation Loop:**
- Investigators and end users can rate agent outputs directly in Langfuse UI
- Human feedback is logged and can trigger retraining or prompt refinement

**Interview One-Liner:**
> "I integrated Langfuse across every agent execution path. Every LLM call is traced with tokens, cost, latency, and model version. Automated scoring checks completeness, helpfulness, trajectory, and faithfulness on every generation. I detect quality regressions by comparing scores before and after every deployment against a fixed test set."

---

### System Design Interview: "Design a Multi-Tenant LLM Platform"

**Step 1 — Scope it:**
> "Before I design, let me ask: how many tenants? How strict is the isolation requirement? Do we need real-time streaming or batch responses? What are the AI surfaces (Q&A, SQL, automation)?"

**Step 2 — The 7-Block Walk:**

1. **Problem:** Multi-tenant enterprise LLM platform, strict data isolation, real-time streaming, 6 AI surfaces
2. **Ingestion:** Document upload pipeline with chunking, dual indexing (dense + sparse), tenant_id on every chunk
3. **Agent Orchestration:** LangGraph StateGraph with intent-based conditional routing, plan-and-execute framework
4. **Retrieval:** Hybrid RAG (pgvector + BM25) with RRF fusion, tenant-filtered at retrieval layer
5. **Serving:** WebSocket streaming, Redis-backed memory, LLM error recovery with intent reformulation
6. **Security:** JWT/OAuth2, Vault secrets, PostgreSQL RLS for tenant isolation
7. **Observability:** Langfuse traces every LLM call; automated quality scoring; regression detection on deployment

**Step 3 — Tradeoffs to mention:**
- Plan-and-execute vs ReAct: predictability and auditability vs flexibility
- RLS vs separate databases: operational simplicity vs absolute isolation
- WebSocket vs REST: streaming UX vs infrastructure simplicity
- Redis memory vs in-process: resilience vs latency overhead

---

### DESIGN X — Source Text Alignment (prod.txt)

Keep the exact resume language in mind when whiteboarding this design:

> "I lead the AI engineering work on Axtria's enterprise GenAI platform. My focus: turning LLMs into reliable, observable, multi-tenant production systems."

| prod.txt Bullet | Maps to DESIGN X Block |
|---|---|
| "Architected a multi-agent AI orchestration platform on LangGraph StateGraph with conditional routing... serving 6 production AI surfaces... through a unified FastAPI backend with 30+ REST endpoints." | Block 1 (scope), Block 3 (orchestration) |
| "Designed a plan-and-execute agent framework where LLMs emit structured JSON execution plans with sequential steps, cross-step result chaining, and dynamic module loading." | Block 3 (plan-and-execute) |
| "Engineered a hybrid RAG pipeline combining dense vector search (ChromaDB / pgvector) with sparse BM25 retrieval — end-to-end document processing from upload through chunking, embedding, and LLM-augmented generation." | Block 2 (ingestion), Block 4 (retrieval) |
| "Built real-time AI streaming over WebSocket with chunked LLM responses, async keepalive pings, and Redis-backed chat memory for stateful multi-turn conversations." | Block 5 (serving, memory) |
| "Implemented an LLM-powered error recovery layer with Redis-persisted state and intent reformulation — agents recover gracefully via automatic query rewriting when execution fails." | Block 5 (error recovery) |
| "Integrated Langfuse across every agent execution path: generation-level tracing, token usage tracking, automated quality scoring (completeness, helpfulness, trajectory)." | Block 7 (observability) |
| "Secured the platform with JWT, OAuth2, Vault-managed secrets, and Row-Level Security for tenant data isolation." | Block 6 (security) |
| "Lead a cross-functional team of 8+ engineers and product folk; mentored 5+ engineers on agentic AI patterns, evaluation, and production LLM hygiene." | Block 1 (team / scale) |

**30-Second Pitch for DESIGN X:**
> "I lead the AI engineering work on Axtria's enterprise GenAI platform, turning LLMs into reliable, observable, multi-tenant production systems. The platform serves 6 AI surfaces through a FastAPI backend with 30+ endpoints, uses LangGraph StateGraph for conditional agent routing, hybrid RAG for retrieval, WebSocket streaming with Redis memory, Langfuse observability, and PostgreSQL RLS for tenant isolation."
