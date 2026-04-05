# eBay ML System Design — Interview Cheat Sheet

## 🏗️ The Universal Framework (Use for EVERY question)

```
┌──────────────────────────────────────────────────────────────┐
│  1. CLARIFY  →  2. DATA  →  3. MODEL  →  4. SERVE  →  5. EVAL │
│     │              │            │            │            │      │
│  Requirements   Features    Selection    Architecture  Metrics   │
│  Constraints    Pipeline    Training     Scaling      A/B Test  │
│  Metrics        Storage     Offline Eval Latency      Monitoring│
└──────────────────────────────────────────────────────────────┘
```

---

## Step 1: CLARIFY Requirements (2-3 min)

Always ask:
- **Latency budget?** (real-time: <200ms, near-real-time: <5s, batch: hours)
- **QPS / Scale?** (how many requests per second)
- **Primary metric?** (CTR, conversion, GMV, fraud catch rate)
- **Guardrail metrics?** (latency p99, seller fairness, user satisfaction)
- **Cold start?** (new users, new items — how many per day?)
- **Online/Offline?** (batch predictions vs. real-time inference)

---

## Step 2: DATA & Features (5-7 min)

### Data Sources at eBay
| Source | Examples |
|--------|----------|
| **User signals** | Click history, search queries, purchase history, dwell time |
| **Item metadata** | Title, description, category, price, images, seller info |
| **Contextual** | Time of day, device, location, session depth |
| **Aggregated** | User avg spend, item conversion rate, seller rating |
| **Real-time** | Current session clicks, items in cart, pages viewed |

### Feature Engineering Patterns
```
Static features   → computed offline, stored in Feature Store
Dynamic features   → computed in real-time at serving time
Cross-features     → user × item interactions (e.g., user_category_affinity)
Embedding features → learned representations (BERT for text, ResNet for images)
```

### Feature Store Architecture
```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│ Offline Jobs │────►│ Feature Store │────►│ Online Serving │
│ (Spark/Flink)│     │  (Redis/NuKV) │     │   (Model API)  │
└─────────────┘     └──────────────┘     └────────────────┘
```

**Why Feature Store?** Prevents **train-serve skew** (features computed differently in training vs. serving).

---

## Step 3: MODEL Selection (5-7 min)

### Decision Matrix

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Ranking/CTR** | GBDT (XGBoost/LightGBM) → DNN | Handles tabular data well, fast inference |
| **Candidate Retrieval** | Two-Tower (DNN) → ANN | Scalable to millions of items |
| **NLP (query/title)** | BERT / DistilBERT | Semantic understanding |
| **Fraud Detection** | GNN + GBDT ensemble | Captures graph relationships |
| **Recommendations** | Matrix Factorization + DNN | Combines collaborative + content-based |
| **Image Classification** | ResNet / EfficientNet | Pre-trained, fine-tunable |
| **Text Generation** | LLM (fine-tuned) | Listing descriptions, Q&A |
| **Anomaly Detection** | Isolation Forest / Autoencoder | Unsupervised, handles novelty |

### Multi-Stage Ranking Pipeline (The Gold Standard for Search/Recs)

```
                    Millions of items
                          │
                          ▼
              ┌───────────────────────┐
              │  CANDIDATE RETRIEVAL  │  ← ANN (Approximate Nearest Neighbor)
              │  (~1000 candidates)   │    Two-tower model, embedding lookup
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │      SCORING          │  ← GBDT or DNN
              │  (rank by P(click)    │    Full feature set (user+item+context)
              │   or P(conversion))   │    Latency budget: ~50ms
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │     RE-RANKING        │  ← Business rules + diversity
              │  (~20 final results)  │    Boost: promoted listings, freshness
              └───────────────────────┘    Demote: duplicates, low-quality
```

---

## Step 4: SERVING Architecture (5-7 min)

### Real-Time Serving Stack
```
User Request
     │
     ▼
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│ API      │───►│ Feature      │───►│ Model Server │───►│ Response │
│ Gateway  │    │ Assembly     │    │ (Triton/TF   │    │ Builder  │
│ (K8s)    │    │ (Feature     │    │  Serving)    │    │          │
│          │    │  Store+RT)   │    │              │    │          │
└──────────┘    └──────────────┘    └──────────────┘    └──────────┘
```

### Key Decisions
| Decision | Options | Trade-off |
|----------|---------|-----------|
| **Batch vs Real-time** | Batch: higher throughput, stale. RT: fresh, expensive | Use batch for slow-changing features, RT for session signals |
| **Model size** | Large: accurate, slow. Distilled: fast, slight accuracy loss | Distill for serving, use large model offline for labels |
| **Caching** | Cache popular queries/items | Cache invalidation complexity |
| **GPU vs CPU** | GPU: DNN inference. CPU: GBDT, simple models | Cost vs. latency |

### Scaling Patterns
- **Horizontal scaling**: More replicas behind load balancer
- **Model sharding**: Split large models across machines
- **Batched inference**: Group requests for GPU efficiency
- **Quantization**: INT8 instead of FP32 (2-4x speedup, <1% accuracy loss)
- **Feature pre-computation**: Pre-join features, store in Redis/NuKV

---

## Step 5: EVALUATION & Monitoring (3-5 min)

### Offline Metrics
| Metric | Use Case | Formula |
|--------|----------|---------|
| **AUC-ROC** | Binary classification (fraud) | Area under ROC curve |
| **AUC-PR** | Imbalanced classification | Area under Precision-Recall curve |
| **NDCG@K** | Ranking quality | Normalized Discounted Cumulative Gain |
| **MAP@K** | Ranking (multiple queries) | Mean Average Precision |
| **MRR** | Ranking (first relevant result) | Mean Reciprocal Rank |
| **Recall@K** | Candidate retrieval | Relevant items in top K / total relevant |

### Online Metrics
| Metric | What it Measures |
|--------|-----------------|
| **CTR** | Click-through rate (engagement) |
| **Conversion Rate** | Purchase / impression ratio |
| **GMV** | Total marketplace revenue |
| **Add-to-Cart Rate** | Consideration intent |
| **Time-to-Purchase** | Decision speed |
| **Bounce Rate** | User dissatisfaction |

### A/B Testing Checklist
- [ ] Define hypothesis clearly
- [ ] Choose unit of randomization (user-level, not session-level for marketplace)
- [ ] Set MDE (Minimum Detectable Effect), α=0.05, power=0.8
- [ ] Calculate sample size needed
- [ ] Run for minimum 1-2 weeks (capture weekly patterns)
- [ ] Check for SRM (Sample Ratio Mismatch)
- [ ] Check guardrail metrics
- [ ] Account for network effects (marketplace SUTVA violations)

### Monitoring Dashboard
```
┌──────────────────────────────────────────────────┐
│                  MODEL HEALTH                     │
│                                                  │
│  📊 Prediction distribution drift (PSI / KL-div) │
│  📈 Feature distribution drift (per feature)     │
│  ⚡ Latency p50/p95/p99                          │
│  ❌ Error rate                                   │
│  📉 Online metric degradation (CTR, conversion)  │
│  🔄 Data freshness (staleness of feature store)  │
│  🚨 Alert: auto-rollback if metric < threshold   │
└──────────────────────────────────────────────────┘
```

---

## 📋 Quick Reference: 3 Most-Likely Design Questions

### 1. Design eBay Search Ranking

```
Requirements: <200ms latency, 10K QPS, optimize GMV & relevance
Data: query, user history, item metadata, click/purchase logs
Model: Two-tower retrieval → LightGBM ranker → business re-rank
Serving: Feature store (NuKV), model on Triton, result cache
Eval: Offline NDCG@10, Online CTR + conversion + GMV via A/B test
Monitor: Query-level latency, zero-result rate, ranking drift
```

### 2. Design eBay Fraud Detection

```
Requirements: <100ms, recall > 95%, false positive < 5%
Data: transaction, device fingerprint, user history, velocity features
Model: Rule layer (instant block) → GNN on transaction graph → GBDT ensemble
Serving: Streaming pipeline (Kafka), real-time feature lookup
Eval: AUC-PR (imbalanced!), recall@precision=95%, manual review rate
Monitor: Fraud rate trend, false positive rate, model staleness
```

### 3. Design eBay Recommendation System

```
Requirements: <300ms, personalized, handle cold-start
Data: purchase history, view/click logs, item metadata, user demographics
Model: Collaborative filtering (ALS) + Content-based (embeddings) → hybrid
Cold-start: Popular items → Content-based → Transition to CF as data accumulates
Serving: Pre-computed recs (batch) + real-time session signals
Eval: Recall@20, catalog coverage, diversity, Online: CTR + conversion
Monitor: Coverage drift, popularity bias, cold-start fallback rate
```

---

## 🧠 Power Phrases for System Design Interviews

Use these to sound structured and experienced:

| Situation | Say This |
|-----------|----------|
| Starting | *"Let me first clarify the requirements and constraints before diving into the design."* |
| Feature engineering | *"I'd use a feature store to ensure consistency between training and serving."* |
| Model choice | *"I'd start with a simpler model like LightGBM as a baseline, then iterate toward a DNN if the data and latency budget support it."* |
| Scaling | *"For this QPS, I'd use horizontal scaling with pre-computed candidates and a real-time scoring layer."* |
| Trade-offs | *"There's a tension between latency and model complexity here — I'd quantize the model and use distillation to maintain quality within the latency budget."* |
| Evaluation | *"Offline metrics like NDCG give us a signal, but the real validation comes from a well-designed A/B test with appropriate guardrail metrics."* |
| Monitoring | *"I'd set up automated drift detection and alert on prediction distribution shifts using PSI or KL-divergence."* |
