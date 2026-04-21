# ⚡ Production Architecture Patterns — Cross-Cutting
### DDS Round: Patterns That Show Up Across All System Design Questions

> Master these patterns and you can answer any system design question by assembling them.

---

## PATTERN 1: THE DUAL-SPEED SERVING ARCHITECTURE

```
Real-World Problem: Fraud detection needs <100ms for transaction blocking
but thorough investigation can take hours.

Solution: Two parallel paths

┌─────────────────────────────────────────────────────────────┐
│              DUAL-SPEED FRAUD ARCHITECTURE                  │
│                                                             │
│   FAST PATH (<100ms)                                        │
│   ├── Pre-computed features (Redis online feature store)    │
│   ├── Lightweight GBM/XGBoost (< 20ms inference)           │
│   ├── Simple rules (IP blacklist, device fingerprint)       │
│   └── Decision: ALLOW / BLOCK / STEP-UP AUTH               │
│                                                             │
│   SLOW PATH (Hours — runs async after transaction)          │
│   ├── Full RAG pipeline (retrieve fraud patterns)           │
│   ├── Graph Neural Network (seller/buyer network)           │
│   ├── LLM investigation report generation                   │
│   └── Decision: Queue for investigator review               │
│                                                             │
│   SHARED STORE                                              │
│   ├── Kafka: event bus connecting both paths                │
│   ├── Redis: shared real-time feature store                 │
│   └── BigQuery: shared offline store + audit log           │
└─────────────────────────────────────────────────────────────┘
```

**When to Use:** Any time you need both real-time decisions AND deep analysis.

**At Flipkart Scale:**
- Fast path: 10M daily transactions × <100ms = feasible with pre-computed features
- Slow path: Priority queue based on fast-path risk score — highest risk → investigated first

---

## PATTERN 2: ONLINE + OFFLINE FEATURE STORE

```
The Core Problem: Training features (batch) ≠ Serving features (real-time)
→ Training-serving skew → Model degrades instantly in production

Solution Architecture:

                    Source Data
                         │
         ┌───────────────┼──────────────────┐
         ▼               ▼                  ▼
  Batch Pipeline    Stream Pipeline    On-demand
  (Spark/Airflow)   (Kafka/Flink)      (direct query)
         │               │
         ▼               ▼
  Offline Store     Online Store
  (BigQuery/Hive)   (Redis/DynamoDB)
  - Historical      - Real-time
  - Training data   - Serving features
  - Feature eng     - Low latency (<10ms)
         │               │
         └───────────────┘
                 │ SAME FEATURES (point-in-time correct)
                 ▼
           Model Training / Serving

Key Design Principle: SAME CODE computes features for both paths
→ Ensures no training-serving skew
```

**Feature Types by Store:**

| Feature | Computation | Store | Latency |
|---|---|---|---|
| User lifetime stats | Daily batch Spark | BigQuery → Redis for serving | 24h staleness OK |
| Rolling 1-hour velocity | Stream (Flink) | Redis (TTL=2h) | <1 min freshness |
| Current session behavior | On-request | Direct compute at inference | Real-time |
| Historical fraud flags | Weekly batch | BigQuery (offline) | Training only |

---

## PATTERN 3: SHADOW → CANARY → CHAMPION-CHALLENGER

```
PROBLEM: How do you deploy a new model safely at scale?
Never go straight from notebook to 100% traffic!

PHASE 1: SHADOW MODE (2-4 weeks, 0% influence)
├── New model runs in parallel with existing system
├── New model's outputs are LOGGED but NOT ACTED ON
├── Compare new model outputs vs. existing system vs. ground truth
├── Discover failure modes: data format issues, edge cases, latency
└── SUCCESS: New model agrees with ground truth ≥ existing system

PHASE 2: CANARY (2-4 weeks, 5-10% traffic)
├── Route small % of traffic exclusively to new model
├── Monitor: business KPIs, latency, error rates
├── A/B test statistical comparison vs. existing system
├── Rollback trigger: any KPI degrades > threshold
└── SUCCESS: New model meets or beats existing on primary KPIs

PHASE 3: CHAMPION-CHALLENGER STEADY STATE
├── Champion (best model): 85-90% of traffic
├── Challenger(s): 10-15% of traffic (new model candidates)
├── Continuous: weekly performance comparison
├── Auto-promote challenger if it wins on primary KPI for 2 weeks
└── This pattern: enables continuous model improvement safely
```

**Rollback Triggers (define these BEFORE deployment):**
- False positive rate on confirmed good users > X%
- Latency p99 > Y ms
- Error rate > Z%
- Primary metric (AUC, recall) drops > 5% vs. baseline

---

## PATTERN 4: FEATURE ENGINEERING FOR FRAUD (The Velocity Pattern)

```python
# The three pillars of fraud feature engineering:

# PILLAR 1: VELOCITY FEATURES (most predictive)
# "How much is happening in a short window?"
velocity_features = {
    'txn_count_1h': 'COUNT(*) in last hour',
    'txn_count_24h': 'COUNT(*) in last 24 hours',
    'amount_sum_1h': 'SUM(amount) in last hour',
    'distinct_merchants_24h': 'COUNT(DISTINCT merchant) in last 24h',
    'amount_velocity_ratio': 'amount_sum_1h / (user_avg_hour_spend + ε)',
}

# PILLAR 2: BEHAVIORAL ANOMALY (z-score from user baseline)
anomaly_features = {
    'amount_zscore': '(amount - user_avg_amount) / user_std_amount',
    'hour_deviation': 'abs(hour - user_avg_transaction_hour)',
    'merchant_novelty': '1 if first time at this merchant else 0',
    'device_change_flag': '1 if device_id != user_primary_device',
}

# PILLAR 3: NETWORK FEATURES (graph-based)
network_features = {
    'shared_device_count': 'How many users share this device fingerprint?',
    'shared_address_count': 'How many seller accounts at same address?',
    'fraud_neighbor_ratio': 'What % of connected accounts are flagged?',
    'ring_size': 'Size of connected component this user belongs to',
}
```

---

## PATTERN 5: EVALUATION LADDER (From Notebook to Business Impact)

```
Every model needs to pass 4 evaluation checkpoints:

CHECKPOINT 1: OFFLINE TECHNICAL
├── Primary metric (PR-AUC for imbalanced, AUC-ROC, RMSE)
├── CV methodology (temporal split for time-series, stratified for classification)
├── Calibration (Brier score, reliability diagram)
└── Subgroup fairness (AUC by demographic segments)

CHECKPOINT 2: OFFLINE BUSINESS PROXY
├── Decile lift chart (does top 10% have 5x+ actual positive rate?)
├── Threshold analysis (precision/recall at operating threshold)
├── Business constraint satisfied ("flags ≤ 500/day at recall ≥ 75%")
└── Cost-benefit analysis (TP value vs. FP cost × rates)

CHECKPOINT 3: ONLINE VALIDATION
├── Shadow mode comparison (new vs. existing on same data)
├── A/B test (primary metric + guardrails)
├── Statistical significance (powered experiment)
└── Novelty effect check (not just catching easy ones new model hasn't seen)

CHECKPOINT 4: BUSINESS KPI TRACKING (post-deployment)
├── Revenue impact / cost savings
├── Operational efficiency (investigator workload, case resolution time)
├── Customer impact (false positive rate on good customers)
└── Longitudinal: is the model improving or degrading over time?
```

---

## PATTERN 6: HANDLING CLASS IMBALANCE (The Complete Playbook)

```
SITUATION: Fraud rate = 0.1% (1 in 1000 transactions)

STRATEGY 1: Algorithm-level (preferred for tree models)
├── XGBoost: scale_pos_weight = neg_count/pos_count = 999
├── LightGBM: is_unbalance=True or class_weight parameter
└── Effect: Up-weights minority class in loss function

STRATEGY 2: Data-level (if algo-level insufficient)
├── SMOTE: Synthetic minority oversampling
│   x_new = xi + λ(x_neighbor - xi), λ~U(0,1)
│   Risk: creates noisy synthetic samples if feature space is complex
├── ADASYN: Adaptive SMOTE (more samples near decision boundary)
└── Random undersampling: Only if massive dataset (lose info otherwise)

STRATEGY 3: Threshold calibration (ALWAYS do this)
├── Default 0.5 threshold is NEVER right for imbalanced data
├── Plot PR curve; find threshold satisfying business constraint
├── Example: "flag top 1% of transactions" → use 99th percentile score
└── Combine with Platt scaling for well-calibrated probabilities

STRATEGY 4: Evaluation (critical — accuracy is useless here)
├── NEVER use accuracy (99.9% by predicting all non-fraud)
├── Primary: PR-AUC (Precision-Recall Area Under Curve)
├── Secondary: Recall @ fixed precision threshold (business-defined)
├── Operational: KS statistic (separation between fraud/non-fraud CDFs)
└── Business: Precision @ top-k% flagged (matches investigation capacity)
```

---

## PATTERN 7: GRAPH-BASED FRAUD DETECTION

```
WHY GRAPHS: Fraudsters operate in networks, not isolation
- Multiple accounts sharing a device fingerprint
- Multiple sellers at the same address
- Accounts that transfer between each other

GRAPH REPRESENTATION:
Nodes: Users, Sellers, Devices, Addresses, Bank Accounts
Edges: 
  - User → Device (used_device relationship)
  - User → Address (registered_at relationship)
  - User → BankAccount (payment_method relationship)
  - User ↔ User (same_order, P2P transfer)

ALGORITHMS:
1. Connected Components → Find fraud rings (linked account clusters)
   - Simple: Union-Find (O(α(n)) ≈ O(1) per operation)
   - At Flipkart: seller accounts sharing bank + device = ring candidate

2. PageRank / Risk Propagation → Spread fraud score through network
   - Fraudulent account → connected accounts receive elevated risk
   - Iterative: score_v = α × own_score + (1-α) × mean(neighbor_scores)

3. Graph Neural Networks (GNN) → Learn embeddings from graph structure
   - GraphSAGE: aggregates neighborhood features into node embedding
   - Use case: detect structurally suspicious patterns even in new accounts
   
IMPLEMENTATION NOTE:
- For <1M nodes: NetworkX + Spark GraphX
- For 100M+ nodes: Custom distributed graph processing (Pregel model)
- At Flipkart scale: Domain-specific graph engine
```

---

*See companion files: 06_Feature_Engineering_Deep_Dive.md, 07_Model_Selection_Defense.md, 08_Evaluation_Deep_Dive.md*
