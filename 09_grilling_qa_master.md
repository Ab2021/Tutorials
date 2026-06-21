# 🔥 GRILLING MASTER — System Design, Deployment & Marketing Q&A
### Abhishek Bhardwaj | Solutions Architect ML/AI @ Huge
### Format: Question → Model Answer → Cross-Questions → Traps → Verdict Criteria

---

> [!IMPORTANT]
> This is the **active practice file**. Cover the model answer first, attempt your own,
> then compare. Questions marked 🔴 are near-certain. 🟡 = likely. 🟢 = differentiator.

---

## 🏗️ PART 1: SYSTEM DESIGN — LIVE WHITEBOARD SCENARIOS

---

### 🔴 SD-1: "Design a real-time personalization engine for McDonald's mobile app"
**Level**: Principal Architect | **Time**: 35-45 min

#### CLARIFYING QUESTIONS TO ASK FIRST (Do NOT skip this)
1. "What is the primary goal — increase order value, increase frequency, or reduce churn?"
2. "What data do we have — loyalty purchase history, location, weather API access?"
3. "Is this real-time (app open) or batch (push notification at 7am)?"
4. "What's the latency budget? Sub-100ms, sub-500ms?"
5. "What % of users are new with no history? How do we handle cold-start?"
6. "What's the peak traffic event? Super Bowl Sunday, breakfast rush?"
7. "Multi-region? EU/APAC users need local latency?"
8. "A/B testing required? Who manages experiment config?"

#### ASSUMPTIONS
- 69M daily active users, 50K RPS at peak breakfast rush (7-9am EST)
- Recommendation must appear within 200ms of app open
- Have: 18 months purchase history, GPS location, time-of-day, weather, local promotions
- 15% new users (cold-start pool)
- A/B testing: max 3 concurrent experiments
- GCP-native stack, 99.9% availability SLA

#### ARCHITECTURE (ASCII)

```
USER OPENS APP (mobile)
        |
        v
[Cloud Load Balancer]  ← Global anycast, latency-based routing
        |
        v
[API Gateway / Cloud Endpoints]  ← JWT auth, rate limiting, request logging
        |
        v
[Personalization Service — Cloud Run, min-instances=3]
        |
   +----+-------------------------------+
   |                                    |
   v                                    v
[Feature Lookup]                 [Candidate Generation]
[Vertex AI Feature Store]        [Vertex AI Vector Search]
  - customer_embedding              ANN: top-50 item candidates
  - visit_frequency                 filtered by: location,
  - preferred_daypart               inventory, daypart,
  - avg_order_value                 active promotions
   |                                    |
   +----+-------------------------------+
        |
        v
[Re-Ranking — Cloud Run (LightGBM)]
  Scores 50 candidates → top 3
  Features: {affinity, context_match, margin_weight, promo_flag}
        |
        v
[A/B Experiment Layer]  ← 5% to experimental strategies
        |
        v
USER SEES RECOMMENDATION (3 items)

BACKGROUND PIPELINES:
[App Events] → [Pub/Sub] → [Cloud Dataflow streaming]
                                    |
                              [Vertex AI Feature Store]
                              - Online: Bigtable-backed (<10ms)
                              - Offline: BigQuery (training)
```

#### SCALE MATH
```
Peak: 50,000 RPS
Each request → 3 downstream calls: Feature Store + Vector Search + Reranker
Total downstream QPS: 150,000/second

Vertex AI Vector Search: handles 1M+ QPS (managed infra)
Vertex AI Feature Store: Bigtable-backed, 10M+ reads/second
Cloud Run: 500 instances × 100 RPS/instance = 50K RPS capacity

Storage:
  - 69M users × 256-dim × 4 bytes = ~70GB embeddings
  - Feature Store online: ~500 bytes/user × 69M = ~34GB

Latency budget:
  Load Balancer: 5ms
  API Gateway: 10ms
  Feature Store: 10ms
  Vector Search: 20ms
  Re-ranker: 50ms
  Network: 15ms
  TOTAL: ~110ms  ← within 200ms budget ✅
```

#### FAILURE MODES & MITIGATIONS

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Feature Store outage | No personalisation | Fallback: cached features from Redis (30min TTL) |
| Vector Search timeout | No candidates | Fallback: pre-computed top items by location from BigQuery |
| Re-ranker crash | No re-ranking | Fallback: serve ANN score directly (Vector Search output) |
| Dataflow lag >15min | Stale features | Alert + serve cached; do NOT crash the recommendation |
| LLM blurb timeout | No natural language | Skip blurb, serve item name + price only |

#### COST ESTIMATION
```
Cloud Run (optimised): $40-80K/month
Vertex AI Feature Store: $104K/month (40M active users)
Vertex AI Vector Search: $130K/month (with 90% caching)
BigQuery: $5K/month
Total realistic: $280-320K/month (full production)
Pilot (1M users): ~$12K/month
```

#### CROSS-QUESTIONS THE INTERVIEWER WILL ASK

**Q: "You mentioned Gemini for natural language blurbs. Gemini alone is 500ms. How?"**
Answer: The blurb is async/optional. Core recommendation (3 items, photos, prices) renders in
<200ms. The blurb loads asynchronously 500ms later — user sees it appear after items.
Better: pre-generate blurbs for top 1,000 item combinations (cache hit rate >80%).

**Q: "How does A/B testing work without biasing the recommendation model?"**
Answer: Users bucketed by hash(user_id) mod N into experiment arms at API Gateway level —
BEFORE any ML inference. The model sees the same user. We log arm assignment and use it
as a stratum in evaluation. This prevents the model from learning arm as a feature.

**Q: "McDonald's is on AWS, not GCP. What changes?"**
Answer: Architecture is cloud-agnostic. Replace:
  Vertex AI Feature Store → DynamoDB + SageMaker Feature Store
  Vector Search → OpenSearch k-NN or Pinecone
  Dataflow → AWS Glue Streaming
  Cloud Run → AWS Fargate
Same pattern, different services. I'd advocate for GCP given Huge's stack but adapt.

---

### 🔴 SD-2: "Design the Huge India ML Platform — 10 Fortune 500 clients, shared infra"
**Level**: Principal | **This is THE question for this specific role**

#### CLARIFYING QUESTIONS
1. "Security model — do clients know infra is shared, or do they expect dedicated?"
2. "Data classification — all Tier 1 (highest security) or mixed?"
3. "Is per-client cost attribution needed for Huge's billing?"
4. "Target onboarding time for a new client?"
5. "What's the expected query volume per client? All similar?"

#### ARCHITECTURE (ASCII)

```
HUGE-ORG (GCP Organization)
│
├── huge-shared-services (Project)  ← DevOps team owns this
│   ├── Artifact Registry (shared base Docker images)
│   ├── Cloud Build (CI/CD for all clients)
│   ├── Cloud Monitoring + Looker (shared dashboards)
│   ├── Security: Cloud Armor, Identity-Aware Proxy
│   └── Agent Gateway (semantic router → client endpoints)
│
├── huge-google-proj ──── VPC SC Perimeter ────────────────────┐
│   ├── BigQuery: google_campaign_data, google_attribution      │
│   ├── Vertex AI: google-campaign-agent-endpoint               │
│   ├── Vector Search: google-brand-index                       │
│   └── SA: sa-google@huge-google.iam (can ONLY access this project)
│                                                               │
├── huge-mcdonalds-proj ── VPC SC Perimeter ─────────────────── same isolation
│   ├── BigQuery: mcd_loyalty_data, mcd_menu_performance
│   ├── Vertex AI: mcd-personalization-endpoint
│   └── [identical isolated structure]
│
└── huge-{client}-proj × 10 clients

USER QUERY FLOW:
[Client User] → API Gateway → [Agent Gateway, huge-shared-services]
                                    │
                              JWT: client_id=mcdonalds
                                    │
                              Semantic Router → routes to huge-mcdonalds-proj
                                    │
                              [MCD Agent endpoint] → [MCD Vector Search]
                                                   → [MCD BigQuery]
```

#### PROVING DATA ISOLATION TO A CISO

```
Level 1 — GCP Project isolation:
  Each client = separate GCP project. IAM is project-scoped.
  sa-mcdonalds@ has ZERO bindings in huge-google-proj by default.

Level 2 — VPC Service Controls:
  Each project wrapped in a VPC SC perimeter.
  Even if code has a bug and tries to call BigQuery cross-project,
  VPC SC blocks at the network level — not just IAM.
  This is cryptographic isolation, not trust-based.

Level 3 — Data residency:
  EU client BigQuery datasets have region=EU. Data never leaves EU.

Level 4 — Audit trail:
  Cloud Audit Logs: every data access event logged.
  Monthly report per client: "zero cross-client data access events."

Command to show Nike's CISO:
  gcloud access-context-manager perimeters describe huge-nike-perimeter
  → shows: restricted services, no ingress from huge-mcdonalds-proj
```

#### CLIENT ONBOARDING IN <2 WEEKS

```
WEEK 1:
Day 1-2: Terraform apply huge-{client} module
  Creates: GCP project, VPC, service accounts, IAM bindings
  Provisions: BigQuery datasets, GCS buckets, Vertex AI endpoints (empty)
  Sets up: Cloud Build trigger, Artifact Registry repo

Day 3-4: Data ingestion
  Fivetran connector to client's data sources
  Dataflow streaming job for live events
  Dataform/dbt transformations

Day 5: Baseline agent deployment
  Deploy Huge standard LangGraph+RAG template to client endpoint
  Configure semantic routes for client domain
  Load brand docs into Vector Search index

WEEK 2:
Day 8-10: Client customisation
  Fine-tune prompts for client brand voice + domain vocabulary
  Load all brand guidelines into Vector Search
  Configure client-specific tool integrations (BI APIs, CRM)

Day 11-12: Testing & security review
  Golden test set against client domain (50 Q&A pairs)
  VPC SC validation: attempt cross-project access, confirm blocked

Day 13-14: Soft launch
  10% traffic → new platform
  Monitor: latency, error rate, LLM-as-judge quality
  Full launch with client sign-off
```

---

### 🔴 SD-3: "Design an auto-updating MMM platform"
**Direct connection to your Axtria experience**

#### ARCHITECTURE (ASCII)

```
DATA INGESTION
Google Ads API ──┐
Meta Ads API ────┤
TV/OOH data ─────┤──→ [Fivetran] ──→ [BigQuery: raw_marketing]
CRM revenue ─────┤
Competitor data ─┘
                        │
                        │ [Dataform/dbt]
                        v
               [BigQuery: mart_mmm_features]
               (weekly spend by channel, pre-computed
                adstock features, saturation curves)
                        │
                        │ [BigQuery CDC → Pub/Sub trigger]
                        v
               [Vertex AI Pipelines — auto-triggered]
               ┌──────────────────────────────────┐
               │ Step 1: Data validation           │
               │         (Great Expectations)      │
               │ Step 2: MMM Training              │
               │         (PyMC / Meridian on GPU)  │
               │ Step 3: Evaluation                │
               │         (MAPE, R², decomposition) │
               │ Step 4: Champion/Challenger gate  │
               │         (must beat current model) │
               │ Step 5: Budget optimisation       │
               │         (NSGA-II genetic algo)    │
               └──────────────────────────────────┘
                        │
               [Vertex AI Model Registry]
               (champion ← promoted if gate passes)
                        │
               [Looker Dashboard]
               - Channel ROI by week
               - Budget reallocation recommendations
               - Scenario planning: "What if we shift 10% to CTV?"
               - Uncertainty bounds (Bayesian credible intervals)
```

#### HANDLING THE 2-3 WEEK MMM TRAINING TIME MYTH
```
Bayesian MMM (PyMC + NUTS MCMC): 4-8 hours for typical weekly dataset.
NOT 2-3 weeks — that's for full econometric models with panel data.

For faster iteration:
  Option A: Variational Inference (ADVI): 10x faster, slightly less accurate
  Option B: Meridian (Google, JAX-based): GPU-accelerated, 30 min standard
  Option C: Vertex AI Training with A100: reduces PyMC NUTS from 4h to 45min

Production schedule:
  - Full Bayesian retrain: weekly (Sunday 2am UTC)
  - Quick Meridian sanity check: daily (catch major anomalies fast)
  - Client recommendation update: weekly post-retrain + human review
```

#### MODEL VALIDATION GATES (before client sees results)
```
Automated gates (ALL must pass):
1. MAPE on holdout period < 15%
2. Decomposition integrity: baseline + paid media + organic = 100% of sales
3. All channel coefficients positive (spend cannot reduce sales in this model)
4. Adstock decay rates in plausible range: 0.1 < λ < 0.95
5. Budget optimizer converges to a feasible solution

Human gate (always required):
- Data scientist reviews: trace plots, R-hat < 1.01 (chain convergence)
- Sanity-checks business story: does TV ROI change vs last quarter make sense?
- Signs off in Vertex AI Experiments UI before Looker dashboard updates
```

---

## 🚀 PART 2: DEPLOYMENT DEEP DIVE Q&A

---

### 🔴 DEP-1: Multi-Region Failure Scenario

**Q: "Your Vertex AI endpoint serving McDonald's goes down in us-central1 at 7:30am EST. What happens?"**

**Model Answer**:

This is a pre-planned scenario, not a surprise. Here's the complete runbook:

```
NORMAL STATE:
US Users → Cloud LB → us-central1 Cloud Run (90%) + us-east1 Cloud Run (10% warm)

FAILURE DETECTION (T+0 to T+30sec):
  Cloud LB health check fires every 5 seconds
  At T+0: us-central1 endpoint returns 503
  At T+15sec: 3 consecutive health check failures → LB marks us-central1 unhealthy
  At T+30sec: 100% traffic shifted to us-east1 (automatic, no human action)

RECOVERY (T+30sec to T+2min):
  us-east1 has min-instances=5 pre-warmed (no cold start)
  Auto-scaling: 5 → 50 instances in ~90 seconds
  Error rate: spike from T+0 to T+30sec, then recovers

DURING FAILOVER GAP (T+0 to T+30sec):
  Cache layer (Redis/Memorystore): serves last-known recommendations for returning users
  New users: popularity-based fallback (no ML, instant from BigQuery)
  User sees: slight delay, then recommendations from fallback

ALERTS:
  PagerDuty alert: "us-central1 Vertex AI endpoint DOWN" at T+15sec
  On-call engineer investigates root cause
  Client SLA: "<2 minute failover" — we're within SLA at T+30sec

RECOVERY AFTER FIX:
  Canary 5% back to us-central1, monitor 15 min
  Restore 90/10 split after no errors

RTO: <2 minutes. RPO: 0 (stateless serving, no data loss).
```

**Cross-question: "What if BOTH regions fail simultaneously?"**
Answer: Multi-region failure is catastrophic and requires a third failover region (europe-west1) or
a pre-cached static fallback (pre-computed top 10 items by location, served from Cloud CDN with
24h TTL). The CDN fallback has no personalisation but maintains app functionality. For a client
like McDonald's with $50M+ daily digital revenue, a third active region is worth the cost.

---

### 🔴 DEP-2: SLA Design for AI Systems

**Q: "How do you define an SLA for an AI system? Uptime alone is not enough."**

**Model Answer**:

Traditional software SLAs measure availability. AI systems need three dimensions:

```
DIMENSION 1 — AVAILABILITY SLA (standard)
SLI: % of requests returning non-5xx response within 2 seconds
SLO: 99.9% (8.7 hours downtime budget/year)
Contractual SLA: 99.5% (buffer for remediation)

DIMENSION 2 — QUALITY SLA (AI-specific, rare in junior interviews)
SLI: LLM-as-judge faithfulness score, sampled daily on 2% of traffic
SLO: Faithfulness ≥ 0.82 (7-day rolling average)
Alert: Faithfulness < 0.78 for 2 consecutive days → P2 incident
  → Root cause: embedding model update? New prompt? Data change in index?

SLI: RAG context precision (% retrieved docs actually used in answer)
SLO: Context precision ≥ 0.75
Alert: < 0.65 → RAG index may contain corrupted or irrelevant documents

DIMENSION 3 — LATENCY SLA
SLI: P95 end-to-end latency
SLO: P95 < 3 seconds (RAG + LLM)
Contractual: P95 < 5 seconds

ERROR BUDGET TRACKING:
Monthly error budget (99.9% availability): 43.8 minutes
If burn rate > 2x expected rate: freeze new feature releases
  → Engineers must focus on reliability, not new features
Review in weekly SRE meeting
```

---

### 🔴 DEP-3: Cost Architecture — $50K/Month Budget

**Q: "The client's AI budget is $50K/month. What can you build?"**

**Model Answer**:

```
WHAT $50K/MONTH BUYS ON GCP:

Tier 1 — RAG Knowledge Base (core): $8K/month
  Vertex AI Vector Search: $2K (1M vectors, 10K queries/day)
  Cloud Storage (raw documents, embeddings): $500
  Embedding API calls: $500 (1M embeddings at text-embedding-004 pricing)
  Cloud Run (RAG serving): $5K (handles ~500K queries/month)

Tier 2 — LLM Inference: $14K/month (post-caching)
  Gemini 1.5 Pro via Vertex AI: ~$12K raw
  Semantic caching (30% hit rate) → saves ~$4K → net $8K
  Claude 3.5 Sonnet (for premium outputs): $6K

Tier 3 — Data & Infrastructure: $10K/month
  BigQuery storage + queries: $3K
  Cloud Build CI/CD: $500
  Cloud Monitoring + Logging: $1.5K
  Networking (egress, LB): $2K
  Pub/Sub, Cloud Functions, misc: $3K

Tier 4 — Reserve: $7K/month (spikes, experimentation)
  Vertex AI Training jobs (monthly model updates): $3K
  Testing + staging environment: $2K
  Buffer: $2K

WHAT YOU GET:
✅ Production RAG system: 500K queries/month
✅ LangGraph agent with 6-8 tools
✅ Real-time observability (Langfuse)
✅ CI/CD pipeline with prompt regression testing
✅ A/B testing: 2 concurrent experiments
✅ Monthly model evaluation + retraining
❌ NOT included: multi-region redundancy (add $15K/month)
❌ NOT included: dedicated GPU serving (add $20K+/month)
❌ NOT included: >500K queries/month without renegotiation

RECOMMENDATION: Start with $50K → demonstrate value → negotiate $150K for full
multi-region, GPU serving, and multi-agent architecture.
```

---

### 🔴 DEP-4: Multi-Tenant Security

**Q: "How do you prove to Nike's CISO that their data is isolated from McDonald's?"**

**Model Answer**:

I don't just claim it — I demonstrate it with three layers of evidence:

```
LAYER 1 — ARCHITECTURE DOCUMENTATION (show, don't tell)
  Provide: GCP project topology diagram showing Nike and McDonald's in
  separate projects with separate VPC SC perimeters.
  Share: IAM policy export showing Nike service accounts have ZERO bindings
  in McDonald's project.

LAYER 2 — LIVE DEMONSTRATION
  In Nike's presence:
  Step 1: Use Nike's service account credentials
  Step 2: Attempt to list BigQuery datasets in McDonald's project:
    gcloud bigquery datasets list --project=huge-mcdonalds-proj
    → Result: "PERMISSION_DENIED: Access Denied"
  Step 3: Attempt to read Nike's data from McDonald's service account:
    → Same result. Cross-project access is impossible.
  Step 4: Show VPC Service Controls perimeter config:
    gcloud access-context-manager perimeters describe huge-nike-perimeter
    → Shows restricted services, no ingress from other projects

LAYER 3 — ONGOING AUDIT TRAIL
  Monthly: export Cloud Audit Logs for Nike's project
  Report shows: every data access event, by whom, from where
  Contractual commitment: zero cross-client data access events in any month
  If any event occurs: automatic alert + incident report within 24h
```

---

## 📊 PART 3: MARKETING AI — DEEP GRILLING

---

### 🔴 MKT-1: MMM Mathematical Foundation

**Q: "Walk me through the mathematical likelihood function in your Bayesian MMM. What are you actually maximising?"**

**Model Answer**:

In Bayesian MMM, we compute the **posterior distribution** of all parameters given observed data — we don't maximise a likelihood, we sample from the posterior.

```
THE MODEL:
Sales(t) = α + Σᵢ βᵢ × f_sat(f_adstock(Spend_i(t))) + γ·Trend(t) + δ·Season(t) + ε(t)

WHERE:
  α         = baseline intercept (organic sales with zero marketing)
  βᵢ        = channel i coefficient (response multiplier)
  f_adstock = carryover: Adstock(t) = Spend(t) + λᵢ × Adstock(t-1)
              λᵢ = decay rate (TV: high ~0.7, digital: low ~0.3)
  f_sat     = Hill saturation: sat(x) = x^n / (x^n + K^n)
              K = half-saturation point, n = Hill coefficient
  ε(t)      ~ Normal(0, σ²) = observation noise

LIKELIHOOD:
  P(data | θ) = ∏_t Normal(Sales(t) | μ(t), σ²)

PRIORS:
  P(λᵢ_TV)      = Beta(3, 1)     ← strong prior: high carryover for TV
  P(λᵢ_digital) = Beta(1, 3)     ← strong prior: low carryover for digital
  P(βᵢ)         = HalfNormal(1)  ← must be positive (spend → sales)
  P(σ)          = HalfNormal(1)  ← positive noise scale

POSTERIOR (what MCMC samples from):
  P(θ | data) ∝ P(data | θ) × P(θ)
              ∝ [∏_t Normal(Sales(t) | μ(t), σ²)] × [priors above]

MCMC SAMPLER: NUTS (No-U-Turn Sampler) in PyMC
  4 chains × 2,000 samples = 8,000 posterior draws
  Convergence check: R-hat < 1.01 for all parameters
  Output: distribution over each βᵢ, λᵢ, K, n — not point estimates
```

**Why this matters for Huge clients**: The credible interval on each channel's ROI is what
clients actually need. "TV's ROI is 3.2x (95% CI: 2.1x–4.8x)" is a business-ready answer.
A point estimate without uncertainty hides model risk.

**Cross-question: "How did you validate the adstock decay rate?"**

Three ways:
1. **Prior sensitivity analysis**: ran with Beta(2,1) vs Beta(4,1) for TV. Posterior barely shifted
   → data was informative, not dominated by prior.
2. **Holdout validation**: withheld last 8 weeks, fit on 44 weeks. Model-implied sales matched
   holdout within credible intervals on 7/8 weeks.
3. **Industry benchmark check**: TV carryover implied 23% effect carried into week after campaign.
   Industry benchmarks: 15-30%. Media planners confirmed this matched their experience.

**Trap: "Isn't adstock just a made-up parameter you can tune to fit anything?"**
Answer: This is a legitimate critique of unconstrained MMM. The defence: (1) priors constrain
the parameter range to economically plausible values — a Beta(3,1) prior on TV carryover means
the model believes high carryover a priori and needs strong data evidence to update away from that;
(2) we validate adstock-implied behaviour against held-out data and known campaign timings
(e.g., does the model correctly attribute the 2-week post-Super-Bowl lift to TV carry?).

---

### 🔴 MKT-2: Multi-Touch Attribution Under Pressure

**Q: "Markov chains assume memoryless transitions. Real journeys aren't memoryless. Defend your choice."**

**Model Answer**:

You're right — this is a real limitation I was aware of and addressed. Order-1 Markov assumes:
P(state_n | state_n-1, state_n-2, ...) = P(state_n | state_n-1)

A user who saw TV → Social → Search has a different conversion probability than Social → Search alone,
but order-1 Markov treats them identically.

I addressed this two ways at Axtria:

**First: Higher-order Markov chains (order-2 and order-3)**
State space becomes {previous_touch, current_touch} pairs.
For order-2: state = (Social, Search) → captures that Search after Social has different conversion
probability than Search after Display.
Cost: exponentially more states, needs exponentially more journey data.
Practical limit: order-2 with min 5,000 journeys per state pair.

**Second: Attention mechanism layer (my addition)**
Modelled journey as sequence input to a Transformer encoder (BERT-style).
Self-attention weights let each touchpoint attend to all other touchpoints in the journey.
Result: early-funnel Medical Congress visits got high attention weight on conversion even
when they weren't the most recent touch — exactly what order-1 Markov missed.
This is fundamentally a higher-order model that handles variable-length sequences.

**When I would still use order-1 Markov**:
When data is scarce (<50K journeys), when interpretability is critical (a Markov transition
matrix is easy to audit), or when speed of implementation matters (week-1 deliverable).

**Q: "What is the difference between correlation and causation in attribution?"**

Answer: Correlation: users who see TV ads have higher conversion rates.
Causation: showing users TV ads CAUSED incremental conversions that wouldn't have happened.
Attribution models (including Markov) measure correlation — they attribute conversions to
touchpoints that appear in converting journeys. But high-intent users may be more likely
to convert AND more likely to engage with every ad. TV didn't cause the conversion —
the user's high intent caused both the ad engagement and the conversion.

To get causation: **incrementality experiments**.
- Geo-holdout: turn off TV in matched DMAs, compare conversion rates
- Ghost bidding: simulate winning an ad auction but don't show the ad, compare converters
- PSA testing: show control ads (Public Service Announcements) to holdout group

At Axtria, we validated the MMM's causal channel claims using geo-matched experiments
for J&J's pharma brands — comparing sales in states that ran vs. didn't run DTC TV
after controlling for demographic differences. The experiment confirmed TV drove
approximately 18% incremental sales, which matched the MMM estimate within 4%.

---

### 🔴 MKT-3: CLV Architecture

**Q: "BG/NBD vs gradient boosting for CLV — when do you use each?"**

```
BG/NBD MODEL:
✅ Use when: transaction data only, small dataset (<50K customers)
✅ Output: P(customer alive) × E(future transactions | alive)
✅ Works with 3-6 months history
✅ Probabilistic, interpretable
❌ Ignores covariates (demographics, product category)
❌ Assumes stationary behaviour (no seasonality)

GRADIENT BOOSTING (LightGBM):
✅ Use when: rich feature set, large dataset (>100K customers)
✅ Captures non-linear interactions between features
✅ Integrates external signals (economic indicators, competitor events)
❌ Needs 12+ months history for meaningful label window
❌ Black box (need SHAP for explanation)

MY HYBRID APPROACH (CVS Health):
  Step 1: BG/NBD → outputs P(active), E(visits) as FEATURES
  Step 2: GBM takes BG/NBD outputs + demographics + health plan type
  Step 3: GBM predicts 12-month CLV (spend)
  Result: 8% MAPE improvement vs either model alone

WHY THIS WORKS:
  BG/NBD captures the churn/buy process from transaction timing
  GBM captures spend heterogeneity across customer segments
  Combined: better than either individually
```

**Q: "How do you use CLV in a marketing context — what decisions does it actually drive?"**

Answer: CLV informs four key marketing decisions:
1. **Acquisition bidding**: Max CPA for acquiring a new customer = CLV × gross margin %.
   If CLV=$500, margin=40%, max CPA=$200. Without CLV, teams use ROAS which optimises
   for the first purchase only.
2. **Retention investment**: Prioritise retention spend on high-CLV customers. Don't spend $50
   on a retention offer for a customer with $60 predicted CLV.
3. **Segment definition**: \"High CLV but at-risk of churn\" is your single most valuable segment —
   personalized retention effort here has the highest ROI.
4. **Channel attribution weighting**: Weight conversions by CLV, not by count. A channel that
   acquires 100 low-CLV customers is worth less than one that acquires 40 high-CLV customers.

---

### 🔴 MKT-4: Huge Client Scenarios

**Q: "Verizon wants to reduce churn. Design the end-to-end AI system."**

```
DATA INPUTS:
  Customer tenure, plan type, payment history, support calls (count + sentiment),
  data usage trends (declining = churn signal), device age, competitor offer exposure,
  network quality scores by location, contract end date proximity

LABEL ENGINEERING:
  Label: churned (contract cancelled or ported number) in next 90 days
  Source: CRM cancellation records + number portability records
  Class imbalance: ~3% churn rate → use stratified sampling, adjust threshold

MODEL:
  LightGBM (tabular + mixed types + fast inference + handles imbalance well)
  Features: 80+ engineered features including rolling trends (30/60/90 day)
  Training: monthly retrain on 24 months history
  Validation: time-based split (train on months 1-20, validate on 21-24)

SERVING:
  Batch scoring: daily, all 10M+ customers → scores in BigQuery
  Real-time trigger: score immediately when customer calls support
    (support call is top-3 churn predictor — intervene during the call)

ACTION LAYER:
  Score < 0.3 (low risk): no intervention
  Score 0.3-0.6 (medium): proactive SMS/email with loyalty offer
  Score > 0.6 (high risk): route to retention specialist + personalised offer
  Score > 0.85 AND high CLV: executive outreach (account manager call)

LLM LAYER:
  For high-risk customers: Gemini generates personalised retention message
  Input: customer's specific risk factors (device age, usage trend, contract expiry)
  Output: personalised message explaining offer relevance to their situation

MEASUREMENT:
  Holdout 10% of high-risk customers from intervention (control group)
  Compare actual 90-day churn rate: treatment vs control
  Report: incremental churn reduction × average CLV saved = ROI
```

---

### 🔴 SA-1: Client Discovery Process

**Q: "A CMO says 'we want AI.' What are your first 5 questions?"**

**Model Answer**:

Before asking any questions, I say: "Before we talk about what AI can do, help me understand
what problem keeps you up at night. Then we'll find where AI actually solves it."

**Question 1: "What decision do you wish you could make faster or better today?"**
This surfaces the actual pain — not "AI" but "I spend 3 days every week pulling reports
to prepare for the board meeting." That's an agentic analytics assistant, not a chatbot.

**Question 2: "What data do you have, and how clean is it?"**
Most critical question. 80% of AI projects fail on data quality, not models. If they say
"our CRM data is incomplete and inconsistent," the first AI project is data quality — not ML.

**Question 3: "What does success look like in 6 months? What number moves?"**
Forces specificity. "AI transforms our business" is not a success metric.
"Reduce time-to-insight from 3 days to 30 minutes for CMO reporting" is measurable and
achievable in 6 months.

**Question 4: "Who owns the AI output? Who acts on it?"**
Identifies the user persona and the change management challenge. If the person acting on
recommendations is a 55-year-old VP who distrusts AI outputs, you need explainability and
a "human in the loop" design. If it's a junior analyst, full automation is acceptable.

**Question 5: "What's happened before when you've tried technology projects here?"**
Surfaces political and organisational context. If they say "we bought Salesforce 2 years ago
and 30% of the team uses it" — adoption is the real problem, not technology. Build that into
your design: simpler UI, change management plan, executive champion required.

**After these 5 questions → 2-week discovery sprint:**
- Data audit (what exists, quality, access controls)
- Workflow mapping (where AI creates most value in their decision-making process)
- Quick win identification (lowest risk, highest visibility)
- 6-week POC scope with clear success metrics both parties agree to upfront

---

### 🔴 SA-2: Effort Estimation

**Q: "Estimate effort for a RAG system for a mid-size enterprise knowledge base."**

```
SCOPE: 50,000 documents, 20 → 200 users, SharePoint + Confluence integration

PHASE 1 — Data & Indexing: 3 weeks (1 ML eng + 1 data eng)
  Data discovery + access permissions: 3 days
  Connector development (SharePoint/Confluence APIs): 5 days
  Document processing (OCR, chunking, embedding): 4 days
  Vector index build + validation: 3 days

PHASE 2 — RAG Core: 2 weeks (1 ML eng)
  Hybrid retrieval (dense + BM25 + RRF): 4 days
  Re-ranking integration: 2 days
  LLM integration + prompt engineering: 4 days

PHASE 3 — Evaluation: 2 weeks (1 ML eng + 1 DS)
  Golden test set with domain experts (100 Q&A pairs): 4 days
  RAGAS baseline + iterative improvement: 6 days

PHASE 4 — Production: 3 weeks (1 BE + 1 ML eng)
  FastAPI/Cloud Run deployment: 3 days
  Chat UI or Slack integration: 3 days
  SSO/auth integration: 2 days
  Monitoring (Langfuse + Cloud Monitoring): 2 days
  Pilot user testing: 5 days

TOTAL: 10 weeks / 2.5 months
TEAM: 3-4 people
COST: ~$165-215K (people + infra)
GCP infra ongoing: $5K/month

WHAT ADDS TIME:
  +2 weeks: Poor data quality (needs cleaning pipeline)
  +3 weeks: Multi-language support required
  +2 weeks: Strict security (CMEK, VPC Service Controls)
  -2 weeks: Client provides pre-cleaned data + clear evaluation criteria upfront
```

---

## ⚡ PART 4: RAPID-FIRE — 15 HARDEST TRAP QUESTIONS

---

**T-1: "What is endogeneity in MMM and how do you handle it?"**
Answer: Endogeneity = ad spend correlated with the error term because companies spend more
when they expect high demand. Standard regression then overestimates ad effectiveness.
Fix: instrumental variables (use media CPM costs as instrument for actual spend —
costs are exogenous to demand), or validate the model with geo-experiments that
provide causal estimates to calibrate against.

**T-2: "Your RAG returns hallucinated answers in production. How do you detect this?"**
Answer: LLM-as-judge on 2% sampled traffic with faithfulness rubric. Also: NLI model
that checks if each claim in the answer is entailed by retrieved context. Alert if
faithfulness < 0.80 on daily sample. On alert: human review of flagged traces in LangSmith.

**T-3: "Explain PagedAttention in vLLM."**
Answer: Standard LLM serving pre-allocates a fixed KV cache per request (e.g., 2048 tokens).
If a request uses only 500 tokens, 75% of that memory is wasted. PagedAttention manages KV
cache like OS virtual memory — allocates in pages (16-32 token blocks) on demand.
Result: 3-5x more concurrent requests on the same GPU. Same memory, more throughput.

**T-4: "Shapley values for attribution — mathematical intuition."**
Answer: From cooperative game theory. Each channel is a "player", the coalition is the set
of touchpoints in a journey, the outcome is conversion probability.
Shapley value for channel C = average marginal contribution to conversion probability,
averaged across ALL possible orderings of channels joining the coalition.
Property: Efficient (values sum to total conversion), Symmetric (same channels get same credit),
Dummy (unused channels get zero). Unlike Markov chains, Shapley is provably fair.

**T-5: "What is VPC Service Controls and why does it matter?"**
Answer: Creates a security perimeter around GCP services. Even if code has a bug and
calls BigQuery cross-project, VPC SC blocks at the network level regardless of IAM.
For multi-tenant AI: each client project wrapped in its own perimeter.
Cross-project data access is cryptographically impossible — not just policy-controlled.

**T-6: "Online vs offline feature store — what's the difference?"**
Answer: Online store (Bigtable-backed): real-time prediction serving, row-based lookup by
entity ID, <10ms latency SLA. Offline store (BigQuery-backed): training data generation,
column-oriented, supports point-in-time correct queries (prevents label leakage —
"what was feature X's value for customer Y at time T in the past?").

**T-7: "What is RLHF? Would you use it for Nike's content generation?"**
Answer: Train a reward model on human preference judgments (A vs B copy ratings from Nike's
creative team), then fine-tune LLM via PPO to maximise reward model score.
For Nike: I would NOT start with RLHF. Start with prompt engineering + RAG (brand guidelines).
RLHF is warranted when: (a) you need consistent brand voice that prompting can't capture,
(b) you have 5,000+ rated copy pairs for the reward model, (c) you have ML infra for
multi-GPU training. That's month 6+, not month 1.

**T-8: "Delta Lake — what is it and when do you need it for ML?"**
Answer: Storage layer adding ACID transactions + time travel to Parquet files on GCS/S3.
For ML: time travel enables point-in-time correct feature lookups (prevents training/serving
skew). ACID transactions: concurrent writes to feature table don't corrupt data.
Need it when: multiple pipelines write to the same feature table, or you need to audit
"what features did the model use to make this decision on 2024-03-01?"

**T-9: "How do you handle PII in a marketing AI system?"**
Answer: PII never enters the vector index raw. Pipeline:
(1) NER-based PII detection (names, emails, SSNs) before indexing
(2) Replace with pseudonymous tokens {CUSTOMER_7843}
(3) Access controls on the vector index per user role
(4) Maintain customer_id → chunk_id mapping for right-to-erasure requests
(5) Data minimisation: index aggregate summaries, not raw transaction records

**T-10: "What is the EU AI Act's risk classification?"**
Answer: Tiers: Unacceptable (banned), High Risk (regulated: hiring, credit, healthcare),
Limited Risk (chatbots must disclose AI nature), Minimal Risk (most recommendation engines).
For Huge: marketing personalisation = Minimal Risk. A chatbot = Limited Risk (disclose AI).
If a client wants AI-assisted hiring screening → High Risk: requires conformity assessment,
bias testing, human oversight design, documentation. Build an AI governance checklist
into Huge's project onboarding to classify every system.

**T-11: "RAG vs fine-tuning — when do you choose each?"**
```
RAG: use when knowledge changes frequently, knowledge base is large,
     citations are required, limited training data. Fast to deploy.
Fine-tuning: use when domain vocabulary is specialised, specific output
             format required, reasoning pattern needed, knowledge is stable.
Both: fine-tune for domain behaviour + RAG for factual grounding.
Rule: "RAG for facts, fine-tuning for behaviour."
```

**T-12: "How do you handle iOS ATT (App Tracking Transparency) in attribution?"**
Answer: iOS14+ requires explicit user opt-in for cross-app tracking. ~60-70% of iOS users
opt out. For attribution:
(1) Segment opted-in vs opted-out users separately
(2) For opted-out: use SKAdNetwork (Apple's privacy-preserving attribution API) —
    gives aggregate conversion reports with noise and delay
(3) Model-based extrapolation: estimate opted-out users' attribution by training a model
    on opted-in users' journeys and extrapolating to opted-out pool (adjusting for
    the selection bias — opted-out users may behave differently)
(4) First-party data focus: server-side events (website purchases recorded on your server)
    don't require ATT — prioritise server-side over client-side tracking

**T-13: "What's the NSGA-II algorithm and why did you use it for budget optimisation?"**
Answer: NSGA-II (Non-dominated Sorting Genetic Algorithm II) is a multi-objective optimisation
algorithm that maintains a Pareto-optimal front across competing objectives.
In my Axtria MMM: the objectives were (a) maximise revenue, (b) maximise reach, (c) minimise
cost. A single-objective optimizer (scipy.optimize) can't handle these simultaneously —
it needs a weighted sum which buries the trade-offs.
NSGA-II outputs a Pareto front: a set of budget allocations where no objective can be improved
without degrading another. The client then picks their preferred point on the front:
"I care more about reach than revenue this quarter" → pick different Pareto point.
This is fundamentally more useful than a single optimal budget — it shows trade-offs explicitly.

**T-14: "You join Huge Day 1. What do you do in Week 1?"**
Answer: Week 1 is entirely listening and learning. No code, no architecture proposals.
Day 1-2: Meet every team I'll interact with (ML engineers, client services, product, creative).
Day 3: Shadow a client call — understand how Huge sells and delivers AI.
Day 4: Audit current technical landscape (what's deployed, what's the CI/CD setup,
        what monitoring exists, what's the biggest pain point for ML engineers today).
Day 5: Write a 1-page "what I heard, what surprised me, what I'll do about it" memo
        for my manager.
Why: The most expensive mistake a new Solutions Architect makes is confidently building
the wrong thing. Week 1 ensures I build what's actually needed.

**T-15: "What is your unique value proposition for this role vs other candidates?"**
Answer: "I am one of very few professionals who can design the ML infrastructure to produce
marketing insights AND understand the marketing science behind which insights matter.
Most ML engineers know Vertex AI but not MMM. Most marketing scientists know ROAS and CLV
but not LangGraph and RAG. I've done both — built production agentic AI at Chubb and
built production MMM with Bayesian budget optimisation at Axtria. At Huge, where the value
is at the intersection of AI engineering and marketing intelligence for brands like
Nike, McDonald's, and Google, that combination is the differentiator."

---

> [!TIP]
> **The 3 answers that win this US interview:**
> 1. System design: Always start with clarifying questions. State assumptions. Draw ASCII architecture. Discuss trade-offs at the end.
> 2. MMM/attribution: Anchor to J&J/Stemline at Axtria. Quantify: "10% revenue improvement, Bayesian credible intervals, NSGA-II Pareto front."
> 3. Production: Show you've felt real pain. "The agent confidently returned wrong numbers — only LLM-as-judge sampling caught it."

> [!NOTE]
> **Opening statement to memorise:**
> "I'm a Senior Data Scientist and AI Engineering Leader with 9+ years. Currently at Chubb leading AI initiatives in insurance fraud detection using RAG and agentic AI. I'm excited about Huge because my background bridges both the hard AI engineering side — LangGraph, Vertex AI, RAG infrastructure — and the marketing science side — Bayesian MMM, multi-touch attribution, CLV modelling at Axtria. That intersection is precisely what Huge builds for brands like Nike, McDonald's, and Google. I'm here to architect those intelligent experiences."
