# 07 — Live System Design Scenarios
## Solutions Architect ML/AI · Huge · Interview Prep
### Abhishek Bhardwaj · June 2026

> **How to use this document**: In a live whiteboard session, you have ~45 minutes per scenario.
> The first 5 minutes are the most important: ask clarifying questions BEFORE drawing anything.
> Interviewers at design agencies like Huge care about *client empathy* as much as technical depth.

---

## GROUND RULES FOR EVERY DESIGN SESSION

| Phase | Time | What to do |
|---|---|---|
| Clarify | 0–5 min | Ask scope, scale, SLA, budget questions |
| Assumptions | 5–7 min | State what you're assuming out loud |
| High-level sketch | 7–15 min | Draw boxes and arrows, name services |
| Deep-dive | 15–35 min | Pick 2–3 components, go deep |
| Trade-offs | 35–42 min | Proactively surface alternatives you rejected |
| Questions for them | 42–45 min | Ask about their actual stack/constraints |

**The Huge twist**: always ask "who owns the data?" and "how does the client see results?" — they are a *services* company, not a product company. Client dashboards, data isolation, and billing attribution matter more here than at Google.

---

---

# SCENARIO 1: Real-Time Personalization Engine for McDonald's Mobile App

---

## 1.1 The Question

> *"McDonald's has 40,000+ locations, 69 million daily customers, and a loyalty app. Design a real-time personalization engine that recommends menu items at the point of ordering. Recommendations must feel instant."*

---

## 1.2 Clarifying Questions to Ask First

```
Before drawing anything, ask these — they signal architecture maturity:

1. "What does 'personalization' mean here — recommendations at app open,
   at checkout, or both?" → Defines latency requirements per touchpoint

2. "Is 'real-time' defined as user-level or session-level?
   Can we use a pre-computed embedding refreshed daily?" → Changes
   whether we need a streaming pipeline or batch is enough

3. "Do we have labeled conversion data (did user buy the recommendation)?"
   → Determines if we train supervised models vs collaborative filtering

4. "Is there a daypart / location context?" → Yes, Big Mac at 7am vs 11pm
   are very different signals; location = nearest store = local menu

5. "What's the existing data infrastructure? GA4? Firebase? CRM?"
   → Avoid re-platforming what already works

6. "Who owns the ML model — Huge builds and hands over, or Huge
   operates it?" → Changes the MLOps complexity dramatically

7. "What's the acceptable cold-start behavior for a brand-new user?"
   → Popularity-based fallback? Category-based?

8. "Is there a budget envelope? 69M users × daily recompute is expensive."
   → Helps scope the feature store refresh cadence
```

---

## 1.3 Assumptions (State These Out Loud)

```
- 69M DAU, ~50K RPS at peak (Super Bowl Sunday order surge)
- P99 recommendation latency < 200ms end-to-end
- 99.9% uptime (~8.7 hours downtime/year acceptable)
- We serve recommendations at: (a) app open, (b) cart page
- Customer has a profile if they've used the loyalty app ≥1 session
- Cold start: ~15% of daily users are first-timers → fallback to
  location-aware popularity
- GCP is the preferred cloud (McDonald's/Huge partnership context)
- Embeddings refreshed every 6 hours (near-real-time, not millisecond)
- A/B testing is required for model iterations
```

---

## 1.4 Architecture (ASCII)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     McDONALD'S LOYALTY APP                          │
│                  (iOS / Android — Firebase SDK)                     │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ User event (view, add-to-cart, order)
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  EVENT INGESTION LAYER                              │
│  ┌──────────────┐    ┌───────────────────┐    ┌───────────────────┐ │
│  │  Firebase    │───▶│  Google Pub/Sub   │───▶│  Pub/Sub Dead     │ │
│  │  Analytics   │    │  (fanout topics)  │    │  Letter Queue     │ │
│  └──────────────┘    └─────────┬─────────┘    └───────────────────┘ │
└────────────────────────────────┼────────────────────────────────────┘
                                 │ stream
                    ┌────────────▼─────────────┐
                    │   FEATURE PIPELINE        │
                    │   Dataflow Streaming      │
                    │   (Apache Beam jobs)      │
                    │   - Session windowing      │
                    │   - Feature normalization  │
                    │   - Embedding lookup       │
                    └────────────┬─────────────┘
                                 │ write features
          ┌──────────────────────▼────────────────────────┐
          │            FEATURE STORE                       │
          │         Vertex AI Feature Store               │
          │  ┌──────────────┐  ┌───────────────────────┐  │
          │  │  Online Store │  │   Offline Store        │  │
          │  │  (Bigtable)  │  │   (BigQuery)           │  │
          │  │  <10ms reads  │  │   batch training       │  │
          │  └──────┬───────┘  └───────────────────────┘  │
          └─────────┼──────────────────────────────────────┘
                    │ on-demand feature read
          ┌─────────▼──────────────────────────────────────┐
          │          RECOMMENDATION SERVING LAYER           │
          │                                                  │
          │  Stage 1: RETRIEVAL                              │
          │  ┌──────────────────────────────────────────┐   │
          │  │  Vertex AI Vector Search (ANN)            │   │
          │  │  - Customer embedding → 500 candidates    │   │
          │  │  - ScaNN index, 10ms P99                  │   │
          │  └──────────────────┬───────────────────────┘   │
          │                     │ 500 candidates             │
          │  Stage 2: RERANKING                              │
          │  ┌──────────────────▼───────────────────────┐   │
          │  │  Two-Tower Reranker (Vertex AI Endpoint)  │   │
          │  │  - Context: time, weather API, location   │   │
          │  │  - Price sensitivity, dietary flags       │   │
          │  │  - Top-K output (K=5 for UI)              │   │
          │  └──────────────────┬───────────────────────┘   │
          └─────────────────────┼──────────────────────────-┘
                                │
          ┌─────────────────────▼──────────────────────────┐
          │              SERVING API                        │
          │           Cloud Run (autoscaling)               │
          │  - Orchestrates feature fetch + retrieval        │
          │  - Adds A/B test assignment (Optimizely SDK)     │
          │  - Returns ranked item list + experiment ID      │
          └─────────────────────┬──────────────────────────┘
                                │ JSON response <200ms
          ┌─────────────────────▼──────────────────────────┐
          │           OBSERVABILITY & A/B LAYER             │
          │  Looker dashboards │ Cloud Monitoring │ BigQuery │
          │  - CTR, conversion, avg basket per variant       │
          └─────────────────────────────────────────────────┘
```

---

## 1.5 Data Model

```
Customer Embedding Vector:
  - user_id: string
  - embedding: float[256]        ← trained Two-Tower model
  - last_updated: timestamp
  - visit_frequency: int          ← behavioral segment
  - dietary_flags: string[]       ← ["no-beef", "vegetarian"]

Item Embedding Vector:
  - item_id: string
  - embedding: float[256]
  - category: string              ← "breakfast", "burger", "dessert"
  - daypart_affinity: float[4]    ← breakfast/lunch/dinner/late-night
  - location_availability: bool

Context Features (real-time):
  - current_hour: int (0-23)
  - day_of_week: int
  - store_id: string
  - weather_condition: string     ← via Tomorrow.io API
  - local_sports_event: bool      ← event calendar lookup
```

---

## 1.6 Scale Math (Back-of-Envelope)

```
DAILY LOAD:
  - 69M DAU × 2.5 sessions/day = 172.5M recommendation requests/day
  - 172.5M / 86400s = 1,996 RPS average
  - Peak (12pm lunch rush): 5× average = ~10K RPS
  - Super Bowl Sunday peak: 10× average = ~20K RPS

FEATURE STORE:
  - 69M users × 256 floats × 4 bytes = ~70 GB for user embeddings
  - 12K menu items × 256 floats × 4 bytes = ~12 MB (trivial)
  - Bigtable online store: ~$0.65/GB/month → ~$45/month for embeddings

VECTOR SEARCH (RETRIEVAL):
  - ScaNN index: 12K items, 256 dims — tiny, fits in memory
  - Latency: <5ms for ANN retrieval at this catalog size

RERANKER MODEL:
  - Input: 500 candidates × feature vector
  - Vertex AI Endpoint, 2 vCPU nodes, autoscale 5→50 nodes at peak
  - ~8ms P50 inference per request

END-TO-END LATENCY BUDGET:
  - Network (app → Cloud Run): 30ms
  - Feature fetch (Bigtable): 8ms
  - ANN retrieval (Vector Search): 5ms
  - Reranking (Vertex AI): 8ms
  - Serialization + response: 10ms
  TOTAL: ~61ms P50 ✓ — well under 200ms even at P99

COST (ROUGH ORDER OF MAGNITUDE):
  - Pub/Sub: 172.5M events/day × 1KB = ~172 GB/day → ~$86/month
  - Dataflow streaming: 5 workers × $0.06/hr → ~$215/month
  - Vertex AI Feature Store online: ~$45/month
  - Vertex AI Vector Search: 2 nodes → ~$400/month
  - Vertex AI Endpoint (reranker): 10 nodes avg → ~$2,000/month
  - Cloud Run: auto-scaled → ~$500/month
  TOTAL: ~$3,250/month for serving layer (ex. model training)
```

---

## 1.7 Super Bowl Sunday Spike Handling

```
Normal: 10K RPS → Peak: 100K RPS (10× for 4-hour window)

Strategy:
1. PREDICTIVE AUTO-SCALE: Cloud Run configured with min-instances=50
   pre-scaled 30 min before kickoff using Cloud Scheduler trigger

2. TIERED DEGRADATION:
   Tier 1 (normal): Full two-stage retrieval + reranker
   Tier 2 (80% load): Skip reranker → return ANN results directly
   Tier 3 (95% load): Return pre-computed "top-10 by store" from Redis cache

3. CACHE WARMING: At 5:30pm on Super Bowl day, batch-compute
   top-20 recommendations per store × customer segment → cache in
   Memorystore Redis with 1hr TTL

4. CIRCUIT BREAKER: If Feature Store latency > 50ms, fall back to
   user's last-ordered items (stored in Cloud Firestore, always available)
```

---

## 1.8 Failure Modes & Mitigations

```
Failure 1: Feature Store Outage
  Impact: Cannot fetch user embeddings → 0 personalization
  Mitigation: 
    - Cache last-known embedding in Cloud Firestore (TTL=24hr)
    - Fallback to store-level popularity model (pre-computed hourly)
    - Circuit breaker pattern in Cloud Run service

Failure 2: Cold Start (New User)
  ~15% of 69M DAU = 10.35M new/unrecognized users daily
  Strategy:
    - No loyalty history → use: store_id + current_daypart + weather
    - Segment-based recommendations (young adult urban = different from 
      family suburban) — use postal code proxy if location permission granted
    - Passive warm-up: after 2 interactions, compute lightweight embedding

Failure 3: Reranker Model Drift
  Symptom: CTR drops 10%+ vs control in A/B test
  Mitigation:
    - Canary deploy: 5% → 20% → 100% rollout gated by CTR threshold
    - Auto-rollback in Vertex AI Model Registry if evaluation metric fails
    - Shadow mode: new model scores silently for 24hr before any traffic
```

---

## 1.9 Three Key Trade-offs the Interviewer Will Probe

```
1. Real-time vs Near-real-time embeddings:
   Q: "Why refresh embeddings every 6 hours, not on every event?"
   A: Embedding recompute requires a full pass of user history.
      Streaming updates create consistency issues across the Two-Tower
      model's user and item towers. 6-hour refresh captures 99% of
      meaningful preference shifts. True real-time would require
      DML-aware incremental training — 3× cost for marginal gain.

2. Two-stage vs Single-stage retrieval:
   Q: "Why not just run the reranker over all 12K items?"
   A: 12K × feature vector × reranker inference at 50K RPS is infeasible
      (12K × 8ms = 96 seconds per user). Two-stage: ANN retrieves 500
      candidates in 5ms, reranker scores 500 in 8ms. Orders of magnitude
      cheaper while losing <2% accuracy vs exhaustive reranking.

3. Vertex AI Feature Store vs Redis:
   Q: "Why Vertex AI Feature Store vs just using Redis?"
   A: Feature Store provides: (1) training-serving skew elimination —
      the same feature definitions used in training are served at inference,
      (2) built-in monitoring for feature drift, (3) time-travel for 
      point-in-time correct training data. Redis would require us to
      build all of that ourselves. Cost difference is ~$500/month —
      worth paying for operational simplicity.
```

---

## 1.10 Week 1 vs Month 3 Roadmap

```
WEEK 1 (Prove the Foundation):
  - Stand up Pub/Sub ingestion from Firebase Analytics
  - Train baseline popularity model (no personalization) as baseline
  - Deploy Cloud Run serving with popularity model
  - Establish latency and CTR baseline metrics in Looker

MONTH 3 (Full Personalization):
  - Launch Two-Tower model trained on 90 days of conversion data
  - Deploy Vertex AI Feature Store with user + item embeddings
  - Enable A/B test: popularity vs personalized recommendations
  - Implement cold-start handling + circuit breakers
  - Present CTR lift report to McDonald's stakeholders
```

---

## 1.11 Cross-Questions the Interviewer WILL Ask

```
Q: "How do you handle the case where a customer has dietary restrictions 
    not explicitly set in the app?"
A: "Implicit signal detection: if a user has NEVER ordered beef in 200 
    sessions, we infer beef-avoidance and suppress beef recommendations.
    This is handled as a soft filter in the reranker — a penalty weight,
    not a hard exclusion, to avoid over-constraining recommendations."

Q: "McDonald's operates in 100+ countries with local menus. How does this
    affect your design?"
A: "Each market gets its own item embedding space — the McAloo Tikki in 
    India has no embedding relationship to a Big Mac. We partition the 
    Vector Search index by market_id. User embeddings are market-scoped 
    too. The serving API routes to the correct market partition first."

Q: "What's your success metric? How do you know personalization is working?"
A: "Primary: Add-to-cart rate for recommended items (vs control).
    Secondary: Average basket size per session, repeat visit rate within 
    7 days. We explicitly track 'personalization-influenced revenue' by 
    multiplying conversion rate lift × basket value × DAU."
```

---
---

# SCENARIO 2: Agentic AI Content Platform for Nike's Global Campaigns

---

## 2.1 The Question

> *"Nike needs to produce campaign content for 190+ countries with consistent brand voice but local relevance. Design an agentic AI platform that takes a creative brief and produces campaign-ready assets, with human approval before publish."*

---

## 2.2 Clarifying Questions to Ask

```
1. "What is the definition of 'campaign assets'? Copy only?
    Social posts? Display banners? Video scripts? Product shots?"
    → Scope of multimodal generation

2. "What's the latency expectation? Is 2 hours acceptable for a
    full campaign package, or does the creative team need it in 10 min?"
    → Determines synchronous vs async architecture

3. "Who are the users of this platform? Nike's in-house creative team,
    or Huge's strategists acting as intermediaries?"
    → Changes the UX and approval workflow design

4. "What constitutes brand compliance? A rulebook document? Or does
    Nike have an existing brand style guide we RAG over?"
    → RAG corpus design

5. "Are there regulatory constraints per market?
    (Alcohol advertising in Islamic countries, child advertising laws in EU)"
    → Compliance checker scope

6. "How many simultaneous campaigns should the system support?"
    → Concurrency requirements for the agent orchestration layer

7. "Is there an existing DAM (Digital Asset Management) system?
    (Brandfolder, Bynder, etc.)" → Integration complexity
```

---

## 2.3 Architecture (ASCII)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CREATIVE BRIEF INPUT                              │
│  (Huge strategist UI — Next.js app on Cloud Run)                    │
│  Brief: product, target audience, markets, budget, timeline          │
└─────────────────────────┬────────────────────────────────────────────┘
                          │ POST /campaigns/new
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   CAMPAIGN PLANNING AGENT                            │
│                (LangGraph on Cloud Run Jobs)                         │
│  - Decomposes brief into market-specific tasks                       │
│  - Assigns localization: 190 markets → language/culture clusters     │
│  - Orchestrates downstream agents via Pub/Sub task queue             │
└──────┬──────────────────┬─────────────────────┬───────────────────--┘
       │                  │                      │
       ▼                  ▼                      ▼
┌────────────┐   ┌────────────────┐   ┌──────────────────────────────┐
│  BRAND     │   │  CONTENT GEN   │   │   LOCALIZATION AGENT         │
│  RAG       │   │  AGENT         │   │   (Gemini + DeepL API)        │
│  (Vertex   │   │  (Gemini 1.5   │   │   - Translate to 60+ langs   │
│   Vector   │   │   Pro)         │   │   - Cultural adaptation       │
│   Search)  │   │  - Ad copy     │   │   - RTL language support      │
│  - Brand   │   │  - Social post │   │   - Date/currency formatting  │
│    voice   │   │  - OOH copy    │   └──────────────┬───────────────┘
│  - Visual  │   │  - Email body  │                  │
│    style   │   └───────┬────────┘                  │
│  - Tone    │           │                            │
│    guide   │           └─────────────┬──────────────┘
└────────────┘                         │ generated content
                                       ▼
                      ┌────────────────────────────────┐
                      │    BRAND COMPLIANCE CHECKER    │
                      │    (LLM-as-Judge: Gemini 1.5)  │
                      │  Scores each asset on:          │
                      │  - Brand voice alignment (0-1)  │
                      │  - Regulatory compliance flag   │
                      │  - Visual consistency flag      │
                      │  Threshold: score < 0.85 →     │
                      │  auto-reject + explain          │
                      └────────────────┬───────────────┘
                                       │ approved drafts
                                       ▼
                      ┌────────────────────────────────┐
                      │    HUMAN-IN-THE-LOOP (HITL)    │
                      │    Approval Workflow            │
                      │  - Slack notification to        │
                      │    campaign lead               │
                      │  - Review UI with side-by-side  │
                      │    diff: AI draft vs brand ref  │
                      │  - Approve / Edit / Reject      │
                      │  - Approved → publish queue     │
                      └────────────────┬───────────────┘
                                       │
                                       ▼
                      ┌────────────────────────────────┐
                      │     DELIVERY & DAM             │
                      │  - Cloud Storage (asset store)  │
                      │  - Webhook → Bynder/Brandfolder │
                      │  - CDN distribution per market  │
                      └────────────────────────────────┘
```

---

## 2.4 Brand RAG — Versioning Brand Guidelines

```
Problem: Nike's brand guidelines change. "Just Do It" campaigns 2023 vs 2024
may have different tone rules. RAG must be version-aware.

Solution:
  - Each brand guideline document tagged with: {version, effective_date, market}
  - Vertex AI Vector Search index has metadata filter support:
    query filter: version = "2024-Q4" AND market IN ["US", "EU"]
  - Retrieval always uses "latest effective version ≤ campaign_start_date"
  - Old versions retained for 3 years → compliance audit trail
  - Update workflow: brand team uploads new guideline PDF →
    Dataflow job re-chunks + re-embeds → incremental index update (no full rebuild)
  - Semantic diff alert: if cosine similarity between old chunk and
    new chunk < 0.7, flag for brand manager review
```

---

## 2.5 Multi-Tenancy: Nike + McDonald's on Same Infrastructure

```
ISOLATION LEVELS (choose based on compliance requirements):

Level 1: Namespace Isolation (cheapest)
  - Single GCP project, separate Vertex AI datasets + endpoints per client
  - VPC-SC perimeter per client dataset
  - Risk: noisy neighbor on shared GPU pools
  - Cost: ~$5K/month base + per-client compute

Level 2: Separate GCP Projects (recommended for Huge)
  - Nike in project: huge-nike-prod
  - McDonald's in project: huge-mcd-prod  
  - Shared services: Huge internal project (CI/CD, monitoring aggregator)
  - Data NEVER crosses project boundary
  - IAM: Nike engineers have ZERO visibility into McDonald's project
  - Cost: ~$2K/month overhead per client for base infra

Level 3: Separate GCP Organizations (max isolation)
  - Full organization-level isolation
  - Overkill for most clients, but right for regulated industries
  - Nike's legal team can audit their own Organization Policy constraints

RECOMMENDATION: Level 2 (project-per-client) for Huge.
  Justification: clients pay for exclusivity perception + real isolation.
  The incremental $2K/month is trivially passed through in SoW.
```

---

## 2.6 Latency Budget for Campaign Generation

```
Acceptable latency: campaign content is NOT a real-time use case.
Creative campaigns take days manually → AI takes minutes → still a win.

ASYNC PIPELINE TIMING:
  - Campaign brief parsing: 2–5 sec
  - Brand RAG retrieval (50 chunks): 1–2 sec
  - Content generation per asset (Gemini 1.5 Pro): 15–45 sec
  - Compliance check per asset: 10–20 sec
  - Localization (60 languages): parallelized → 60–120 sec
  - HITL notification: immediate (Slack webhook)
  - Human review time: 15–60 min (not in system budget)

TOTAL SYSTEM TIME (excl. human): 5–10 minutes for a full 60-market campaign
Compare to: 2–3 weeks manually. ROI is obvious.

Pattern: Fire-and-forget + polling/webhook
  - POST /campaigns → returns campaign_id immediately (202 Accepted)
  - Client polls GET /campaigns/{id}/status OR receives webhook on completion
  - Pub/Sub drives agent pipeline asynchronously
```

---

## 2.7 Scale Considerations

```
Concurrency: 50 simultaneous campaigns (Nike runs global + regional campaigns)
  - Each campaign = 190 market tasks × 5 asset types = 950 generation tasks
  - 50 campaigns = 47,500 concurrent Gemini API calls (batched, not simultaneous)
  - Gemini API quota: negotiate enterprise tier (100K+ QPM) with Google

Agent Reliability:
  - LangGraph checkpointing: every agent step checkpointed to Cloud Firestore
  - If agent crashes mid-run: resume from last checkpoint, not start-over
  - Idempotency: task_id prevents duplicate generation on retry
```

---

## 2.8 Three Key Trade-offs

```
1. Gemini API vs Fine-tuned model:
   Q: "Why not fine-tune a model on Nike's historical campaign copy?"
   A: "Fine-tuning is a valid future state. We start with Gemini + RAG
      because: (1) faster to deploy, (2) brand guidelines in RAG are 
      interpretable and updateable without retraining, (3) Nike's legal
      team can audit what instructions the model receives. Fine-tuning
      is a month-3 optimization if RAG quality is insufficient."

2. LLM-as-Judge vs Deterministic Compliance Rules:
   Q: "How can you trust an LLM to judge compliance?"
   A: "LLM-as-judge is layered on top of deterministic rules, not instead.
      Hard rules (e.g., banned phrases, required disclaimers) are regex/
      keyword checks first. LLM-as-judge handles soft rules: tone,
      cultural sensitivity, brand voice. Score threshold is tuned on a
      held-out set of human-judged examples. We report judge agreement
      rate with humans weekly."

3. Human approval on every asset vs sampling:
   Q: "Reviewing 950 assets per campaign doesn't scale. What do you do?"
   A: "Two-tier HITL: (1) Auto-approve assets scoring ≥ 0.95 in compliance
      check AND matching a previously-approved template. (2) Human review
      for novel creative, low-scoring assets, or regulated markets. Target:
      human reviews 15% of assets, auto-approves 85%. Track override rate —
      if humans override > 20% of auto-approvals, tighten the threshold."
```

---

## 2.9 Interviewer Cross-Questions

```
Q: "How do you prevent the AI from generating something that goes viral
    for the wrong reasons — culturally insensitive content?"
A: "Three layers: (1) Localization agent uses culture-specific system 
    prompts with explicit taboo lists per market. (2) Compliance checker
    runs a cultural sensitivity sub-check using Gemini with a 
    country-specific persona prompt. (3) All markets flagged as 'high 
    sensitivity' (e.g., Middle East, China) are mandatory human review,
    no auto-approve. Nike's regional marketing leads are notified directly."

Q: "What happens to rejected content? Is there a feedback loop?"
A: "Every rejection is logged with: asset_id, rejection_reason, 
    compliance_score, human_override. This forms a golden dataset for
    fine-tuning the compliance checker. Monthly: we retrain the judge
    model on accumulated rejections. This is the flywheel — the system
    gets better the more campaigns it runs."
```

---
---

# SCENARIO 3: Marketing Analytics Intelligence Platform for Huge India

---

## 3.1 The Question

> *"Design the shared infrastructure that powers Huge India's Marketing Analytics practice — serving 10 Fortune 500 clients from a single platform, with data isolation, per-client AI capabilities, and a billing model."*

---

## 3.2 Architecture (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CLIENT-FACING LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ...  ┌──────────────┐             │
│  │  Google      │  │  McDonald's  │       │  Nike         │             │
│  │  Portal      │  │  Portal      │       │  Portal       │             │
│  │  (SSO+RBAC)  │  │  (SSO+RBAC)  │       │  (SSO+RBAC)  │             │
│  └──────┬───────┘  └──────┬───────┘       └──────┬───────┘             │
└─────────┼─────────────────┼───────────────────────┼─────────────────────┘
          │                 │                         │
          └─────────────────┼─────────────────────────┘
                            │ Authenticated API calls
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    AGENT GATEWAY (API Layer)                            │
│              Cloud Run + Cloud Endpoints + Apigee                       │
│  - JWT validation → extract client_id                                   │
│  - Semantic router: classify intent → route to correct agent            │
│  - Rate limiting per client (SLA tiers)                                 │
│  - Request logging → Cloud Logging (per-client log bucket)              │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │ Routed request + client_id
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   SHARED ML PLATFORM (Vertex AI)                        │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  AGENT ORCHESTRATION LAYER (LangGraph)                             │ │
│  │  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │ │
│  │  │  Analytics Agent  │  │  Insight Agent   │  │  Forecast Agent  │  │ │
│  │  │  (SQL generation) │  │  (NL → insight) │  │  (Prophet/BQML)  │  │ │
│  │  └──────────────────┘  └─────────────────┘  └──────────────────┘  │ │
│  │  All agents receive: {client_id, allowed_datasets[], query}         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  CLIENT-SPECIFIC RAG INDEXES (Vertex AI Vector Search)             │ │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │ │
│  │  │  google-idx      │  │  mcd-idx          │  │  nike-idx        │  │ │
│  │  │  (Google data)  │  │  (MCD data)       │  │  (Nike data)     │  │ │
│  │  └─────────────────┘  └──────────────────┘  └──────────────────┘  │ │
│  │  Index isolation enforced by: separate VPC-SC perimeters            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA LAYER (BigQuery)                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  google-prod.    │  │  mcd-prod.       │  │  nike-prod.          │  │
│  │  analytics.*     │  │  analytics.*     │  │  analytics.*         │  │
│  │                  │  │                  │  │                      │  │
│  │  GCP Project:    │  │  GCP Project:    │  │  GCP Project:        │  │
│  │  huge-google-    │  │  huge-mcd-prod   │  │  huge-nike-prod      │  │
│  │  prod            │  │                  │  │                      │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘  │
│  VPC-SC: Each project in its own Access Context Manager perimeter        │
└─────────────────────────────────────────────────────────────────────────┘
          │                        │                          │
          ▼                        ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              MONITORING, BILLING & COMPLIANCE                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Cloud Billing export → BigQuery → Looker (per-client view)     │   │
│  │  Labels: client_id, agent_name, model_name on every resource    │   │
│  │  Cloud Monitoring: per-client dashboards + SLA alerting         │   │
│  │  Audit logs: Cloud Audit Logs → immutable per-client log sink   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3.3 Data Isolation: Ensuring Google's Data Never Touches McDonald's Path

```
ENFORCEMENT MECHANISMS:

1. GCP PROJECT SEPARATION:
   - Each client = separate GCP project
   - Service accounts are project-scoped → no cross-project access by default
   - BigQuery dataset IAM: huge-google-prod's service account has ZERO
     role on huge-mcd-prod datasets

2. VPC SERVICE CONTROLS (VPC-SC):
   - Each client project is in a separate Access Context Manager perimeter
   - Data exfiltration protection: a Vertex AI job in mcd project cannot
     read data from google project even with a stolen credential

3. AGENT GATEWAY ENFORCEMENT:
   - Agent receives client_id from JWT (server-side, not user-supplied)
   - BigQuery query template: always includes "WHERE client_id = ?" bind param
   - SQL-injection mitigation: parameterized queries only, no string concat
   - Allowed datasets list is server-side configured per client_id

4. AUDIT TRAIL:
   - All BigQuery reads logged to Cloud Audit Logs
   - Automated daily check: flag any query from client_A's service account
     touching client_B's dataset → PagerDuty alert to Huge CTO

5. PENETRATION TESTING:
   - Quarterly third-party pen test of the isolation boundaries
   - Required by enterprise SoW with Fortune 500 clients
```

---

## 3.4 Client Onboarding in Under 2 Weeks

```
AUTOMATED ONBOARDING PIPELINE (Terraform + Cloud Build):

Day 1-2: Infrastructure Provisioning (automated)
  terraform apply -var="client_id=new_client" -var="tier=enterprise"
  Creates:
    - New GCP project: huge-{client}-prod
    - BigQuery dataset: analytics.raw, analytics.curated, analytics.ml_features
    - Service accounts + IAM bindings
    - VPC-SC perimeter membership
    - Vertex AI Vector Search index (empty)
    - Client RAG index initialized

Day 3-5: Data Ingestion Setup
  - Fivetran connectors configured for client's data sources
    (Google Ads, Meta, Salesforce, CRM, website analytics)
  - Dataform pipelines cloned from template, parameterized with client schema
  - First data load validation: automated dbt tests

Day 6-8: AI Customization
  - Upload client's brand documents → chunked + embedded → RAG index populated
  - Client-specific agent prompts configured (brand voice, industry context)
  - KPI definitions loaded (client defines "conversion" for their business)

Day 9-10: UAT + Handover
  - Client's analytics team gets portal access (SSO configured)
  - Guided walkthrough of 3 core use cases
  - SLA monitoring dashboard activated

Day 11-14: Buffer for hypercare, edge cases, custom requests
```

---

## 3.5 Cost Attribution per Client

```
TAGGING STRATEGY:
  All GCP resources tagged with:
    - Labels: {client_id, environment, team, feature}
    - Example: {client_id: "mcdonalds", environment: "prod",
                team: "huge-india", feature: "mmm-pipeline"}

BILLING EXPORT:
  - Cloud Billing → BigQuery daily export → huge-internal-billing project
  - Looker report: per-client monthly cost breakdown by service
  - Margin calculation: (contracted price - actual GCP cost) per client

CHARGEBACK MODEL:
  - Base platform fee: flat $X/month (covers shared services: gateway, monitoring)
  - Variable: actual GCP spend × 1.4 markup (40% margin)
  - AI usage: Gemini API calls billed at cost + margin, tracked per client_id
  - Clients see their spend dashboard → builds trust, avoids billing disputes
```

---

## 3.6 Trade-offs

```
1. Shared Vertex AI cluster vs per-client clusters:
   Q: "If Google's ML job is on the same GPU cluster as McDonald's,
      is there a noisy neighbor risk?"
   A: "Yes. Mitigation: Vertex AI Workbench and Training jobs are in 
      separate projects, so they use separate resource pools. For inference
      endpoints, we use Cloud Run (CPU-based) which auto-scales independently
      per deployment. GPU training is batch, not real-time — less sensitivity
      to noisy neighbor. If a client needs dedicated GPU: dedicated node pools
      at 2× cost, available as add-on."

2. Single Looker instance vs per-client Looker:
   A: "Single Looker instance with row-level security (RLS) enforced via 
      user attributes. Each client user gets a user_attribute: client_id.
      All LookML models filter on: ${TABLE}.client_id = '{{ _user_attributes["client_id"] }}'.
      Tested rigorously — but we document this as a 'logical isolation'
      not 'physical isolation'. Clients who require physical isolation 
      get their own Looker instance (premium tier)."

3. Latency for shared Agent Gateway vs direct Vertex AI calls:
   A: "The gateway adds ~15-20ms of overhead (auth, routing, logging).
      For analytics use cases (response in seconds, not milliseconds), 
      this is negligible. The gateway's value — security, audit trail,
      rate limiting — far outweighs the latency overhead."
```

---
---

# SCENARIO 4: Event-Driven Marketing Mix Model (MMM) Platform

---

## 4.1 The Question

> *"Design an MMM platform that automatically reruns when new marketing spend data arrives, evaluates the new model, and auto-deploys if it's better. Connect this to budget optimization recommendations."*

---

## 4.2 Architecture (ASCII)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION                                    │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐ │
│  │  Fivetran    │   │  Google Ads  │   │  Salesforce / CRM        │ │
│  │  (connectors)│   │  API         │   │  (offline conversions)   │ │
│  └──────┬───────┘   └──────┬───────┘   └──────────────┬───────────┘ │
│         └──────────────────┴──────────────────────────┘             │
│                             │ raw data                              │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  BigQuery Raw Layer (partitioned by date)                   │   │
│  │  Dataform transformations → curated spend/revenue tables    │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
└───────────────────────────────┼─────────────────────────────────────┘
                                │ new data partition detected
                                ▼
                 ┌──────────────────────────────┐
                 │  BigQuery → Pub/Sub Trigger   │
                 │  (Cloud Scheduler + BQ Jobs  │
                 │   OR BigQuery Change History) │
                 └──────────────┬───────────────┘
                                │ trigger event
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  VERTEX AI PIPELINES (Kubeflow)                     │
│                                                                      │
│  Step 1: Data Validation (Great Expectations)                        │
│    - Check spend data completeness > 95%                             │
│    - Check revenue data lag < 3 days                                 │
│    - FAIL → alert to data engineering team, halt pipeline            │
│                                                                      │
│  Step 2: Feature Engineering                                          │
│    - dbt transformations: adstock (Weibull decay), saturation        │
│      (Hill function), seasonal decomposition                         │
│    - Output: feature matrix → BigQuery ML_FEATURES table             │
│                                                                      │
│  Step 3: MMM Training                                                 │
│    ┌────────────────────────────────────────────────────┐            │
│    │  Google Meridian (PyMC-based Bayesian MMM)         │            │
│    │  - MCMC sampling: 4 chains × 2000 iterations       │            │
│    │  - Priors: informed by industry benchmarks         │            │
│    │  - Outputs: contribution %, ROI per channel,       │            │
│    │    diminishing returns curves                      │            │
│    └────────────────────────────────────────────────────┘            │
│                                                                      │
│  Step 4: Model Evaluation                                             │
│    - MAPE < 10% on holdout last 4 weeks                              │
│    - RHAT < 1.1 (MCMC convergence diagnostic)                        │
│    - Coefficient signs consistent with prior knowledge               │
│    - Compare vs challenger: if new MAPE < current MAPE - 0.5%       │
│      → promote to production                                         │
│                                                                      │
│  Step 5: Budget Optimizer                                             │
│    - NSGA-II multi-objective optimization                            │
│    - Objectives: maximize revenue, minimize spend variance           │
│    - Constraints: total budget, channel min/max, brand spend floor   │
│    - Output: Pareto-optimal budget allocations                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
             ┌─────────────────────────────────────┐
             │     MODEL REGISTRY & VERSIONING      │
             │     Vertex AI Model Registry         │
             │  - Champion / Challenger tracking    │
             │  - Auto-promote if evaluation passes │
             │  - Rollback: one-click to previous   │
             └─────────────────────────────────────-┘
                               │
                               ▼
             ┌─────────────────────────────────────┐
             │     INSIGHTS DASHBOARD              │
             │     Looker + Looker Studio          │
             │  - Channel contribution waterfall   │
             │  - ROI by channel + confidence int  │
             │  - Budget recommendation table      │
             │  - "What if" scenario planner       │
             └─────────────────────────────────────┘
```

---

## 4.3 Handling 2-3 Week MMM Training Time

```
PROBLEM: Bayesian MCMC sampling for a large MMM (3 years weekly data,
20 channels) can take hours to days on CPU.

SOLUTIONS:

1. GPU-ACCELERATED MCMC:
   - Meridian/PyMC supports JAX backend → run on GPU (A100 or L4)
   - Reduces 10-hour CPU run to ~45 minutes on A100
   - Vertex AI Training Jobs: request GPU node, terminate after training
   - Cost: A100 ~$3.67/hr × 0.75hr = ~$2.75 per model run

2. INCREMENTAL TRAINING (not full refit):
   - For weekly data updates: use warm-start from previous MCMC posterior
   - Treat previous posterior as the prior for new data
   - Reduces iterations needed: 500 samples vs 2000 for full fit
   - Caveat: can accumulate posterior drift over time → full refit monthly

3. PIPELINE SCHEDULING:
   - Daily trigger: run data validation + feature engineering only
   - Weekly trigger (Sunday 2am): full MMM refit (low-traffic window)
   - Monthly trigger: full refit + model review meeting with client

4. ASYNC NOTIFICATION:
   - Pipeline runs in background; CMO dashboard shows "Model updating..."
   - Email/Slack notification when new model is ready + key change summary
   - "Your TV ROI estimate changed from $2.1 to $1.9 — here's why"
```

---

## 4.4 Validation Before Pushing to Client

```
VALIDATION GATE (all must pass before promotion):

Technical Gates:
  ✓ MAPE < 10% on last 4 weeks (out-of-sample)
  ✓ RHAT < 1.1 on all parameters (MCMC convergence)
  ✓ No sign flips vs previous model (TV can't suddenly go negative ROI
    without business explanation)
  ✓ Confidence intervals overlap with previous model (stability check)

Business Logic Gates:
  ✓ Channel contributions sum to ~100% (sanity)
  ✓ Total model revenue within 5% of actual reported revenue
  ✓ No channel's contribution changed > 20% week-over-week without
    a corresponding known spend change

Human Gate (monthly):
  - Huge analyst reviews model diagnostics report
  - Client CMO reviews "what changed this month" summary
  - Sign-off recorded in audit log before production promotion

EXPLAINABILITY:
  - Auto-generated model change summary:
    "TV contribution increased 3.2% due to higher spend in weeks 8-10.
     Digital ROI decreased 0.4% — likely diminishing returns at current budget."
  - This is sent to client alongside the new budget recommendations
```

---

## 4.5 Connection to Abhishek's Axtria MMM Experience

```
At Axtria, you built MMM for pharmaceutical clients (pharma channel attribution
is analogous — DTC advertising, HCP promotion, conference spend).

TALKING POINTS FOR THE INTERVIEW:
  "At Axtria, I built MMM pipelines for pharma clients that had to handle
   - Long-lag effects: physician prescriptions take 12-16 weeks to convert
     from a sales rep visit — this maps to the adstock/carryover problem
   - Regulatory constraints on promotional spend — same pattern as brand
     spend floors in the optimizer
   - Weekly data cadence with month-end adjustments — same trigger logic
     as the BigQuery CDC → Pub/Sub pattern I'd design here
   - I'd leverage Google Meridian here specifically because it handles
     Bayesian uncertainty intervals natively, which is critical for
     giving clients confidence bounds on their budget recommendations,
     not just point estimates."
```

---
---

# SCENARIO 5: Multi-Channel Attribution System (Post-Cookie)

---

## 5.1 The Question

> *"With third-party cookies gone, design an attribution system that tells clients which marketing channels actually drove conversions — without relying on cross-site tracking."*

---

## 5.2 Architecture (ASCII)

```
┌──────────────────────────────────────────────────────────────────────┐
│              FIRST-PARTY DATA COLLECTION                             │
│                                                                      │
│  ┌────────────────┐   ┌──────────────┐   ┌────────────────────────┐ │
│  │  Website       │   │  Server-side │   │  Offline CRM           │ │
│  │  (GA4 + GTAG)  │   │  Event API   │   │  (Salesforce, POS)     │ │
│  │  1P cookies    │   │  (bypass iOS │   │  (in-store, call ctr)  │ │
│  │  only          │   │   ITP)       │   │                        │ │
│  └───────┬────────┘   └──────┬───────┘   └───────────┬────────────┘ │
└──────────┼───────────────────┼───────────────────────┼──────────────┘
           └──────────────────┬┘                       │
                              │                         │
                              ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  CUSTOMER DATA PLATFORM (CDP)                       │
│              (Segment or custom on BigQuery)                        │
│                                                                      │
│  IDENTITY RESOLUTION:                                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Deterministic matching: email hash, phone hash, login ID    │   │
│  │  Probabilistic matching: device fingerprint, IP+UA cluster   │   │
│  │  Output: Golden Record per customer (unified_customer_id)    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  JOURNEY STITCHING:                                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Sequence: [Paid Search] → [Email] → [Direct] → [Purchase]  │   │
│  │  Time-ordered by unified_customer_id                         │   │
│  │  Stored in BigQuery: customer_journeys table                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │ journeys
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  ATTRIBUTION MODELS                                  │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Rule-based (baseline): Last-touch, First-touch, Linear     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              +                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Markov Chain Attribution:                                   │    │
│  │  - Build transition matrix from observed journeys           │    │
│  │  - Removal effect: credit = P(convert) - P(convert|remove) │    │
│  │  - Handles 4-15 touchpoint journeys well                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              +                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  ML Attribution (BQML Logistic / XGBoost):                  │    │
│  │  - Features: channel sequence, time gaps, device type,      │    │
│  │    content engaged, recency to purchase                      │    │
│  │  - Shapley values for channel attribution scores            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              +                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  MODELED CONVERSION (for unobservable users):               │    │
│  │  - iOS users with ATT=off: we observe 30% of their journey  │    │
│  │  - Use statistical scaling: if 30% of iOS users convert at  │    │
│  │    rate R, model total iOS conversions = R / 0.30           │    │
│  │  - Uncertainty bands on all iOS-influenced channel credits  │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                              │ attribution results
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              TRIANGULATION WITH MMM                                  │
│  Attribution → BOTTOM-UP (individual journey level)                  │
│  MMM → TOP-DOWN (aggregate channel contribution)                     │
│                                                                      │
│  If both agree within 15%: high confidence in channel credit        │
│  If they disagree > 15%: flag for analyst review                     │
│  Combined view: Bayesian posterior blend (weighted by model MAPE)    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
             ┌──────────────────────────────────┐
             │   ATTRIBUTION DASHBOARD (Looker) │
             │   - Channel credit table         │
             │   - CPA by channel (attributed)  │
             │   - iOS vs Android comparison    │
             │   - MMM vs MTA reconciliation    │
             └──────────────────────────────────┘
```

---

## 5.3 Handling Apple ATT (iOS Users Not Trackable)

```
PROBLEM: After iOS 14.5, ~60-70% of iOS users opt out of ATT.
This means no IDFA → cannot link ad impressions to app installs/events.

SOLUTIONS:

1. SERVER-SIDE EVENT API (Meta CAPI, Google Enhanced Conversions):
   - Instead of pixel firing in browser, server sends conversion events
   - Matched on: email hash, phone hash, first-party login
   - Recovers ~40-60% of "lost" iOS conversions where user has an account

2. PROBABILISTIC MODELING:
   - On opted-in users (30%): measure conversion rate by channel, creative
   - Apply learned rates to opted-out population
   - Uncertainty: 30% observation means confidence intervals are wider
   - Report as: "Email drove 15K ± 4K conversions (iOS adjustment applied)"

3. AGGREGATED EVENT MEASUREMENT (AEM):
   - Apple's privacy-preserving framework: reports top 8 conversion events
     with delay and noise (DP)
   - Ingest AEM data into attribution model as a signal (not ground truth)

4. INCREMENTALITY TESTING (gold standard):
   - Run geo-holdout experiments: turn off a channel in 20% of markets
   - Measure lift in conversion rate in active vs holdout
   - This is iOS-agnostic — measures aggregate causal effect
   - Schedule quarterly per channel, results inform MMM priors

TRANSPARENCY TO CLIENT:
   - Every attribution report includes "data coverage" metric:
     "73% of conversions are directly observed; 27% are modeled"
   - Trend over time: if modeled % increases, data collection needs attention
```

---

## 5.4 Abhishek's Axtria Attribution Experience

```
TALKING POINTS:
  "At Axtria I built omnichannel attribution for pharma clients where:
   - HCP (physician) touchpoints: rep visits, email, congress, webinar
     - directly analogous to digital + offline channel attribution
   - Patient touchpoints: DTC TV, digital, co-pay programs
     - we used Markov chain attribution here — same methodology
   - Privacy constraints: HIPAA meant no individual-level patient tracking
     → we used aggregated zip-code level data + modeled conversion
     → exactly the post-cookie world problem, just with HIPAA not GDPR
   - I'd apply the same triangulation approach: MMM for top-down,
     Markov chain MTA for bottom-up, reconciliation dashboard for client"
```

---
---

# SCENARIO 6: IKEA AI Shopping Assistant (12,000 Product Catalog)

---

## 6.1 The Question

> *"IKEA has 12,000 products. Design an AI shopping assistant that understands text AND images — so a customer can say 'I need a blue sofa under 220cm for 6 people' or upload a room photo and get furniture recommendations."*

---

## 6.2 Clarifying Questions

```
1. "Is this a web app, mobile app, or both? Does it replace or
    augment the existing IKEA.com search?" → Integration scope

2. "What's the success metric — time to purchase? Basket size?
    Return rate reduction?" → Model optimization target

3. "Do we have access to IKEA's full product catalog with
    structured attributes (dimensions, colors, weight, materials)?"
    → Determines filter quality

4. "What's the volume? IKEA.com gets ~3 billion visits/year
    (~100 RPS average, 1K RPS peak)" → Serving scale

5. "Is the room photo feature available to all users or premium?"
    → Feature flag and rollout strategy

6. "Are product images consistent (white background studio shots)
    or user-generated too?" → Image embedding quality
```

---

## 6.3 Architecture (ASCII)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    USER INTERACTION LAYER                            │
│  ┌──────────────────────────────┐  ┌───────────────────────────────┐ │
│  │   TEXT QUERY                 │  │   IMAGE UPLOAD                │ │
│  │   "blue sofa, 220cm, 6 ppl"  │  │   [room photo uploaded]       │ │
│  └───────────────┬──────────────┘  └───────────────┬───────────────┘ │
└──────────────────┼───────────────────────────────────┼───────────────┘
                   │                                   │
                   ▼                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  QUERY UNDERSTANDING LAYER                           │
│                                                                      │
│  TEXT PATH:                          IMAGE PATH:                     │
│  ┌────────────────────────────┐      ┌─────────────────────────────┐│
│  │  Gemini 1.5 Flash          │      │  Gemini Vision              ││
│  │  - Extract structured      │      │  - Detect furniture style   ││
│  │    filters from NL query:  │      │  - Extract room dimensions  ││
│  │    {color: blue,           │      │  - Identify color palette   ││
│  │     max_width: 220,        │      │  - Style keywords: modern,  ││
│  │     min_seats: 6,          │      │    Scandinavian, cozy       ││
│  │     category: sofa}        │      │  → structured JSON output   ││
│  │  - Generate text embedding │      │  → image embedding (1408d)  ││
│  └─────────────┬──────────────┘      └──────────────┬──────────────┘│
└────────────────┼────────────────────────────────────┼───────────────┘
                 │ text embed                         │ img embed
                 └─────────────────┬──────────────────┘
                                   │ combined query vector
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│               VERTEX AI VECTOR SEARCH (Multimodal Index)            │
│                                                                      │
│  Index contains:                                                     │
│  - 12,000 products × (text_embedding + image_embedding) concatenated │
│  - Metadata: {width_cm, height_cm, color[], material, price_eur,    │
│               category, in_stock_store_ids[], rating}               │
│                                                                      │
│  ANN Search: top-100 candidates                                      │
│  Post-filter: metadata filter (color=blue, width < 220, seats >= 6) │
│  → filtered to top-20                                                │
└────────────────────────────────────┬─────────────────────────────────┘
                                     │ top-20 candidates + metadata
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│              CONVERSATIONAL RERANKER + RESPONSE GENERATOR           │
│              Gemini 1.5 Pro (with conversation history)             │
│                                                                      │
│  Input:                                                              │
│  - User query (original NL)                                          │
│  - Top-20 candidate products (name, description, price, dims)        │
│  - Conversation history (multi-turn)                                 │
│  - User's basket (what they already have)                            │
│                                                                      │
│  Output:                                                             │
│  - Ranked recommendations with natural language explanations:        │
│    "The KARLSTAD sofa fits your space at 190cm wide, seats 6,       │
│     and comes in blue. It pairs well with the LACK coffee table     │
│     you already have in your cart."                                  │
│                                                                      │
│  Cross-sell: "You might also need cushions and a rug to complete    │
│               the look — here are 3 options that match."            │
└────────────────────────────────────┬─────────────────────────────────┘
                                     │
                                     ▼
                     ┌──────────────────────────┐
                     │   IKEA.COM PRODUCT PAGE  │
                     │   + Add to Cart CTA      │
                     └──────────────────────────┘
```

---

## 6.4 Handling Catalog Updates (12K Products Change Daily)

```
CATALOG UPDATE PIPELINE:

Trigger: IKEA ERP system publishes product update events to Pub/Sub
  - New product added
  - Product discontinued
  - Attribute update (price, dimensions, color)
  - Stock status change

For NEW/UPDATED products:
  1. Fetch product data (name, description, images, attributes) from catalog API
  2. Generate text embedding: Gemini text-embedding-004 API
  3. Generate image embedding: Gemini Vision on product studio images
  4. Upsert into Vertex AI Vector Search index (streaming upsert supported)
  5. Update metadata in product catalog database (Cloud Spanner)
  Latency: 2-5 minutes from ERP event to searchable in Vector Search

For DISCONTINUED products:
  1. Delete document from Vector Search index
  2. Mark as out-of-catalog in metadata store
  3. If returned in ongoing conversation: system detects and says
     "That product is no longer available, here's an alternative"

DAILY RECONCILIATION:
  - Nightly job: compare Vector Search index count vs catalog count
  - Flag discrepancies > 0.1% for manual review
  - Full index rebuild: monthly (Sunday 3am) to clear any corruption
```

---

## 6.5 Success Metrics

```
PRIMARY METRICS (North Star):
  - Conversion rate: assistant sessions → purchase
    Target: > 2× baseline (non-assistant sessions)
  - Average Order Value: assistant-influenced vs baseline
    Target: +15% AOV (cross-sell effectiveness)
  - Return rate: products bought via assistant vs search
    Target: -10% return rate (better fit recommendations)

SECONDARY METRICS:
  - Session engagement: messages per session (target: 4+)
  - Query resolution rate: % of sessions ending with product added to cart
  - Recommendation relevance: thumbs up/down on recommendations
  - P90 response latency: < 3 seconds for text queries, < 8 sec for image

GUARDRAIL METRICS (must NOT regress):
  - Customer satisfaction (CSAT): must not drop below current baseline
  - Accessibility: screen reader compatibility maintained
  - Privacy: no PII stored in conversation without consent
```

---

## 6.6 Scale Math

```
IKEA.COM TRAFFIC:
  ~3 billion visits/year = ~95 RPS average
  Peak (Black Friday, back-to-school): 10× = 950 RPS
  Assume 20% engage with AI assistant = 190 RPS peak on assistant

TEXT EMBEDDING (Gemini text-embedding-004):
  - On catalog update: 12,000 products × 2 embeddings (text+image) = 24K calls
  - 24K × ~100ms = 40 minutes for full catalog re-embed
  - Cost: 24K calls × ~$0.00013/call = $3.12 for full rebuild

VECTOR SEARCH:
  - Index size: 12K vectors × 1408 dims × 4 bytes = 67 MB (tiny)
  - Latency: <5ms ANN retrieval at this scale
  - Can run on smallest Vector Search node configuration

GEMINI API FOR CONVERSATIONS:
  - 190 RPS × avg 3 turns/session × avg 500 tokens/turn = 285K tokens/sec
  - At $0.075/1M input tokens → ~$1,850/month
  - At 20% assistant engagement on 3B visits → very manageable

CLOUD RUN (serving):
  - Autoscale 10→100 instances based on RPS
  - Cost: ~$1,200/month at average load
```

---

## 6.7 Three Key Trade-offs

```
1. Concatenated embedding vs separate indexes:
   Q: "Why concatenate text and image embeddings vs maintain separate indexes?"
   A: "For room-planning use case, concatenated captures cross-modal relationships:
      a 'blue Scandinavian' style chair should be near both blue chairs AND
      Scandinavian-style chairs. Separate indexes require score fusion (tricky
      to normalize). Concatenated embedding loses some modality specificity but
      gains holistic relevance. Alternative: ColPali-style late interaction —
      worth exploring in month 3 once we have relevance data."

2. Gemini as reranker vs fine-tuned cross-encoder:
   Q: "Is using Gemini for reranking too expensive/slow?"
   A: "At 190 RPS with avg 20 candidates each: 3,800 product descriptions
      sent to Gemini per second. That's ~2M tokens/second — too expensive
      for synchronous reranking. Solution: use a lighter cross-encoder
      (fine-tuned bi-encoder) for reranking, and Gemini only for
      final response generation (top-5 candidates). Gemini handles the
      natural language output, not the scoring of 20 items."

3. Conversation history storage:
   Q: "If a user has a 15-turn conversation and closes the browser, 
      what happens?"
   A: "Conversation stored in Cloud Firestore with TTL=7 days.
      If user returns within 7 days: 'Welcome back! You were looking for
      a blue sofa. Want to continue?' — drives re-engagement.
      After 7 days: conversation purged (privacy by default).
      Cross-device: tied to logged-in IKEA Family ID, not browser session."
```

---
---

# MASTER CROSS-SCENARIO QUESTIONS

> *These are questions the interviewer will ask to test your breadth and ability to connect scenarios.*

---

## "How would you productize all 6 systems into a single Huge platform?"

```
ANSWER FRAMEWORK:
  The 6 systems share 80% of infrastructure:
  
  SHARED COMPONENTS:
  ┌────────────────────────────────────────────────────────────┐
  │  Component              | Shared Across                   │
  │─────────────────────────┼─────────────────────────────────│
  │  Vertex AI Feature Store| Scenarios 1, 4, 5              │
  │  Vertex AI Vector Search| Scenarios 2, 3, 6              │
  │  Pub/Sub + Dataflow     | Scenarios 1, 4, 5              │
  │  LangGraph agents       | Scenarios 2, 3, 6              │
  │  BigQuery data lake     | All 6                          │
  │  Looker dashboards      | Scenarios 3, 4, 5              │
  │  Cloud Run serving      | All 6                          │
  │  Vertex AI Pipelines    | Scenarios 4, 5                 │
  └────────────────────────────────────────────────────────────┘
  
  MODULAR ARCHITECTURE:
  - Platform team at Huge builds and maintains the SHARED layer
  - Client-specific team builds only the domain logic (the 20%)
  - New client onboarding = terraform module + domain configuration
  
  This is the actual platform play for Huge India:
  "We're not selling 6 separate projects — we're selling access
   to a platform that has already solved the hard infrastructure
   problems. A new client's MMM is live in 2 weeks, not 6 months."
```

---

## "What's your biggest risk in this architecture and how do you mitigate it?"

```
TOP 3 RISKS:

1. GCP VENDOR LOCK-IN:
   Risk: Client wants to move to AWS. Entire stack is GCP-native.
   Mitigation: Abstract the ML serving layer behind internal APIs.
   BigQuery-specific SQL is the hardest lock-in — document all
   non-standard syntax. Vertex AI Pipelines use standard Kubeflow
   components — portable to any Kubeflow cluster.

2. LLM QUALITY REGRESSION:
   Risk: Gemini model update changes output format or quality.
   Mitigation: Prompt versioning in Cloud Storage. Automated eval
   suite runs on every model version change (detected via API
   version header). Canary testing with 5% traffic before full rollout.

3. DATA BREACH / CLIENT DATA LEAKAGE:
   Risk: A bug in the Agent Gateway passes client_A's query to
   client_B's RAG index.
   Mitigation: Defense in depth — gateway enforcement + VPC-SC +
   BigQuery row-level security are three independent layers.
   Monthly automated data isolation testing (red team exercise).
```

---

## "You're presenting to Huge's CEO and McDonald's CMO. What's your 1-slide summary?"

```
ANSWER (what you'd actually say):
  "We built McDonald's a system that knows what you want to order
  before you do — based on where you are, what time it is, and 
  what you've loved before. It serves 69 million customers daily,
  responds in under 200 milliseconds, and learns every time someone
  taps 'order'. In the first 90 days, we target a 12% lift in average
  basket size. That's $X million in incremental annual revenue.
  The infrastructure pays for itself in 8 weeks."

  Always anchor to BUSINESS IMPACT, not technology.
  The CMO doesn't care about Vertex AI Feature Store.
  They care about: revenue, cost, risk.
```

---

## QUICK REFERENCE: GCP Service Cheat Sheet

| Use Case | Service | Why |
|---|---|---|
| Event streaming | Pub/Sub | Managed, 99.99% SLA, global |
| Stream processing | Dataflow | Managed Apache Beam, exactly-once |
| Feature store | Vertex AI Feature Store | Training-serving consistency |
| Vector search | Vertex AI Vector Search | Managed ScaNN, <10ms |
| Model serving | Vertex AI Endpoints | Auto-scaling, A/B traffic split |
| Serverless API | Cloud Run | Per-request billing, instant scale |
| Data warehouse | BigQuery | Petabyte-scale, multi-client isolation |
| Data transforms | Dataform + dbt | SQL-native, version controlled |
| Orchestration | Vertex AI Pipelines | Kubeflow-compatible, managed |
| LLM | Gemini 1.5 Pro/Flash | State of art, within GCP ecosystem |
| Agent framework | LangGraph | Stateful, checkpointing, production-ready |
| Caching | Memorystore (Redis) | Managed Redis, VPC-native |
| NoSQL | Cloud Firestore | Real-time, mobile SDKs, TTL support |
| Dashboards | Looker + Looker Studio | BigQuery-native, embedded analytics |
| CI/CD | Cloud Build + Terraform | IaC for client onboarding automation |

---

## CLOSING: THE HUGE-SPECIFIC MINDSET

```
In every design session, weave in these Huge-specific angles:

1. "Who is the client stakeholder seeing this?"
   → Always design the business dashboard, not just the ML pipeline

2. "How does this get handed over?"
   → Document the operational runbook as part of the design

3. "What's the 90-day proof point?"
   → Huge sells in retainers; month 3 renewal depends on a visible win

4. "Can we pitch this to the next client?"
   → Every solution should have reusable components that become Huge IP

5. "What's the Huge India differentiation?"
   → Same global platform quality at local cost efficiency
   → "We can build this for $500K; a US agency would charge $2M"

The best Solutions Architects at design agencies aren't just engineers.
They're trusted advisors who speak the client's language while building
the engineer's architecture. In every answer, show you can do both.
```

---

*Document created: June 2026 | Abhishek Bhardwaj | Solutions Architect ML/AI — Huge Interview Prep*
*Total scenarios: 6 | Total components designed: 40+ | GCP services referenced: 20+*
