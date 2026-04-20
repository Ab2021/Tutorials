# 🚀 Flipkart — Advanced & Emerging Projects (2025–2026)
### The Full Picture: What Flipkart Is Actively Building Beyond Core E-Commerce

> **Purpose:** Deep research-backed briefing on Flipkart's most exciting frontier AI projects.
> Know these cold — they signal you understand where the company is *going*, not just where it *is*.
> Use these in conversation to show strategic awareness and research depth.

---

## 🗺️ THE MACRO PICTURE: FLIPKART'S "AI-FIRST" TRANSFORMATION

Flipkart in 2025–2026 is not iterating — it is **restructuring from the ground up** for an IPO,
unified under an initiative called **"OneTech"** and driven by an **"AI Transformation Charter"**.

```
CPTO: Balaji Thiagarajan
AI Charter Lead: Hemant Badri (appointed April 2026)

Quote (CPTO Balaji Thiagarajan):
"We are changing the engines of a flying plane —
replacing legacy infrastructure while maintaining full business continuity."

Approach: AI "sidecar" services run alongside legacy systems,
           gradually taking over functionality without downtime.

Investment: Sixfold increase in AI investment by mid-2025
Hiring: 5,000+ new employees planned for AI and quick commerce
```

**Three Strategic Pillars of the AI Transformation Charter:**

| Pillar | What It Means |
|---|---|
| **Intuitive Customer Experience** | Conversational commerce (SLAP), hyper-personalization, immersive video discovery |
| **Seller Empowerment** | AI cataloging, GenAI content generation, demand forecasting tools, agentic seller assistant |
| **Operational Efficiency** | Supply chain AI, quick commerce Flipkart Minutes, automated back-end, human-in-the-loop |

---

## PROJECT 1: FLIPKART MINUTES — 10-MINUTE QUICK COMMERCE ML ENGINE

### What It Is

Flipkart Minutes (launched August 2024) is Flipkart's entry into quick commerce —
competing with Blinkit, Zepto, and Instamart. The target: **10-minute delivery** for groceries,
electronics, medicines, and daily essentials from a network of AI-optimized dark stores.

**Scale (as of April 2026):**
- Dark store network expanding from 400 → **1,200 dark stores** by mid-2026
- Coverage: Metros + aggressively expanding into Tier-2/3 cities
- Product catalog: Grocery + high-margin categories (electronics, fashion, medicines)

---

### The ML Stack for Quick Commerce

#### Problem 1: Hyper-Local Demand Forecasting (The Hardest Problem)

```
WHAT MAKES IT HARD:
- Must forecast at SKU × Dark Store × 30-minute bucket level
- "Which dark store should stock how many bananas in Koramangala on Tuesday morning?"
- Demand varies by: weather, local events, day-of-week, cricket match,
                   festival proximity, competitor promotions
- Sparse data: new dark stores have < 2 weeks of history

FORECASTING APPROACH — 3-Layer Hierarchy:
┌──────────────────────────────────────────────────────────────┐
│  Layer 1: City-Level Demand Trend (LightGBM + SARIMA)        │
│  Features: city-wide order volume, festival flags, weather   │
│                                                              │
│  Layer 2: Dark Store × Category (Prophet + location signals) │
│  Features: local residential density, competitor presence,   │
│            historical order patterns, geofence population    │
│                                                              │
│  Layer 3: SKU-Level (LSTM / Temporal Fusion Transformer)     │
│  Features: item velocity, days since restock, price,         │
│            promotion flag, shelf life constraints            │
└──────────────────────────────────────────────────────────────┘

KEY INNOVATION: Temporal Fusion Transformer (TFT) for SKU forecasting
- Attention mechanism learns which past time steps matter most
- Handles mixed covariates: static (item weight, category) + dynamic (price, promotions)
- Outputs probabilistic forecast (10th, 50th, 90th percentile) → safety stock calculation
```

#### Problem 2: Real-Time Order Routing & Dispatch Agent

```
TRIGGER: Customer places order on Flipkart Minutes app

ROUTING DECISION (<500ms):
1. Which dark store? (nearest with sufficient stock)
2. Which delivery partner? (availability, proximity, rating)
3. What's the promised ETA? (route optimization + current load)

REAL-TIME OPTIMIZATION:
├── SKU availability check: Redis lookup (< 10ms)
├── Dark store proximity: Geospatial index (< 20ms)
├── Delivery partner assignment: Auction-based dispatch model
│   (partners bid availability; model selects optimal match)
├── Route optimization: Modified TSP with Google Maps API
│   Real-time rerouting every 3 min based on traffic updates
└── ETA prediction: XGBoost model (features: distance, time of day,
                    weather, partner rating, current load)
```

#### Problem 3: Dark Store Micro-Inventory Optimization (In-Store AI)

```
INSIDE THE DARK STORE (fulfillment optimization):
- AI-optimized picking paths: high-velocity items at front + center
- Re-slotting: Weekly rearrangement based on velocity change
- Picking assistant: Worker app shows optimal item sequence
- Out-of-stock prediction: Flag SKU running low before stockout

TARGET: Order packing ready in < 90 seconds
(picking path optimization cuts ~30 seconds vs. unoptimized layout)
```

### Common Challenges — Flipkart Minutes

| Challenge | Why It's Hard | Approach |
|---|---|---|
| **New dark store cold start** | < 2 weeks of local demand history | Transfer learning from nearest similar dark store; cluster-based prior |
| **Intermittent demand** | Many SKUs sell 0-1 units/hour | Zero-inflated Poisson models; stocking minimum viable assortment |
| **Under-10-min SLA pressure** | Any model delay = bad delivery experience | Strict latency budgets; precomputed routing; fallback heuristics |
| **Perishable inventory spoilage** | Wrong forecast → wastage (especially fresh groceries) | Loss-asymmetric optimization (overstock = loss; stockout = larger loss) |
| **Driver supply-demand mismatch** | Rush orders in rain = high demand + low supply | Surge pricing (incentive layer) + predictive driver positioning pre-surge |

---

### 🔗 Your Bridge
> "Flipkart Minutes is a demand forecasting problem at extreme spatio-temporal granularity.
> My hierarchical MMM work at Axtria decomposes revenue by channel/region with seasonality —
> same principle. The key difference at Flipkart Minutes is the time granularity (30-minute
> buckets vs. weekly) and the constraint that being wrong has an immediate physical cost —
> either wasted perishables or a missed 10-minute SLA."

---

## PROJECT 2: SUPER.MONEY — AI-FIRST FINTECH ARM

### What It Is

Launched August 2024, super.money is Flipkart's standalone fintech app — now the **#5 UPI app
in India by transaction volume** (mid-2025). It's a credit-first platform embedded into the
Flipkart ecosystem.

**Products:**
- **superUPI** — rewards-based UPI payments (instant cashback on merchant transactions)
- **Credit-on-UPI** — RuPay co-branded credit cards (partners: Axis Bank, Kotak811, Utkarsh SFB)
- **superDeposit** — Fixed deposits from small amounts
- **Personal Loans** — pre-approved, instant, via lending partners
- **BharatX acquisition (early 2025)** — bought checkout-financing startup → owns entire credit stack now

---

### The AI/ML Stack for super.money

#### Credit Underwriting — "Bureau-Light" Model

```
INDIA-SPECIFIC CHALLENGE:
- 40% of India is "thin file" or "new to credit" — no CIBIL score
- Traditional bureaus are insufficient for young/rural users
- super.money has access to Flipkart's behavioral data → first-party signal is GOLD

CREDIT RISK MODEL:
┌────────────────────────────────────────────────────────────────┐
│                SUPER.MONEY UNDERWRITING ENGINE                 │
│                                                                │
│  Data Sources:                                                 │
│  ├── Flipkart purchase history: frequency, categories, returns │
│  ├── Payment behavior: EMI history, on-time payment rate       │
│  ├── UPI transaction patterns: spend velocity, merchant mix    │
│  ├── App engagement: active usage, feature adoption            │
│  ├── KYC quality signals: PAN, Aadhaar verified, selfie match  │
│  ├── Device trust: phone model, SIM age, app install history   │
│  └── Bureau (if available): CIBIL/Experian score as one feature│
│                                                                │
│  Model: LightGBM ensemble (fast inference, interpretable)      │
│  Output: Credit limit (₹500 – ₹2L) + default probability      │
│  Explainability: SHAP values for regulatory audit trail        │
│                                                                │
│  Special: REAL-TIME limit assignment (< 2 seconds at checkout) │
│  User buys ₹3000 item → system approves Buy Now Pay Later      │
│  in real-time using live behavioral signals                    │
└────────────────────────────────────────────────────────────────┘
```

#### Dynamic Credit Limit & Risk Management

```
POST-APPROVAL (continuous monitoring):
- Daily: update behavioral features (spend pattern, repayment)
- Weekly: re-score user → adjust revolving credit limit up/down
- Trigger-based: missed EMI → immediate limit freeze + collection alert

Collection Propensity Model:
- Who among delinquent users is most likely to repay with outreach?
- Features: past repayment response, contact timing, channel preference
- Output: Priority score → auto-call vs. SMS vs. in-app nudge
- Optimizes limited collection bandwidth across thousands of cases
```

#### Fraud Detection — UPI Transaction Level

```
Every UPI transaction from super.money goes through:
1. Device trust scoring (is this the user's regular device?)
2. Merchant risk scoring (new merchant, high-risk category?)
3. Velocity check (N+ UPI transactions in last hour?)
4. Social engineering detection (unusual payment pattern + call recently?)
5. Amount anomaly: >3σ from user's historical distribution → step-up auth

Goal: < 100ms scoring; <0.1% fraud rate (UPI industry benchmark)
```

### Common Challenges — super.money

| Challenge | Description | Approach |
|---|---|---|
| **Thin-file users** | No credit bureau history for ~40% of India | First-party Flipkart behavioral features as primary signal |
| **Regulatory compliance** | RBI guidelines on BNPL, mandatory KYC, digital lending norms | Explainable models (SHAP), mandatory human review for credit decisions |
| **UPI fraud velocity** | Fraudsters probe UPI for gaps faster than models update | Real-time rule layer + ML; 15-min model update cycle for velocity features |
| **Adverse selection** | Users who need credit most are highest risk | Cohort-based underwriting; first-small-then-grow limit progression |
| **Data freshness** | Behavioral signals must be minutes-fresh for checkout approval | Kafka streaming → Redis feature store → sub-minute freshness |

---

### 🔗 Your Bridge
> "The super.money credit underwriting problem is the consumer fintech analogue of Adhish's
> EMI risk modeling — same core challenge: use first-party behavioral signals to underwrite
> thin-file users where bureau data is sparse. My insurance fraud work used a similar philosophy:
> claims behavioral signals (not just static policyholder attributes) drove the most predictive features."

---

## PROJECT 3: ONETECH — AI-FIRST PLATFORM MODERNIZATION

### What It Is

OneTech is Flipkart's internal initiative to **unify engineering, product, and data teams**
and replace decade-old legacy infrastructure with a modern, AI-native architecture.
This is the invisible but most important project — it enables everything else.

```
THE PROBLEM ("Changing Engines in a Flying Plane"):
Flipkart's legacy systems were built for a 2014-era marketplace.
They were not designed for:
- LLM inference at 350M user scale
- Real-time personalization (batch pipeline → near-real-time)
- Agentic AI workflows requiring multi-service orchestration
- DPDP Act (India's data protection law) compliance requirements
- IPO-ready governance standards

THE APPROACH — AI "Sidecar" Architecture:
┌─────────────────────────────────────────────────────────────┐
│                   ONETECH MIGRATION PATTERN                  │
│                                                             │
│  Legacy System A  ←────── Runs in parallel ──────→         │
│          ↕                                                  │
│  [AI Sidecar Service] ← Intercepts traffic (dark launch)   │
│  - New AI-powered version of same functionality             │
│  - Runs alongside legacy: 1% → 10% → 50% → 100% traffic   │
│  - Fallback: any error → route to legacy automatically     │
│                                                             │
│  Advantage:                                                 │
│  ✓ No big-bang migration risk                               │
│  ✓ A/B test new vs. legacy continuously                     │
│  ✓ Gradual data flywheel for new models                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Technical Components Being Rebuilt

#### 1. Real-Time Data Platform (Batch → Streaming)

```
OLD: User browsing data → daily batch job → recommendations update next day
NEW: User browsing data → Kafka → Spark Streaming → recommendations update in seconds

Architecture:
User Event (click, search, purchase)
    │
    ▼ (< 100ms)
Kafka Topic [user-events]
    │
    ├── Spark Streaming job → update online feature store (Redis)
    │   (velocity features, session context, recent clicks)
    │
    ├── Flink job → update personalization model in near-real-time
    │   (FTRL incremental update on recommendation model)
    │
    └── BigQuery sink → offline feature store for model training

IMPACT: Recommendations now reflect real-time intent within same session
        (user searched "gaming laptop" → homepage immediately shifts to gaming)
```

#### 2. Proprietary GPU Farm for AI Inference

```
PROBLEM: Running LLM inference for 350M users on 3rd-party APIs = 
         massive cost + latency + data privacy risk (IP leakage)

SOLUTION: In-house GPU farm (in partnership with Adani data centers)

Purpose-built for:
├── LLM inference at scale (SLAP shopping assistant, seller AI)
├── Embedding model serving (search, RAG, recommendation)
├── Real-time re-ranking model serving (search + recommendations)
└── Triksha adversarial testing workloads (massive parallel probing)

Infrastructure:
├── AdaniConneX Chennai (existing Tier-IV data center since 2021)
├── Second AI-focused data center (under development, 2026)
│   Purpose-built for high-density compute + liquid cooling
│   Powered by renewable energy (Adani's Khavda solar plant)
└── Google Cloud (burst capacity for peak loads like Big Billion Days)

Governance:
├── DPDP Act compliance: User data never leaves Indian data centers
├── PCI compliance: Payment data isolated with hardware security modules
└── IPO-ready audit trails: All AI decisions logged + explainable
```

#### 3. Unified Agentic Framework

```
PROBLEM: Every team building their own agent architecture →
         fragmentation, duplicated tool definitions, inconsistent safety

SOLUTION: Internal agentic framework (standardized across teams)

Components:
├── Tool Registry: Centralized catalog of approved tools
│   Each tool: schema-validated inputs, rate-limited, access-controlled
├── Agent Template Library: Pre-built ReAct, Plan-and-Execute, MRKL templates
├── Safety Layer: Mandatory input/output validation for all agents
│   (Triksha integration — automatic adversarial scanning on deploy)
├── Observability: Full trace logging (thought → action → observation)
│   Searchable by query, user segment, failure mode
└── Cost Manager: Token budget enforcement; small model routing for simple queries

USED BY: SLAP (shopping), Seller AI, Fraud Investigator, Analytics Agent
```

---

## PROJECT 4: VIDEO COMMERCE & CREATOR CITIES — AI LAYER

### What It Is

Flipkart saw **200M users engage with video in H1 2025** (up from 75M the year before),
with a **17x increase in daily livestream engagement YoY**. This is now a strategic pillar.

**Creator Cities Initiative (June 2025):**
- 18,000 sq. ft. of dedicated studio infrastructure across Mumbai, Bengaluru, Gurugram
- 300+ production experts, hosting 200+ creators monthly
- Goal: Make shoppable video the primary product discovery surface

---

### The AI Stack Behind Video Commerce

#### 1. Real-Time Trend Detection → Shoppable Moments

```
PROBLEM: A fashion trend goes viral on Instagram at 8pm.
         Flipkart must surface matching products on its platform by 8:05pm.

PIPELINE:
External Trend Signals (social media, fashion editorials, celebrity posts)
    │
    ▼
Trend Detection Model (NLP + Vision):
- Extract trend entity ("neon green sneakers", "boho maxi dress")
- Classify trend velocity (slow burn vs. viral spike)
    │
    ▼
Catalog Matching:
- Map trend to Flipkart product catalog via multimodal embedding similarity
- Find closest visual + semantic matches
    │
    ▼
Dynamic Curation:
- Generate "trending now" shoppable page automatically
- Personalize product sequence per user within the trend page
    │
    ▼
TTM (Trend-to-Market): < 5 minutes from viral signal to live shoppable page
```

#### 2. AI-Powered Live Streaming (2026 Frontier)

```
AI LIVE HOSTS (launched 2026):
- Synthetic AI presenters run 24/7 livestreams without human fatigue
- Real-time product catalog sync: product goes out-of-stock → AI host pivots instantly
- Dynamic script generation: LLM generates live commentary based on products
- Viewer engagement adaptation: if engagement drops → AI host switches product/style

RECOMMENDATION DURING LIVE STREAM:
- Real-time "show similar" and "buy this look" during stream
- Multimodal stream analysis: detect product being shown → surface catalog match
- Purchase trigger: "4 people just bought this" → social proof injection
- Personalization by viewer: same stream, different products surfaced per user
```

#### 3. Video Recommendation Algorithm

```
UNLIKE YOUTUBE (maximize watch time):
Flipkart video optimization target: maximize Purchase Probability × Engagement

Multi-objective recommendation:
- Primary signal: add-to-cart during or within 30min of watch
- Secondary: watch completion rate, replay rate
- Guardrail: don't over-optimize for purchase at cost of bad UX

Model: Two-Tower Deep Neural Network
- User tower: user embedding (purchase history, video watch history, demographics)
- Video tower: video embedding (products shown, creator, category, style)
- Match: cosine similarity in shared embedding space → ranked feed
- Online update: FTRL (Follow The Regularized Leader) on real-time engagement signals

Diversity injection:
- Pure relevance → user gets trapped in same category bubble
- Add exploration budget: 20% of feed = slightly-outside-comfort-zone content
- Serendipity metric: track cross-category discovery rate as health signal
```

### Common Challenges — Video Commerce

| Challenge | Description | Approach |
|---|---|---|
| **Video cold start** | New creator/video has no engagement history | Content-based embedding only initially; fast ramp via early signal boosting |
| **Engagement ≠ purchase** | Videos that entertain don't always sell | Separate engagement model + purchase model; optimize weighted combination |
| **Trend latency** | By the time AI identifies trend, it's peaking | Pre-position inventory for predicted viral categories based on leading indicators |
| **AI host authenticity** | Synthetic hosts may feel uncanny → trust deficit | Hybrid: human co-host + AI segments; brand disclosure mandated |
| **Stream reliability** | Live stream + real-time inventory sync = many failure points | Pre-buffered segments; graceful degradation; inventory cache with 5-min staleness OK |

---

## PROJECT 5: V3 STRATEGY — VERNACULAR, VIDEO, VOICE FOR BHARAT

### What It Is

Flipkart's next 200 million users are in Tier-2/3 India. They speak **11 Indian languages**,
shop on low-end Android devices, and prefer voice over typing. This requires fundamentally
different AI than English-first systems.

**The V3 Framework:**
- **Vernacular** — 11 language support with Hinglish transliteration
- **Video** — Creator Cities, short-form video discovery (above)
- **Voice** — Multimodal voice-first commerce interface

---

### The AI Stack for Bharat

#### 1. Vernacular NLP — Beyond Translation

```
PROBLEM: Simple translation fails — "mujhe accha phone chahiye under 15k"
         Translation gives: "I want a good phone under 15k"
         But the cultural context: "accha" in electronics = reliable brand,
         budget is ₹15,000, "good" includes service center availability

APPROACH:
├── Language Detection: FastText multilingual classifier (11 languages)
├── Transliteration: Hinglish (Hindi in Roman script) model → Hindi text
├── NLU (Natural Language Understanding):
│   IndicBERT / MuRIL (multilingual BERT fine-tuned on Indian languages)
│   Fine-tuned on: Flipkart search queries in regional languages
│   Task: Intent classification + entity extraction in native language
├── Cultural Context Adaptation:
│   Domain-specific fine-tuning: "local brands", regional size standards
│   (Indian clothing sizes ≠ Western; regional electronics brands matter)
└── Code-Switching: Handle mid-sentence language switches
    "Yeh phone ka battery life kitna hai? Is it good for gaming?"
```

#### 2. Voice Commerce Interface

```
MULTIMODAL INPUT PIPELINE:
Voice Input (regional accent + dialect)
    │
    ▼
Speech-to-Text → IndicWave2Vec / Whisper fine-tuned on Indian accents
    │
    ▼
NLU → Intent + Entities (regional language model)
    │
    ▼
Action (search, filter, add to cart, ask question)
    │
    ▼
TTS Response → Natural-sounding regional language voice
    (using regional language TTS models for 11 languages)

DESIGN PRINCIPLES:
- Handle background noise (street noise, family sounds) → robust ASR
- Low-bandwidth: model inference on-device where possible (TensorFlow Lite)
- Graceful fallback: voice fail → text; text fail → menu-based navigation
```

#### 3. Inclusive AI — Low-End Device Optimization

```
PROBLEM: Next 200M users have ₹5000-10000 phones; 2G/3G connectivity in rural areas

SOLUTIONS:
├── Model Quantization: Full precision → INT8 quantized models
│   (4x smaller, 3x faster on mobile CPUs; < 5% accuracy loss)
├── Progressive Web App: Critical features work offline/cached
├── Image Compression: Product images optimized per connection speed
├── Edge Inference: Simple NLU models run on-device
│   (reduces latency from 500ms server → 100ms local inference)
├── Predictive Prefetch: Pre-download likely-next content based on browse pattern
└── Adaptive UI: Feature degradation based on device capability detection
```

### Common Challenges — V3/Bharat

| Challenge | Description | Approach |
|---|---|---|
| **Accent diversity** | 11 languages × dozens of regional accents | Large-scale accent-diverse ASR training data; user-specific adaptation |
| **Code-switching** | Mid-sentence language switch → parser fails | Joint multilingual NLU model; language-agnostic intermediate representation |
| **Low-resource languages** | Limited training data for Tamil, Odia, Assamese | Cross-lingual transfer from Hindi/English; data augmentation |
| **Informal language** | "Yaar, koi sasta gaming phone?" → informal → slang | Large corpus of informal e-commerce chat; colloquial training data |
| **On-device model constraints** | Quantized models lose accuracy on complex queries | Cascade: try on-device first, fallback to server if confidence low |

---

## PROJECT 6: AI DATA GOVERNANCE & DPDP COMPLIANCE

### What It Is

India's **Digital Personal Data Protection (DPDP) Act** came into full effect in 2025, requiring:
- Explicit user consent before collecting/processing personal data
- Data minimization (collect only what's needed)
- Right to erasure (user can request deletion of their data)
- Data localization (certain sensitive data cannot leave India)

For an AI-first company like Flipkart, this is a **massive engineering challenge**.

---

### Technical Implementation for AI Systems

#### 1. Consent Management & Data Lineage

```
CONSENT MANAGEMENT PLATFORM:
Every data point used in ML models must be traceable to a user consent event

├── Consent Store: Tracks per-user, per-purpose consent status
│   purpose: search_personalization | recommendation | fraud_detection | ads
│   Given: Y/N + timestamp + version of policy user agreed to
│
├── Data Lineage DAG:
│   For every ML model feature, trace the lineage:
│   User clicks → raw event store → feature engineering → model feature
│   Each hop: which consent purpose covers this flow?
│
├── Automated Consent Check: At model training time
│   Filter training data to only include users with active relevant consent
│   Alert: if consent change affects > 5% of training set → retrain trigger
│
└── Right to Erasure Pipeline:
    User requests deletion →
    1. Delete from raw event stores (Kafka, data lake)
    2. Remove from feature store
    3. Retrain models if affected user was in training set (or use unlearning techniques)
    4. Audit log: prove deletion completed
```

#### 2. Machine Unlearning (Frontier Research)

```
PROBLEM: "Forget" a specific user from a trained model without full retraining
         Full retraining every time a user requests deletion = too expensive

APPROACHES BEING EXPLORED:
├── Influence Functions: Identify which training samples most influenced model weights
│   Remove those samples mathematically (approximation of retraining)
├── SISA Training (Sharded, Isolated, Sliced, Aggregated):
│   Train model on shards; when deletion requested, retrain only affected shard
├── Differential Privacy: Train with DP-SGD → individual user influence bounded by ε
│   Deletion requests can't leak individual data → no retraining needed
└── Federated Unlearning: For on-device models, push model update
    that "forgets" specific user's contribution without server data

STATUS: Active R&D area; not yet fully productionized at Flipkart's scale
```

---

## PROJECT 7: AGENTIC COMMERCE — THE NEXT FRONTIER

### What Flipkart Is Building Toward

**"Agentic Commerce"** is Flipkart's stated 2026-2027 vision:
AI agents that don't just answer questions — they **act on your behalf** across the
full shopping lifecycle.

```
CURRENT STATE (2026): Conversational AI (you ask, it explains/recommends)
FUTURE STATE (2027+): Agentic AI (you authorize, it executes autonomously)

EXAMPLE AGENTIC COMMERCE FLOW:
User: "Find me a good washing machine under ₹25,000 with same-week delivery,
       compare top 3, buy the best one, and schedule installation"

Agent Actions:
1. search_products("washing machine", filters={budget: 25000})
2. check_availability(top_results, user_pincode, delivery_mode="this_week")
3. get_reviews_summary(product_ids=[...])  → LLM synthesis
4. compare_products(top_3)  → structured comparison table
5. get_user_payment_preference(user_id)  → preferred EMI / UPI
6. place_order(product_id, payment_method)  → confirm with user
7. schedule_installation(order_id, user_preferred_slot)

CHALLENGES BEING SOLVED:
├── Authorization scope: How much authority does user give the agent?
│   (view only? add to cart? purchase? set spending limit?)
├── Error recovery: Order placed wrong item → agent-initiated return?
├── Trust calibration: High-value purchases need human confirmation
└── Multi-session memory: Agent remembers user's past preferences across sessions
```

### Agentic Commerce Technical Stack

```
MEMORY SYSTEMS (Enabling Multi-Session Agents):
├── Short-term: Session context (Redis, in-memory)
├── Medium-term: User preference store (past product interactions, explicit preferences)
└── Long-term: User profile graph (purchase history, size/brand preferences, household)

PLANNING & EXECUTION:
├── Task decomposition: LLM plans sub-tasks from user natural language goal
├── Tool use: Function calling / Code Interpreter for data operations
├── Reflection: After each step, agent evaluates if on-track; re-plans if not
└── Safety checks: High-value actions (purchase > ₹5000) require explicit user confirmation

GUARDRAILS (Critical for consumer trust):
├── Spending limits: User sets max per-session spend authority for agent
├── Category restrictions: "Only buy from my wishlist" or "only electronics"
├── Undo capability: Agent actions reversible within 30 min (order cancellation)
└── Audit trail: Every agent action logged; user can review agent's reasoning
```

---

## 📊 SUMMARY: FLIPKART'S FRONTIER AI PROJECT MAP

```
┌─────────────────────────────────────────────────────────────────────────┐
│              FLIPKART ADVANCED PROJECTS — QUICK REFERENCE               │
│                                                                         │
│  🏃 FLIPKART MINUTES     → Quick commerce ML: hyper-local demand        │
│     Status: Live (800+ dark stores); Scaling to 1,200                  │
│     Key ML: Temporal Fusion Transformer, dark store routing agent       │
│                                                                         │
│  💳 SUPER.MONEY          → Fintech arm: bureau-light credit AI          │
│     Status: Live (#5 UPI app in India)                                  │
│     Key ML: First-party behavioral credit model + BharatX credit stack  │
│                                                                         │
│  🔧 ONETECH              → Platform modernization + AI infrastructure   │
│     Status: Active (multi-year transformation)                          │
│     Key: GPU farm (Adani data centers), sidecar migration, real-time    │
│                                                                         │
│  🎬 VIDEO COMMERCE       → Creator Cities + AI live streams             │
│     Status: 200M video users H1 2025; AI hosts launching 2026           │
│     Key ML: Trend detection, multimodal rec, synthetic AI hosts         │
│                                                                         │
│  🇮🇳 V3 / BHARAT AI      → Vernacular + Voice + Video for Tier-2/3     │
│     Status: 11 languages live; voice commerce expanding                 │
│     Key ML: IndicBERT, multilingual ASR, on-device quantized models     │
│                                                                         │
│  🔒 DPDP COMPLIANCE      → Data governance for AI systems               │
│     Status: Required by law (2025); active build                        │
│     Key: Consent tracking, data lineage, machine unlearning R&D         │
│                                                                         │
│  🤖 AGENTIC COMMERCE     → Full lifecycle autonomous shopping agents    │
│     Status: R&D / Early pilots; target 2027 broad rollout               │
│     Key: Multi-session memory, spending authority, undo capability      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 HOW TO USE THESE IN THE INTERVIEW

### Power Phrases to Drop Naturally

| Context | What to Say |
|---|---|
| When asked about Flipkart's strategy | *"The OneTech initiative — rebuilding the entire platform as AI-native while maintaining business continuity — that's the hardest engineering challenge I've seen: changing engines in a flying plane."* |
| When asked about quick commerce | *"Flipkart Minutes' hyper-local demand forecasting at the pincode × 30-minute granularity — that's where Temporal Fusion Transformers really shine. The spatio-temporal version of what I built for forecasting at Axtria."* |
| When asked about fintech | *"super.money's bureau-light credit model using first-party Flipkart behavioral signals to underwrite thin-file users — that's a brilliant example of data moat monetization. Directly parallels the EMI risk modeling work Adhish published."* |
| When asked about future of AI shopping | *"Agentic commerce — where the agent acts across the full shopping lifecycle, not just recommends — that's what I believe is the 2-year trajectory. It's where my Agentic BI work at Chubb has been heading."* |
| On India-specific challenge | *"The V3 strategy is brilliant — Vernacular, Video, Voice — because the next 200M users aren't English-first and aren't desktop-first. Every ML system needs to be designed for a ₹8000 phone on 2G. That's a hard constraint that changes architecture fundamentally."* |

---

## 🔑 KEY NUMBERS TO REMEMBER

| Fact | Number |
|---|---|
| Flipkart AI investment increase | **6x** in mid-2025 |
| Video users H1 2025 | **200M** (from 75M prior year) |
| Livestream engagement YoY growth | **17x** |
| Dark stores (target mid-2026) | **1,200** |
| super.money UPI app rank in India | **#5 by transaction volume** |
| Indian languages supported | **11** |
| Creator Cities studio space | **18,000 sq. ft.** across 3 cities |
| Hiring commitment for AI/QC | **5,000+** employees |
| Adani data center investment | Part of **$100B** commitment by 2035 |
| SLAP launch date | **January 2026** |
| Minivet AI acquisition | **December 2025** |

---

*End of Flipkart Advanced Projects Document*
