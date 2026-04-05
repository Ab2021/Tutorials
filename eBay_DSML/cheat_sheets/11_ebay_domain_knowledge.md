# eBay DS/ML — eBay Domain Knowledge Deep Dive

> Knowing eBay's business, technology, and challenges shows genuine interest
> and helps you give context-aware answers in every round.

---

## 🏢 Part 1: eBay Company Profile

### Key Facts (2025)
| Metric | Value |
|---|---|
| **Active Buyers** | 130+ million globally |
| **Active Sellers** | 18+ million |
| **Live Listings** | 1.9+ billion |
| **GMV (Annual)** | ~$73 billion |
| **Revenue (Annual)** | ~$10.1 billion |
| **Markets** | 190+ countries |
| **Founded** | 1995, San Jose, CA |
| **CEO** | Jamie Iannone |
| **Bengaluru GCC** | 4,000+ employees, AI-first engineering hub |

### Business Model
```
Revenue Sources:
├── Transaction Fees (Take Rate ~13.5%)
│   ├── Final Value Fee (% of sale price)
│   └── Payment Processing Fee
├── Promoted Listings (Advertising)
│   ├── Cost-per-click ads
│   └── Cost-per-sale ads
├── Store Subscriptions
│   └── Basic / Premium / Anchor / Enterprise
└── Other (shipping labels, managed payments)
```

### Strategic Pillars (2024-2026)
1. **Reimagining the Platform** — Focus on enthusiast buyers (luxury, collectibles, auto parts)
2. **AI-First Commerce** — GenAI for listings, search, discovery, and seller tools
3. **Trusted Marketplace** — Authenticity guarantees, condition grading, buyer protection
4. **Category Focus** — Vertical deep-dives: Sneakers, Watches, Trading Cards, Auto Parts

---

## 🔍 Part 2: How eBay Search Works

### The "Best Match" Algorithm

```
User Query: "vintage rolex submariner"
         │
         ▼
┌──────────────────────┐
│ QUERY UNDERSTANDING   │
│ ├── Spell correction  │
│ ├── Query expansion   │ "vintage rolex submariner" → 
│ ├── Intent detection  │   + "rolex sub" + "rolex 5513"
│ └── Entity extraction │   entities: {brand: Rolex, model: Submariner}
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ CANDIDATE RETRIEVAL   │  ← Billions → Thousands
│ ├── Inverted index    │  keyword matching
│ ├── Vector search     │  semantic embedding (eBERT)
│ └── ANN (FAISS)       │  approximate nearest neighbor
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ SCORING / RANKING     │  ← Thousands → Hundreds
│ ├── Relevance score   │  query-item semantic match
│ ├── Listing quality   │  images, description, attributes
│ ├── Seller quality    │  rating, ship time, return rate
│ ├── Price competitive │  vs category median
│ ├── Conversion prob   │  P(click), P(purchase)
│ └── Personalization   │  user history, preferences
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ RE-RANKING            │  ← Hundreds → Page (20-50)
│ ├── Promoted listings │  paid ads inserted
│ ├── Diversity rules   │  avoid same seller dominating
│ ├── Freshness boost   │  newer listings get exposure
│ └── Trust signals     │  authenticity guarantee, top-rated
└──────────────────────┘
```

### Key Ranking Features
- **Text relevance:** eBERT cosine similarity between query and title embeddings
- **Behavioral signals:** Historical CTR, conversion rate for this query-item pair
- **Item quality:** Number of photos, attribute completeness, description length
- **Seller signals:** Average rating, % positive feedback, on-time shipping rate
- **Price signal:** Price vs median in category, free shipping flag
- **Recency:** Days since listing, freshness decay factor
- **User personalization:** Past categories, price range, brand preferences

---

## 🛡️ Part 3: eBay Fraud & Trust

### Types of Fraud eBay Fights

| Type | Description | ML Approach |
|---|---|---|
| **Account Takeover** | Hacker accesses legitimate account | Device fingerprinting, login anomaly detection |
| **Shill Bidding** | Fake bids to inflate auction price | Graph analysis, bid pattern detection |
| **Transaction Fraud** | Stolen payment methods | Real-time risk scoring, velocity features |
| **Counterfeit Goods** | Fake branded items | Image classification, price anomaly, seller history |
| **Non-Delivery** | Seller never ships | Shipping event monitoring, seller risk score |
| **Feedback Manipulation** | Fake positive reviews | NLP sentiment analysis, reviewer graph analysis |
| **Fraud Rings** | Coordinated fake buyer-seller networks | Graph Neural Networks (GNN), community detection |

### xFraud — eBay's Fraud Detection Framework
- **Architecture:** GNN-based heterogeneous graph model
- **Nodes:** Users, devices, IP addresses, payment methods, listings
- **Edges:** Relationships (bought from, logged in with, used payment, etc.)
- **Key feature:** Explainability — generates human-readable explanations for why a transaction was flagged
- **Real-time:** <100ms inference per transaction

---

## 🤖 Part 4: eBay's AI Technology Stack

### Internal AI Platforms

| System | Purpose | Technology |
|---|---|---|
| **Krylov** | GPU cluster for ML training | Distributed PyTorch, multi-tenant |
| **NuKV** | Key-value store for embeddings | Cloud-native, low-latency |
| **eBERT** | eBay-pretrained BERT | Domain-specific NLP understanding |
| **MicroBERT** | Distilled eBERT for production | Knowledge distillation, 10x faster |
| **eBayCoder** | Internal coding assistant | Fine-tuned Code Llama |
| **e-Llama** | eBay's custom LLM | Fine-tuned Llama for marketplace tasks |
| **Mercury** | Agentic RAG platform | LLM + tools + product retrieval |
| **Triton** | GPU inference serving | NVIDIA Triton, batch/stream inference |

### Data Infrastructure
- **Batch:** Apache Spark, Airflow for ETL, data lake on cloud storage
- **Stream:** Apache Kafka for event streaming, Flink for stream processing
- **Storage:** Hadoop/HDFS legacy, cloud data lake (Parquet/Delta), NuKV
- **Experimentation:** Internal A/B testing platform, supports geo and user-level randomization

---

## 📊 Part 5: eBay's Key Business Challenges

### 1. Marketplace Liquidity
**Problem:** Need enough buyers AND sellers in every category for healthy marketplace dynamics.
**DS opportunity:** Supply-demand matching models, seller onboarding optimization, category expansion recommendations.

### 2. Search Relevance at Scale
**Problem:** 1.9B listings, highly diverse inventory (new+used, $1 → $100K), varied seller quality.
**DS opportunity:** Embedding-based semantic search, personalized ranking, zero-result query mining.

### 3. Trust & Safety
**Problem:** Open marketplace = attracting bad actors (counterfeits, scams, fraud rings).
**DS opportunity:** Real-time fraud scoring, graph-based fraud ring detection, image-based counterfeit classification.

### 4. Seller Experience
**Problem:** Complex listing process, pricing uncertainty, competition from simpler platforms.
**DS opportunity:** AI listing generation, dynamic pricing suggestions, seller analytics dashboards.

### 5. Buyer Engagement
**Problem:** Competing with Amazon for convenience, with niche apps for specific categories.
**DS opportunity:** Personalized recommendations, agentic shopping assistant, price tracking/alerts.

### 6. Structured Data
**Problem:** Many listings have unstructured titles/descriptions, making filtering and comparison hard.
**DS opportunity:** NLP-based attribute extraction, image-based category classification, listing quality scoring.

---

## 🗣️ Part 6: eBay Interview Power Phrases

Use these naturally in your answers to show domain awareness:

| Situation | Power Phrase |
|---|---|
| Discussing scale | *"With 1.9 billion live listings across 190 markets..."* |
| System design | *"We'd store precomputed embeddings in NuKV for sub-millisecond retrieval..."* |
| Search | *"The Best Match algorithm needs to balance relevance, seller fairness, and ad revenue..."* |
| Fraud | *"A GNN-based approach like xFraud captures the interconnected nature of fraud rings..."* |
| GenAI | *"eBay's Mercury platform uses agentic RAG for hyper-personalized recommendations..."* |
| Two-sided marketplace | *"Any change needs to balance buyer and seller incentives — we can't optimize one side at the expense of the other..."* |
| Experimentation | *"In a marketplace, SUTVA violations from shared supply mean we might need geo-based clustering rather than user-level randomization..."* |
| Production ML | *"I'd train on Krylov, serve embeddings via NuKV, and deploy the ranking model on Triton for GPU inference..."* |
| Metrics | *"GMV, conversion rate, and seller NPS are the primary health metrics for the marketplace..."* |

---

## 📖 Part 7: Must-Read Before Interview

### eBay Tech Blog Posts (prioritize these)
1. How eBay uses AI for search ranking
2. Krylov: eBay's ML training platform
3. NuKV: Cloud-native key-value store for ML serving
4. eBay's approach to GenAI: multi-track strategy
5. xFraud: Explainable fraud detection using GNN
6. Two-tower models for personalized recommendations
7. GitHub Copilot and eBayCoder: developer productivity

### Recommended Reading Order (Day Before Interview)
```
Morning:  eBay investor presentation (latest quarterly) — understand GMV trends
Midday:   Tech blog: Krylov + NuKV + search ranking posts
Evening:  Review your STAR stories + system design framework
Night:    Light review of SQL window functions + Python pandas cheat sheet
```
