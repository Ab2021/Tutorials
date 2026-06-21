# 🎨 Marketing AI, Huge Domain & Solution Architecture — Interview Prep
### Abhishek Bhardwaj | Solutions Architect ML/AI @ Huge

---

> [!IMPORTANT]
> This document bridges your **marketing AI background** (Axtria MMM, attribution, CVS Health CLV) with **Huge's core business** (design + technology for brands like McDonald's, Nike, IKEA, Verizon). This is your unique differentiator — most technical candidates don't have both deep AI engineering AND marketing science expertise. Lead with this in your opening.

---

## SECTION 1: Understanding Huge's Business

---

### Q1.1: "What do you know about Huge as a company? Why do you want to work here?"

**Model Answer**:

Huge is a global design and technology company founded in 1999, headquartered in New York City. What makes Huge genuinely distinctive is that it sits at the intersection of brand strategy, experience design, and technology delivery — it's not just a tech consultancy, and it's not just a creative agency. The combination is rare and increasingly valuable.

Huge's concept of "Intelligent Experiences" is the right framing for where AI needs to go in marketing and consumer-facing applications. It's not just about efficiency (using AI to generate content faster) — it's about making every interaction smarter, more personal, and more relevant. When Nike uses Huge to design a digital experience, "intelligent" means the experience adapts to who you are, what you've done before, and what you're likely to want next. That requires the intersection of ML, product design, and data infrastructure that Huge is uniquely positioned to deliver.

I want to work at Huge specifically because my background sits exactly at this intersection. At Axtria, I built Marketing Mix Models and attribution systems for J&J and Stemline — understanding how marketing channels interact to drive outcomes. At Chubb, I built agentic AI and RAG systems. Huge is the company where both skill sets matter simultaneously: you need the marketing science depth to understand what insights clients need, and the AI engineering depth to build production systems that deliver those insights intelligently. That combination is rare, and I believe I can be genuinely valuable here from day one.

The 2025 AI Breakthrough Award and the Creativepool recognition for Agency of the Year tell me that Huge is winning on both the creative and technical dimensions — exactly the culture where I'd thrive.

---

### Q1.2: "What does 'Intelligent Experiences' mean technically?"

**Model Answer**:

"Intelligent Experiences" is Huge's framework for AI-powered digital products and services. Technically, it translates to four core capabilities:

**1. Personalization at scale**: Dynamically adapting UI, content, and recommendations to each individual based on their behavior, preferences, and context. Technical stack: feature stores for real-time user features, recommendation models (collaborative filtering + content-based), A/B testing infrastructure for continuous optimization, and LLM-powered content adaptation.

**2. Conversational and agentic interfaces**: Moving beyond click-based interactions to natural language interfaces where users achieve goals through conversation or where the system proactively acts on their behalf. Technical stack: RAG-powered knowledge bases, agentic workflows (LangGraph/Vertex AI Agent Builder), MCP-based tool integration.

**3. Predictive intelligence**: Anticipating user needs before they're expressed. Suggesting the next product, identifying churn risk, recommending the optimal next touchpoint in a customer journey. Technical stack: ML models for propensity, next-best-action, CLV; real-time feature pipelines; model serving infrastructure.

**4. Adaptive content generation**: Creating brand-consistent content at scale — personalized emails, localized ad copy, dynamic website content. Technical stack: LLMs fine-tuned or RAG-augmented with brand guidelines, content moderation, multimodal models for image/video.

**Client portfolio mapping**:

| Client | Primary Intelligent Experience | Key AI Capabilities |
|--------|-------------------------------|---------------------|
| **McDonald's** | Menu personalization, drive-through prediction | Recommendation, CLV, real-time inference |
| **Nike** | Personalized fit/style recommendations | Computer vision, embeddings, collaborative filtering |
| **IKEA** | Room planning AI, product discovery | Multimodal (image+text), recommendation |
| **Verizon** | Plan recommendation, churn prevention | Propensity modeling, CLV, NLP |
| **Google** | Broad AI capability (data infrastructure, GenAI tools) | Broad ML/AI platform work |

---

## SECTION 2: Marketing AI — Deep Technical Questions

---

### Q2.1: "How would you build MMM for McDonald's marketing across 40,000+ global locations?"

**Model Answer**:

McDonald's at 40,000+ locations across 100+ countries is a fundamentally different MMM problem than the J&J pharma MMM I built. The challenges multiply:

**Challenge 1 — Hierarchical structure**: McDonald's has global media (Super Bowl ads, global brand campaigns), regional media (US West Coast radio, APAC digital), and local media (individual restaurant promotions, local radio). A single MMM can't capture all these levels. The solution is **hierarchical Bayesian MMM** — a model that estimates both global parameters (shared across all regions) and local parameters (specific to each region/market), with partial pooling between levels (local estimates are "shrunk" toward the global estimate proportionally to the local data volume).

```python
import pymc as pm
import numpy as np

with pm.Model() as hierarchical_mmm:
    n_regions = 50  # 50 major markets
    
    # Global priors (hyperpriors)
    mu_tv_decay = pm.Beta("mu_tv_decay", alpha=3, beta=1)  # Global TV adstock rate
    sigma_tv_decay = pm.HalfNormal("sigma_tv_decay", sigma=0.1)
    
    # Regional parameters (partial pooling)
    tv_decay_region = pm.Beta("tv_decay_region", 
                               alpha=mu_tv_decay * 10, 
                               beta=(1 - mu_tv_decay) * 10,
                               shape=n_regions)
    
    # Regional sales models
    for r in range(n_regions):
        sales_r = model_sales(region_media[r], tv_decay_region[r], ...)
```

**Challenge 2 — Menu and price variation**: McDonald's menu and pricing varies by country, season, and location. The model needs to account for price elasticity (a $0.50 Big Mac increase in the US affects sales differently than the same % increase in Thailand) and product mix effects.

**Challenge 3 — Incrementality testing at scale**: At J&J, we could validate MMM with geo-level incrementality tests. At McDonald's scale, I'd design a **geo-matched experiment program**: systematically vary media spend across matched pairs of DMAs (Designated Market Areas) to generate causal estimates that validate the MMM's coefficient estimates. Google's Meridian framework specifically supports this calibration workflow.

**Tool recommendation for McDonald's**: Google's **Meridian** (open-sourced 2024) is the right starting point — it's built specifically for large-scale Bayesian MMM with hierarchical structure, incrementality test calibration, and budget optimization built in. I'd customize it with McDonald's-specific components: footfall data (transactions at individual restaurants from POS systems), competitive advertising data (fast food competitor spend), and local demographic features.

---

### Q2.2: "Compare Meridian, Robyn, and your custom approach."

**Model Answer**:

| Dimension | Meridian (Google) | Robyn (Meta) | Abhishek's Custom Approach |
|-----------|-------------------|-------------|---------------------------|
| **Framework** | Bayesian (JAX-based) | Frequentist (NSGA-II optimization) | Bayesian (PyMC) |
| **Open source** | ✅ (2024) | ✅ (2022) | Custom |
| **Hierarchical** | ✅ Strong | Limited | ✅ |
| **Experiment calibration** | ✅ Native | Limited | Manual |
| **Saturation model** | Hill, Carryover (Weibull) | Hill, Geometric adstock | Hill, Weibull, custom S-curves |
| **Budget optimizer** | Constrained optimization | NSGA-II genetic | NSGA-II multi-objective |
| **Best for** | Large-scale, GCP, many touchpoints | Fast iteration, Meta-heavy media | Complex pharmaceutical, small touchpoints |

For Huge's clients, I'd recommend **Meridian as the default** (GCP-native, Bayesian, well-maintained, Google-backed) with custom modifications for client-specific saturation curves and data sources. Robyn is excellent if the client has heavy Meta (Facebook/Instagram) spend and wants attribution that aligns with Meta's own estimates.

**Adstock models comparison**:
- **Geometric decay**: `Adstock(t) = spend(t) + λ × Adstock(t-1)`. Simple, only one parameter. Use for fast-decaying digital channels (paid search, display).
- **Weibull CDF decay**: `Adstock(t) = Σ spend(t-k) × Weibull(k; shape, scale)`. Two parameters — can model both concave and S-shaped decay. Use for TV and brand-building channels with complex carryover patterns.
- **L2-regularized delayed adstock**: Useful when you believe there's a "warmup" period before ads take effect (e.g., new product launch with slow awareness build).

---

### Q2.3: "How do you handle attribution in a post-cookie, post-ATT world?"

**Model Answer**:

Apple's App Tracking Transparency (ATT) and Google's planned deprecation of third-party cookies represent the most significant shift in digital attribution since the web was invented. The impact: identity resolution across devices and platforms becomes much harder, and deterministic multi-touch attribution (which relied on third-party cookies) loses precision.

**The unified measurement framework**:

Rather than relying on any single attribution approach, the modern answer is a **triangulation** of three methodologies:

1. **MMM (Marketing Mix Modeling)**: Always privacy-safe because it uses aggregate data (total spend by channel, total sales/conversions) without individual user data. Provides strategic budget allocation across channels. Operates weekly/monthly granularity.

2. **Incrementality experiments (geo-tests, holdout tests)**: The gold standard for causal attribution. Run controlled experiments (turn off spend in matched geographies) to measure true incremental lift. Doesn't require user tracking. Operates on campaign timescales.

3. **Modeled attribution (cookieless MTA)**: Use first-party data (server-side events, logged-in user journeys, CRM data) to build probabilistic attribution models on the identifiable fraction of users and extrapolate to the broader population. Google's Consent Mode and Facebook's Conversions API facilitate server-side event collection that doesn't require client-side tracking.

For Huge's clients, I'd architect this as a **Measurement Intelligence Platform**: a BigQuery-based data lake combining first-party signals (CRM, loyalty apps, website server-side events), experimental results, and MMM outputs into a unified dashboard that provides channel-level ROI estimates with confidence intervals. This is exactly the kind of platform Huge should offer as a value-added service.

---

## SECTION 3: GenAI for Marketing & Creative AI

---

### Q3.1: "Design a GenAI content engine for Nike's global campaigns."

**Model Answer**:

Nike operates in 190+ countries, with campaigns that must be globally consistent in brand values yet locally relevant in language, cultural references, and market specifics. A GenAI content engine addresses the $X/piece cost and weeks-long timelines of traditional content production.

**Architecture**:

```
INPUT LAYER
Campaign Brief (human-authored) → Structured brief template:
  {campaign_concept, target_audience, key_message, channels, markets, tone, restrictions}

KNOWLEDGE LAYER (RAG)
Nike Brand Guidelines Vector Store (updated with every brand refresh):
  - Brand voice principles ("Just Do It" ethos, authentic, aspirational, athlete-first)
  - Visual identity guidelines (color palette, typography, imagery style)
  - Legal restrictions (can't show injuries in aspirational context, no health claims)
  - Approved athlete endorsements and their associated brand attributes

GENERATION LAYER
Campaign Content Agent (LangGraph):
  Node 1: Market Research Agent
    → Retrieves cultural context for target markets (holidays, cultural sensitivities, sports seasons)
    → Tools: market research database, cultural sensitivity checker, sports calendar API
  
  Node 2: Content Generator
    → Generates copy variants per channel: hero headline, body copy, CTA, social posts
    → Retrieves from Brand Guidelines RAG to maintain voice
    → Generates 5 variants per market
  
  Node 3: Brand Compliance Checker
    → Scores each variant against brand guidelines (LLM-as-judge with Nike-specific rubric)
    → Flags violations (off-brand language, restricted claims, cultural sensitivity issues)
    → Returns only compliant variants
  
  Node 4: Localization Agent  
    → Translates approved variants into target market languages
    → Cultural adaptation (not just translation — adapts idioms, imagery references)
    → Native speaker review flag (for high-stakes markets)

OUTPUT LAYER
Campaign Content Package:
  - Per market, per channel: 3-5 approved copy variants
  - Brand compliance scores
  - Localization notes
  - Asset briefs for creative team (what imagery to pair with each copy variant)
```

**Brand voice consistency** is maintained through RAG over the brand guidelines + an LLM-as-judge evaluator trained on Nike's historical content (approved vs rejected). The evaluator scores each piece 0-1 on "Nike brand alignment" — pieces below 0.8 are automatically regenerated.

**A/B testing infrastructure**: Rather than guessing which variant is best, deploy all variants to a small traffic split (5% per variant), collect engagement signals (CTR, time-on-ad, conversion), and automatically promote the winner using multi-armed bandit optimization (Thompson sampling).

---

### Q3.2: "Design an agentic system for McDonald's menu personalization."

**Model Answer**:

McDonald's collects rich behavioral data through the MyMcDonald's Rewards app: order history, visit frequency, location, time of day, weather at time of order. The personalization challenge is real-time: when a customer opens the app, the recommended items should reflect who they are right now.

**Real-time personalization architecture**:

```
Event: Customer opens McDonald's app at 7:42am, Chicago, -5°C

Feature Pipeline (real-time, <100ms):
→ Customer feature lookup (Vertex AI Feature Store):
  - order_history_embedding: [item vectors from last 90 days]
  - visit_frequency: 3.2x/week
  - preferred_daypart: BREAKFAST (78% of orders)
  - avg_order_value: $6.40
  - location_cluster: Chicago_Downtown

→ Contextual features (real-time):
  - time_of_day: 7:42am → BREAKFAST
  - weather: cold (-5°C) → warm_food preference signal
  - local_promotions: "McRib is back in Chicago"
  - inventory: McRib available at nearest location ✓

Recommendation Model (<50ms):
→ Two-stage retrieval:
  Stage 1: ANN retrieval of candidate items using customer embedding (Vertex AI Vector Search)
    → Candidates: {McRib, Sausage McMuffin, Hash Browns, Oatmeal, Hot Coffee, ...}
  
  Stage 2: Re-ranking model (LightGBM trained on conversion signals):
    Features: {customer_history_affinity, weather_context_match, 
               promotion_flag, margin_optimization_weight}
    → Top 3 ranked: Hot Coffee (cold weather boost), Sausage McMuffin (breakfast preference), McRib (promotion)

Explanation Layer (LLM, optional for app display):
→ "Good morning! It's cold out there — here's what looks good right now: ..."
→ Natural language explanation of recommendation rationale
```

**My connection**: This directly maps to my CLV and propensity modeling work at CVS Health/Aetna — predicting which products individual customers are most likely to engage with based on their behavioral history. The core architecture (feature store + two-stage retrieval + re-ranking) is the same; the domain is different. For McDonald's, I'd additionally incorporate **menu item embeddings** trained on co-occurrence data (items frequently ordered together) to enable complementary recommendations ("Fries go well with your McRib").

---

## SECTION 4: HDBSCAN — Deep Dive

---

### Q4.1: "What is HDBSCAN? How does it differ from DBSCAN and K-means? When would you use it?"

**Why they're asking**: HDBSCAN is explicitly listed in the JD skills list. This is a must-nail topic.

**Model Answer**:

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that finds clusters of arbitrary shape in data with varying density, without requiring you to specify the number of clusters.

**The K-means problem**: K-means requires specifying k (number of clusters) upfront, assumes spherical clusters of roughly equal size, and is sensitive to outliers. For customer segmentation, you rarely know k beforehand, and customer segments are rarely spherically distributed in feature space.

**The DBSCAN problem**: DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds arbitrary-shape clusters but requires two hyperparameters (ε: neighborhood radius, minPts: minimum points for a core point) that must be set consistently across the entire dataset. This fails for **varying-density clusters** — a natural marketing reality where some segments are tightly packed (power users with very similar behavior) and others are diffuse (occasional shoppers with varied patterns). A single ε can't work for both.

**HDBSCAN's innovation**: It builds a hierarchy of clusters by varying ε from small to large (creating a dendrogram), then extracts the most persistent, stable clusters from this hierarchy. The "stability" measure rewards clusters that exist across a wide range of density thresholds.

**Mathematical intuition**:
1. Compute mutual reachability distance between points (accounts for varying local density)
2. Build a minimum spanning tree (Prim's algorithm) on the mutual reachability distances
3. Convert MST to a cluster hierarchy (dendrogram)
4. Extract flat clusters from the hierarchy by cutting at maximum stability (FOSC: Framework for Optimal Selection of Clusters)

**Hyperparameters**:
- `min_cluster_size`: The minimum number of points required to form a cluster. If a cluster would have fewer points, its members are classified as noise. Higher value → fewer, larger clusters.
- `min_samples`: Controls conservatism — how many neighbors are required for a point to be a core point. Higher value → more points classified as noise.
- `cluster_selection_epsilon`: Can add a minimum cluster merging distance (like DBSCAN's ε) to prevent over-splitting.

```python
import hdbscan
import numpy as np
from umap import UMAP

# Customer segmentation workflow
# Step 1: Reduce dimensionality first (HDBSCAN struggles in high dimensions)
umap_reducer = UMAP(n_components=10, min_dist=0.0, n_neighbors=30, metric='cosine')
customer_embedding_reduced = umap_reducer.fit_transform(customer_embeddings)  # n_customers × 10

# Step 2: HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,      # At least 50 customers per segment
    min_samples=10,            # Require 10 neighbors to be a core point
    cluster_selection_method='eom',  # Excess of mass: default, finds stable clusters
    metric='euclidean',        # After UMAP, euclidean is appropriate
    prediction_data=True       # Enable soft clustering
)
cluster_labels = clusterer.fit_predict(customer_embedding_reduced)

# -1 = noise (customers that don't fit any segment clearly)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = sum(1 for label in cluster_labels if label == -1)
print(f"Found {n_clusters} customer segments, {n_noise} customers unclassified")

# Soft clustering: probability each customer belongs to each cluster
soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
# soft_clusters[i, j] = probability customer i belongs to segment j
```

**HDBSCAN + UMAP for customer segmentation**: HDBSCAN in raw feature space (100+ features) doesn't work well — the curse of dimensionality makes all distances similar. The solution is to first apply UMAP for non-linear dimensionality reduction (preserves local neighborhood structure much better than PCA), then cluster the UMAP-reduced representation. This combination is the standard approach for high-dimensional customer data.

**Marketing applications**:
1. **Customer segmentation**: Finding natural behavioral segments without assuming a fixed number. Reveal power users, occasional buyers, lapsed customers, etc. as the data dictates.
2. **Anomaly detection**: Noise points (-1 labels) are potential anomalies — claims that don't fit any known pattern, customers with unusual behavior.
3. **Topic modeling in documents**: Use HDBSCAN on document embeddings to find thematic clusters in a corpus (e.g., clustering campaign copy to identify brand voice drift).
4. **RAG pipeline enhancement**: Cluster query embeddings to discover what types of questions users are asking — inform which knowledge base sections need more coverage.

**Use HDBSCAN over K-means when**:
- You don't know the number of clusters upfront
- Clusters have irregular shapes or varying densities
- Some data points genuinely don't belong to any cluster (noise is meaningful)
- You want soft cluster memberships (probability of belonging to each cluster)

**Use K-means when**:
- You have a specific business reason to want exactly k segments
- Data is roughly spherically distributed
- Computational efficiency is critical (K-means is much faster for very large datasets)
- Cluster centroids are meaningful (K-means centroids represent the "average" customer in each segment)

⚠️ **Trap**: "Have you used HDBSCAN directly?"  
Answer: "I've used density-based clustering approaches in customer segmentation work. For the CLV prediction and propensity modeling at CVS Health, I explored DBSCAN-family approaches for anomaly detection in customer behavioral data. I've studied HDBSCAN specifically and would apply it in preference to K-means for the customer segmentation work at Huge — particularly for clustering customer journey embeddings where the natural number of segments is unknown."

---

## SECTION 5: Solution Architecture for Digital Experiences

---

### Q5.1: "Design a real-time personalization platform for IKEA's e-commerce."

**Model Answer**:

IKEA's e-commerce challenge is unique: they sell large, considered-purchase items (furniture) where the customer journey is long (weeks to months), cross-channel (website, app, showroom visits), and highly contextual (I need a sofa for a specific room with specific dimensions). Pure recommendation engines fail here — they'd recommend more sofas to someone who just bought a sofa.

**Architecture**:

```
DATA FOUNDATION
IKEA First-Party Data (CDP — Customer Data Platform):
  - Online behavior: product views, saved items, room planner sessions
  - Transactional: purchase history, return history
  - Showroom: mobile app check-ins at stores, staff interactions
  - Life events (inferred): new home signals, life stage changes

Real-time event stream (Pub/Sub / Kafka):
  - Product view events
  - Cart additions/removals
  - Room planner saves
  - Search queries

FEATURE LAYER (Vertex AI Feature Store)
Customer features (refreshed hourly via Dataflow):
  - room_planning_stage: {browsing, planning, ready_to_buy}
  - style_affinity_vector: embeddings of liked/saved products
  - home_profile: {apartment/house, estimated_size, ownership}
  - purchase_cycle_stage: {research, comparison, decision}

INTELLIGENCE LAYER
1. Intent Detection Model:
   "Is this customer browsing inspiration or actively shopping?"
   → Classifies session intent: DISCOVERY | PLANNING | PURCHASING
   → Routes to appropriate recommendation strategy

2. Recommendation Engine (strategy varies by intent):
   DISCOVERY: Semantic style clustering (HDBSCAN on product embeddings) → show diverse style inspiration
   PLANNING: Complementary item recommendations (room set completion)
   PURCHASING: Similar-but-cheaper alternatives, financing options, in-stock at nearest store

3. Generative Room Visualization (multimodal):
   "Show me this sofa in my room" → AI-powered room visualization
   → Customer uploads room photo → Stable Diffusion inpainting places the furniture

4. Conversational Shopping Assistant (Agentic RAG):
   "I need a dining table for 6 people, max 180cm wide, under $1500"
   → Agent: queries product database → checks measurements → filters budget → presents options
   → Handles: follow-up questions, availability, delivery time, complementary items

SERVING LAYER
- Vertex AI Endpoints: recommendation models
- Cloud Run: conversational agent (auto-scales, serverless)
- Cloud CDN: pre-computed recommendations for popular item combinations
- Redis: session-level feature cache (<10ms feature retrieval)
```

**The key insight for IKEA**: Personalization must respect the **purchase cycle**. Someone who viewed sofas 3 times in the past week is in a different intent state than someone who viewed sofas once 6 months ago. The intent detection model is the most critical component — getting the right content to the right intent state matters more than the recommendation algorithm itself.

---

## SECTION 6: Client-Facing Presentation Skills

---

### Q6.1: "Explain RAG to a McDonald's marketing VP who has never heard of AI."

**Model Answer** (business-friendly language, no jargon):

"Imagine you hired the world's best research analyst. They've read every internal document McDonald's has ever produced — every campaign brief, every market research report, every brand guidelines document, every performance dashboard. When you ask them a question, they don't just guess — they instantly flip to the exact pages that answer your question, read them, and give you a precise, cited answer in plain English.

That's essentially what RAG does. We're not teaching the AI to memorize facts — we're giving it access to your company's knowledge library and training it to find the right documents and synthesize the answer for you.

In practice, for McDonald's this could mean: a marketing manager asks 'What did we learn about breakfast daypart performance in the Midwest markets from our 2023 campaigns?' Instead of spending hours searching through reports or emailing analysts, they get a precise answer in 30 seconds, with links to the source documents.

The magic is that the AI is always working from your actual data — it can't hallucinate facts it found in your own documents. Every answer comes with citations. Your team is still making the decisions; the AI just does the research faster and more completely than any human could."

---

### Q6.2: "How do you present a technical architecture to a US-based client executive?"

**Model Answer** (frameworks for US business communication):

US executives value directness and business outcomes over technical depth. My approach:

**The Pyramid Principle**: Start with the answer (what we're building and why it matters to the business), then support with logic (how it works at a high level), then detail (technical specifics for those who want them). Never lead with technical architecture.

**The "So What?" Test**: After every technical statement, ask "so what for the business?" Example: don't say "We're using a hybrid dense-sparse retrieval system with cross-encoder re-ranking." Say: "Our search system finds the exact relevant information from thousands of documents in under a second, and double-checks it for accuracy before showing it to users — this means your team spends time on decisions, not document hunting."

**The 3-level explanation**:
- **For the CEO**: "This system cuts your marketing team's time-to-insight from days to minutes, and the recommendations are based on your actual company data, not guesses."
- **For the CMO**: "We index all your campaign history, brand guidelines, and market research, and the AI pulls exactly the relevant pieces to answer any marketing question. It cites sources so you can verify."
- **For the technical lead**: "RAG architecture with Vertex AI Vector Search, hybrid BM25+dense retrieval, cross-encoder re-ranking, served via Cloud Run with P95 latency < 800ms."

**Handling pushback**: When a US client pushes back on an AI recommendation ("We tried AI before, it didn't work"), the response is: "Tell me more about that — what went wrong?" Listen fully, then address their specific concern. Don't defend AI in the abstract; defend the specific approach based on their specific concern. Clients trust architects who listen before prescribing.

---

## SECTION 7: Emerging Trends for Huge

---

### Q7.1: "Gemini vs GPT-4o vs Claude 3.5 — which would you recommend for Huge's GenAI stack?"

**Model Answer**:

I'd recommend a **multi-model strategy** with Gemini 1.5 Pro / Gemini 2.0 as the primary model, for these reasons:

**Why Gemini/Vertex AI for Huge**:
1. **GCP-native**: Huge's infrastructure appears GCP-centric. Vertex AI provides Gemini access with native integrations (Cloud Storage, BigQuery, Vertex AI Pipelines) — no cross-cloud data transfer, unified IAM, unified billing.
2. **Long context**: Gemini 1.5 Pro's 1M-token context window is genuinely differentiated for long-document analysis (entire campaign histories, full brand guidelines in context).
3. **Multimodal**: Gemini's native multimodal capabilities (text + image + video + audio) are critical for a design company — analyzing creative assets, understanding campaign visuals, generating multimodal content.
4. **Cost**: Vertex AI provides committed use discounts and enterprise pricing; Gemini on Vertex is competitive with OpenAI enterprise pricing.

**When to use Claude (Anthropic via Vertex)**:
- Complex reasoning, long-form content generation where Claude's writing quality is superior
- Code generation tasks (Claude 3.5 Sonnet is the best coding model)
- Available on Vertex AI (Anthropic partnership) — no infrastructure complexity

**When to use GPT-4o**:
- Specific use cases where clients have existing OpenAI agreements
- Function calling and structured outputs (OpenAI's implementation remains the most mature)

**My architectural recommendation**: Vertex AI Model Garden as the abstraction layer — it hosts Gemini, Claude (Anthropic on Vertex), Llama 3, and other models behind a unified API. Route to the optimal model per task using semantic routing. This prevents vendor lock-in and lets you optimize model selection by task type.

---

### Q7.2: "What is the EU AI Act and how does it affect Huge's clients?"

**Model Answer**:

The EU AI Act (fully effective August 2026) is the world's first comprehensive AI regulation, applying to any AI system used in the EU. Huge's European clients (IKEA is Swedish, many others have EU operations) must comply.

**Risk-based classification**:
- **Unacceptable risk** (banned): Biometric mass surveillance, social scoring
- **High risk**: AI in hiring, credit scoring, healthcare diagnosis, insurance (relevant to Chubb background), law enforcement — requires conformity assessment, data governance documentation, human oversight mechanisms
- **Limited risk**: Chatbots (must disclose AI to users), deepfakes (must label)
- **Minimal risk**: Most recommendation engines, spam filters — no obligations

**Implications for Huge's deliverables**:
1. **Chatbots for EU users** (McDonald's EU, IKEA EU): Must disclose "you're talking to an AI" — UI design consideration
2. **Personalization systems**: If using EU resident data for personalization, GDPR + AI Act compliance: right to explanation, right to object, no profiling for certain vulnerable groups
3. **AI-generated content**: Must label deepfake/synthetic content — campaigns using AI-generated visuals in EU markets need disclosure
4. **High-risk AI systems for clients**: If Huge builds an AI hiring tool or insurance scoring tool for a client, significant compliance burden — model documentation, human oversight, bias testing

**Practical advice for Huge**: Build an **AI governance checklist** into the project onboarding process for EU-market clients — classify the AI system by risk level, document compliance requirements, and design the human oversight mechanism from the start rather than retrofitting it.

---

*End of Marketing AI & Huge Domain Document*

---

> [!TIP]
> Your killer opening line: *"I'm the candidate who bridges the gap between the data science that builds the intelligence and the marketing science that understands what intelligence clients actually need. My Axtria work gives me the domain depth — I've literally built the models that measure ROI across every marketing channel. My Chubb work gives me the production AI depth. Huge is where both come together."*
