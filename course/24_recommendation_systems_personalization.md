# Module 24 — Recommendation Systems and Personalization

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 30 hours (14 lecture, 10 lab, 6 project)
> **Track fit:** Retail & E-commerce track; open to all
> **Prerequisites:** Modules 06, 09, 17, 21
> **Tools:** Python, implicit/ALS or LightFM, PyTorch, DuckDB, A/B test tooling
> **Industry question:** "What should we show this customer, given what we have, what it costs, and what we want to sell?"

---

## 1. Why industry cares

Recommendation systems are usually taught as a media problem. In supply chain, they are a **demand-shaping and inventory-allocation system**:

- A recommendation changes what is bought, not only what is clicked.
- Recommending an item that is out of stock in the customer's node creates a broken promise.
- Recommending a high-margin, slow-moving item can reduce markdowns.
- A substitution recommendation can save a sale when the first choice is unavailable.
- A personalized assortment can increase basket size but also complexity and fairness risk.

The supply chain question is not only "what will this customer like?" It is "what should we show, given availability, margin, capacity, and the promise we can keep?"

## 2. Learning outcomes

By the end of this module, you can:

- Frame a recommendation problem as ranking, retrieval, or substitution.
- Build popularity, content-based, collaborative, and hybrid recommenders.
- Explain matrix factorization and two-tower retrieval.
- Handle cold start, sparsity, and feedback loops.
- Evaluate offline with ranking metrics and online with experiments.
- Add business constraints: availability, margin, diversity, and freshness.
- Design inventory-aware recommendations and substitutions.
- Measure the causal effect on demand, basket, and margin.
- Assess fairness, privacy, and filter-bubble risk.
- Evaluate LLM-based recommenders and their failure modes.

## 3. Mental model

```text
   CANDIDATES ──> RANK ──> CONSTRAIN ──> EXPLAIN ──> SERVE
       │            │          │            │          │
   popularity    score      availability  reason     API/cache
   content       model      margin        copy       experiment
   collaborative  features   diversity     trust      feedback
       │            │          │            │          │
       └────────────┴──────────┴────────────┴──────────┘
                          FEEDBACK LOOP
                    impressions → clicks → orders → returns
```

A recommender is a funnel: retrieve many candidates, rank them, apply business rules, and serve. Most production failures happen in the constraints and feedback loop, not in the model.

### 3.1 Recommendation tasks

| Task | Question | Metric |
|---|---|---|
| Retrieval | Which 1,000 items from 1M? | Recall@k |
| Ranking | Which order for this user? | NDCG, MAP |
| Substitution | What replaces an unavailable item? | Acceptance, margin |
| Cross-sell | What goes with the basket? | Basket lift |
| Next-best-offer | What promotion? | Incremental margin |
| Replenishment | When will they need it again? | Repeat rate |

## 4. Core concepts

### 4.1 Popularity and content-based baselines

Always build a baseline:

- **Global popularity:** most sold items.
- **Segment popularity:** by category or region.
- **Content-based:** similarity by attributes (brand, category, size, price).
- **Recently viewed:** session context.

A complex model that does not beat a popularity baseline is not ready.

### 4.2 Collaborative filtering

User-based and item-based nearest neighbors use the user-item matrix.

| Method | Strength | Weakness |
|---|---|---|
| User-based | Intuitive | Scalability |
| Item-based | Stable | Popularity bias |
| Matrix factorization | Latent factors | Cold start |
| Implicit ALS | Implicit feedback | No negatives |

For implicit feedback, confidence weighting (ALS) is common:

\[
\min_{X,Y} \sum_{u,i} c_{ui}(p_{ui} - x_u^T y_i)^2 + \lambda(\|X\|^2 + \|Y\|^2)
\]

where \(p_{ui}=1\) for observed interactions and \(c_{ui}\) is confidence.

### 4.3 Two-tower retrieval

A user tower and an item tower produce embeddings; retrieval is nearest-neighbor search.

| Component | Inputs |
|---|---|
| User tower | History, demographics, context |
| Item tower | Attributes, text, image |
| Loss | Sampled softmax or contrastive |
| Serving | ANN index |

Two-tower models scale to millions of items and support cold items through content features.

### 4.4 Sequential and session models

- GRU4Rec and transformer-based sequential models capture order.
- Session-based recommendation handles anonymous users.
- Next-item prediction uses the last \(k\) interactions.

Sequence models are useful when intent changes quickly, as in grocery or fashion.

### 4.5 Cold start

| Cold-start type | Approach |
|---|---|
| New user | Popularity, context, onboarding |
| New item | Content similarity, exploration |
| New market | Transfer, local popularity |
| New session | Session-based model |

Cold start is a data problem before it is a model problem.

### 4.6 Evaluation

Offline:

| Metric | Meaning |
|---|---|
| Recall@k | Share of relevant items retrieved |
| Precision@k | Share of retrieved items relevant |
| NDCG@k | Rank-weighted relevance |
| MAP | Mean average precision |
| Coverage | Share of catalog recommended |
| Novelty | Non-obvious recommendations |
| Diversity | Variety within a list |

Online:

| Metric | Meaning |
|---|---|
| CTR | Click-through rate |
| Conversion | Orders per impression |
| Basket size | Units or value |
| Margin | Contribution per session |
| Return rate | Quality of fit |
| Long-term retention | Repeat purchase |

Offline gains often fail online. Pre-register the online metric and run a proper experiment.

### 4.7 Feedback loops and bias

- **Popularity bias:** popular items get more exposure, becoming more popular.
- **Position bias:** top slots get more clicks regardless of relevance.
- **Selection bias:** only observed interactions are recorded.
- **Exposure bias:** the model cannot learn about items it never shows.

Controls: inverse propensity weighting, exploration, unbiased evaluation, and holdout traffic.

### 4.8 Business constraints

| Constraint | Implementation |
|---|---|
| Availability | Filter by node ATP |
| Margin | Adjust score by contribution |
| Inventory | Boost or suppress slow stock |
| Diversity | Max items per brand/category |
| Freshness | Boost new items |
| Fairness | Exposure constraints |
| Contract | Minimum placement |

Constraint-aware ranking is often a re-ranking step, not a change to the model.

### 4.9 Inventory-aware recommendation

Recommendation changes demand, which changes inventory. Close the loop:

1. Score items by relevance.
2. Filter by availability in the customer's node.
3. Adjust by margin and markdown risk.
4. Cap exposure for items with fragile supply.
5. Substitute when unavailable.
6. Measure the downstream inventory effect.

A recommendation that sells out a low-supply item can destroy the experience for the next customer.

### 4.10 Substitution

When an item is unavailable, offer a substitute:

- Same attributes and price band.
- Available in the node.
- Acceptable to the customer.
- Profitable.

Measure substitution acceptance, not only click rate. A substitute that is returned is not a save.

### 4.11 LLM-based recommenders

LLMs can:

- Generate explanations.
- Parse natural-language intent.
- Recommend from a catalog with retrieval.
- Personalize copy.

Risks:

- Hallucinated products.
- Ignoring inventory.
- Prompt injection from reviews.
- Cost and latency.
- Inconsistent output.

Use retrieval with structured filters and validate every recommendation against the catalog.

### 4.12 Privacy and fairness

- Minimize personal data.
- Explain why an item is shown.
- Avoid sensitive attributes.
- Provide opt-outs.
- Audit exposure by demographic group where lawful.
- Do not infer sensitive traits.

### 4.13 Evaluation, worked

A holdout of 10,000 sessions, each with one purchased item. The baseline recommends the ten most popular items in the category.

| System | Recall@10 | NDCG@10 | Coverage | Margin per session |
|---|---:|---:|---:|---:|
| Global popularity | 0.061 | 0.048 | 0.4% | €1.42 |
| Segment popularity | 0.074 | 0.059 | 2.1% | €1.51 |
| Item-item collaborative | 0.112 | 0.084 | 8.7% | €1.63 |
| Two-tower + re-rank | 0.128 | 0.097 | 14.2% | €1.71 |

The two-tower model wins offline, but the decision is not settled. Check whether the gain survives when the re-ranker sees only items that are in stock at the customer's node. Check whether coverage improved because the model explores, or because it recommends long-tail items that are frequently returned. Then run an online experiment with margin per session and return rate as pre-registered metrics.

A useful rule: report the baseline, the candidate, and the candidate under production constraints. The gap between the last two is the real model contribution.

### 4.14 Two-tower architecture, worked

| Layer | User tower | Item tower |
|---|---|---|
| Inputs | Last 20 interactions, segment, node, session context | Attributes, text, image embedding, price band |
| Embedding | 64–256 dimensions | Same dimension |
| Training | Sampled softmax with in-batch negatives | Shared item embeddings |
| Serving | ANN index over item vectors | Refreshed on catalogue change |
| Constraint | Filter by node ATP after retrieval | Availability flag in the index |

The architecture is simple; the hard parts are negative sampling, freshness, and the join between the embedding space and the inventory system. If the item tower is refreshed weekly but stock changes hourly, the retriever will confidently recommend items that cannot be shipped.

### 4.15 Reference data model

| Table | Grain | Key fields | Use |
|---|---|---|---|
| impression | request × item | user_id, node_id, position, model_version | Exposure and bias |
| interaction | user × item × time | event_type, session_id, dwell | Training |
| order_line | order × item | sku, quantity, margin, return_flag | Business outcome |
| item_attributes | sku | category, brand, price, size, substitute_group | Content and cold start |
| availability | sku × node × hour | atp, lead_time | Constraint filter |
| experiment | user × variant | assignment, exposure, conversion | Causal measurement |

The `impression` table is the one most teams omit. Without position and model version, offline evaluation cannot correct for exposure bias and online gains cannot be attributed.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Grain | Use |
|---|---|---|
| Impressions | Event | Exposure and CTR |
| Clicks | Event | Engagement |
| Orders | Order line | Conversion |
| Returns | Return | Quality |
| Catalog | Item | Content features |
| Inventory | Item × node | Availability |
| Margin | Item | Business score |
| Customer profile | Customer | Personalization |

### 5.2 Schema for events

```text
impression_id, session_id, customer_id, item_id, position,
timestamp, device, node_id, model_version, score
```

Log the model version and score. Without them, you cannot debug or evaluate.

### 5.3 Workflow

1. Define the business objective and constraint.
2. Build a baseline.
3. Build retrieval and ranking.
4. Add constraints and substitution.
5. Evaluate offline on temporal holdout.
6. Run an online experiment.
7. Monitor exposure, conversion, returns, and inventory.
8. Retrain and re-evaluate.

## 6. Industry case studies

### 6.1 Amazon — item-to-item recommendations

**Context:** Amazon popularized item-to-item collaborative filtering.
**Intervention:** Similar-item recommendations on product pages.
**Evidence:** The 2003 Linden, Smith, and York paper describes the approach; current systems are far more complex.
**Outcome:** A canonical retail recommender.
**What transferred:** item-to-item similarity is scalable and effective.
**What did not:** the 2003 method is a baseline, not the state of the art.

### 6.2 Netflix — ranking and experimentation

**Context:** Netflix uses recommendations to drive engagement.
**Intervention:** Personalization, ranking, and a strong experimentation culture.
**Evidence:** Netflix technology blog and public talks describe the system; metrics are company claims.
**Outcome:** A reference for online evaluation.
**What transferred:** offline metrics are not enough; experimentation is the decision.
**What did not:** media consumption differs from physical product availability.

### 6.3 Instacart — substitution and availability

**Context:** Grocery delivery must handle unavailable items.
**Intervention:** Substitution recommendations and customer choice.
**Evidence:** Company engineering blog and public materials describe the problem.
**Outcome:** A supply-chain-native recommendation problem.
**What transferred:** availability and substitution are first-class ranking constraints.
**What did not:** grocery substitution tolerance varies by category.

### 6.4 Alibaba — personalization at scale

**Context:** Large-scale e-commerce personalization.
**Intervention:** Deep interest networks and ranking systems.
**Evidence:** Public papers from Alibaba describe the methods; performance is company-reported.
**Outcome:** A reference for large-scale retail ranking.
**What transferred:** user behavior sequences improve ranking.
**What did not:** the infrastructure cost is prohibitive for small retailers.

## 7. Labs

### Lab 24.1 — Baseline recommender

Build popularity, item-item, and content-based recommenders on a public retail dataset. Evaluate Recall@10 and NDCG@10 on a temporal holdout.

**Deliverable:** Code, metrics table, and error analysis.

### Lab 24.2 — Implicit ALS

Train an implicit ALS model. Compare to item-item. Measure coverage and novelty.

**Deliverable:** Model, metrics, and trade-off note.

### Lab 24.3 — Constraint-aware re-ranking

Add availability and margin constraints to a ranking. Measure the change in conversion proxy, margin, and catalog coverage.

**Deliverable:** Re-ranker and trade-off table.

### Lab 24.4 — Substitution

Build a substitution model for unavailable items. Evaluate acceptance and return rate in a simulated or historical setting.

**Deliverable:** Substitution model and evaluation.

### Lab 24.5 — Online experiment design

Design an A/B test for a recommender change. Define the primary metric, guardrails, sample size, and stopping rule.

**Deliverable:** Experiment plan and power calculation.

## 8. Assignment / project

Design a recommendation system for a retail or e-commerce decision.

**Required components:**

1. Business objective and constraints.
2. Data and event schema.
3. Baseline and candidate models.
4. Offline evaluation with temporal holdout.
5. Constraint-aware ranking.
6. Online experiment design.
7. Monitoring and feedback-loop controls.
8. Privacy and fairness assessment.
9. Inventory impact analysis.

**Length:** 10–14 pages plus code.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Problem framing | 15% | "Recommend products" | Task and metric defined | Business constraint and supply link |
| Baselines | 15% | None | Popularity and content | Beats baseline with uncertainty |
| Model | 20% | Off-the-shelf | Tuned collaborative model | Retrieval and ranking design |
| Evaluation | 20% | Random split | Temporal holdout | Online design and bias controls |
| Constraints | 15% | Ignores availability | Filters stockouts | Margin, diversity, inventory loop |
| Ethics | 10% | Generic | Privacy noted | Fairness and explainability |
| Communication | 5% | Descriptive | Clear | Decision-ready |

## 10. Regulatory, ethical, and commercial lens

- **Privacy:** personalization uses behavioral data; minimize and disclose.
- **Consumer protection:** recommendations must not mislead about availability or price.
- **Fairness:** exposure can disadvantage sellers or groups.
- **Competition:** self-preferencing may be scrutinized.
- **Dark patterns:** urgency and scarcity claims must be truthful.
- **EU AI Act:** certain recommender uses may be in scope; assess with counsel.
- **Labor:** warehouse demand effects of demand shaping.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Popularity spiral | Same items everywhere | Feedback loop | Weak discovery | Exploration |
| Offline-online gap | Good NDCG, flat CTR | Proxy mismatch | Wasted work | Online test |
| Stockout recommendations | Cancellations | No ATP filter | Broken promise | Availability constraint |
| Margin blindness | High revenue, low margin | Revenue objective | Profit loss | Contribution score |
| Cold-start neglect | New items never shown | No exploration | Catalog stagnation | Content and exploration |
| Position bias | Top slots win | Logged clicks | Biased model | Propensity weighting |
| Filter bubble | Low diversity | Narrow personalization | Customer fatigue | Diversity constraint |
| Hallucinated products | LLM recommends nonexistent item | No catalog validation | Trust loss | Retrieval and validation |
| Return blindness | High return rate | Fit not modeled | Margin loss | Return-aware objective |
| Privacy overreach | Sensitive inference | Proxy features | Legal risk | Feature review |

## 12. Interview and discussion questions

1. What is the difference between retrieval and ranking? *(Hint: candidate generation vs ordering.)*
2. Why is a popularity baseline necessary? *(Hint: it is hard to beat and cheap.)*
3. How do you evaluate a recommender offline? *(Hint: temporal holdout, ranking metrics, coverage.)*
4. Why do offline gains fail online? *(Hint: proxy mismatch, position bias, feedback.)*
5. How do you handle cold start? *(Hint: content, popularity, exploration.)*
6. What is a feedback loop? *(Hint: exposure changes future data.)*
7. How do you make recommendations inventory-aware? *(Hint: ATP filter, margin, supply caps.)*
8. How do you measure substitution success? *(Hint: acceptance and return rate.)*
9. What fairness risks exist? *(Hint: exposure, seller equity, sensitive inference.)*
10. How would you use an LLM safely? *(Hint: retrieval, catalog validation, no free generation.)*
11. What constraints would you add for a grocery app? *(Hint: availability, substitutions, perishability.)*
12. What is the business metric? *(Hint: contribution, not CTR.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Candidate generation | Retrieving a shortlist of items |
| Ranking | Ordering candidates for a user |
| Collaborative filtering | Using user-item interactions |
| Matrix factorization | Latent user and item factors |
| Two-tower model | Separate user and item encoders |
| Implicit feedback | Clicks, views, purchases without ratings |
| Cold start | New user or item with little data |
| Recall@k | Share of relevant items in top k |
| NDCG | Rank-weighted relevance |
| Coverage | Share of catalog recommended |
| Novelty | Non-obvious recommendations |
| Diversity | Variety in a recommendation list |
| Position bias | Top slots get more attention |
| Feedback loop | Model output shapes future data |
| Exploration | Showing uncertain items to learn |
| Substitution | Replacing an unavailable item |
| Re-ranking | Adjusting order with business rules |
| ATP | Available to promise |
| Guardrail metric | Metric that must not degrade |
| Self-preferencing | Favoring own products |

## 14. Sources and further reading

### Standards and regulation

- EU AI Act, Regulation (EU) 2024/1689. https://eur-lex.europa.eu/
- FTC, endorsement and advertising guidance. https://www.ftc.gov/
- GDPR. https://eur-lex.europa.eu/

### Peer-reviewed and professional

- Linden, G., Smith, B., York, J., "Amazon.com Recommendations: Item-to-Item Collaborative Filtering," *IEEE Internet Computing*, 2003.
- Koren, Y., Bell, R., Volinsky, C., "Matrix Factorization Techniques for Recommender Systems," *Computer*, 2009.
- Rendle, S. et al., "BPR: Bayesian Personalized Ranking," UAI, 2009.
- Covington, P., Adams, J., Sargin, E., "Deep Neural Networks for YouTube Recommendations," RecSys, 2016.
- Hidasi, B. et al., "Session-based Recommendations with Recurrent Neural Networks," ICLR, 2016.
- Castells, P. et al., "Novelty and Diversity in Recommender Systems," *Recommender Systems Handbook*.

### Industry

- Netflix Technology Blog. https://netflixtechblog.com/
- Instacart Engineering Blog. https://tech.instacart.com/
- Alibaba research papers on DIN and ranking.
- Amazon Science. https://www.amazon.science/

### Tools

- `implicit` (ALS).
- LightFM.
- PyTorch.
- FAISS for nearest-neighbor retrieval.
- DuckDB for event analysis.

> **Next:** [Module 25](25_product_economics_leadership.md) turns these capabilities into a funded, adopted product.
