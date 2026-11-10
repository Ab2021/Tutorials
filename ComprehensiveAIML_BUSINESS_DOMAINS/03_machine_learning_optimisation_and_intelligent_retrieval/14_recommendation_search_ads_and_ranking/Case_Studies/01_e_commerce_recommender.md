---
case_id: "CASE-M14-01"
case_title: "E-commerce recommender: retrieval upgrade versus full re-architecture"
case_version: "1.0-draft"
created: "2026-11-10"
last_reviewed: null
status: "draft"
content_status: "draft"
source_status: "OPEN"
case_type: "predictive-ml"
domain: "E-commerce and marketplaces — recommendation retrieval and ranking"
primary_pattern: "retrieval-ranking"
secondary_patterns: ["experimentation", "metric-alignment"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Recommendation product owner, with merchandising and finance sign-off"
source_index: "INDEX.md"
review_trigger: "Any change to the contribution-margin definition, returns-cost attribution, eligibility rule, or the serving latency budget"
specialization: "recommendation retrieval and ranking (module 14)"
---

# E-commerce recommender: retrieval upgrade versus full re-architecture

> **Reading note on evidence.** Every quantity in this file — traffic, catalog size, model scores, experiment results, costs — is an **illustrative teaching value** built on a coherent arithmetic spine. The arithmetic has been recomputed and is internally consistent; none of it is a measurement of any company, product, or team. Where the prose states a mechanism rather than a number, the claim is a **DERIVED** consequence of the stated model. Nothing here is a `published` company result and nothing is `observed`. The file is a design-and-decision exercise whose numbers a reader can re-derive with a calculator.

## 1. The One-Line Framing

Given **one browsing session on a commerce site, evaluated at the moment a product rail must render** (unit of analysis: session, with item-level candidates inside it), decide **which eight eligible stock-keeping units to place in a ranked recommendation rail**, for **the category merchandising owner and the recommender platform team**, to improve **contribution margin per session net of returns and fulfilment cost**, subject to **a 250 ms p99 serving budget, catalog-eligibility rules, and a returns-rate guardrail that may not rise by more than 50 basis points**, compared with **the incumbent co-visitation retrieval plus gradient-boosted-tree ranker**.

The task category is **retrieval then ranking** (candidate generation followed by pointwise-to-listwise scoring), executed at request time, roughly seventy times per second on average and about 267 times per second at peak. It is not a batch scoring job, not a forecasting problem, and not a generative task; no text is produced and no free-form recommendation is written.

The primary success condition in plain language: **shipping the change must raise contribution margin per session without pushing the returns rate or the latency budget past their guardrails.** The full metric definition, including numerator, denominator, population, window, and comparator, appears in Section 9. The headline decision in this case is not "does the model rank better" but "which of two plausible shipping configurations should go to peak-season traffic."

Primary reusable pattern: `retrieval-ranking`. Secondary patterns: `experimentation` (the decision is only resolvable with a properly powered online test) and `metric-alignment` (the offline ranking metric and the commercial objective disagree, and the disagreement is the whole case).

**The supported surprise, stated up front.** The configuration with the stronger offline ranking metric — full two-tower retrieval plus a retrained ranker — is the configuration that **loses contribution margin**, while the configuration with the weaker offline metric — retrieval upgrade with the incumbent ranker left untouched — **gains** it. The offline metric improves by roughly 12.6% and 11.5% respectively in unconditional terms; the commercial outcomes move in opposite directions (+1.46% and −3.80%). A team that gated on NDCG alone would have shipped the losing arm with confidence.

**Why the obvious simpler framing misses the point.** The obvious framing is "recommendations should be more relevant, relevance is measured by click-through or NDCG, so maximise NDCG." That framing is blind to three commerce mechanics that dominate the economics: a recommendation that changes *which* products sell changes the **product mix** and therefore the blended gross margin; a recommendation that surfaces higher-priced or higher-variance products changes the **returns rate**, and returns destroy the margin on the sale while leaving the fulfilment cost spent; and a recommendation that changes basket composition changes **fulfilment cost per order**. In this case those three effects together are worth more than the entire volume gain, and only one of them is visible in a ranking metric.

### 1.1 The decision contract in one paragraph

The decision owner must choose, before a six-week peak trading window, between (a) launching the retrieval upgrade alone, (b) launching retrieval plus a newly trained ranker that consumes fresh behavioural features, or (c) shipping nothing and keeping the incumbent. Option (b) was the team's prior, because it had the best offline numbers and because the architecture slide looked like the "real" recommender. Option (a) was the sceptic's position: it changed one component and left the calibrated ranker that merchandising had spent two years tuning against margin-aware labels. The evidence in Section 6 shows the sceptic is right for this workload, and shows precisely why — the retrained ranker optimised engagement-weighted labels, and engagement-weighted labels over-reward high-consideration, high-return categories.

### 1.2 What a reader should be able to do afterwards

Reconstruct the offline evaluation and explain why an unconditional metric can rise while a conditional metric falls; compute the contribution-margin decomposition and attribute a total change to five named levers; size an experiment with clustering, a design effect, and a minimum detectable effect; explain position bias and why naive click labels mislead; describe the online request path and its failure behaviour under a dependency timeout; distinguish four plausible causes of a returns-rate increase and name the check that separates them; and defend or reject the shipping decision when the returns guardrail is tightened.

**Takeaway:** the framing that survives contact with the business is "which ranked slate maximises margin after returns and fulfilment," not "which ranker has the better relevance score."

## 2. Context & Background

### 2.1 Domain and product

The setting is a general-merchandise online retailer with a large third-party marketplace. The surface in scope is the **recommendation rail** that appears on product detail pages (a "customers also viewed" strip) and immediately after add-to-cart (a "complete your order" strip). Each render places eight items in ranked order. The rail is not the search results page and not the home feed; those have separate owners, separate objectives, and separate eligibility rules, and mixing them into this decision would make the case unbounded.

The rail is a **placement** decision, not a pricing decision. The retailer does not change prices here; it changes which items are shown and in what order. That matters for the cost model: the lever is exposure, and exposure propagates into mix, returns, and fulfilment.

Terminology used throughout, defined once so the arithmetic is unambiguous:

- **SKU** — a stock-keeping unit, the finest sellable identifier; a product with two colours is two SKUs.
- **Session** — a contiguous visit by one authenticated or cookie-identified visitor, ended by thirty minutes of inactivity or by logout.
- **Render** — one execution of the rail for one session; a session may contain several renders, but in this case one render per session is assumed for sizing.
- **Eligible item** — an SKU that passes the eligibility filter at request time: in stock at a fulfilling node, shippable to the destination, permitted for the channel, not recalled, not restricted, and within the merchandising inclusion list. Eligibility is a hard filter applied before any scoring.
- **Contribution margin (CM)** — revenue net of returns, multiplied by gross margin rate, minus variable fulfilment cost. Defined exactly in Section 6.8.
- **Gross order value (GOV)** — order value before returns and before cost, the number merchandising quotes in reviews.
- **Returns rate** — returned units or returned order value divided by shipped order value, measured on a mature window so that late returns are counted.

### 2.2 Workflow before and after the decision

Before the rail renders, the visitor has viewed at least one product page. The serving path then resolves identity (logged-in customer, or an anonymous cookie with a short history), loads the session context, retrieves a bounded candidate set from several retrieval sources, filters by eligibility and policy, scores and orders the survivors, applies business rules (diversity, seller caps, margin floors), and returns eight items with metadata. After the render, the visitor may click, add to cart, purchase, keep, or return. Those downstream events arrive with different delays: clicks in seconds, orders in minutes to days, returns in weeks.

The decision changes the middle of that chain — which candidates are retrieved and how they are ordered — and therefore changes everything downstream. That is why the evaluation cannot stop at click-through: the rail is upstream of margin, so a metric measured at the click is measured too early.

### 2.3 Stakeholders and their conflicting objectives

| Stakeholder | What they optimise | What they fear | Authority in this decision |
|---|---|---|---|
| Category merchandising owner | Margin per session, sell-through of committed inventory | Cannibalisation of full-price lines, margin dilution | Business sign-off; can veto on mix grounds |
| Recommender platform team | Ranking quality, latency, operational simplicity | A model they cannot retrain or explain under pressure | Technical owner of the shipping decision |
| Finance / commercial analytics | Contribution margin, returns cost, fulfilment cost per order | Growth reported on GOV that does not convert to margin | Defines the metric; audits the decomposition |
| Marketplace seller operations | Fair exposure across sellers | Concentration of impressions on a few large sellers | Consulted; can escalate through policy |
| Customer support and reverse logistics | Return volume, return reasons, fraud | A returns spike they cannot staff for | Owns the guardrail measurement |
| Site reliability | p99 latency, error budget, capacity headroom | A ranking service that consumes the page budget | Holds the latency veto |

The conflict that drives the case is between the platform team's instinct (better model, better metric, ship it) and finance's constraint (margin after returns). Both are legitimate; they are measuring at different points in the causal chain.

### 2.4 Baseline and its documented weaknesses

The incumbent is a two-stage system. Retrieval is **co-visitation**: item-item pairs scored by session co-occurrence over a rolling window, with popularity and category-affinity fallbacks. Ranking is a **gradient-boosted tree** ensemble over about 140 features — item popularity, price, rating, category affinity, session history aggregates, and margin-aware features such as item gross margin band and historical return propensity.

The incumbent's documented weaknesses, stated by its owners before this project began:

1. Co-visitation is **sparse for long-tail and new items**; a SKU with few co-occurrences falls back to popularity, which is the least personalised option available.
2. The ranker was trained on **labels from the incumbent's own candidate distribution**. Its calibration is only meaningful for candidates the incumbent retrieves; it has never seen the score distribution of a denser, more semantically diverse candidate set.
3. Freshness is **coarse**: item embeddings and co-visitation counts refresh nightly, so a viral item or a same-day promotion is invisible for up to a day.
4. The margin-aware features are **static bands**, not learned interactions, so the model does not trade relevance against margin the way the business objective requires.

These weaknesses motivate the project but do not by themselves justify the re-architecture. That distinction is the discipline this case tries to teach: a weakness is a hypothesis about where value is, not evidence that a particular fix captures it.

### 2.5 Scale, timing, and boundary conditions

All figures below are **illustrative** and chosen to make the arithmetic checkable, not to describe a real retailer.

| Quantity | Illustrative value | Unit | How it is used later |
|---|---|---|---|
| Active SKUs in the inclusion list | 2,100,000 | SKUs | Index size, embedding memory |
| Monthly active customers | 41,000,000 | customers | Feature-store footprint |
| Rail renders | 6,000,000 | per day | Traffic and cost |
| Rail renders per month | 180,000,000 | per month | Cost and annualisation |
| Share of renders passing eligibility | 58% | fraction | Experiment sample |
| Average item gross margin rate | 34.02% | fraction of net revenue | Margin arithmetic |
| Average order value | 60.88 | currency units per order | Margin arithmetic |
| Order rate per session | 7.90% | fraction | Margin arithmetic |
| Returns rate on shipped value | 8.10% | fraction | Guardrail baseline |
| Variable fulfilment cost | 4.20 | currency per order | Margin arithmetic |
| Rail slots | 8 | items | Ranking cut-off |

Timing matters in three ways. First, the annual peak trading window is six weeks long and starts in about ten weeks; any change must be validated and rollable-back before then, which caps the experiment duration at fourteen days. Second, returns mature slowly — a meaningful share of returns arrive fifteen to forty-five days after purchase — so a fourteen-day test can measure the *direction* of a returns change but not its final level. Third, the experiment must not run through a promotional event whose mix shift would confound the comparison; the team therefore chose a fourteen-day window between two campaigns, and this is a genuine limitation recorded in Section 9 rather than an inconvenience to be waved away.

Geography and channel are held constant: one country, one currency, desktop and mobile web and app combined, authenticated and anonymous visitors included. Cross-border, marketplace-only, and subscription surfaces are out of scope.

### 2.6 Prior attempts, including one that failed for an instructive reason

Twelve months earlier the team had tried a "semantic retrieval" pilot: replace co-visitation with an embedding index built from item titles and descriptions, keep the incumbent ranker, and measure click-through. Click-through rose. Margin did not, and the pilot was quietly retired. The retrospective explanation offered at the time was "the model found similar items but not complementary ones" — plausible, and partly true, but not the whole story.

The pilot's post-mortem had two gaps that this case repairs. It never computed margin, so it could not see that the new candidates skewed towards substitutable items in high-return categories; and it never checked whether the incumbent ranker's scores were calibrated on the new candidate distribution, so the ordering it produced was effectively arbitrary in the region that mattered. Those two gaps are exactly the two failure mechanisms that reappear, at larger scale, in Section 12.

**What the organisation learned from the pilot, correctly and incorrectly.** Correctly: retrieval changes propagate further than the retrieval metric shows. Incorrectly: that semantic similarity is the wrong idea. The idea was fine; the evaluation was insufficient.

**Takeaway:** the incumbent is a tuned system with a calibrated ranker, so any replacement must be judged on the objective it was tuned against, not on a metric it never optimised.

## 3. The Business Question

The question below is an **illustrative stakeholder question**, reconstructed for teaching. It is not a quotation from a real person and should not be cited as one.

> "For the peak season, do we ship the new retrieval with the new ranker, ship the new retrieval on its own, or leave the rail alone? I need to know which one makes us more money after returns, not which one has the prettier offline number — and I need to know before the freeze."

### 3.1 Decision contract

| Field | Content |
|---|---|
| Question | Which of three configurations should serve the product rail for the peak-season window: retrieval upgrade alone, retrieval plus retrained ranker, or no change? |
| Decision owner | Recommendation product owner, with merchandising sign-off and finance audit of the metric |
| Trigger and deadline | Peak-season freeze begins in ten weeks; the experiment must complete and be analysed before it. The trigger is the availability of a funded retrieval upgrade, not a production incident |
| Available actions | (A) launch retrieval-only; (B) launch retrieval plus retrained ranker with fresh features; (C) no change; (D) defer and run a longer or larger test |
| Value and downside | Upside is margin per session; downside is margin erosion through mix and returns, plus operational risk during the highest-traffic weeks of the year |
| Objective | Contribution margin per eligible session, net of returns and variable fulfilment, aggregated over the window and population |
| Hard constraints | 250 ms p99 for the rail's share of the page budget; no increase above +50 bp in returns rate; eligibility rules enforced before scoring; no use of post-purchase information at request time |
| Scope | One country, all channels, authenticated and anonymous, product rail only |
| Out of scope | Search ranking, home feed, pricing, promotions, seller onboarding, cross-border |

### 3.2 The proxy-versus-objective conflict, named explicitly

The team's habitual proxy is **click-through rate on the rail**, and its offline surrogate is **normalised discounted cumulative gain at eight (NDCG@8)**. Both are legitimate measurements of relevance. Neither is the objective. The objective is margin. Three specific ways they can disagree, each of which is live in this case:

**Substitution versus complementarity.** A click on a substitute is a click that may replace a higher-margin purchase rather than add to it. NDCG rewards the click; the margin ledger sees a swap at a lower margin rate.

**Price and variance effects on returns.** Higher-priced items raise order value and therefore GOV, and higher-priced apparel-like categories return more often. A relevance model trained on engagement learns to prefer items that attract attention, which correlates with items that are considered and then returned.

**Fulfilment composition.** Recommending a complementary item from the same warehouse lowers cost per order; recommending a heavy or bulky item from a distant node raises it. No ranking metric sees the warehouse.

The guardrail structure in Section 9 exists because these conflicts are predictable and directional: we can state in advance which way a metric could move while the objective moves the other way, and we can measure both.

### 3.3 What an expensive wrong decision looks like

Two asymmetric errors are possible and they are not equally priced.

**Wrongly shipping arm B** costs, on the illustrative arithmetic of Section 6, about 4.45 hundredths of a currency unit of contribution margin per session, which annualises to roughly −55.8 million over 1.25 billion sessions. It also exposes the busiest trading weeks to a new model, a new feature pipeline, and a new failure surface, and it consumes the organisational credibility that a second attempt would need.

**Wrongly shipping nothing** forgoes about 1.71 hundredths per session, roughly +21.5 million annualised on the same base, and leaves the long-tail and freshness weaknesses in place through the peak. It is the cheaper error, but not free.

The asymmetry argues for a decision rule that prefers the reversible, better-understood change when the evidence for the larger change is contaminated — which is exactly the rule the case ends up applying. It also argues against a third error that is easy to make and hard to see: **shipping both at once**, so that no result is attributable.

**Takeaway:** the business question is a three-way choice under a hard returns guardrail, and the cost of the two errors is asymmetric enough to be decided explicitly rather than left to whichever arm has the larger offline number.

## 4. Assumptions & Constraints

Every assumption below has an owner-facing consequence. The three marked **critical** are the ones that, if wrong, reverse the recommendation; they are revisited in Section 6.9 and Section 12.

### 4.1 Assumption and constraint register

| ID | Assumption or constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | **Critical.** The order rate and returns rate observed in the fourteen-day window are representative of the peak window within ±10% relative | Business | Compare against the same weeks in the prior two years; segment by category | Unverified | The sign of the margin conclusion could flip if the peak mix differs materially | Re-measure during peak with a 5% holdback | Peak start / commercial analytics |
| A-02 | **Critical.** Returns are attributable to the session that generated the order, at the item level, not only at the order level | Data | Reconcile item-level return records to order lines; count unattributable lines | Partially verified (see 5d) | Margin effect of a mix shift is mis-estimated, typically understated | Use order-level returns with category priors | Any change to the returns feed schema |
| A-03 | **Critical.** The ranker's scores are only calibrated for the candidate distribution it was trained on | Model | Score-distribution comparison between incumbent and new candidate sets; reliability plot by decile | Verified in this dataset (Section 6.5) | Arm B's ordering is arbitrary where it matters most | Recalibrate or retrain before use | Any retrieval change / platform team |
| A-04 | Eligibility is evaluated from a consistent snapshot within a request | Infrastructure | Inject a mid-request stock change; assert single-snapshot reads | Verified by design (read the snapshot once) | Out-of-stock items shown; conversion loss and support cost | Re-check eligibility at render | Any change to inventory service |
| A-05 | Session-level metrics are correlated within a customer, with an intra-customer correlation near 0.05 | Statistics | Estimate ICC from historical sessions by customer | Assumed; measured post hoc | Understated standard errors and false confidence | Cluster-robust variance; larger sample | Any change to identity resolution |
| A-06 | The fourteen-day window contains no promotional event large enough to shift category mix | Business | Calendar review of the window | Verified for the chosen window | Confounded comparison; mix effect misattributed to the model | Choose a different window; stratify by campaign | Campaign calendar change |
| A-07 | New items with no interaction history can be served by content-based retrieval | Data | Cold-start share of impressions; recall on cold items in a held-out slice | Assumed | Cold-start share is overstated or understated, changing the value of the retrieval upgrade | Popularity fallback with exploration | Catalog churn change |
| A-08 | Variable fulfilment cost per order is stable over the window and does not depend on which items are recommended | Cost | Warehouse cost-per-order series by week | Violated in part: basket composition changes cost | Understates the cost of bulky or split-shipment recommendations | Item-level fulfilment cost model | Logistics cost change |
| A-09 | The 250 ms budget is the rail's allocation and is not shared with above-the-fold content | Infrastructure | Page performance trace ownership map | Verified by contract | Latency contention with the product summary module | Degrade the rail first | Page architecture change |
| C-01 | Rail p99 latency must remain at or below 250 ms measured at the edge for the rail's own span | Latency | Continuous measurement; error budget burn alert | Confirmed | Page-level abandonment; SEO and revenue harm | Reduce candidate count or slots | Any added network hop |
| C-02 | Returns rate may not rise by more than 50 basis points over baseline | Commercial | Item-level returns joined to exposure cohort | Confirmed | Margin gains that depend on returns are not admissible | Cap exposure of high-return categories | Finance policy revision |
| C-03 | Eligibility and channel-permission rules are enforced before scoring, not after | Compliance | Policy test suite; synthetic restricted item | Confirmed | Restricted or recalled goods exposed | Hard filter at candidate boundary | Any catalog policy change |
| C-04 | No post-purchase or post-return data may be used as a request-time feature | Leakage | Feature-availability audit against decision timestamps | Confirmed | Unavailable-at-serving-time features; leakage; invalid evaluation | Rebuild features from pre-purchase events | Feature-pipeline change |
| C-05 | Seller exposure must not concentrate beyond the policy cap on any single seller in a category | Fairness | Concentration metric per category per day | Confirmed | Marketplace trust erosion; seller escalation | Re-rank with a seller-diversity constraint | Seller policy change |
| C-06 | Compute and storage spend must stay inside the existing recommendation budget | Cost | Monthly cost attribution | Confirmed | Project cannot launch | Reduce replicas or index precision | Budget cycle |

### 4.2 The three assumptions that can reverse the recommendation

**A-01, representativeness of the window.** The margin conclusion in this case depends on the *mix* of what sells. Mix is the most seasonal quantity in retail. If the peak window shifts sales towards categories with materially different margin rates and return propensities, the size — and in a bad case the sign — of the arm comparison changes. The check is a prior-year comparison of the same calendar weeks, and the mitigation is a small holdback during peak so the comparison can be re-measured in the real window rather than assumed.

**A-02, item-level returns attribution.** If returns could only be attributed at the order level, the margin effect of a mix shift would be smeared across items and the decomposition in Section 6.10 would be unavailable. Worse, the direction would be systematically understated, because a mix shift toward high-return categories would be partially attributed to the whole basket. The check is a reconciliation rate: what share of returned value maps to a specific order line.

**A-03, ranker calibration.** This is the assumption that decides the case. The incumbent ranker's output is a probability-like score trained on a specific candidate distribution. When the retrieval stage changes, the score *ranking* may still look sensible while the score *level* becomes meaningless. Any downstream rule that compares a score to a threshold, or that fuses scores with a business term, breaks. The check is a score-distribution comparison and a reliability plot by decile, shown in Section 6.5.

### 4.3 What is deliberately not assumed

The case does not assume that a better offline metric implies a better online outcome. It does not assume that the new ranker is better because it is newer. It does not assume that the retrieval upgrade is harmless; Section 12 lists the ways it can hurt. And it does not assume that the fourteen-day test is sufficient to certify the returns guardrail — it is sufficient to detect a large deterioration and insufficient to certify a small one, which is stated as a limitation rather than resolved by optimism.

**Takeaway:** three assumptions carry the decision — window representativeness, item-level returns attribution, and ranker calibration — and each has a concrete check rather than a vote of confidence.
