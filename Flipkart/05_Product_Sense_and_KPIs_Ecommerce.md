# 🛒 Product Sense & KPIs in the E-Commerce Domain (Flipkart)
### Flipkart Senior Data Scientist — Interview Preparation

> **Why this matters:** At Flipkart, a Senior Data Scientist is expected to be deeply commercial. You aren't just optimizing log-loss; you are optimizing GMV and Conversion. "Product Sense" interviews at Flipkart test your ability to tie ML metrics (like NDCG or Precision) directly to business outcomes (like Revenue or RTO reduction), and how you manage trade-offs when those metrics conflict.

---

## PART 1: THE E-COMMERCE KPI GLOSSARY
Before designing an AI product for Flipkart, you must speak the language of retail. These are the core metrics every e-commerce product ultimately tries to improve.

### 1. Fundamental Business KPIs
*   **GMV (Gross Merchandise Value):** The total value of merchandise sold over a given period. *AI Goal: Increase GMV through better recommendations and search relevance.*
*   **AOV (Average Order Value):** The average amount spent each time a customer places an order. *AI Goal: Cross-selling (e.g., "Frequently Bought Together") directly targets AOV.*
*   **Conversion Rate (CVR):** The percentage of visitors to a product page who actually make a purchase.
*   **RTO (Return to Origin) Rate:** Highly critical in India due to Cash on Delivery (COD). It's the percentage of orders that are shipped but returned because the customer refused delivery or wasn't found. *AI Goal: Predict RTO risk and disable COD for high-risk users.*
*   **CAC (Customer Acquisition Cost) & LTV (Lifetime Value):** How much it costs to acquire a user vs. how much profit they generate over time. LTV must be > CAC.

### 2. Search & Discovery Metrics
*   **CTR (Click-Through Rate):** Percentage of impressions that result in a click.
*   **Zero-Result Rate:** The percentage of search queries that yield 0 products. *AI Goal: Use LLMs or vector search for query expansion to reduce this to near zero.*
*   **Query Reformulation Rate:** How often a user has to type a new search term because their first search gave bad results. (Lower is better).

---

## PART 2: AI-SPECIFIC PRODUCT METRICS
When you deploy an ML model at Flipkart, you track technical metrics alongside product metrics.

### Ranking & Recommendation Metrics
*   **NDCG (Normalized Discounted Cumulative Gain):** Measures ranking quality. Are the most relevant/purchased items at the very top of the list?
*   **MRR (Mean Reciprocal Rank):** Evaluates search. How far down the list did the user have to scroll before clicking the first item?
*   **Diversity & Novelty:** If you recommend 10 identical black t-shirts, CTR drops. You must measure how diverse the recommendations are, and how often you show "novel" (newly discovered) items vs. historical favorites.

### Generative AI / LLM Metrics (For Reviews & Support)
*   **Task Completion Rate:** For customer support bots—did the bot resolve the refund, or did it hand off to a human agent?
*   **Helpfulness Score / Upvotes:** For LLM-generated review summaries—do users click "Helpful" on the AI summary?
*   **Faithfulness / Hallucination Rate:** Did the AI summary invent a product feature that doesn't exist?

---

## PART 3: THE PRODUCT SENSE INTERVIEW FRAMEWORK
When asked "How would you build an AI product for X?", use the **GAME** framework.

1. **G**oal: What is the core business objective? (Revenue? Efficiency? User Experience?)
2. **A**udience: Who are the users? (Tier 2/3 city shoppers? Sellers? Delivery agents?)
3. **M**etrics: What are the North Star and Counter metrics?
4. **E**xecution & Trade-offs: How do we build it, and what are the risks/trade-offs?

---

## PART 4: DETAILED FLIPKART INTERVIEW SCENARIOS

### Scenario 1: The "Frequently Bought Together" (FBT) Widget
**Question: "Flipkart wants to launch a 'Frequently Bought Together' widget on the checkout page. How do you measure its success, and what are the trade-offs?"**

**The Answer:**
> "I would break the metrics down into Business, Model, and Guardrail metrics.
>
> **1. Business / North Star Metric:**
> *   **AOV (Average Order Value):** The primary goal of FBT is to get the user to add one more item (e.g., a phone case with a phone), increasing the total cart value.
> *   **Attach Rate:** The percentage of primary product purchases that included the FBT recommended product.
>
> **2. Counter Metrics (The Guardrails):**
> *   **Checkout Abandonment Rate:** If the FBT widget is too slow to load, or clutters the UI, users might abandon the cart entirely. We cannot sacrifice the primary conversion to chase a cross-sell.
> *   **Cannibalization:** Are we just shifting spend? If they buy the phone case, did they drop the earphones they already had in the cart?
>
> **3. The Trade-off (Relevance vs. Margin):**
> Do we recommend the most *relevant* item (highest probability of click), or the item with the highest *profit margin*? If a $5 screen protector has a 90% chance to sell, but a $20 Flipkart Assured case has a 60% chance to sell, Expected Value dictates we show the $20 case. I would optimize for Expected Margin (Probability of Purchase × Item Margin)."

---

### Scenario 2: Handling India-Specific RTO (Return to Origin)
**Question: "Cash on Delivery (COD) is popular on Flipkart but leads to high Return to Origin (RTO) rates, costing us shipping fees. How do you build an ML product to fix this?"**

**The Answer:**
> "RTO prediction is a classic uplift/risk modeling problem, but the product execution requires delicate handling to avoid destroying our Conversion Rate.
>
> **1. The Model:** 
> I'd build an XGBoost model predicting `P(RTO)` based on user history (past RTOs), location (Tier 3 pin codes have higher RTO), item category (apparel has higher RTO than electronics), and order value.
>
> **2. The Product Execution (The Trade-off):**
> *   **Bad Product Sense:** 'If `P(RTO) > 80%`, block the user.' (This is bad because it creates massive user friction and hurts GMV).
> *   **Good Product Sense:** We use dynamic friction based on the risk score:
>     *   *Low Risk (0-30%):* Show COD as the default option.
>     *   *Medium Risk (31-70%):* Allow COD, but add a ₹50 non-refundable 'COD Handling Fee', or require OTP verification to confirm intent.
>     *   *High Risk (>70%):* Disable COD entirely; require prepaid via UPI/Card.
>
> **3. Metrics to Track:**
> *   *North Star:* Net RTO Cost Reduction.
> *   *Counter Metric:* Drop in overall Conversion Rate (CVR). If blocking COD causes a 10% drop in genuine sales, the cost of lost GMV might exceed the shipping savings from preventing RTOs."

---

### Scenario 3: GenAI for Review Summarization
**Question: "We want to use LLMs to summarize 10,000 user reviews for popular mobile phones. How do you evaluate if this product is actually working?"**

**The Answer:**
> "Evaluating GenAI in e-commerce requires measuring both offline model quality and online business impact.
>
> **1. Offline Metrics (Quality & Safety):**
> *   *Faithfulness:* Does the summary only contain facts present in the reviews? (e.g., If the LLM says 'Great battery', we must verify a review actually mentioned battery).
> *   *Toxicity/Bias check:* Ensure the summary doesn't surface hate speech from troll reviews.
>
> **2. Online Product Metrics (Business Impact):**
> *   *Time on Page:* A good summary should *decrease* the time a user spends reading reviews, allowing them to make a faster decision.
> *   *Conversion Lift:* In an A/B test (Control: normal reviews vs. Treatment: AI summary), does the AI summary increase the CVR for that product?
> *   *Return Rate:* Interestingly, better summaries set clearer expectations. I would track if items with AI summaries have lower return rates due to 'item not as described'.
>
> **3. The Trade-off:**
> Do we summarize *all* reviews, or only verified purchases? To prevent competitor sabotage, I would restrict the LLM context exclusively to 'Flipkart Verified Purchase' reviews."

---

### Scenario 4: The Search Ranking Conflict
**Question: "Your new Deep Learning search ranking model increases CTR by 5%, but the business team complains that it's surfacing cheaper, low-margin products. How do you handle this?"**

**The Answer:**
> "This is the classic relevance vs. profitability conflict. An ML model optimizing purely for clicks (CTR) will inevitably surface cheap items, clickbait, or massive discounts. 
>
> **1. Multi-Objective Optimization:**
> We cannot optimize solely for `P(Click)`. The ranking score must be a composite function:
> `Final Score = w1 * P(Click) + w2 * P(Conversion | Click) + w3 * Expected_Margin`
>
> **2. A/B Testing the Weights:**
> I would run an A/B test with different weight configurations. We measure the trade-off curve: 'How much CTR are we willing to sacrifice to gain a 2% lift in Gross Margin?'
>
> **3. Sponsored Listings (Ad Revenue):**
> Another layer is ad revenue. Sometimes the most profitable item to show isn't the one with the highest margin, but the one a seller paid Flipkart to promote. We have to incorporate the 'Bid Price' into the expected value calculation of the ranking slot."

---

### Scenario 5: The Cold Start Problem (New Sellers)
**Question: "A new seller lists a smartphone on Flipkart. It has zero clicks, zero reviews, and zero purchase history. How does your search algorithm rank it?"**

**The Answer:**
> "If a ranking algorithm relies purely on historical CTR, new items will be trapped at the bottom of page 10 forever, creating a terrible experience for new sellers.
>
> **1. Content-Based Embeddings (The Baseline):**
> Since we have no interaction data, we rely on metadata. We use an embedding model to encode the product title, description, and image. If its embedding vector is 99% similar to a highly-selling Samsung phone, we borrow the 'prior probability' of the Samsung phone to give the new item a starting score.
>
> **2. Explore & Exploit (Multi-Armed Bandit):**
> We allocate a specific 'exploration budget' (e.g., 5% of search traffic). We artificially boost the new item to Page 1 for a small fraction of users. 
> *   *Exploration:* We gather real CTR and Conversion data from this 5% traffic.
> *   *Exploitation:* Once the statistical confidence of its true CTR is established, we let the standard ranking algorithm take over. If it's a bad product, it sinks back to page 10. If it's good, it stays on page 1 organically."
