# eBay Bengaluru — Interview Questions & Answers (2024–2025)
> Sourced from: Glassdoor, LeetCode Discuss, Reddit, InterviewQuery, DataInterview.com, Prepfully, GeeksForGeeks
> Updated: April 2026 | Roles: Data Scientist, ML Engineer, SDE II/III

---

## 📋 Interview Process Overview

| Round | Format | Duration | What They Test |
|-------|--------|----------|----------------|
| 1. Recruiter Screen | Phone/Video | 20–30 min | Background, motivation, role fit |
| 2. Online Assessment (OA) | CodeSignal / HackerRank | 90 min | 3–4 DSA problems (Easy → Hard) |
| 3. Technical Round 1 | Video call | 45–60 min | Coding / DSA (LeetCode Medium-Hard) |
| 4. Technical Round 2 | Video call | 45–60 min | System Design / ML System Design |
| 5. Technical Round 3 | Video call | 45–60 min | Project Deep Dive + SQL/Stats |
| 6. Behavioral / Director | Video call | 45–60 min | Leadership, culture fit, trade-offs |

---

## 🔵 SECTION 1: CODING / DATA STRUCTURES & ALGORITHMS (DSA)

### Q1. Implement an LRU Cache

**Question:** Design and implement a Least Recently Used (LRU) cache with `get(key)` and `put(key, value)` operations in O(1) time.

**Answer:**
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()  # maintains insertion order

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)   # mark as recently used
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # evict LRU (first item)
```

**Complexity:** O(1) for both get & put  
**eBay Context:** Used to cache product metadata (title, price, seller score) for high-traffic listing pages.

---

### Q2. Three-Sum (3Sum)

**Question:** Given an integer array `nums`, return all unique triplets `[a, b, c]` such that `a + b + c == 0`.

**Answer:**
```python
def threeSum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:   # skip duplicates
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]: left += 1
                while left < right and nums[right] == nums[right-1]: right -= 1
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return result
```

**Complexity:** O(n²) time, O(1) extra space  
**Pattern:** Two-pointer after sorting — a classic pivot + two-pointer pattern.

---

### Q3. Find K-th Largest Element in a Stream

**Question:** Design a class to find the k-th largest element in a stream of integers.

**Answer:**
```python
import heapq

class KthLargest:
    def __init__(self, k: int, nums: list):
        self.k = k
        self.heap = []
        for n in nums:
            self.add(n)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]   # min of top-k = k-th largest
```

**Complexity:** O(log k) per insertion  
**eBay Context:** Streaming top-K sellers by rating or top-K bids in real-time auction scenarios.

---

### Q4. Binary Search on Rotated Sorted Array

**Question:** Find a target in a rotated sorted array.

**Answer:**
```python
def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

**Complexity:** O(log n) — always identify the sorted half first.

---

### Q5. Word Ladder (BFS)

**Question:** Find the shortest transformation sequence from `beginWord` to `endWord`, changing one letter at a time, using a given word list.

**Answer:**
```python
from collections import deque

def ladderLength(beginWord, endWord, wordList):
    wordSet = set(wordList)
    if endWord not in wordSet:
        return 0
    queue = deque([(beginWord, 1)])
    while queue:
        word, steps = queue.popleft()
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                newWord = word[:i] + c + word[i+1:]
                if newWord == endWord:
                    return steps + 1
                if newWord in wordSet:
                    wordSet.remove(newWord)
                    queue.append((newWord, steps + 1))
    return 0
```

**Pattern:** BFS for shortest path in unweighted graphs.

---

## 🟢 SECTION 2: SQL & DATA WRANGLING

### Q6. Top Sellers by GMV with Window Functions

**Question:** Find the top 3 sellers by Gross Merchandise Value (GMV) in each product category for the last 30 days.

**Answer:**
```sql
WITH seller_gmv AS (
    SELECT
        seller_id,
        category,
        SUM(sale_price * quantity) AS gmv,
        RANK() OVER (
            PARTITION BY category
            ORDER BY SUM(sale_price * quantity) DESC
        ) AS rnk
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '30 days'
      AND status = 'completed'
    GROUP BY seller_id, category
)
SELECT seller_id, category, gmv, rnk
FROM seller_gmv
WHERE rnk <= 3
ORDER BY category, rnk;
```

---

### Q7. User Retention Cohort Analysis

**Question:** Write a SQL query to compute Day-7 retention rate by weekly signup cohort.

**Answer:**
```sql
WITH cohort AS (
    SELECT
        user_id,
        DATE_TRUNC('week', signup_date) AS cohort_week
    FROM users
),
activity AS (
    SELECT DISTINCT
        user_id,
        DATE_TRUNC('week', activity_date) AS activity_week
    FROM user_events
),
joined AS (
    SELECT
        c.cohort_week,
        COUNT(DISTINCT c.user_id) AS cohort_size,
        COUNT(DISTINCT CASE
            WHEN a.activity_week = c.cohort_week + INTERVAL '1 week' THEN c.user_id
        END) AS retained_users
    FROM cohort c
    LEFT JOIN activity a ON c.user_id = a.user_id
    GROUP BY c.cohort_week
)
SELECT
    cohort_week,
    cohort_size,
    retained_users,
    ROUND(100.0 * retained_users / cohort_size, 2) AS day7_retention_pct
FROM joined
ORDER BY cohort_week;
```

---

### Q8. Detect Price Anomalies with LAG

**Question:** Identify listings where the price dropped by more than 30% compared to the previous listing from the same seller.

**Answer:**
```sql
WITH price_lag AS (
    SELECT
        listing_id,
        seller_id,
        list_price,
        LAG(list_price) OVER (
            PARTITION BY seller_id
            ORDER BY created_at
        ) AS prev_price,
        created_at
    FROM listings
)
SELECT *
FROM price_lag
WHERE prev_price IS NOT NULL
  AND (prev_price - list_price) / prev_price > 0.30;
```

---

### Q9. Median Sale Price per Category

**Question:** Calculate the median sale price per product category (no MEDIAN() function available).

**Answer:**
```sql
WITH ranked AS (
    SELECT
        category,
        sale_price,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY sale_price) AS rn,
        COUNT(*) OVER (PARTITION BY category) AS cnt
    FROM transactions
    WHERE status = 'completed'
)
SELECT
    category,
    AVG(sale_price) AS median_price
FROM ranked
WHERE rn IN (FLOOR((cnt + 1) / 2.0), CEIL((cnt + 1) / 2.0))
GROUP BY category;
```

---

### Q10. First Purchase Funnel Drop-off

**Question:** How many users who viewed a listing also added it to cart, and then made a purchase (funnel)?

**Answer:**
```sql
WITH funnel AS (
    SELECT
        user_id,
        MAX(CASE WHEN event_type = 'view'     THEN 1 ELSE 0 END) AS viewed,
        MAX(CASE WHEN event_type = 'add_cart' THEN 1 ELSE 0 END) AS carted,
        MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchased
    FROM user_events
    GROUP BY user_id
)
SELECT
    SUM(viewed)    AS total_viewers,
    SUM(carted)    AS total_carted,
    SUM(purchased) AS total_purchased,
    ROUND(100.0 * SUM(carted) / NULLIF(SUM(viewed), 0), 2)     AS view_to_cart_pct,
    ROUND(100.0 * SUM(purchased) / NULLIF(SUM(carted), 0), 2)  AS cart_to_purchase_pct
FROM funnel;
```

---

## 🟠 SECTION 3: MACHINE LEARNING & MODELING

### Q11. Build a CTR Prediction Model for New Listings

**Question:** How would you build a Click-Through Rate (CTR) prediction model for newly listed items on eBay?

**Answer:**

**Problem Framing:**
- **Goal:** Predict P(click | user, listing, context) to rank listings in search results.
- **Challenge:** Cold-start problem — new listings have zero historical click data.

**Feature Engineering:**
| Feature Type | Examples |
|---|---|
| Listing features | Title embedding, category, price, condition, seller_score, image quality score |
| Seller features | Historical CTR, GMV, average response time, seller rating |
| User features | Search history, purchase history, demographics, session context |
| Contextual | Time of day, device type, query-listing semantic similarity (cosine sim) |

**Handling Cold-Start:**
- Use content-based features (NLP embeddings of title/description) as a proxy.
- Leverage transfer learning: pre-train on similar category items.
- Exploration strategies: epsilon-greedy or UCB banding for new listings.

**Model Selection:**
```
Candidate Generation → Lightweight model (logistic regression, GBDT)
Re-ranking          → XGBoost or Wide & Deep Network
                       Wide part: memorization (sparse features)
                       Deep part: generalization (dense embeddings)
```

**Evaluation:**
- **Offline:** AUC-ROC, Log Loss, Precision@K, NDCG
- **Online:** CTR lift, GMV impact, A/B test significance

**Production Considerations:**
- Feature store (Feast/Tecton) for low-latency real-time features
- Model versioning and shadow deployment before full rollout
- Monitor for feature drift and concept drift

---

### Q12. Handling Imbalanced Datasets in Fraud Detection

**Question:** eBay fraud transactions are < 1% of all transactions. How do you handle such class imbalance?

**Answer:**

**Techniques (ordered by preference):**

1. **Threshold Tuning**: Use Precision-Recall curve instead of ROC. Set threshold based on business cost of FP vs. FN.
2. **Resampling:**
   - Oversample minority: `SMOTE` (Synthetic Minority Oversampling Technique)
   - Undersample majority: Random undersampling or `NearMiss`
3. **Class Weights:** Set `class_weight='balanced'` in sklearn models — most practical starting point.
4. **Algorithm Choice:** Tree-based ensembles (XGBoost, LightGBM) handle imbalance better than logistic regression.
5. **Anomaly Detection:** Treat fraud as anomaly (Isolation Forest, One-Class SVM) if labels are very sparse.
6. **Cost-Sensitive Learning:** Assign higher misclassification cost to minority class in the loss function.

**Key Metric:** Use **F1 Score**, **AUC-PR**, or **Average Precision** — NOT accuracy.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, average_precision_score

model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05)
model.fit(X_train, y_train, sample_weight=compute_sample_weight('balanced', y_train))

y_proba = model.predict_proba(X_test)[:, 1]
print(f"AUC-PR: {average_precision_score(y_test, y_proba):.4f}")
```

---

### Q13. Bias-Variance Tradeoff

**Question:** Explain bias-variance tradeoff. How does it apply to model selection?

**Answer:**

| | Bias | Variance |
|---|---|---|
| **High when** | Model too simple (underfitting) | Model too complex (overfitting) |
| **Symptoms** | High train error, high test error | Low train error, high test error |
| **Example** | Linear model on non-linear data | Deep tree with no pruning |
| **Fix** | More features, complex model | Regularization, pruning, dropout |

**Mathematical decomposition:**
```
Expected Error = Bias² + Variance + Irreducible Noise
```

**eBay Example:** A logistic regression for listing quality prediction might underfit (high bias). A 10,000-leaf decision tree might overfit to historical seller behavior noise (high variance). An XGBoost with tuned `max_depth`, `min_child_weight`, and `lambda` regularization hits the sweet spot.

**Regularization:**
- **L1 (Lasso):** Drives some coefficients to zero → feature selection
- **L2 (Ridge):** Shrinks all coefficients → handles multicollinearity

---

### Q14. XGBoost vs. Wide & Deep for Recommendations

**Question:** When would you pick XGBoost over a Wide & Deep neural network for eBay recommendations?

**Answer:**

| Criterion | XGBoost | Wide & Deep |
|---|---|---|
| **Data size** | Works well on < 10M rows | Shines with 100M+ rows |
| **Feature types** | Structured/tabular features | Mixed: sparse (wide) + dense (deep) |
| **Training speed** | Faster iteration, easier to debug | Slower, needs GPU |
| **Cold-start** | Poor — needs historical signal | Better — can learn content embeddings |
| **Interpretability** | High (SHAP values) | Lower (black box) |
| **Latency** | Very fast inference | Slower (neural network layers) |

**eBay Recommendation:**
- Use **XGBoost** for seller quality scoring, listing quality ranking (audit-friendly, fast).
- Use **Wide & Deep** for the main search/recommendation feed where scale and embedding power matter.

---

### Q15. Online Learning Pipeline for Search Re-ranking

**Question:** Design an online learning pipeline to re-rank eBay search results in near-real-time.

**Answer:**

```
[User Query] → Feature Extraction → Candidate Generation → Re-ranker
                     ↑                                          ↓
              Feature Store                               Ranked Results
              (Redis / Feast)                                   ↓
                                                         Clickstream Log
                                                               ↓
                                              Online Learning Update (mini-batch)
                                                               ↓
                                                     Model Registry → Deploy
```

**Key Decisions:**
1. **Feature Store:** Pre-compute and cache item embeddings, user profiles in Redis for <5ms lookup.
2. **Model Architecture:** Start with a Ranking SVM or LambdaMART; evolve to a two-tower neural model.
3. **Online Update Strategy:** Mini-batch gradient descent every 15–30 minutes using fresh click/purchase data.
4. **Exploration vs. Exploitation:** Use contextual bandits (LinUCB) to balance showing known-good vs. new listings.
5. **Monitoring:** Track NDCG@10, CTR, and GMV per ranked position. Alert on distributional shift.

---

## 🟡 SECTION 4: STATISTICS & EXPERIMENTATION (A/B TESTING)

### Q16. Design an A/B Test for a New Shipping Badge

**Question:** eBay wants to add a "Free Shipping" badge on listing cards. Design an A/B test.

**Answer:**

**Step 1 — Define Hypothesis:**
> H₀: The shipping badge has no effect on click-through rate.  
> H₁: The shipping badge increases CTR by at least 2% (MDE).

**Step 2 — Select Unit of Randomization:**
- Randomize at **user level** (not listing level) to avoid cross-contamination.
- Use a **cookie-based** or **user_id hash** assignment.

**Step 3 — Define Metrics:**
| Type | Metric |
|---|---|
| Primary | CTR on listing card |
| Secondary | Add-to-cart rate, Conversion rate |
| Guardrail | Page load time (badge shouldn't slow down rendering), Seller GMV (don't hurt sellers without free shipping) |

**Step 4 — Sample Size Calculation:**
```python
import scipy.stats as stats
import math

def sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta  = stats.norm.ppf(power)
    effect  = mde * baseline_rate
    p1, p2  = baseline_rate, baseline_rate + effect
    p_bar   = (p1 + p2) / 2
    n = (z_alpha + z_beta)**2 * (p1*(1-p1) + p2*(1-p2)) / (p2 - p1)**2
    return math.ceil(n)

# Example: baseline CTR = 5%, MDE = 2% relative lift
n = sample_size(0.05, 0.02)
print(f"Required sample per arm: {n}")  # ~188,000 users per arm
```

**Step 5 — Duration:**
- Run for at least **2 full weeks** to account for weekday/weekend effects.
- Do not peek at results early (multiple testing issue).

**Step 6 — Pitfalls to Watch:**
- **Novelty Effect:** Users click badge because it's new, not because they care about free shipping.
- **Network Interference (SUTVA violation):** If sellers react to badge treatment and change prices.
- **Simpson's Paradox:** Badge might help for electronics but hurt for collectibles — always segment.

---

### Q17. p-Value, Statistical Significance & Power

**Question:** Explain p-value, Type I & Type II error, and statistical power in plain English.

**Answer:**

| Concept | Definition | eBay Example |
|---|---|---|
| **p-value** | P(seeing data this extreme | H₀ is true) | p = 0.03 → only 3% chance we'd see this CTR diff if badge had no effect |
| **Type I Error (α)** | False positive — rejecting H₀ when it's true | Declaring badge works when it doesn't (wasted rollout) |
| **Type II Error (β)** | False negative — failing to reject H₀ when H₁ is true | Missing a real improvement |
| **Statistical Power** | 1 - β, probability of detecting a real effect | Power = 80% means 80% chance to detect MDE if it truly exists |
| **Confidence Interval** | Range containing true parameter with 95% probability | CTR lift 95% CI: [0.8%, 3.2%] means effect is likely positive |

**Key rule of thumb:** p < 0.05 → statistically significant, but check **effect size** (Cohen's d) too — small effects with large samples can be statistically significant but **not practically meaningful**.

---

### Q18. CUPED — Variance Reduction in A/B Tests

**Question:** What is CUPED and why does eBay use it?

**Answer:**

**CUPED** (Controlled-experiment Using Pre-Experiment Data) reduces variance of the estimator by adjusting the post-experiment outcome using a pre-experiment covariate.

**Formula:**
```
Y_cuped = Y_post - θ * (X_pre - E[X_pre])

Where:
  Y_post = observed metric (e.g., conversion rate)
  X_pre  = pre-experiment metric (e.g., past 30-day conversion)
  θ      = cov(Y_post, X_pre) / var(X_pre)
```

**Why it works:** Uses the correlation between pre and post metrics to "explain away" background noise, making the estimator more precise.

**Benefit:** Can reduce sample size requirements by 20–50%, allowing faster, more sensitive experiments.

**When to use:** When you have a strong pre-experiment baseline metric that correlates with your primary metric.

---

### Q19. Investigating a Sudden GMV Drop

**Question:** eBay's GMV drops by 15% on a Tuesday. How do you diagnose it?

**Answer:**

**Step 1 — Rule Out Data Issues:**
- Check data pipeline logs for ingestion failures or delays.
- Compare raw event counts (views, clicks, purchases) with expected baselines.

**Step 2 — Segment the Drop:**
```
By Dimension        → Category? Geography? Device? Seller type?
By User Cohort      → New vs. returning? High-value vs. casual?
By Funnel Stage     → Drop in traffic? Conversion? AOV?
By Time             → Which hour did it start? Is it sustained?
```

**Step 3 — Correlate with Changes:**
- Was there a code deployment or A/B test launched Tuesday?
- Check competitor activity or external events (holiday, payment outage).
- Check SEO traffic changes (Google update?).

**Step 4 — Quantify Root Cause:**
```sql
SELECT category, SUM(gmv) AS gmv_today, 
       LAG(SUM(gmv), 7) OVER (PARTITION BY category ORDER BY date) AS gmv_last_week,
       (SUM(gmv) - LAG(SUM(gmv), 7) OVER (PARTITION BY category ORDER BY date)) 
           / LAG(SUM(gmv), 7) OVER (PARTITION BY category ORDER BY date) AS pct_change
FROM transactions
GROUP BY category, date;
```

**Step 5 — Escalate & Fix:**
- If code-related: rollback or hotfix.
- If payment gateway: coordinate with ops team.
- Document root cause for post-mortem.

---

## 🔴 SECTION 5: SYSTEM DESIGN

### Q20. Design a Real-Time Auction System (eBay-style)

**Question:** Design a scalable real-time bidding/auction system like eBay's core auction feature.

**Answer:**

**Requirements Clarification:**
- Scale: 100M active auctions, 50K bids/second at peak.
- Consistency: Bids must be strictly ordered. Highest bid wins.
- Latency: Bid confirmation < 200ms. Live bid display < 500ms.

**High-Level Architecture:**
```
Client → API Gateway → Bid Service → Kafka → Bid Processor → DB
                                           ↓
                                    WebSocket Hub → Client (live updates)
                                           ↓
                                    Auction Engine (winner determination)
```

**Key Components:**

| Component | Technology | Reason |
|---|---|---|
| API Layer | Spring Boot / FastAPI | Stateless, horizontally scalable |
| Message Queue | Apache Kafka | High throughput, durable, ordered per partition |
| Auction State | Redis (sorted set) | O(log n) bid insertion, O(1) max bid lookup |
| Persistent Store | Cassandra or CockroachDB | High write throughput, distributed ACID |
| Real-time Push | WebSockets / Server-Sent Events | Push bid updates to all auction viewers |
| Cache | Redis | Cache auction metadata (title, reserve price) |

**Critical Design Decisions:**
1. **Kafka Partition Key = auction_id** → Ensures all bids for same auction are ordered.
2. **Redis Sorted Set** stores `(bid_amount, user_id)` — ZADD for O(log n) insert, ZREVRANGE for current max.
3. **Optimistic Locking** in DB: Compare-and-swap to prevent race conditions on final winner write.
4. **Reserve Price Logic:** Auction engine checks bid vs. reserve price before accepting.

---

### Q21. Design a Product Recommendation System

**Question:** Design a recommendation system for eBay's homepage "Items you might like" section.

**Answer:**

**System Components:**

```
Offline Pipeline:
User activity logs → Feature Store → 
  ┌─ Collaborative Filtering (ALS matrix factorization)
  ├─ Content-Based (item2vec / BERT embeddings)
  └─ Popularity-Based (cold-start fallback)
         ↓
     Candidate Pool (top-1000 items per user)
         ↓
     Re-ranking Model (XGBoost / Wide&Deep)
         ↓
     Final Top-20 Recommendations → CDN Cache

Online Pipeline:
User request → User profile (Redis) → 
    Candidate retrieval (ANN: FAISS / ScaNN) → 
    Re-ranker (低 latency inference) → 
    Served recommendations
```

**Feature Store Schema:**
- **User features:** recent views (last 7d), purchase history, price sensitivity, preferred categories
- **Item features:** embedding vector, price, seller score, category, listing age
- **Cross features:** user-item affinity score, query-item similarity

**Evaluation:**
- **Offline:** NDCG@10, Hit Rate@10, Coverage
- **Online:** CTR lift, Add-to-cart rate, Revenue per session

**Cold-Start Strategy:**
- New users: Use geographic popularity + trending items.
- New items: Use content embeddings from title/description.

---

### Q22. Design a Notification Service at Scale

**Question:** Design a multi-channel notification system (email, push, SMS) for eBay to notify 100M users.

**Answer:**

**Requirements:**
- Channels: Email, Mobile Push (APNs/FCM), SMS (Twilio)
- Use cases: Bid outbid, auction won, shipping update, price drop alert
- Scale: 50M notifications/day, < 30s end-to-end latency

**Architecture:**
```
Event Producers (Auction, Order, Shipping) 
    → Kafka Topics (per event type)
    → Notification Router (reads user prefs + event rules)
    → Channel Workers (email, push, SMS) → 3rd party gateways
    → Delivery Status → DB (for auditing + retry)
```

**Key Design Points:**
1. **User Preference Service:** Respects "no push after 10PM" opt-outs, unsubscribes.
2. **Rate Limiting:** Cap at 5 notifications/hour per user to avoid spam.
3. **Retry Logic:** Exponential backoff with dead-letter queue (DLQ) for failed deliveries.
4. **Idempotency:** Use `notification_id` to prevent duplicate sends.
5. **Template Engine:** Jinja2-style templates stored in DB with personalization tokens.

---

## 🟣 SECTION 6: BEHAVIORAL / LEADERSHIP

### Q23. Tell me about a time you made a data-driven decision with imperfect data.

**Framework (STAR):**

> **Situation:** Our team needed to decide whether to expand a new seller incentive program to all regions within 2 weeks, but we only had 3 weeks of data from a pilot in 2 cities.
> 
> **Task:** I was asked to make a recommendation on whether to proceed with expansion or wait for more data.
> 
> **Action:** I used a Bayesian approach to quantify our uncertainty — the posterior credible interval for GMV lift was [2%, 8%] with 80% confidence. I combined this with a sensitivity analysis showing that even the lower bound was ROI-positive. I flagged risks: potential novelty effect, insufficient data on returns/fraud rate.
> 
> **Result:** Recommended a staged rollout (3 more regions, not all). This allowed us to collect more data while capturing upside. The final lift was 4.2%, and fraud rates were within acceptable bounds.

---

### Q24. How do you handle a disagreement with a product manager on metrics?

**Good Answer:**
> "I start by understanding their perspective — usually the disagreement comes from different views on what the user values. I bring data: I'd show how my proposed metric directly correlates with long-term user retention, while their metric may capture short-term engagement. I'd suggest running the experiment and measuring both metrics. Data usually resolves the debate. If we're still in disagreement, I'd escalate to the team lead and let the post-experiment results adjudicate."

---

### Q25. Why eBay? What excites you about the role?

**Good Answer:**
> "eBay is one of the few platforms that operates a genuine two-sided marketplace at massive scale — over 132M buyers in 190 markets. The data problems are uniquely complex: you're simultaneously optimizing for buyer satisfaction, seller success, and marketplace trust. I'm particularly excited by the experimentation culture — running causal inference at a platform where seller decisions can interfere with buyer outcomes is a fascinating problem space. The AI/ML work in search ranking, fraud detection, and GenAI-powered listing tools aligns exactly with the problems I want to work on."

---

## 📌 SECTION 7: QUICK-FIRE CONCEPTS (RAPID ROUND)

| Question | Answer |
|---|---|
| What is p-hacking? | Running multiple tests and selectively reporting the one with p < 0.05 |
| L1 vs L2 regularization? | L1 → sparse features (Lasso); L2 → shrinks all weights (Ridge) |
| What is AUC-ROC? | Area Under ROC curve; measures discrimination ability regardless of threshold |
| Precision vs Recall trade-off? | Precision = TP/(TP+FP); Recall = TP/(TP+FN); use F1 to balance |
| What is data leakage? | Using future info to predict the past; causes unrealistically high train accuracy |
| ACID properties? | Atomicity, Consistency, Isolation, Durability — transactional DB guarantees |
| CAP theorem? | Distributed systems can guarantee only 2 of: Consistency, Availability, Partition Tolerance |
| Explain NDCG | Normalized Discounted Cumulative Gain — ranking metric that weights relevant results higher when they appear earlier |
| What is a feature store? | Centralized repo for computed ML features with low-latency serving (e.g., Feast, Tecton) |
| What is Model Calibration? | Aligning predicted probabilities with actual outcomes; use Platt scaling or isotonic regression |

---

## 🗂️ PREPARATION CHECKLIST

```
CODING (40%)
  ✅ LeetCode Medium/Hard: Arrays, Trees, Graphs, DP, Heaps
  ✅ Classic: LRU Cache, 3Sum, K-th Largest, BFS/DFS, Sliding Window
  ✅ Language: Python (preferred), Java also accepted

SQL (20%)
  ✅ Window functions: ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, SUM OVER
  ✅ CTEs, subqueries, self-joins
  ✅ Aggregations + GROUP BY + HAVING
  ✅ Funnel analysis, cohort analysis, time-window queries

ML / MODELING (20%)
  ✅ End-to-end ML pipeline: feature eng → model → evaluation → deployment
  ✅ XGBoost, LightGBM, Wide & Deep, Two-tower models
  ✅ Imbalance handling, bias-variance, regularization
  ✅ GenAI: RAG, embeddings, fine-tuning, hallucination mitigation

STATS / EXPERIMENTATION (10%)
  ✅ Hypothesis testing, p-values, CI, Type I/II error, Power
  ✅ A/B test design: unit of randomization, metric selection, sample size
  ✅ CUPED, Bayesian vs frequentist, Simpson's paradox

SYSTEM DESIGN (10%)
  ✅ Auction system, Recommendation system, Notification service
  ✅ Kafka, Redis, Cassandra, API design, rate limiting
  ✅ CAP theorem, sharding, caching strategies

BEHAVIORAL
  ✅ STAR format for 5–6 stories: conflict, failure, decisions with data, leadership
  ✅ Know eBay's mission, GMV breakdown, product areas
```

---

*Sources: Glassdoor (2024–2025), LeetCode Discuss, Reddit r/cscareerquestions, InterviewQuery.com, DataInterview.com, Prepfully, GeeksForGeeks, YouTube interview breakdowns*
