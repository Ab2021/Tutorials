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

## 🐍 SECTION 8: DSA PYTHON — FULL QUESTION BANK

> **Files:** `python_practice/04_dsa_coding_patterns.py` (Patterns 1–8) · `python_practice/05_dsa_advanced_patterns.py` (Patterns 9–15)

---

### 📦 PATTERN REFERENCE MAP

| Pattern | Topics Covered | Key Problems |
|---|---|---|
| HashMap | Two Sum, Group Anagrams, Prefix Sum | Q-P1 |
| Two Pointers | Palindrome, 3Sum, Container Water | Q-P2 |
| Sliding Window | Min Window, Longest Unique Substr | Q-P3 |
| Sorting/Intervals | Merge Intervals, Meeting Rooms | Q-P4 |
| Heaps | Kth Largest, Merge K Sorted | Q-P5 |
| Binary Search | Rotated Array, Find Peak | Q-P6 |
| Stacks | Valid Parens, Daily Temperatures | Q-P7 |
| Greedy/DP | Kadane, Coin Change, Max Profit | Q-P8 |
| Linked Lists | Reverse, Cycle, Merge, Reorder | Q-P9 |
| Trees | BFS, DFS, LCA, Diameter, Serialize | Q-P10 |
| Graphs | Islands, Topo Sort, Dijkstra, Union-Find | Q-P11 |
| DP Classic | LCS, LIS, Word Break, Edit Distance, Knapsack | Q-P12 |
| Backtracking | Permutations, N-Queens, Subsets, Word Search | Q-P13 |
| Trie | Autocomplete, Word Search II | Q-P14 |
| Bit Manipulation | XOR, Count Bits, Power of Two | Q-P15 |

---

### LINKED LIST QUESTIONS

---

#### Q26. Reverse a Linked List

**Pattern:** Iterative pointer swap · **TC:** O(n) · **SC:** O(1)

```python
def reverse_linked_list(head):
    prev, cur = None, head
    while cur:
        nxt = cur.next      # save next node
        cur.next = prev     # reverse the link
        prev = cur          # move prev forward
        cur = nxt           # move cur forward
    return prev             # prev is the new head
```

---

#### Q27. Detect and Find Cycle Start (Floyd's Algorithm)

**Pattern:** Fast/Slow pointers · **TC:** O(n) · **SC:** O(1)

```python
def find_cycle_start(head):
    slow = fast = head
    # Phase 1: detect cycle
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            # Phase 2: find start — reset slow to head
            slow = head
            while slow is not fast:
                slow = slow.next
                fast = fast.next
            return slow   # cycle start node
    return None
```

**Key Insight:** After meeting, resetting one pointer to head and advancing both one step at a time leads them to meet exactly at the cycle start. This works because the distance from head→cycle_start equals meeting_point→cycle_start.

---

#### Q28. Remove N-th Node from End (One Pass)

**Pattern:** Two-pointer gap · **TC:** O(L) · **SC:** O(1)

```python
def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n + 1):   # gap of n+1 between fast and slow
        fast = fast.next
    while fast:
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next  # skip the target
    return dummy.next
```

---

#### Q29. Reorder List (L0→Ln→L1→Ln-1→...)

**Pattern:** Find mid → Reverse 2nd half → Interleave · **TC:** O(n) · **SC:** O(1)

```python
def reorder_list(head):
    # Step 1: find middle (slow/fast pointers)
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next; fast = fast.next.next

    # Step 2: reverse second half
    prev, cur = None, slow.next
    slow.next = None
    while cur:
        nxt = cur.next; cur.next = prev; prev = cur; cur = nxt
    second = prev

    # Step 3: merge (interleave)
    first = head
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second; second.next = tmp1
        first = tmp1; second = tmp2
```

---

### TREE QUESTIONS

---

#### Q30. Level-Order Traversal (BFS)

**Pattern:** BFS with queue, group by level · **TC:** O(n) · **SC:** O(n)

```python
from collections import deque

def level_order(root):
    if not root: return []
    result, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):       # process all nodes at this depth
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        result.append(level)
    return result
```

---

#### Q31. Diameter of Binary Tree

**Pattern:** Post-order DFS, track max path through each node · **TC:** O(n) · **SC:** O(h)

```python
def diameter_of_binary_tree(root):
    ans = [0]
    def depth(node):
        if not node: return 0
        l, r = depth(node.left), depth(node.right)
        ans[0] = max(ans[0], l + r)   # diameter through this node
        return 1 + max(l, r)
    depth(root)
    return ans[0]
```

**eBay Context:** Finding the longest dependency chain in a seller's supply graph.

---

#### Q32. Lowest Common Ancestor (LCA)

**Pattern:** Post-order DFS — if both subtrees return non-null, current node is LCA · **TC:** O(n) · **SC:** O(h)

```python
def lowest_common_ancestor(root, p, q):
    if not root or root is p or root is q:
        return root
    left  = lowest_common_ancestor(root.left,  p, q)
    right = lowest_common_ancestor(root.right, p, q)
    # If both sides found something, this node is the LCA
    return root if left and right else (left or right)
```

---

#### Q33. Serialize / Deserialize Binary Tree

**Pattern:** Pre-order DFS with '#' as null marker · **TC:** O(n) · **SC:** O(n)

```python
def serialize(root):
    if not root: return '#'
    return f"{root.val},{serialize(root.left)},{serialize(root.right)}"

def deserialize(data):
    vals = iter(data.split(','))
    def build():
        v = next(vals)
        if v == '#': return None
        node = TreeNode(int(v))
        node.left  = build()
        node.right = build()
        return node
    return build()
```

---

#### Q34. All Root-to-Leaf Paths Summing to Target

**Pattern:** DFS backtracking with path tracking · **TC:** O(n) · **SC:** O(h)

```python
def path_sum_ii(root, target):
    result = []
    def dfs(node, remaining, path):
        if not node: return
        path.append(node.val)
        if not node.left and not node.right and remaining == node.val:
            result.append(list(path))
        else:
            dfs(node.left,  remaining - node.val, path)
            dfs(node.right, remaining - node.val, path)
        path.pop()     # backtrack
    dfs(root, target, [])
    return result
```

---

### GRAPH QUESTIONS

---

#### Q35. Number of Islands (DFS Flood Fill)

**Pattern:** DFS, mark visited in-place · **TC:** O(m×n) · **SC:** O(m×n)

```python
def num_islands(grid):
    rows, cols = len(grid), len(grid[0])
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        grid[r][c] = '0'   # sink the land
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            dfs(r+dr, c+dc)

    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c); count += 1
    return count
```

---

#### Q36. Course Schedule — Topological Sort (Kahn's BFS)

**Pattern:** In-degree array + BFS · **TC:** O(V+E) · **SC:** O(V+E)

```python
from collections import defaultdict, deque

def can_finish(numCourses, prerequisites):
    graph  = defaultdict(list)
    in_deg = [0] * numCourses
    for dest, src in prerequisites:
        graph[src].append(dest)
        in_deg[dest] += 1

    q = deque(c for c in range(numCourses) if in_deg[c] == 0)
    processed = 0
    while q:
        node = q.popleft(); processed += 1
        for nbr in graph[node]:
            in_deg[nbr] -= 1
            if in_deg[nbr] == 0:
                q.append(nbr)
    return processed == numCourses  # True = no cycle
```

**eBay Context:** Validate that microservice deployment pipelines have no circular dependencies.

---

#### Q37. Shortest Path — Dijkstra's Algorithm

**Pattern:** Min-heap (priority queue) · **TC:** O((V+E) log V) · **SC:** O(V)

```python
import heapq
from collections import defaultdict

def dijkstra(graph, start):
    # graph = {node: [(neighbor, weight), ...]}
    dist = defaultdict(lambda: float('inf'))
    dist[start] = 0
    heap = [(0, start)]   # (cost, node)

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]: continue   # stale entry — skip
        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    return dict(dist)
```

---

#### Q38. Number of Connected Components — Union-Find

**Pattern:** DSU with path compression · **TC:** O(E·α(n)) ≈ O(E) · **SC:** O(n)

```python
def count_components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return False
        parent[ra] = rb; return True

    components = n
    for a, b in edges:
        if union(a, b): components -= 1
    return components
```

---

### DYNAMIC PROGRAMMING QUESTIONS

---

#### Q39. Longest Common Subsequence (LCS)

**Pattern:** 2D DP · **TC:** O(m×n) · **SC:** O(m×n)

```python
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

**eBay Context:** Detect near-duplicate listing titles (similarity score via LCS).

---

#### Q40. Longest Increasing Subsequence (O(n log n))

**Pattern:** Patience sorting with binary search · **TC:** O(n log n) · **SC:** O(n)

```python
import bisect

def lis(nums):
    tails = []
    for n in nums:
        pos = bisect.bisect_left(tails, n)
        if pos == len(tails): tails.append(n)
        else: tails[pos] = n
    return len(tails)
```

---

#### Q41. Edit Distance (Levenshtein)

**Pattern:** 2D DP · **TC:** O(m×n) · **SC:** O(m×n) (or O(n) with rolling array)

```python
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],   # delete
                                   dp[i][j-1],   # insert
                                   dp[i-1][j-1]) # replace
    return dp[m][n]
```

---

#### Q42. 0/1 Knapsack

**Pattern:** 1D DP (iterate capacity in reverse to ensure each item used once) · **TC:** O(n×W) · **SC:** O(W)

```python
def knapsack(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):   # RIGHT to LEFT avoids reuse
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[capacity]
```

---

### BACKTRACKING QUESTIONS

---

#### Q43. All Subsets (Power Set)

**Pattern:** Include/exclude each element · **TC:** O(2ⁿ × n) · **SC:** O(n)

```python
def subsets(nums):
    result = []
    def bt(start, current):
        result.append(list(current))
        for i in range(start, len(nums)):
            current.append(nums[i])
            bt(i + 1, current)
            current.pop()        # backtrack
    bt(0, [])
    return result
```

---

#### Q44. Combination Sum (Reuse Allowed)

**Pattern:** Backtracking with pruning (sorted candidates) · **TC:** O(2^(t/min)) · **SC:** O(t/min)

```python
def combination_sum(candidates, target):
    candidates.sort()
    result = []
    def bt(start, remaining, path):
        if remaining == 0:
            result.append(list(path)); return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining: break   # pruning
            path.append(candidates[i])
            bt(i, remaining - candidates[i], path)  # i allows reuse
            path.pop()
    bt(0, target, [])
    return result
```

---

#### Q45. N-Queens

**Pattern:** DFS with column + diagonal sets for O(1) conflict checking · **TC:** O(n!) · **SC:** O(n)

```python
def n_queens(n):
    result = []
    cols = set(); d1 = set(); d2 = set()
    def bt(row, board):
        if row == n:
            result.append([''.join(r) for r in board]); return
        for col in range(n):
            if col in cols or (row-col) in d1 or (row+col) in d2: continue
            cols.add(col); d1.add(row-col); d2.add(row+col)
            board[row][col] = 'Q'
            bt(row + 1, board)
            board[row][col] = '.'; cols.discard(col)
            d1.discard(row-col); d2.discard(row+col)
    bt(0, [['.']*n for _ in range(n)])
    return result
```

---

### TRIE QUESTION

---

#### Q46. Trie — Insert, Search, Autocomplete

**Pattern:** Linked TrieNodes with children dict · **TC:** O(L) per op · **SC:** O(Σ × L × W)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end   = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True

    def search(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children: return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children: return False
            node = node.children[ch]
        return True

    def autocomplete(self, prefix):
        """Return all words with given prefix."""
        node = self.root
        for ch in prefix:
            if ch not in node.children: return []
            node = node.children[ch]
        results = []
        def dfs(n, path):
            if n.is_end: results.append(prefix + path)
            for ch, child in n.children.items():
                dfs(child, path + ch)
        dfs(node, '')
        return results
```

**eBay Context:** Powers search bar autocomplete — as user types "iph", instantly suggests "iphone 15", "iphone 14 pro", etc.

---

### BIT MANIPULATION QUESTIONS

---

#### Q47. Single Number (XOR)

**TC:** O(n) · **SC:** O(1)

```python
def single_number(nums):
    result = 0
    for n in nums:
        result ^= n     # a^a=0, a^0=a → all pairs cancel out
    return result
```

---

#### Q48. Count Set Bits for 0..n (DP)

**TC:** O(n) · **SC:** O(n)

```python
def count_bits(n):
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i >> 1] + (i & 1)  # right-shift + check last bit
    return dp
```

---

## 🗺️ DSA PATTERN DECISION TREE

```
What type of problem is this?
│
├─ "Find/count pairs or subarrays"     → HashMap / Prefix Sum
├─ "Shortest path, fewest steps"       → BFS (unweighted) / Dijkstra (weighted)
├─ "All solutions / arrangements"      → Backtracking
├─ "Optimize over overlapping subproblems" → Dynamic Programming
├─ "Find in sorted array"             → Binary Search
├─ "Top-K / running median"           → Heap
├─ "Matching brackets / next greater" → Stack (Monotonic)
├─ "Connected components / cycles"    → Union-Find / DFS
├─ "Prefix queries / autocomplete"    → Trie
├─ "Reverse / detect cycle (linked)"  → Two Pointers (slow/fast)
└─ "Deduplicate / check uniqueness"   → HashSet / XOR (bits)
```

---

## ⏱️ COMPLEXITY CHEAT SHEET

| Algorithm | Time | Space |
|---|---|---|
| Two Sum (HashMap) | O(n) | O(n) |
| Binary Search | O(log n) | O(1) |
| BFS / DFS | O(V+E) | O(V) |
| Merge Sort | O(n log n) | O(n) |
| Heap push/pop | O(log n) | O(k) |
| Dijkstra | O((V+E) log V) | O(V) |
| DP — LCS/Edit Distance | O(m×n) | O(m×n) |
| LIS (patience sort) | O(n log n) | O(n) |
| Union-Find (path compress) | O(α(n)) ≈ O(1) | O(n) |
| Trie insert/search | O(L) | O(Σ×L×W) |
| Backtracking (permutations) | O(n! × n) | O(n) |

---

*Sources: Glassdoor (2024–2025), LeetCode Discuss, Reddit r/cscareerquestions, InterviewQuery.com, DataInterview.com, Prepfully, GeeksForGeeks, YouTube interview breakdowns*
