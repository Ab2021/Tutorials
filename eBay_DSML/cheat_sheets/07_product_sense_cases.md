# eBay DS/ML — Product Sense & Case Studies

> Product sense questions test your ability to think like a business owner.
> Always use frameworks, never jump to solutions.

---

## 🧩 Part 1: The Diagnostic Framework

### When Asked: "Metric X Dropped by Y%. Investigate."

```
Step 1: VERIFY THE DATA
  ├── Is the data pipeline working correctly?
  ├── Did tracking/logging change recently?
  ├── Compare internal vs external analytics
  └── Check for ETL delays or data duplication

Step 2: SCOPE THE PROBLEM
  ├── How large is the drop? (5% vs 50%)
  ├── Sudden cliff or gradual decline?
  ├── When did it start? (correlate with releases)
  └── Is it global or localized?

Step 3: DECOMPOSE THE METRIC (Metric Tree)
  GMV = Traffic × Conversion Rate × AOV
  ├── Which component changed?
  ├── Traffic: unique visitors, sessions, page views
  ├── Conversion: view→cart→checkout→purchase
  └── AOV: price × quantity per order

Step 4: SEGMENT THE DATA
  ├── Platform: mobile app / mobile web / desktop
  ├── Geography: US / UK / DE / AU / IN
  ├── User type: new vs returning
  ├── Traffic source: organic / paid / email / direct
  ├── Category: electronics / fashion / collectibles
  └── Seller type: casual / power seller / business

Step 5: CHECK EXTERNAL FACTORS
  ├── Seasonality (post-holiday, back-to-school)
  ├── Competitor actions (Amazon Prime Day)
  ├── Macroeconomic shifts
  ├── Outages (payment processor, shipping partner)
  └── Regulatory changes

Step 6: FORM HYPOTHESIS → VERIFY → RECOMMEND
```

---

## 💡 Part 2: 15 Product Sense Case Studies

### Case 1: GMV Drop Investigation

**Q:** *"eBay's GMV dropped 8% week-over-week. Walk me through your investigation."*

**Model Answer:**
```
1. VERIFY: Check data pipeline → no ETL issues found

2. DECOMPOSE:
   GMV = Traffic × Conversion × AOV
   - Traffic: ↓3% (modest decline)
   - Conversion: ↓4% (significant)
   - AOV: ↓1% (minor)
   → Primary driver: Conversion rate decline

3. SEGMENT conversion by platform:
   - Desktop: -1% (within normal range)
   - Mobile Web: -2% (slight)
   - Mobile App (iOS): -12% ← ANOMALY!
   → Isolated to iOS app

4. CORRELATE with timeline:
   - iOS app v5.2 released 5 days ago
   - Checkout funnel on iOS: cart→checkout step is -15%
   → Likely a bug in the iOS checkout flow

5. VERIFY: Check crash logs, error rates on iOS checkout
   → Confirm: JS error on payment selection screen for iOS 17+

6. RECOMMEND:
   - Immediate: Rollback to v5.1 or hotfix for payment screen
   - Short-term: Add automated checkout flow tests to CI/CD
   - Impact estimate: Fixing will recover ~$X million/week in GMV
```

### Case 2: Define Metrics for New Feature

**Q:** *"eBay launches an AI Shopping Assistant chatbot. How would you measure success?"*

```
NORTH STAR: Task Completion Rate
  = sessions where user completes a purchase after using assistant

DRIVER METRICS:
  - Adoption Rate: % of eligible sessions that engage with assistant
  - Queries per Session: depth of engagement
  - Click-Through Rate: % of assistant suggestions that get clicked
  - Conversion Lift: compare assisted vs non-assisted sessions

QUALITY METRICS:
  - Answer Relevance Score: human-rated sample of responses
  - Hallucination Rate: % of factually incorrect answers
  - Latency: p50/p95 response time (target: <2 seconds)

GUARDRAIL METRICS:
  - Overall Conversion Rate: must not decrease for non-users
  - Customer Satisfaction (NPS/CSAT): post-session surveys
  - Return Rate: ensure assistant isn't overselling
  - Support Ticket Volume: should decrease, not increase

LONG-TERM:
  - Repeat Usage Rate: do users come back to the assistant?
  - LTV Impact: do assisted users have higher lifetime value?
```

### Case 3: Two-Sided Marketplace Balance

**Q:** *"eBay's buyer count is growing 10% YoY but seller count is declining 5%. What's happening and what would you do?"*

```
INVESTIGATION:
1. Segment seller decline:
   - New seller registration: down or up?
   - Seller churn: which sellers are leaving?
   - Power sellers vs casual sellers?

2. Hypotheses:
   a) Fees too high → sellers moving to competitors (Mercari, Poshmark)
   b) Buyer protection policies favor buyers → sellers feel disadvantaged
   c) Listing complexity → high friction for new sellers
   d) Market consolidation → fewer sellers handling more volume

3. Data check:
   - If GMV is flat despite fewer sellers → existing sellers are larger
   - If listings per seller ↑ → concentration risk
   - If seller satisfaction (NPS) ↓ → policy/fee issues

4. Recommendations:
   - Run seller exit surveys (qualitative)
   - Analyze cohort: sellers who churned vs stayed (feature differences)
   - Test reduced fees for new sellers (A/B test by geo)
   - Simplify listing flow (reduce steps from 8 to 3)
   - AI listing assistant (auto-generate descriptions from photos)
```

### Case 4: Search Ranking Metrics

**Q:** *"You're the DS lead for eBay's search team. Define your metric framework."*

```
PRIMARY METRICS:
  - GMV per Search: total value of purchases attributed to search
  - Conversion Rate per Search: searches that lead to purchase

QUALITY METRICS:
  - NDCG@10: ranking quality (position of clicked/purchased items)
  - Zero-Result Rate: % of queries returning no results
  - Click-Through Rate: CTR on first page results
  - Mean Reciprocal Rank (MRR): position of first clicked result

ENGAGEMENT METRICS:
  - Pages per Search Session
  - Search Refinement Rate: how often users modify their query
  - Bounce Rate: searches with no click

GUARDRAIL METRICS:
  - Search Latency p99: must stay under 200ms
  - Seller Impression Fairness: ensure diverse sellers get exposure
  - Ad Revenue: search ads should not degrade organic relevance

LONG-TERM:
  - Buyer Retention: do better search results drive repeat visits?
  - Category Coverage: are niche categories well-served?
```

### Case 5-15: Quick-Fire Product Questions

5. *"How would you measure the success of 'Buy It Now' vs Auction format?"*
   → Compare: sell-through rate, GMV per listing, time-to-sale, seller preference

6. *"eBay introduces free shipping on orders over $50. What metrics do you track?"*
   → AOV distribution (do orders cluster at $50?), shipping cost absorption, seller margins, conversion lift

7. *"Define a seller quality score. What features would you include?"*
   → Ship time, item accuracy, return rate, response time, review sentiment, listing quality

8. *"Cart abandonment rate increased 15%. How do you investigate?"*
   → Funnel stage analysis, payment methods, shipping cost visibility, mobile vs desktop, competitor comparison

9. *"Should eBay launch a subscription program for frequent buyers?"*
   → Analyze: purchase frequency distribution, willingness-to-pay, competitive landscape (Amazon Prime), cannibalization risk

10. *"How do you detect if a seller is gaming the search ranking algorithm?"*
    → Anomaly detection on: keyword stuffing, fake reviews, self-purchasing, abnormal return patterns

11. *"eBay's mobile app conversion is 30% lower than desktop. Why and what to do?"*
    → Screen size → simplified checkout, saved payment, one-click buy, push notifications for abandoned carts

12. *"How would you evaluate the impact of adding user reviews to listing pages?"*
    → A/B test: conversion rate, time on page, return rate, review submission rate, seller response rate

13. *"Design a notification system. When should eBay send push notifications?"*
    → Price drops on watched items, back-in-stock, auction ending soon, abandoned cart, personalized recommendations
    → Guardrail: unsubscribe rate, notification-to-engagement ratio, do NOT over-notify

14. *"eBay wants to enter the live shopping/streaming market. How do you evaluate feasibility?"*
    → Market size, competitor analysis (TikTok Shop), user survey, pilot in one category, success metrics (engagement, GMV from streams)

15. *"A new recommendation algorithm shows +3% CTR but -1% conversion. Ship or not?"*
    → CTR up + conversion down = users clicking more but buying less = recommendations are clickbait-y but not relevant
    → Do NOT ship. Optimize for purchase, not just clicks. Check: are recommendations lower quality? Are they in wrong categories?

---

## 📊 Part 3: eBay Key Metrics Cheat Sheet

| Metric | Formula | Why It Matters |
|---|---|---|
| **GMV** | Σ(sale_price × quantity) for completed orders | Total marketplace throughput |
| **Take Rate** | eBay revenue / GMV | eBay's monetization efficiency |
| **Conversion Rate** | Purchases / Unique Visitors | Purchase funnel health |
| **AOV** | GMV / Number of Orders | Per-order revenue |
| **CAC** | Marketing Spend / New Buyers Acquired | Customer acquisition efficiency |
| **LTV** | avg_order_value × purchase_frequency × customer_lifespan | Long-term customer value |
| **Sell-Through Rate** | Items Sold / Items Listed | Supply-demand match |
| **Listing Quality Score** | f(title, images, description, attributes, price) | Listing completeness |
| **Search Relevance (NDCG)** | Normalized Discounted Cumulative Gain | Ranking quality |
| **Buyer Retention (M+1)** | Buyers in month M who purchase in M+1 / Total M buyers | Repeat purchase health |
| **Seller NPS** | Net Promoter Score from seller surveys | Seller satisfaction |
| **Defect Rate** | (Returns + Complaints) / Total Transactions | Trust & quality signal |
