# 🗺️ Flipkart Senior Data Scientist — Interview Game Plan
### Process, Strategy & Execution Playbook

---

## 🔍 ABOUT YOUR INTERVIEWER

| Attribute | Detail |
|---|---|
| **Name** | Alex Rivera |
| **Current Role** | Research Director, Data Science @ Flipkart (since June 2021) |
| **Education** | PhD (Machine Learning, University of Copenhagen) + M.Tech (IIT Bombay) |
| **PhD Research** | Voxel classification for medical image segmentation using CNNs; Co-author of *"Deep feature learning for knee cartilage segmentation using triplanar CNN"* (MICCAI 2013) |
| **Published Work (Flipkart)** | *"Portfolio Risk Management Model for EMI-based loan in E-Commerce"* — 300+ features, transaction-level delinquency prediction, post-acquisition credit risk |
| **Previous** | SVP Data Science @ Info Edge (Naukri, Jeevansathi, 99acres) |
| **Style** | Academic rigor + industry pragmatism — will PROBE depth on fundamentals |

### 🧠 What Alex Is Looking For (inferred from background):
1. **Mathematical depth** — Can you derive things? Do you understand WHY algorithms work?
2. **Production mindset** — Have you shipped things? Do you understand deployment challenges?
3. **Creative problem framing** — Can you turn a fuzzy business ask into a precise ML problem?
4. **Risk/credit/fraud domain** — He has deep expertise here from his EMI paper; he'll connect your fraud experience to Flipkart's context
5. **Research instinct** — He comes from academia; he respects novel thinking and intellectual curiosity
6. **Scale thinking** — Info Edge + Flipkart experience means he knows what works at 100M+ user scale

---

## 📋 FLIPKART INTERVIEW PROCESS (Senior DS)

Based on research and LeetCode/Glassdoor experiences:

```
Round 1: Recruiter Screening (30 min)
├── Your background, salary expectations, notice period
├── Why Flipkart? Why this role?
└── Basic cultural fit

Round 2: Technical Screening (45-60 min) — Possible via phone/video
├── Resume deep-dive: Pick any 2 projects and drill deep
├── ML fundamentals: 3-4 concept questions
└── Probability/Statistics: 1-2 problems

Round 3: ML / Mathematics Modeling (60 min) [THE HARDEST]
├── Mathematical derivations (logistic regression, GBM)
├── Advanced ML theory (regularization, optimization, loss functions)
└── For GenAI teams: transformers, fine-tuning, RLHF, RAG deep dives

Round 4: ML System Design / Case Study (60 min)
├── Open-ended business problem → design ML solution end-to-end
├── Justify EVERY decision: feature choice, model choice, evaluation metric
├── Cover: data pipeline, training, serving, monitoring, feedback loop
└── Typical problems: Fraud detection, recommendation, credit risk, pricing

Round 5: Coding Round (45 min) [May be earlier or combined]
├── LeetCode Medium-Hard DSA (Python)
├── Data manipulation (Pandas/SQL window functions)
└── May include: implement an ML algorithm from scratch (e.g., logistic regression)

Round 6: Presentation Round (60 min) [Sometimes included for Senior roles]
├── Present your best project in detail
├── 15-20 min presentation + 40-45 min Q&A
└── Focus on: Problem → Data → Model → Results → Business Impact

Round 7: Hiring Manager / Culture (45 min)
├── Behavioral: Ownership, bias-for-action, cross-functional collaboration
├── Leadership: Managing teams, influencing without authority
└── Fit: Why Flipkart specifically, long-term career goals
```

---

## 🎯 YOUR COMPETITIVE EDGE (How to Position Yourself)

### Unique Differentiators vs. Average Senior DS Candidate:

1. **RAG + Agentic AI in Production (Fraud Context)**
   - Very few senior DS candidates have shipped agentic AI for high-stakes fraud detection
   - Connect directly to Flipkart's Triksha framework and their human-in-the-loop fraud strategy

2. **Cross-Domain Depth (Healthcare + Insurance + Pharma)**
   - Shows adaptability — you can learn Flipkart's e-commerce domain quickly
   - Healthcare ML (BERT + clinical NLP) is rigorous; Flipkart knows this

3. **Full-Stack MLOps Experience**
   - MLflow + Kubeflow + Airflow + GCP + Docker/K8s — shows production mindset
   - Alex's research background means HE values deployment experience (researchers often lack this)

4. **Knowledge Graph (Neo4j) Experience**
   - Directly relevant: Flipkart's fraud linkage verification uses graph-based approaches
   - Seller ring detection, fake account clustering — graph ML is critical here

5. **9 Years = Senior Maturity**
   - Can lead 4-5 person teams, manage stakeholders, own P&L impact
   - Awards track record (5 awards across Chubb + EXL) = proven excellence

---

## 🗓️ 72-HOUR PREP CHECKLIST

### Day 1 (Technical Deep Dive)
- [ ] Review logistic regression derivation + XGBoost math (Section A of Q&A doc)
- [ ] Practice RAG evaluation framework (Ragas metrics — faithfulness, relevance, recall)
- [ ] Revise BERT attention mechanism + transformer architecture
- [ ] Read Alex's MICCAI 2013 triplanar CNN paper abstract (know his academic context)

### Day 2 (System Design + Coding)
- [ ] Practice fraud system design out loud (use system design in Q&A doc — time yourself 40 min)
- [ ] Solve 2 LeetCode mediums (graph BFS/DFS + dynamic programming)
- [ ] Practice SQL: window functions (LAG, LEAD, RANK, PARTITION BY)
- [ ] Read Flipkart's fraud detection blog posts (Triksha framework, X-ray returns)

### Day 3 (Behavioral + Mock Run)
- [ ] Do full mock self-introduction (record yourself — check for filler words)
- [ ] Prepare 5 STAR stories from resume (Situation, Task, Action, Result)
- [ ] Review Flipkart's values: Customer-First, Audacity, Bias for Action, Ownership, Integrity
- [ ] Research Flipkart recent news: AI investments 2025-26, Walmart partnership, quick commerce

---

## 💬 FLIPKART VALUES — HOW TO DEMONSTRATE AUTHENTICALLY

| Value | How You've Lived It |
|---|---|
| **Customer-First** | "At CVS/Aetna, I built survival analysis tools for healthcare providers to reduce patient readmissions — that's customer-first with life stakes. Same principle applies to Flipkart customers." |
| **Audacity** | "Built an Agentic AI fraud system when the technology was barely production-ready — took the risk because the business value was clear." |
| **Bias for Action** | "At Chubb, I shipped a working fraud detection MVP in 6 weeks — began with the most critical pipeline first, iterated from there rather than planning for 6 months." |
| **Ownership** | "Led DS offshore team of 3 FTEs with 5/5 SLA ratings for 3 consecutive quarters — owned the delivery end-to-end, not just the modeling." |
| **Integrity** | "When the marketing attribution model was misleading business teams, I proactively flagged the limitation and redesigned the experiment — even though it delayed my project timeline." |

---

## 🧮 CODING ROUND PREP

### Key Python DSA Patterns to Review

```python
# Pattern 1: Sliding Window (Fraud velocity features)
def max_fraud_in_window(transactions, k):
    # Count max fraudulent transactions in any k-minute window
    from collections import deque
    fraud_count = sum(transactions[:k])
    max_fraud = fraud_count
    for i in range(k, len(transactions)):
        fraud_count += transactions[i] - transactions[i-k]
        max_fraud = max(max_fraud, fraud_count)
    return max_fraud

# Pattern 2: Graph BFS (Fraud ring detection)
from collections import deque
def find_fraud_ring(graph, start):
    # BFS to find all accounts connected to a fraudulent account
    visited = set()
    queue = deque([start])
    ring = []
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            ring.append(node)
            queue.extend(graph.get(node, []))
    return ring

# Pattern 3: SQL Window Functions (Common in DS interviews)
"""
SELECT 
    user_id,
    transaction_date,
    amount,
    SUM(amount) OVER (PARTITION BY user_id ORDER BY transaction_date 
                      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7day_spend,
    COUNT(*) OVER (PARTITION BY user_id ORDER BY transaction_date 
                   ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS txn_count_7d,
    AVG(amount) OVER (PARTITION BY user_id ORDER BY transaction_date 
                      ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS avg_30d_spend,
    RANK() OVER (PARTITION BY user_id ORDER BY amount DESC) AS spend_rank
FROM transactions
WHERE transaction_date >= '2024-01-01'
"""
```

### Key SQL Patterns for DS Interviews

```sql
-- 1. Find users with suddenly high transaction velocity (fraud signal)
WITH user_daily AS (
    SELECT user_id, DATE(txn_time) as txn_date, COUNT(*) as daily_txns
    FROM transactions
    GROUP BY user_id, DATE(txn_time)
),
user_avg AS (
    SELECT user_id, AVG(daily_txns) as avg_daily_txns, STDDEV(daily_txns) as std_daily_txns
    FROM user_daily
    GROUP BY user_id
)
SELECT u.user_id, ud.txn_date, ud.daily_txns, ua.avg_daily_txns
FROM user_daily ud
JOIN user_avg ua ON ud.user_id = ua.user_id
WHERE ud.daily_txns > ua.avg_daily_txns + 3 * ua.std_daily_txns  -- > 3 sigma anomaly

-- 2. Cohort retention analysis (common Flipkart DS question)
WITH first_purchase AS (
    SELECT user_id, MIN(DATE(order_date)) as cohort_month
    FROM orders GROUP BY user_id
),
cohort_activity AS (
    SELECT fp.user_id, fp.cohort_month,
           DATEDIFF(DATE(o.order_date), fp.cohort_month) as days_since_first
    FROM first_purchase fp
    JOIN orders o ON fp.user_id = o.user_id
)
SELECT cohort_month, 
       COUNT(DISTINCT CASE WHEN days_since_first BETWEEN 0 AND 30 THEN user_id END) as month_1,
       COUNT(DISTINCT CASE WHEN days_since_first BETWEEN 31 AND 60 THEN user_id END) as month_2,
       COUNT(DISTINCT CASE WHEN days_since_since_first BETWEEN 61 AND 90 THEN user_id END) as month_3
FROM cohort_activity
GROUP BY cohort_month
```

---

## 📊 KEY FLIPKART CONTEXT TO WEAVE IN

### E-commerce ML Problems Flipkart Actively Works On:
- **Fraud:** Seller fraud, buyer fraud, return fraud, payment fraud, fake reviews
- **Risk:** EMI credit scoring (Flipkart Pay Later), seller financial risk, inventory risk
- **Recommendation:** Personalized product ranking, search re-ranking, similar item recommendation
- **Pricing:** Dynamic pricing, demand forecasting, promotion optimization
- **Logistics:** Delivery ETA prediction, route optimization, last-mile ML
- **GenAI (2025-26 priority):** Triksha (LLM security), agentic customer support, seller insights
- **NLP:** Product cataloging, review analysis, search understanding, multi-language support

### Flipkart's AI Stack (what you know):
- **Triksha:** Adversarial LLM security framework for GenAI applications
- **Fraud Linkage Verification:** Cross-referencing device IDs, addresses against fraud history (graph-based)
- **AI X-ray scanning:** Computer vision for return fraud detection on high-value products
- **Build vs. Buy:** High-volume generic tasks → buy (API); core competitive differentiation → build in-house

---

## 🚨 COMMON MISTAKES TO AVOID

1. **Don't just say XGBoost — explain WHY** (Alex will push you on math)
2. **Don't ignore evaluation metrics** — always tie model performance to business metric
3. **Don't skip deployment** — always mention how model goes to production and how it's monitored
4. **Don't be generic about Flipkart** — show you understand their specific fraud/risk context
5. **Don't over-claim on scalability** — be honest about scale (Chubb wasn't 350M users; frame as transferable thinking)
6. **Don't forget the "why Flipkart" question** — prep a specific, genuine answer

---

## 💡 YOUR "WHY FLIPKART" ANSWER

> "Flipkart operates at a scale where the data science problems are genuinely hard — 350M+ users, real-time fraud across millions of daily transactions, seller ecosystem risk management with complex network effects. My RAG + agentic AI fraud work at Chubb has given me the methodological toolkit, but I want to apply it at a scale where the feedback loops are faster and the impact is visible in real-time. 
>
> I also find Flipkart's approach to AI particularly thoughtful — the Triksha framework for LLM security, the human-in-the-loop philosophy for high-stakes decisions — this is exactly the responsible AI deployment mindset I believe in. 
>
> And specifically, Alex's published work on EMI risk modeling — that's the kind of rigorous, end-to-end thinking about credit risk at e-commerce scale that I want to learn from and contribute to."

---

*End of Strategy & Process Document*
