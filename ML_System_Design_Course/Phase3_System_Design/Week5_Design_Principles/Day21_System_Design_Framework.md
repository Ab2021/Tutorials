# Day 21: The ML System Design Framework

> **Phase**: 3 - System Design
> **Week**: 5 - Design Principles
> **Focus**: The Interview Framework
> **Reading Time**: 60 mins

---

## 1. The 5-Step Framework

In an ML System Design interview (and in real life), jumping straight to "I'll use a Transformer" is a red flag. You must follow a structured approach.

### Step 1: Requirements & Constraints
*   **Business Goal**: What are we solving? (e.g., Increase watch time, reduce fraud loss).
*   **Constraints**:
    *   **Latency**: 100ms or 1 hour? (Real-time vs. Batch).
    *   **Throughput**: 1k QPS or 1M QPS?
    *   **Budget**: Cost of inference?
    *   **Privacy**: PII handling?

### Step 2: Data Engineering
*   **Sources**: User logs, DB, Third-party APIs.
*   **Labeling**: How do we get ground truth? (Implicit feedback vs. Human raters).
*   **Features**: User profile, Context (Time/Device), Item metadata.

### Step 3: Modeling
*   **Baseline**: Start simple (Logistic Regression).
*   **Model Selection**: Deep Learning? Tree-based?
*   **Objective Function**: Log Loss? Ranking Loss (NDCG)?

### Step 4: Evaluation
*   **Offline**: AUC, RMSE, Precision@k.
*   **Online**: A/B Testing metrics (CTR, Revenue).

### Step 5: Serving & Infrastructure
*   **Architecture**: Microservices? Monolith?
*   **Hardware**: CPU vs. GPU inference.
*   **Monitoring**: Drift detection.

---

## 2. Key Trade-offs

System design is about trade-offs. There is no "correct" answer, only justified choices.

### 2.1 Batch vs. Real-time
*   **Batch**: Precompute predictions for all users every night.
    *   *Pros*: Simple, high throughput, cheap.
    *   *Cons*: Stale data. Cannot react to user's current session.
*   **Real-time**: Compute prediction on-demand.
    *   *Pros*: Fresh. Can use current session context.
    *   *Cons*: Complex, strict latency requirements, expensive.

### 2.2 Accuracy vs. Latency
*   A 100-layer Transformer is accurate but takes 500ms.
*   A Logistic Regression is less accurate but takes 1ms.
*   **Solution**: **Two-Tower Architecture** or **Cascade**. Use a fast model to filter 1M items to 100, then a slow model to rank the top 100.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: The Cold Start Problem
**Scenario**: A new user signs up. You have no history. Recommendation system fails.
**Solution**:
*   **Heuristics**: Recommend "Most Popular" items globally.
*   **Contextual Bandits**: Explore different items to learn preferences quickly.
*   **Metadata**: Use age/location provided during signup.

### Challenge 2: Feedback Loops
**Scenario**: The model recommends "Clickbait". Users click it. The model learns "Clickbait is good". It recommends more. Quality degrades.
**Solution**:
*   **Positional Bias Correction**: Account for the fact that top items get clicked just because they are at the top.
*   **Long-term Rewards**: Optimize for "Retention" or "Watch Time" instead of just "Clicks".

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: What is the difference between Throughput and Latency?**
> **Answer**:
> *   **Latency**: Time taken to process a *single* request (e.g., 50ms). Critical for user experience.
> *   **Throughput**: Number of requests processed per second (e.g., 10,000 QPS). Critical for scaling and cost.
> *   You can have high throughput but poor latency (batch processing).

**Q2: How do you handle a sudden spike in traffic (10x load)?**
> **Answer**:
> 1.  **Auto-scaling**: Kubernetes HPA (Horizontal Pod Autoscaler) adds more replicas.
> 2.  **Caching**: Serve precomputed results from Redis/Memcached.
> 3.  **Degradation**: Switch to a lighter model or a heuristic (fallback) to survive the spike.
> 4.  **Queueing**: Buffer requests in Kafka (increases latency but prevents crash).

**Q3: Why is "Online Evaluation" necessary if Offline metrics are good?**
> **Answer**: Offline metrics (AUC) are proxies. They are calculated on historical data which might be biased (Selection Bias). They cannot measure user behavior changes (e.g., users getting bored). Only Online A/B tests measure the true business impact.

---

## 5. Further Reading
- [Machine Learning System Design Interview (Alex Xu)](https://bytebytego.com/)
- [Rules of Machine Learning (Google)](https://developers.google.com/machine-learning/guides/rules-of-ml)
