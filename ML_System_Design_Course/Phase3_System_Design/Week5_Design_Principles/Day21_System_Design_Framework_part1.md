# Day 21 (Part 1): Advanced System Design Framework

> **Phase**: 6 - Deep Dive
> **Topic**: The Math of Systems
> **Focus**: Capacity Planning, Latency Budgets, and Cost
> **Reading Time**: 60 mins

---

## 1. Capacity Planning Math

"Design for 100M DAU." What does that mean in hardware?

### 1.1 The Formula
*   **QPS**: $\text{DAU} \times \text{Requests/User} / 86400$.
    *   Example: $10^8 \times 10 / 10^5 = 10,000$ QPS.
*   **Peak QPS**: Average QPS $\times$ Multiplier (usually 2x-5x). Design for 50k QPS.
*   **Cores Needed**: $\text{Peak QPS} \times \text{Latency (s)}$.
    *   If Latency = 100ms (0.1s).
    *   $50,000 \times 0.1 = 5,000$ Cores.
*   **Machines**: If 1 machine has 32 cores -> $5000 / 32 \approx 157$ Machines.

### 1.2 Storage Math
*   **Feature Store**: 100M Users $\times$ 1KB features = 100GB (RAM/Redis).
*   **Logs**: 1B requests $\times$ 1KB = 1TB/day. 1PB/3 years.

---

## 2. Latency Budgeting

### 2.1 The Fan-Out Problem
*   Service A calls B, C, D in parallel.
*   Total Latency = $\max(B, C, D)$.
*   **P99 Tail Latency**: If B, C, D have 1% chance of being slow, the chance A is slow is $1 - (0.99)^3 \approx 3\%$.
*   **Conclusion**: Tail latency amplifies in microservices.

---

## 3. Tricky Interview Questions

### Q1: QPS vs. Concurrency?
> **Answer**:
> *   **QPS**: Rate (Requests per second). Flow.
> *   **Concurrency**: Number of requests active *at the same time*. Stock.
> *   **Little's Law**: $\text{Concurrency} = \text{QPS} \times \text{Latency}$.
> *   If QPS=100 and Latency=2s, you need to handle 200 concurrent connections.

### Q2: How to estimate cost of a GPU cluster?
> **Answer**:
> *   **A100 Cost**: ~$3/hour (On-demand).
> *   **Training**: 70B model takes 1 month on 128 GPUs.
> *   $128 \times 24 \times 30 \times 3 = \$276,000$.
> *   **Inference**: 100 Machines $\times$ $3/hr $\times$ 24 $\times$ 365 = $2.6M/year.

### Q3: Availability vs Consistency (CAP Theorem)?
> **Answer**:
> *   **ML Serving**: Usually chooses **Availability** (AP). Better to show a slightly stale recommendation than an error page.
> *   **Feature Store**: Eventual Consistency is usually fine for user profiles, but strict consistency needed for "Balance" or "Inventory".

---

## 4. Practical Edge Case: Thundering Herd
*   **Scenario**: Cache expires. 10,000 requests hit DB simultaneously. DB dies.
*   **Fix**:
    1.  **Jitter**: Add random noise to expiration time.
    2.  **Request Coalescing**: Collapse 10k requests for Key X into 1 DB call.

