# Day 30: Case Study - Fraud Detection (FinTech)

> **Phase**: 3 - System Design
> **Week**: 5 - Design Principles (Case Studies)
> **Focus**: Anomaly Detection & Real-Time Scoring
> **Reading Time**: 60 mins

---

## 1. The Problem

**Goal**: Detect fraudulent transactions in < 500ms.
**Constraint**: Extremely Imbalanced Data (0.1% Fraud). High cost of False Positives (Blocking a legit user).

---

## 2. The Architecture

### 2.1 Feature Engineering (The Secret Sauce)
*   **Aggregations**: "Number of transactions in last 1 hour". "Amount spent in last 24h".
*   **Velocity**: "Time since last transaction".
*   **Geospatial**: "Distance between current location and last location".
*   **Graph Features**: "Is this user connected to known fraudsters?" (PageRank, Connected Components).

### 2.2 The Model
*   **Ensemble**: XGBoost / LightGBM (Handles tabular data and imbalance well).
*   **Unsupervised**: Isolation Forest / Autoencoders (To detect *new* patterns of fraud not in training data).

### 2.3 Real-Time Serving
*   **Feature Store**: Critical. Must fetch "Last 1 hour count" in ms.
*   **Rules Engine**: Hard filters run *before* or *after* the model.
    *   *Pre-filter*: "If Amount > $1M, Block".
    *   *Post-filter*: "If Model Score > 0.9, Block". "If 0.7-0.9, Step-up Auth (SMS)".

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Imbalanced Data
**Solution**:
*   **Sampling**: Undersample majority.
*   **Loss**: Focal Loss.
*   **Metric**: PR-AUC (Precision-Recall Area Under Curve). Do NOT use Accuracy.

### Challenge 2: Adversarial Attacks
**Scenario**: Fraudsters learn the rules. "If I spend < $100, I'm safe."
**Solution**:
*   **Online Learning**: Update model frequently.
*   **Hidden Features**: Use features fraudsters can't control (Device Fingerprint, Typing Cadence).
*   **Graph Neural Networks (GNN)**: Detect rings of colluding accounts.

---

## 4. Interview Preparation

### System Design Questions

**Q1: How do you handle "Label Delay"?**
> **Answer**: A transaction happens today. We only know it's fraud if the user reports it (Chargeback) 30 days later.
> *   **Ingestion**: Ingest labels asynchronously.
> *   **Training**: Train on data from 30 days ago (Mature labels).
> *   **Short-term**: Use "Probable Fraud" labels from manual review or user reports as weak signals.

**Q2: Why use GNNs for Fraud?**
> **Answer**: Fraudsters often work in rings (Sybil attacks). They share devices, IP addresses, or funding sources. A GNN can aggregate information from neighbors. "If my neighbor is fraud, I am likely fraud." Standard tabular models miss this structural info.

**Q3: Design the "Velocity" feature calculation.**
> **Answer**:
> *   **Naive**: `SELECT count(*) FROM txns WHERE user=X AND time > now - 1h`. (Too slow on SQL).
> *   **Optimized**: Streaming (Flink). Maintain a sliding window count in Redis. When a new event arrives, increment count, expire old events. $O(1)$ lookup.

---

## 5. Further Reading
- [Stripe: Radar for Fraud Detection](https://stripe.com/radar/guide)
- [Graph Neural Networks for Fraud Detection](https://arxiv.org/abs/2008.08692)
