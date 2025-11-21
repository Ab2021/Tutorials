# Day 30 (Part 1): Advanced Fraud Detection & Graphs

> **Phase**: 6 - Deep Dive
> **Topic**: Adversarial ML
> **Focus**: GNNs, PageRank, and Active Learning
> **Reading Time**: 60 mins

---

## 1. Graph Features

### 1.1 PageRank
*   Measure of node centrality.
*   **Fraud**: "BadRank". If a node is connected to bad nodes, it is bad.
*   **Personalized PageRank**: Random walk starting from known fraud nodes.

### 1.2 Connected Components
*   Find islands of users.
*   **Fraud Ring**: A dense cluster of 100 nodes all interconnected (Sybil attack).

---

## 2. Graph Neural Networks (GNN)

### 2.1 GraphSAGE
*   **Aggregate**: Mean/Max of neighbor features.
*   **Update**: Combine self-feature with aggregate.
*   **Inductive**: Can generalize to *new* nodes (unlike GCN which requires fixed graph).

---

## 3. Active Learning

Fraud labels are expensive (Manual Review).

### 3.1 Uncertainty Sampling
*   Select samples where model is 50/50 confident.
*   Send to human review.
*   Retrain.

---

## 4. Tricky Interview Questions

### Q1: How to handle "Camouflage" in Graphs?
> **Answer**: Fraudsters connect to popular nodes (Celebrities, Amazon) to look normal.
> *   **Fix**: Weight edges by inverse degree. Connection to Amazon (Degree 1M) means less than connection to User X (Degree 5).

### Q2: Explain "Oversmoothing" in GNNs.
> **Answer**:
> *   If GNN is too deep (many layers), all node embeddings converge to the same value (average of graph).
> *   **Fix**: Use shallow GNNs (2-3 layers) or Skip Connections.

### Q3: Precision vs Recall for Fraud?
> **Answer**:
> *   **Recall** is king. Missed fraud = Loss.
> *   **Precision** matters for User Experience (False Positive = Blocked Card).
> *   **Tradeoff**: Block if Score > 0.9. Step-up Auth (SMS) if Score > 0.5.

---

## 5. Practical Edge Case: Device Fingerprinting
*   **Technique**: Hash (UserAgent + ScreenRes + TimeZone + Fonts).
*   **Use**: Detect 100 accounts from same physical device.

