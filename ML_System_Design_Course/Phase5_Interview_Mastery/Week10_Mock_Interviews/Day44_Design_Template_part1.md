# Day 44 (Part 1): Advanced System Design Template

> **Phase**: 6 - Deep Dive
> **Topic**: Level 6+ Design
> **Focus**: Scale, Multi-Region, and Cost
> **Reading Time**: 60 mins

---

## 1. The "Scale it by 100x" Curveball

Interviewer: "Great design. Now imagine traffic grows 100x. What breaks?"

### 1.1 Bottlenecks
*   **DB**: Single Writer won't suffice. -> Sharding.
*   **Cache**: Redis cluster limits. -> Consistent Hashing.
*   **Network**: Bandwidth limits. -> Compression / Edge CDNs.

---

## 2. Multi-Region Active-Active

### 2.1 Architecture
*   Users routed to nearest Region (GeoDNS).
*   **Write**: Write to local DB.
*   **Replication**: Async replication to other regions.
*   **Conflict Resolution**: Last-Write-Wins (LWW) or CRDTs (Conflict-free Replicated Data Types).

---

## 3. Tricky Interview Questions

### Q1: How to design for Cost?
> **Answer**:
> *   **Spot Instances**: For training (Checkpoints needed).
> *   **Tiered Storage**: Hot (SSD) -> Warm (HDD) -> Cold (Glacier).
> *   **Quantization**: Serve INT8 instead of FP16 (2x less hardware).

### Q2: Build vs Buy?
> **Answer**:
> *   **Core IP**: Build (e.g., Ranking Model).
> *   **Commodity**: Buy (e.g., Auth0 for login, PagerDuty for alerts).

### Q3: Handling "Hot Keys" (Celebrity Problem)?
> **Answer**:
> *   Justin Bieber posts. 10M reads.
> *   **Local Cache**: Cache on the *app server* (L1), not just Redis (L2).
> *   **Replication**: Replicate the key to multiple Redis nodes.

---

## 4. Practical Edge Case: Privacy (GDPR)
*   **Right to be Forgotten**: User deletes account.
*   **ML Impact**: Can we keep the model trained on their data?
*   **Answer**: Usually yes (Aggregated), but strictly: Retrain without their data (Machine Unlearning - Research topic).

