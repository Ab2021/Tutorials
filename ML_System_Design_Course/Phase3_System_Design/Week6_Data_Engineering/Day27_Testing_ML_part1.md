# Day 27 (Part 1): Advanced Testing & Experimentation

> **Phase**: 6 - Deep Dive
> **Topic**: Safe Deployment
> **Focus**: Shadow Mode, Interleaving, and Bias
> **Reading Time**: 60 mins

---

## 1. Shadow Mode Implementation

How to implement Shadow Mode without doubling latency?

### 1.1 Async Shadowing
*   **Request** -> **Load Balancer** -> **Model A (Prod)** -> **Response**.
*   **Load Balancer** -> **Queue (Kafka)** -> **Model B (Shadow)**.
*   **Pros**: Zero latency impact.
*   **Cons**: Shadow model doesn't run in exact real-time context (e.g., if feature store updates).

---

## 2. Advanced A/B Testing

### 2.1 Interleaving (Ranking)
*   **A/B**: User sees either List A OR List B. High variance.
*   **Interleaving**: User sees mixed list (A1, B1, A2, B2).
*   **Metric**: Which item was clicked? A or B?
*   **Pros**: 100x more sensitive. Needs fewer users.

### 2.2 A/A Testing
*   **Goal**: Verify the pipeline.
*   **Setup**: Split traffic 50/50. Give both groups the SAME model.
*   **Result**: Should be no difference. If there is, your splitting logic or metric calculation is buggy.

---

## 3. Tricky Interview Questions

### Q1: Novelty Effect?
> **Answer**: Users click a new feature just because it's new/different, not better. Effect wears off.
> *   **Fix**: Run experiment for longer (Cohort analysis).

### Q2: Primacy Effect?
> **Answer**: Users are used to the old UI. New UI confuses them initially (dip in metrics), but might be better long term.

### Q3: How to test a model that affects the ecosystem (e.g., Uber pricing)?
> **Answer**: **Switchback Testing**.
> *   You can't split Users (Marketplace is shared).
> *   Split by **Time**.
> *   10:00-10:20 -> Model A.
> *   10:20-10:40 -> Model B.
> *   Randomize time slots.

---

## 4. Practical Edge Case: Carryover Effect
*   **Problem**: User in Group A today, Group B tomorrow.
*   **Fix**: Consistent Hashing (`hash(user_id) % 100`). User stays in same group forever.

