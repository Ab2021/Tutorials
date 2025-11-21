# Day 24 Deep Dive: Lambda vs Kappa Architecture

## 1. The Problem
How to build a system that is both low latency (Real-time) and accurate (Batch correction)?

## 2. Lambda Architecture
*   **Concept:** Run two layers in parallel.
    *   **Speed Layer (Stream):** Processes data fast. Approximate results. (Storm/Flink).
    *   **Batch Layer (Hadoop):** Processes all data. Accurate results. Overwrites Speed Layer.
    *   **Serving Layer:** Merges results.
*   **Pros:** Robust. If Stream code has bug, Batch fixes it.
*   **Cons:** Complexity. Maintain two codebases (Java for Hadoop, Scala for Flink).

## 3. Kappa Architecture
*   **Concept:** Stream is the only source of truth.
*   **Design:** Remove Batch Layer. Everything is a Stream.
*   **Reprocessing:** If code has a bug, replay the Kafka topic from the beginning with new code.
*   **Pros:** One codebase. Simpler.
*   **Cons:** Replaying 1PB of Kafka data is hard (Retention limits).

## 4. Case Study: Real-time Fraud Detection
*   **Input:** Credit Card Transaction.
*   **Pipeline:**
    1.  **Ingest:** Kafka.
    2.  **Process:** Flink.
    3.  **State:** Flink stores "User's spending in last 1 hour" in RocksDB (local state).
    4.  **Logic:** If `CurrentTx > 5 * AvgSpending`, flag as Fraud.
    5.  **Output:** Alert Topic.
