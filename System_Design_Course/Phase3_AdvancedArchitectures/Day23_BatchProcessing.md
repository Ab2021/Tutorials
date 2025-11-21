# Day 23: Batch Processing

## 1. What is Batch Processing?
Processing a large volume of data all at once (usually offline/scheduled).
*   **Input:** Bounded dataset (File, DB Table).
*   **Output:** Derived data (Report, Aggregation).
*   **Latency:** Hours/Days.
*   **Throughput:** High.

## 2. MapReduce (The Paradigm)
*   **Map:** Transform input `(k1, v1)` -> `list(k2, v2)`.
*   **Shuffle:** Group by `k2`. `(k2, [v2, v2, ...])`.
*   **Reduce:** Aggregate `(k2, [v2...])` -> `(k2, v3)`.
*   **Example (Word Count):**
    *   Map: `("apple", 1)`, `("banana", 1)`, `("apple", 1)`.
    *   Shuffle: `("apple", [1, 1])`, `("banana", [1])`.
    *   Reduce: `("apple", 2)`, `("banana", 1)`.

## 3. Hadoop (The Pioneer)
*   **HDFS:** Distributed File System. Stores data.
*   **YARN:** Resource Manager. Allocates CPU/RAM.
*   **MapReduce:** Execution Engine. (Writes to disk after every step - Slow).

## 4. Spark (The Modern Standard)
*   **In-Memory:** Keeps intermediate data in RAM. 100x faster than Hadoop.
*   **DAG (Directed Acyclic Graph):** Optimizes execution plan.
*   **RDD (Resilient Distributed Dataset):** Immutable, fault-tolerant collection of objects.
