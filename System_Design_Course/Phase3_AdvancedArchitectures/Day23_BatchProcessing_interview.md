# Day 23 Interview Prep: Batch Processing

## Q1: Hadoop vs Spark?
**Answer:**
*   **Hadoop MapReduce:** Writes to disk after every Map and Reduce. Good for massive jobs where RAM is insufficient.
*   **Spark:** Keeps data in memory. 10-100x faster. Supports SQL, Streaming, ML.

## Q2: What is a Shuffle?
**Answer:**
*   The process of redistributing data across partitions (network transfer).
*   Required for `groupBy`, `join`, `sort`.
*   Most expensive operation in distributed computing.

## Q3: How to handle Data Skew?
**Answer:**
*   **Symptom:** Job stuck at 99%.
*   **Fix:**
    *   **Salting:** Add random numbers to hot keys to split them.
    *   **Broadcast Join:** If joining large table with small table, broadcast the small one.

## Q4: Parquet vs CSV?
**Answer:**
*   **CSV:** Row-based. Text. Slow to parse. No schema.
*   **Parquet:** Columnar. Binary. Compressed. Schema included.
*   **Benefit:** If you only need 1 column (`SELECT age FROM users`), Parquet reads only that column. CSV reads the whole file.
