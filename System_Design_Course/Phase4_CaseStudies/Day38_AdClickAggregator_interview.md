# Day 38 Interview Prep: Design Ad Click Aggregator

## Q1: Row-based vs Column-based DB?
**Answer:**
*   **Row (MySQL):** Good for transactions (`INSERT`, `UPDATE`). Bad for analytics (reads all columns).
*   **Column (Cassandra/Druid):** Stores columns separately. Good for `SUM(clicks)`. Compressible (Run Length Encoding).

## Q2: How to handle "Re-statement" of data?
**Answer:**
*   **Scenario:** We found a bug in the counting logic yesterday.
*   **Solution:**
    *   Since we stored **Raw Logs** in S3 (Immutable), we can re-run the Batch Job (Spark) with fixed logic.
    *   Overwrite the data in Druid for that day.

## Q3: What is the Star Schema?
**Answer:**
*   **Fact Table:** `Clicks (time, user_id, ad_id, geo_id)`. Huge.
*   **Dimension Tables:** `Ads (id, name, budget)`, `Users (id, name)`. Small.
*   **Join:** Join Fact with Dimensions at query time.

## Q4: How to scale the ingestion?
**Answer:**
*   **Kafka:** Partition by `AdID`.
*   **Flink:** Scale out workers. Each worker handles a subset of partitions.
*   **Backpressure:** If Flink is slow, Kafka buffers data.
