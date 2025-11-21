# Day 38 Deep Dive: OLAP & Lambda

## 1. Why not MySQL?
*   `SELECT count(*) FROM clicks WHERE ad_id=123` on 1B rows is slow.
*   **OLAP (Online Analytical Processing):**
    *   **Columnar Storage:** Read only needed columns.
    *   **Pre-aggregation:** Store `Sum` instead of raw rows.
    *   **Examples:** Apache Druid, ClickHouse.

## 2. Lambda Architecture (Revisited)
*   **Speed Layer (Flink):** Real-time. "Last 1 hour".
*   **Batch Layer (Spark):** Nightly. "All time". Re-processes raw logs from S3 to fix any stream errors.
*   **Serving Layer:** Merges Stream + Batch.

## 3. Handling Late Events
*   **Scenario:** User clicks at 12:00. Phone is offline. Sends event at 12:10.
*   **Watermark:** Flink waits for 10 mins.
*   **Correctness:**
    *   If event arrives within 10 mins, update 12:00 bucket.
    *   If event arrives after 10 mins, update "Late" bucket or discard.

## 4. Fraud Detection
*   **Click Farm:** 1000 clicks from same IP.
*   **Pattern:** High CTR (Click Through Rate) but low Conversion.
*   **Real-time Filter:** Flink job checks IP reputation and velocity.
