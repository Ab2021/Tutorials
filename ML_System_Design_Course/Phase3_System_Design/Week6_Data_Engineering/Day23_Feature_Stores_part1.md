# Day 23 (Part 1): Advanced Feature Stores

> **Phase**: 6 - Deep Dive
> **Topic**: Data Consistency
> **Focus**: Point-in-Time Joins, Entity Resolution, and Leakage
> **Reading Time**: 60 mins

---

## 1. Point-in-Time (ASOF) Join

The hardest problem in Feature Stores.

### 1.1 The Problem
*   **Labels**: `User A` clicked ad at `12:05`.
*   **Features**:
    *   `User A` view count updated at `12:00` (Value: 5).
    *   `User A` view count updated at `12:10` (Value: 6).
*   **Naive Join**: Joins on `User A`. Might pick `12:10` value (Leakage!).
*   **Correct Join**: Pick the latest value *strictly before* `12:05`. (Value: 5).

### 1.2 Implementation
*   **Pandas**: `pd.merge_asof(direction='backward')`.
*   **Spark**: Window functions or `AsofJoin` (in newer versions).
*   **Complexity**: Requires sorting both tables by time. Expensive.

---

## 2. Entity Resolution (Identity Graph)

### 2.1 The Graph
*   User logs in on Phone (DeviceID A).
*   User logs in on Laptop (CookieID B).
*   **Feature Store**: Must link A and B to `UserID 123`.
*   **Graph DB**: Stores edges `(Device A, User 123)`, `(Cookie B, User 123)`.
*   **Inference**: Query `Device A` -> Traverse to `User 123` -> Fetch features.

---

## 3. Tricky Interview Questions

### Q1: How to handle "Feature Cold Start"?
> **Answer**: New feature added today. No history.
> *   **Backfill**: Compute the feature for the last 1 year using historical logs.
> *   **Default Value**: If logs don't exist, use Mean/Median imputation, but add a binary flag `is_imputed`.

### Q2: Online Store: Redis vs DynamoDB?
> **Answer**:
> *   **Redis**: In-memory. Sub-millisecond. Expensive. Volatile (needs persistence config). Best for high QPS.
> *   **DynamoDB**: SSD-based. Single-digit ms. Serverless. Cheaper for massive datasets. Best for scale.

### Q3: What is "Label Leakage" in Feature Engineering?
> **Answer**: Including information in the feature that is only known *after* the label event.
> *   Example: "Duration of Session" as a feature to predict "Will Purchase". (Purchase ends the session, so duration correlates perfectly).

---

## 4. Practical Edge Case: Timestamp Mismatch
*   **Problem**: Server time (UTC) vs User time (PST) vs Event time (Client device clock).
*   **Fix**: Always enforce **UTC** on server ingestion. Trust server receipt time over client device time (which can be wrong/hacked), unless precision is critical.

