# Day 17 Deep Dive: Instagram ID Generation

## 1. The Problem
*   Instagram needed IDs for Images, Comments, Users.
*   Needed to be sortable by time (for pagination).
*   Didn't want to introduce a new service (like Zookeeper/Snowflake servers) to keep architecture simple.

## 2. The Solution: PL/PGSQL (Postgres)
*   Instagram uses Postgres sharded into thousands of logical shards.
*   **ID Format (64-bit):**
    *   **41 bits:** Timestamp (ms).
    *   **13 bits:** Shard ID.
    *   **10 bits:** Auto-increment sequence (Local to shard).
*   **Mechanism:**
    *   Each shard generates its own IDs using a Stored Procedure.
    *   `ID = (Timestamp << 23) | (ShardID << 10) | (Sequence % 1024)`.

## 3. Why this is brilliant?
*   **No new infrastructure:** Uses existing DB.
*   **Locality:** The ID contains the Shard ID. If you have the ID, you know exactly which shard to query. `ShardID = ID >> 10`.
*   **Sortable:** Time is the leading component.

## 4. Comparison: YouTube IDs
*   YouTube uses Base64 (A-Z, a-z, 0-9, -, _).
*   11 characters. `dQw4w9WgXcQ`.
*   Not time-ordered. (Prevents people from guessing how many videos are uploaded).
