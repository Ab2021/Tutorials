# Day 4 Deep Dive: WAL & SSTables

## 1. Write-Ahead Log (WAL)
*   **Problem:** Writing to disk is slow. If we write to RAM and crash, data is lost.
*   **Solution:** Append the command to a log file (WAL) on disk *before* applying to memory.
*   **Mechanism:** Sequential Write (Fast).
*   **Recovery:** On restart, replay WAL to restore state.
*   **Used by:** Postgres, MySQL, Cassandra.

## 2. SSTables (Sorted String Tables)
*   **Definition:** Immutable file on disk where keys are sorted.
*   **Structure:**
    *   `Key: Value` pairs sorted by Key.
    *   Index file (Sparse index) to jump to offsets.
*   **Compaction:**
    *   Over time, you get many SSTables (`Data_1.sst`, `Data_2.sst`).
    *   Background process merges them (Merge Sort) and removes deleted keys (Tombstones).

## 3. Bloom Filters
*   **Problem:** LSM Trees have slow reads (checking many files).
*   **Solution:** Probabilistic Data Structure.
*   **Query:** "Is Key X in this SSTable?"
    *   **No:** Definitely No. (Skip file).
    *   **Yes:** Maybe. (Check file).
*   **Benefit:** Drastically reduces disk lookups for non-existent keys.

## 4. Isolation Levels (Dirty Reads vs Phantom Reads)
*   **Read Uncommitted:** Dirty reads allowed. (Fastest, Dangerous).
*   **Read Committed:** No dirty reads. (Default in Postgres).
*   **Repeatable Read:** No non-repeatable reads. (MySQL Default).
*   **Serializable:** Strict. No concurrency. (Slowest).
