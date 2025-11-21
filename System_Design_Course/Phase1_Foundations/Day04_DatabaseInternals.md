# Day 4: Database Internals I

## 1. ACID Properties
*   **Atomicity:** All or nothing. (Transactions).
*   **Consistency:** Data remains valid according to rules/constraints.
*   **Isolation:** Transactions don't interfere. (Read Committed, Serializable).
*   **Durability:** Once committed, data is saved (even if power fails).

## 2. Storage Engines: B-Trees vs LSM Trees
### B-Trees (Read Optimized)
*   **Used by:** MySQL (InnoDB), PostgreSQL.
*   **Structure:** Balanced Tree. Data in leaf nodes.
*   **Pros:** Fast Reads ($O(\log N)$). Good for Range Queries.
*   **Cons:** Random Writes are slow (disk seeks to update pages).

### LSM Trees (Log-Structured Merge Trees) (Write Optimized)
*   **Used by:** Cassandra, RocksDB, HBase.
*   **Structure:**
    1.  Write to MemTable (RAM).
    2.  Flush to SSTable (Disk, Sorted String Table).
    3.  Compaction merges SSTables.
*   **Pros:** Fast Writes (Append-only).
*   **Cons:** Slower Reads (Check MemTable -> SSTable 1 -> SSTable 2...). Bloom Filters help.

## 3. Indexing
*   **Clustered Index:** Data is stored with the key (Primary Key). Only one per table.
*   **Non-Clustered Index:** Stores Key + Pointer to data. Can have multiple.

## 4. Code: B-Tree Concept (Python)
```python
# Simplified Node
class Node:
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []
        self.children = []

# B-Trees keep keys sorted and balanced.
# Searching is O(log N).
```
