# Day 17: Distributed ID Generation

## 1. The Requirement
*   **Unique:** No collisions.
*   **Sortable:** Time-ordered (useful for DB indexing).
*   **Scalable:** Generate 10k+ IDs/sec without coordination.
*   **Size:** 64-bit (fits in `long`).

## 2. Approaches
### Auto-Increment (MySQL)
*   **Pros:** Simple.
*   **Cons:** SPOF. Hard to shard. Not time-ordered across shards.

### UUID (Universally Unique Identifier)
*   **Format:** 128-bit hex string (`550e8400-e29b...`).
*   **Pros:** No coordination needed.
*   **Cons:** Too big (128-bit). Unordered (Bad for B-Tree fragmentation).

### Snowflake (Twitter)
*   **Format:** 64-bit integer.
    *   **1 bit:** Sign bit (unused).
    *   **41 bits:** Timestamp (milliseconds since epoch).
    *   **10 bits:** Machine ID (Data Center ID + Worker ID).
    *   **12 bits:** Sequence Number (Per millisecond).
*   **Pros:** Sortable, Compact, Distributed.

## 3. Code: Snowflake Concept (Python)
```python
import time

class Snowflake:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.sequence = 0
        self.last_timestamp = -1
        
    def next_id(self):
        timestamp = int(time.time() * 1000)
        
        if timestamp == self.last_timestamp:
            self.sequence = (self.sequence + 1) & 4095
            if self.sequence == 0:
                # Wait for next millisecond
                while timestamp <= self.last_timestamp:
                    timestamp = int(time.time() * 1000)
        else:
            self.sequence = 0
            
        self.last_timestamp = timestamp
        
        return ((timestamp << 22) | (self.worker_id << 12) | self.sequence)
```
