# Day 6: Caching Strategies

## 1. Why Cache?
*   **Latency:** RAM (ns) vs Disk (ms).
*   **Throughput:** Serve more requests.
*   **Cost:** Reduce load on expensive DBs.

## 2. Caching Patterns
### Cache-Aside (Lazy Loading)
*   **Read:** App checks Cache. If miss, read DB, write to Cache, return.
*   **Write:** App writes to DB, deletes Key from Cache.
*   **Pros:** Resilient to cache failure. Only requests data is cached.
*   **Cons:** Cache miss penalty. Stale data (if DB updated directly).

### Write-Through
*   **Write:** App writes to Cache. Cache writes to DB (synchronously).
*   **Pros:** Data consistency.
*   **Cons:** Slow write (2 writes).

### Write-Back (Write-Behind)
*   **Write:** App writes to Cache. Cache writes to DB (asynchronously).
*   **Pros:** Fast writes.
*   **Cons:** Data loss if Cache crashes before flushing.

## 3. Eviction Policies
*   **LRU (Least Recently Used):** Remove the item that hasn't been used for the longest time. (Most common).
*   **LFU (Least Frequently Used):** Remove the item with fewest hits.
*   **FIFO (First In First Out):** Queue.

## 4. Code: LRU Cache (Python)
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key) # Mark as recently used
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False) # Remove first (LRU)
```
