# Day 30 Deep Dive: Implementation (Python)

## 1. Consistent Hashing
```python
import hashlib
import bisect

class ConsistentHash:
    def __init__(self, nodes, replicas=3):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        for node in nodes:
            self.add_node(node)

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node):
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            bisect.insort(self.sorted_keys, key)

    def get_node(self, key):
        if not self.ring: return None
        hash_val = self._hash(key)
        idx = bisect.bisect(self.sorted_keys, hash_val)
        if idx == len(self.sorted_keys):
            idx = 0
        return self.ring[self.sorted_keys[idx]]

# Test
ch = ConsistentHash(["NodeA", "NodeB", "NodeC"])
print(ch.get_node("user_123")) # NodeB
print(ch.get_node("user_456")) # NodeA
```

## 2. Quorum Read/Write
*   **N=3, W=2, R=2.**
*   **Write:**
    *   Coordinator sends `PUT` to Node A, B, C.
    *   Wait for 2 ACKs.
    *   Return Success.
*   **Read:**
    *   Coordinator sends `GET` to Node A, B, C.
    *   Wait for 2 responses.
    *   Compare timestamps (Vector Clock). Return latest.

## 3. Hinted Handoff
*   If Node A is down, write to Node D (temporary).
*   Node D keeps a "Hint": "This data belongs to A".
*   When A comes back, D sends data to A.
