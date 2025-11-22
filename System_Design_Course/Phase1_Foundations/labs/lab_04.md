# Lab 04: Consistent Hashing

## Difficulty
🔴 Hard

## Problem Statement
Implement Consistent Hashing to distribute keys across N servers.
- Map servers to a "ring" (hash space).
- Map keys to the same ring.
- Assign key to the next server on the ring.
- Handle adding/removing servers with minimal key remapping.

## Starter Code
```python
import hashlib

class ConsistentHash:
    def __init__(self, nodes=None, replicas=3):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        
    def add_node(self, node):
        # TODO: Add node and replicas to ring
        pass
        
    def get_node(self, key):
        # TODO: Find nearest node
        pass
```
