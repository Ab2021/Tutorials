# Lab 02: Load Balancer Implementation

## Difficulty
🟡 Medium

## Problem Statement
Implement a Load Balancer with two strategies:
1. **Round Robin**: Distribute requests sequentially.
2. **Random**: Distribute requests randomly.

## Starter Code
```python
import random

class LoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.current_index = 0

    def get_server_round_robin(self):
        # TODO: Implement Round Robin
        pass

    def get_server_random(self):
        # TODO: Implement Random
        pass
```
