# Lab 01: Scalability Simulation

## Difficulty
🟢 Easy

## Problem Statement
Simulate the difference between Vertical Scaling (Scaling Up) and Horizontal Scaling (Scaling Out).
Create a class `Server` that can handle `capacity` requests per second.
1. **Vertical**: Increase `capacity` of a single server.
2. **Horizontal**: Add more `Server` instances.

Calculate the cost if:
- Vertical: Cost doubles for every 2x capacity.
- Horizontal: Cost is linear (number of servers * base cost).

## Starter Code
```python
class Server:
    def __init__(self, capacity, cost):
        self.capacity = capacity
        self.cost = cost

def compare_scaling(target_capacity):
    # TODO: Calculate cost for vertical vs horizontal
    pass
```
