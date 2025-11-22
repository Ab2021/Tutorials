# Lab 05: Monte Carlo Estimation

## Difficulty
🟢 Easy

## Problem Statement
Estimate the value of Pi using Monte Carlo simulation.
1. Sample random points (x, y) in [-1, 1].
2. Check if point is inside unit circle (x^2 + y^2 <= 1).
3. Ratio of points inside / total points approx Pi/4.

## Starter Code
```python
import random

def estimate_pi(num_samples):
    inside = 0
    for _ in range(num_samples):
        # TODO: Sample and check
        pass
    return 4 * inside / num_samples
```
