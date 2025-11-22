# Lab 02: Time Complexity Comparison

## Difficulty
🟢 Easy

## Estimated Time
45 mins

## Learning Objectives
- Compare theoretical time complexity with actual runtime
- Understand the impact of input size on performance
- Visualize O(n^2) vs O(n log n) growth

## Problem Statement
Implement two sorting algorithms: Bubble Sort (O(n^2)) and Merge Sort (O(n log n)). 
Measure their execution time for increasing input sizes (n = 100, 1000, 5000, 10000).
Plot or print the results to verify the theoretical complexity.

## Requirements
1. Implement `bubble_sort(arr)`
2. Implement `merge_sort(arr)`
3. Create a benchmarking function that runs both on random arrays
4. Compare the time taken

## Starter Code
```python
import time
import random
import matplotlib.pyplot as plt  # Optional for plotting

def bubble_sort(arr):
    # TODO: Implement Bubble Sort
    pass

def merge_sort(arr):
    # TODO: Implement Merge Sort
    pass

def benchmark():
    sizes = [100, 1000, 5000, 10000]
    for n in sizes:
        arr = [random.randint(0, 10000) for _ in range(n)]
        
        # Measure Bubble Sort
        start = time.time()
        bubble_sort(arr.copy())
        print(f"Bubble Sort (n={n}): {time.time() - start:.4f}s")
        
        # Measure Merge Sort
        # TODO: Measure Merge Sort
```
