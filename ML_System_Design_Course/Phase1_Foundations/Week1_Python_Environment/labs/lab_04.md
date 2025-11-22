# Lab 04: Matplotlib Dashboard

## Difficulty
🟢 Easy

## Problem Statement
Create a function that takes a DataFrame and generates a dashboard with 4 subplots:
1. Histogram of a numerical column.
2. Scatter plot of two numerical columns.
3. Bar chart of a categorical column.
4. Line plot of a time-series column.

## Starter Code
```python
import matplotlib.pyplot as plt

def create_dashboard(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # TODO: Plot on axes[0,0], axes[0,1], etc.
    plt.show()
```
