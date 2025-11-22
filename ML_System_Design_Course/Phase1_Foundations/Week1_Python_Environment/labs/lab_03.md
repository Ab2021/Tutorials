# Lab 03: Pandas Data Cleaning

## Difficulty
🟢 Easy

## Problem Statement
Given a CSV file with missing values and inconsistent formatting:
1. Load data using Pandas.
2. Fill missing numerical values with the mean.
3. Drop rows with missing categorical values.
4. Convert date strings to datetime objects.

## Starter Code
```python
import pandas as pd

def clean_data(file_path):
    df = pd.read_csv(file_path)
    # TODO: Implement cleaning steps
    return df
```
