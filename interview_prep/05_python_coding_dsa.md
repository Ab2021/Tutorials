# Python Coding, DSA & OOP — Senior Data Scientist Interview Prep
### Target Companies: General Mills India | LSEG (London Stock Exchange Group) India
### Last Updated: November 2026

---

> [!IMPORTANT]
> **General Mills** focuses on clean, practical Python for CPG/retail analytics pipelines — expect data-wrangling tasks, modular functions, and business-context coding.
> **LSEG** tests via Codility/HackerRank, then live coding — emphasises string parsing, financial time-series logic, edge-case handling, and performance optimization.

---

## Table of Contents

1. [Python Comprehensions & Functional Programming](#1-comprehensions--functional-programming)
2. [Generators & Iterators](#2-generators--iterators)
3. [Decorators](#3-decorators)
4. [Context Managers](#4-context-managers)
5. [OOP for Data Science](#5-oop-for-data-science)
6. [Python Internals & Memory Management](#6-python-internals--memory-management)
7. [DSA — Hash Maps & Frequency Counting](#7-dsa--hash-maps--frequency-counting)
8. [DSA — Sorting Algorithms](#8-dsa--sorting-algorithms)
9. [DSA — Sliding Window & Two Pointers](#9-dsa--sliding-window--two-pointers)
10. [Performance: Vectorization vs Loops](#10-performance-vectorization-vs-loops)
11. [Multiprocessing & Threading for Data Tasks](#11-multiprocessing--threading-for-data-tasks)
12. [Memory-Efficient Data Loading](#12-memory-efficient-data-loading)
13. [Google Colab Specific Patterns](#13-google-colab-specific-patterns)
14. [Quick-Fire Interview Questions](#14-quick-fire-interview-questions)

---

## 1. Comprehensions & Functional Programming

### Q1 — List, Dict, Set Comprehensions [Easy]

**Interview Prompt:** *"Rewrite these loops using comprehensions."*

```python
import numpy as np

# ---- List Comprehension ----
# Task: Square all even numbers from 1 to 20
squares_even = [x**2 for x in range(1, 21) if x % 2 == 0]
# [4, 16, 36, 64, 100, 144, 196, 256, 324, 400]

# ---- Dict Comprehension ----
# Task: Build word frequency map from a list of tokens
tokens = ["apple", "banana", "apple", "cherry", "banana", "apple"]
freq = {word: tokens.count(word) for word in set(tokens)}
# {'apple': 3, 'banana': 2, 'cherry': 1}

# Better O(n) approach using Counter:
from collections import Counter
freq_fast = dict(Counter(tokens))

# ---- Set Comprehension ----
# Task: Unique product categories from raw data (may have duplicates/whitespace)
raw_cats = ["  Cereals", "Snacks", "cereals", "Beverages", "SNACKS"]
unique_cats = {cat.strip().title() for cat in raw_cats}
# {'Cereals', 'Snacks', 'Beverages'}

# ---- Nested Comprehension (Matrix Flatten) ----
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [val for row in matrix for val in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# ---- Generator Expression (memory-efficient alternative) ----
total = sum(x**2 for x in range(1_000_000))  # No list built in RAM
```

> [!TIP]
> Generator expressions `(x for x in ...)` save memory vs list comprehensions `[x for x in ...]` — mention this in interviews when processing large datasets.

---

### Q2 — Lambda, map, filter, reduce [Easy]

**Interview Prompt:** *"Use functional programming tools to process this sales data."*

```python
from functools import reduce

sales = [1200, 850, 2300, 450, 1750, 3100, 980]

# --- map: apply transformation to every element ---
# Apply 10% tax on all sales
sales_with_tax = list(map(lambda x: round(x * 1.10, 2), sales))

# --- filter: keep elements satisfying a condition ---
# Keep only high-value transactions (> 1000)
high_value = list(filter(lambda x: x > 1000, sales))

# --- reduce: fold list into single value ---
# Running product of all sales (illustrative; in practice use math.prod or np.prod)
product = reduce(lambda acc, x: acc * x, sales)

# --- Chaining all three: normalize high-value sales ---
normalized = list(map(
    lambda x: (x - min(high_value)) / (max(high_value) - min(high_value)),
    filter(lambda x: x > 1000, sales)
))

# ---- Sorted with custom key ----
products = [("Cheerios", 450), ("Häagen-Dazs", 1200), ("Pillsbury", 300)]
# Sort by revenue descending
products_sorted = sorted(products, key=lambda x: x[1], reverse=True)
# [('Häagen-Dazs', 1200), ('Cheerios', 450), ('Pillsbury', 300)]

# ---- Multiple sort keys ----
records = [("A", 2, 100), ("B", 1, 200), ("A", 1, 150)]
records_sorted = sorted(records, key=lambda x: (x[0], x[1], -x[2]))
```

> [!NOTE]
> `reduce` requires an import from `functools` in Python 3. Interviewers sometimes ask why it was moved — answer: Guido felt it was not Pythonic and hard to read.

---

## 2. Generators & Iterators

### Q3 — Generator for Large File Processing [Medium]

**Interview Prompt (LSEG):** *"You have a 50 GB CSV of financial tick data. How do you process it without running out of memory?"*

```python
import csv
from pathlib import Path

# --- Approach 1: Line-by-line generator ---
def stream_csv(filepath: str, batch_size: int = 1000):
    """
    Lazily yield batches of rows from a large CSV.
    Memory footprint: O(batch_size) rather than O(file_size).
    """
    batch = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:          # Yield the last (possibly partial) batch
            yield batch

# Usage:
# for batch in stream_csv("tick_data.csv", batch_size=5000):
#     df_batch = pd.DataFrame(batch)
#     process(df_batch)

# --- Approach 2: Pandas chunking (simpler for DS work) ---
import pandas as pd

def process_in_chunks(filepath: str, chunk_size: int = 50_000):
    """Read CSV in chunks, compute aggregates, concat results."""
    results = []
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        # Example: compute mean price per symbol per chunk
        agg = chunk.groupby("symbol")["price"].mean()
        results.append(agg)
    return pd.concat(results).groupby(level=0).mean()

# --- Generator pipeline pattern ---
def parse_prices(rows):
    """Stage 1: Parse string prices to float."""
    for row in rows:
        row["price"] = float(row["price"])
        yield row

def filter_active(rows, min_volume: int = 100):
    """Stage 2: Keep only rows with sufficient volume."""
    for row in rows:
        if int(row.get("volume", 0)) >= min_volume:
            yield row

def compute_vwap(rows):
    """Stage 3: Tag each row with rolling VWAP numerator."""
    for row in rows:
        row["pv"] = row["price"] * int(row["volume"])
        yield row

# Compose pipeline lazily — no intermediate lists created
# pipeline = compute_vwap(filter_active(parse_prices(stream_csv("ticks.csv"))))
# for row in pipeline: ...
```

---

### Q4 — Custom Iterator Class [Medium]

**Interview Prompt:** *"Implement a DataBatchIterator class that yields mini-batches from a NumPy array."*

```python
import numpy as np

class DataBatchIterator:
    """
    Iterates over a dataset in fixed-size mini-batches.
    Supports shuffle for stochastic gradient descent.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        assert len(X) == len(y), "X and y must have same length"
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
        self._indices = np.arange(self.n_samples)

    def __len__(self):
        """Number of batches."""
        return int(np.ceil(self.n_samples / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self._indices)
        self._current = 0
        return self

    def __next__(self):
        if self._current >= self.n_samples:
            raise StopIteration
        idx = self._indices[self._current : self._current + self.batch_size]
        self._current += self.batch_size
        return self.X[idx], self.y[idx]


# Usage:
X = np.random.rand(1000, 20)
y = np.random.randint(0, 2, 1000)
loader = DataBatchIterator(X, y, batch_size=64)

for X_batch, y_batch in loader:
    # train_step(X_batch, y_batch)
    pass
print(f"Total batches: {len(loader)}")  # 16
```

---

## 3. Decorators

### Q5 — Timing & Logging Decorator [Easy]

```python
import time
import functools
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")

def timer(func):
    """Measure and log execution time of any function."""
    @functools.wraps(func)   # Preserves __name__, __doc__ of wrapped function
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"[{func.__name__}] completed in {elapsed:.4f}s")
        return result
    return wrapper

@timer
def train_demand_forecaster(data, epochs=10):
    time.sleep(0.5)   # Simulate training
    return {"rmse": 0.034}

result = train_demand_forecaster(data=None, epochs=50)
# 2026-11-13 17:00:00 — [train_demand_forecaster] completed in 0.5003s
```

---

### Q6 — Retry Decorator for Flaky API Calls [Medium]

**Interview Prompt (LSEG):** *"You call a market data API that sometimes times out. How do you make your pipeline resilient?"*

```python
import time
import functools
import requests

def retry(max_attempts: int = 3, delay: float = 1.0, exceptions=(Exception,)):
    """
    Parameterized decorator: retries a function on specified exceptions.
    max_attempts: total tries including the first
    delay: seconds between retries (doubles each time — exponential backoff)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    wait = delay * (2 ** (attempt - 1))   # Exponential backoff
                    print(f"Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
            raise RuntimeError(f"All {max_attempts} attempts failed.") from last_exc
        return wrapper
    return decorator


@retry(max_attempts=4, delay=0.5, exceptions=(requests.Timeout, requests.ConnectionError))
def fetch_market_data(symbol: str) -> dict:
    """Fetch real-time price data from an external API."""
    response = requests.get(f"https://api.example.com/price/{symbol}", timeout=3)
    response.raise_for_status()
    return response.json()
```

---

### Q7 — Memoization / Cache Decorator [Medium]

```python
import functools
import time

# --- Manual memoization ---
def memoize(func):
    """Cache results keyed by arguments. Useful for expensive feature computations."""
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper.cache = cache          # Expose cache for inspection
    wrapper.cache_clear = cache.clear
    return wrapper

@memoize
def compute_tfidf_score(doc_id: int, term: str) -> float:
    time.sleep(0.01)  # Simulate expensive NLP computation
    return hash((doc_id, term)) % 100 / 100.0

# --- Built-in alternative (prefer this in production) ---
from functools import lru_cache

@lru_cache(maxsize=256)
def fibonacci(n: int) -> int:
    """Classic example; in DS: think sliding window feature computation."""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(50))          # Computed instantly after first call
print(fibonacci.cache_info()) # CacheInfo(hits=48, misses=51, maxsize=256, currsize=51)
```

---

## 4. Context Managers

### Q8 — Custom Context Manager for DB Connections & File Handling [Medium]

**Interview Prompt:** *"How do you ensure a database connection is always closed, even if an error occurs?"*

```python
import contextlib
import sqlite3
import time

# --- Class-based context manager ---
class DatabaseConnection:
    """
    Manages SQLite connection lifecycle.
    Guarantees connection.close() is called even on exceptions.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Dict-like row access
        print(f"Connected to {self.db_path}")
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.conn.rollback()
            print(f"Transaction rolled back due to: {exc_val}")
        else:
            self.conn.commit()
        self.conn.close()
        print("Connection closed.")
        return False  # Don't suppress exceptions


# Usage:
# with DatabaseConnection("sales.db") as conn:
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM transactions WHERE amount > 1000")
#     rows = cursor.fetchall()


# --- Generator-based context manager (using contextlib) ---
@contextlib.contextmanager
def timer_context(label: str):
    """Context manager to time a code block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{label}] took {elapsed:.4f}s")

# Usage:
# with timer_context("Feature Engineering"):
#     df["rolling_mean"] = df["sales"].rolling(7).mean()


# --- Temporarily redirect stdout (useful in testing) ---
import io, sys

@contextlib.contextmanager
def suppress_output():
    """Silence verbose library output during batch processing."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

# with suppress_output():
#     model.fit(X_train, y_train)   # No verbose output
```

---

## 5. OOP for Data Science

### Q9 — Custom Scikit-Learn Transformer [Medium]

**Interview Prompt:** *"Build a custom transformer that can be plugged into a sklearn Pipeline."*

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps outliers using IQR method.
    Inherits BaseEstimator → get_params/set_params for GridSearchCV.
    Inherits TransformerMixin → fit_transform for free.
    """
    def __init__(self, factor: float = 1.5):
        self.factor = factor            # Store hyperparameters as attributes
        self.lower_bounds_: dict = {}   # Learned params end with underscore (sklearn convention)
        self.upper_bounds_: dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Learn bounds from training data only (avoid data leakage)."""
        for col in X.select_dtypes(include="number").columns:
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - self.factor * iqr
            self.upper_bounds_[col] = q3 + self.factor * iqr
        return self                      # Always return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned bounds. Never mutate the input."""
        X_out = X.copy()
        for col, lb in self.lower_bounds_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].clip(lower=lb, upper=self.upper_bounds_[col])
        return X_out


class LogTransformer(BaseEstimator, TransformerMixin):
    """Log1p transform for right-skewed features (e.g., revenue, volume)."""
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.columns_ = self.columns or list(X.select_dtypes(include="number").columns)
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out[self.columns_] = np.log1p(X_out[self.columns_].clip(lower=0))
        return X_out


# Full sklearn pipeline — survives GridSearchCV without data leakage
pipe = Pipeline([
    ("cap_outliers", OutlierCapper(factor=1.5)),
    ("log_transform", LogTransformer()),
    ("scaler",        StandardScaler()),
    ("classifier",    LogisticRegression(max_iter=1000)),
])

# pipe.fit(X_train, y_train)
# y_pred = pipe.predict(X_test)
```

---

### Q10 — OOP: ML Model Registry / Factory Pattern [Hard]

**Interview Prompt (General Mills):** *"Design a ModelRegistry class that lets the team plug in different forecasting models without changing downstream code."*

```python
from abc import ABC, abstractmethod
from typing import Dict, Type
import numpy as np

# ---- Abstract Base Class ----
class ForecasterBase(ABC):
    """Interface every forecasting model must implement."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ForecasterBase":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return RMSE (lower is better)."""
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# ---- Concrete Implementations ----
class NaiveForecaster(ForecasterBase):
    """Predict last known value (baseline)."""
    def fit(self, X, y):
        self._last_value = y[-1]
        return self

    def predict(self, X):
        return np.full(len(X), self._last_value)

    def score(self, X, y):
        preds = self.predict(X)
        return float(np.sqrt(np.mean((preds - y) ** 2)))


class MovingAverageForecaster(ForecasterBase):
    """Predict rolling mean of last `window` periods."""
    def __init__(self, window: int = 7):
        self.window = window

    def fit(self, X, y):
        self._mean = np.mean(y[-self.window:])
        return self

    def predict(self, X):
        return np.full(len(X), self._mean)

    def score(self, X, y):
        return float(np.sqrt(np.mean((self.predict(X) - y) ** 2)))


# ---- Registry (Factory Pattern) ----
class ModelRegistry:
    """
    Central store for model classes. Enables plug-and-play model selection.
    Usage: register a class once, instantiate by name later.
    """
    _registry: Dict[str, Type[ForecasterBase]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""
        def decorator(model_cls):
            if not issubclass(model_cls, ForecasterBase):
                raise TypeError(f"{model_cls} must subclass ForecasterBase")
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> ForecasterBase:
        if name not in cls._registry:
            raise KeyError(f"Unknown model: {name}. Available: {list(cls._registry)}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_models(cls):
        return list(cls._registry.keys())


ModelRegistry.register("naive")(NaiveForecaster)
ModelRegistry.register("moving_avg")(MovingAverageForecaster)

# ---- Usage ----
model_name = "moving_avg"   # Comes from config file / experiment tracker
model = ModelRegistry.get(model_name, window=14)

y_train = np.random.rand(100)
y_test  = np.random.rand(20)
model.fit(None, y_train)
rmse = model.score(None, y_test)
print(f"{model_name} RMSE: {rmse:.4f}")
print(f"Available models: {ModelRegistry.list_models()}")
```

---

### Q11 — Dataclass for Feature Config [Easy]

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class FeatureConfig:
    """
    Dataclasses: clean alternative to __init__ boilerplate.
    Auto-generates __init__, __repr__, __eq__.
    """
    name: str
    dtype: str                           # "numeric", "categorical", "datetime"
    impute_strategy: str = "median"
    encode: bool = False
    top_k_categories: Optional[int] = None
    engineered_features: List[str] = field(default_factory=list)   # Mutable default!

    def __post_init__(self):
        """Validate after auto-generated __init__ runs."""
        valid_dtypes = {"numeric", "categorical", "datetime"}
        if self.dtype not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}")
        if self.encode and self.dtype != "categorical":
            raise ValueError("encode=True only valid for categorical features")


# Usage
price_feat = FeatureConfig(name="unit_price", dtype="numeric", impute_strategy="mean")
brand_feat  = FeatureConfig(name="brand", dtype="categorical", encode=True, top_k_categories=50)
print(brand_feat)
# FeatureConfig(name='brand', dtype='categorical', ..., encode=True, top_k_categories=50, ...)
```

---

## 6. Python Internals & Memory Management

### Q12 — Memory Optimization: Downcasting & dtype Selection [Medium]

**Interview Prompt:** *"Your Pandas DataFrame with 10M rows is using 8 GB of RAM. How do you reduce it?"*

```python
import pandas as pd
import numpy as np

def memory_usage_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1024**2

def optimize_dtypes(df: pd.DataFrame, categorical_threshold: float = 0.5) -> pd.DataFrame:
    """
    Reduce DataFrame memory footprint by:
    1. Downcasting int64 → smallest int type
    2. Downcasting float64 → float32
    3. Converting low-cardinality strings → category
    """
    df_opt = df.copy()
    before_mb = memory_usage_mb(df_opt)

    for col in df_opt.columns:
        col_type = df_opt[col].dtype

        if col_type == "object":
            n_unique = df_opt[col].nunique()
            n_total  = len(df_opt[col])
            # Convert to category if < 50% unique values (saves storage via integer codes)
            if n_unique / n_total < categorical_threshold:
                df_opt[col] = df_opt[col].astype("category")

        elif col_type in [np.int64, np.int32]:
            df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")

        elif col_type == np.float64:
            df_opt[col] = df_opt[col].astype(np.float32)
            # Caution: float32 precision ~7 digits; check acceptable for your domain

    after_mb = memory_usage_mb(df_opt)
    print(f"Memory: {before_mb:.1f} MB → {after_mb:.1f} MB ({100*(1-after_mb/before_mb):.1f}% reduction)")
    return df_opt


# Example usage:
# df = pd.read_csv("large_sales.csv")
# df_opt = optimize_dtypes(df)

# ---- Python memory internals (interview talking points) ----
import sys

# Reference counting
x = [1, 2, 3]
print(sys.getrefcount(x))   # 2 (x + temporary ref from getrefcount)

# Object interning (integers -5 to 256 are cached)
a, b = 100, 100
print(a is b)   # True — same object (interned)
c, d = 1000, 1000
print(c is d)   # False in some contexts (not interned)

# Garbage collection
import gc
gc.collect()   # Force collection of cyclic garbage — call after processing large batches
```

---

### Q13 — Python GIL, Mutable Defaults & Common Gotchas [Easy]

```python
# ---- Gotcha 1: Mutable Default Argument ----
# WRONG — the list is shared across ALL calls
def append_to_bug(item, target_list=[]):
    target_list.append(item)
    return target_list

print(append_to_bug(1))   # [1]
print(append_to_bug(2))   # [1, 2]  ← Bug! Expected [2]

# CORRECT — use None as sentinel
def append_to(item, target_list=None):
    if target_list is None:
        target_list = []
    target_list.append(item)
    return target_list

# ---- Gotcha 2: Late binding in closures ----
# WRONG
funcs = [lambda: i for i in range(5)]
print([f() for f in funcs])   # [4, 4, 4, 4, 4] — all closures capture same `i`

# CORRECT — force early binding via default arg
funcs_fixed = [lambda i=i: i for i in range(5)]
print([f() for f in funcs_fixed])   # [0, 1, 2, 3, 4]

# ---- Gotcha 3: Shallow vs Deep Copy ----
import copy
original = {"weights": [0.1, 0.5, 0.9], "lr": 0.01}
shallow = original.copy()
deep    = copy.deepcopy(original)

original["weights"].append(0.99)
print(shallow["weights"])   # [0.1, 0.5, 0.9, 0.99]  ← Shared inner list!
print(deep["weights"])      # [0.1, 0.5, 0.9]         ← Independent copy

# ---- is vs == ----
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)   # True — same value
print(a is b)   # False — different objects in memory
```

---

## 7. DSA — Hash Maps & Frequency Counting

### Q14 — Top-K Frequent Elements [Medium]

**Interview Prompt (General Mills):** *"Given a list of product SKUs sold today, find the top 5 most sold SKUs efficiently."*

```python
from collections import Counter
import heapq

def top_k_products(sku_list: list, k: int = 5) -> list:
    """
    O(n log k) using min-heap.
    Counter builds frequency map in O(n), heapq.nlargest in O(n log k).
    """
    freq = Counter(sku_list)                      # O(n)
    return heapq.nlargest(k, freq, key=freq.get)  # O(n log k)

# Alternatively — most_common is O(n log k) internally
def top_k_v2(sku_list: list, k: int = 5) -> list:
    return [item for item, _ in Counter(sku_list).most_common(k)]

# Example
skus = ["SKU001", "SKU002", "SKU001", "SKU003", "SKU001", "SKU002", "SKU004"]
print(top_k_products(skus, k=3))  # ['SKU001', 'SKU002', 'SKU003']

# ---- Anagram Grouping (NLP/text preprocessing in DS) ----
from collections import defaultdict

def group_anagrams(words: list) -> list:
    """Group words that are anagrams of each other. O(n * m log m)."""
    groups = defaultdict(list)
    for word in words:
        key = "".join(sorted(word))   # Sorted chars = canonical form
        groups[key].append(word)
    return list(groups.values())

tokens = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(group_anagrams(tokens))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

---

### Q15 — Two Sum / Find Missing Values [Easy]

**Interview Prompt:** *"Given a sequence of transaction IDs that should be 1 to N, find the missing ones."*

```python
def find_missing_ids(ids: list, n: int) -> list:
    """O(n) using set difference."""
    expected = set(range(1, n + 1))
    actual   = set(ids)
    return sorted(expected - actual)

ids = [1, 2, 4, 6, 7, 10]
print(find_missing_ids(ids, n=10))   # [3, 5, 8, 9]

# ---- Two Sum (classic hash map pattern) ----
def two_sum(nums: list, target: int):
    """
    Find indices of two numbers that add to target.
    O(n) time, O(n) space.
    """
    seen = {}   # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

print(two_sum([2, 7, 11, 15], target=9))   # [0, 1]

# ---- Subarray sum equals K (frequency map pattern) ----
def subarray_sum_k(nums: list, k: int) -> int:
    """
    Count subarrays with sum exactly k.
    O(n) using prefix sum + hash map.
    Real DS use: count time windows with cumulative sales == budget.
    """
    count = 0
    prefix_sum = 0
    freq = {0: 1}   # prefix_sum → how many times seen
    for num in nums:
        prefix_sum += num
        count += freq.get(prefix_sum - k, 0)
        freq[prefix_sum] = freq.get(prefix_sum, 0) + 1
    return count

sales = [1, 2, 3, 4, 2, 1]
print(subarray_sum_k(sales, k=6))   # 2 → subarrays [1,2,3] and [4,2]... actually let's verify logic
```

---

## 8. DSA — Sorting Algorithms

### Q16 — Sorting Variants Relevant to DS [Medium]

**Interview Prompt:** *"Implement merge sort and explain when you'd use it vs quicksort for data pipelines."*

```python
import numpy as np
from typing import List

# ---- Merge Sort — O(n log n) stable, good for external/linked data ----
def merge_sort(arr: List[float]) -> List[float]:
    """
    Stable sort → preserves order of equal elements (crucial for time-series).
    Space: O(n). Used in pandas .sort_values() on some backends.
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left  = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)

def _merge(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    return result + left[i:] + right[j:]


# ---- Quick Sort — O(n log n) avg, O(n²) worst, in-place ----
def quick_sort(arr: List[float], low: int = 0, high: int = None) -> List[float]:
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot_idx = _partition(arr, low, high)
        quick_sort(arr, low, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, high)
    return arr

def _partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# ---- Partial Sort — when you only need top-k (most common DS need) ----
def top_k_largest(arr: List[float], k: int) -> List[float]:
    """O(n log k) using heap. Faster than full sort for large arrays."""
    import heapq
    return heapq.nlargest(k, arr)

# In practice — always use numpy/pandas built-ins:
arr = np.random.rand(1_000_000)
top10 = np.partition(arr, -10)[-10:]   # O(n) partial sort
```

> [!NOTE]
> **Interview Answer — Merge vs Quick:** Merge sort is stable and predictable O(n log n) — preferred when stability matters (sorting events by time + secondary key). Quicksort is typically faster in practice due to cache efficiency but worst-case O(n²). NumPy uses an **introsort** (hybrid quicksort + heapsort) for arrays.

---

## 9. DSA — Sliding Window & Two Pointers

### Q17 — Moving Average & Rolling Max [Medium]

**Interview Prompt (LSEG):** *"Without using pandas, compute a rolling 7-day average on a price array."*

```python
from collections import deque
from typing import List

# ---- Sliding Window: Moving Average — O(n) ----
def moving_average(prices: List[float], window: int) -> List[float]:
    """
    Single pass, O(1) per step by maintaining running sum.
    Returns list of length (n - window + 1).
    """
    if len(prices) < window:
        return []
    window_sum = sum(prices[:window])
    result = [window_sum / window]
    for i in range(window, len(prices)):
        window_sum += prices[i] - prices[i - window]
        result.append(window_sum / window)
    return result

prices = [10, 20, 30, 40, 50, 60, 70]
print(moving_average(prices, window=3))
# [20.0, 30.0, 40.0, 50.0, 60.0]


# ---- Sliding Window Maximum — O(n) using monotonic deque ----
def sliding_window_max(nums: List[float], k: int) -> List[float]:
    """
    Find max in each window of size k.
    Use case: Max price in each trading day window.
    Deque stores indices in decreasing order of value.
    """
    result = []
    dq = deque()   # Stores indices; front always holds max

    for i, val in enumerate(nums):
        # Remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        # Remove smaller elements from back — they can never be the max
        while dq and nums[dq[-1]] < val:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result

print(sliding_window_max([3, 1, 3, 5, 6, 4, 2, 5], k=3))
# [3, 5, 6, 6, 6, 5]


# ---- Two Pointers: Longest subarray with sum ≤ budget ----
def longest_subarray_budget(revenue: List[int], budget: int) -> int:
    """
    Find longest contiguous sub-period where total cost ≤ budget.
    Two pointer / sliding window: O(n), O(1) space.
    Real use: Max marketing window within quarterly budget.
    """
    left = 0
    current_sum = 0
    max_len = 0
    for right in range(len(revenue)):
        current_sum += revenue[right]
        while current_sum > budget:
            current_sum -= revenue[left]
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len

weekly_costs = [10, 15, 20, 5, 8, 12, 25, 6]
print(longest_subarray_budget(weekly_costs, budget=40))  # 4
```

---

## 10. Performance: Vectorization vs Loops

### Q18 — Loop vs Vectorization Benchmark [Medium]

**Interview Prompt:** *"You have a 10M row DataFrame. Your colleague wrote a for-loop to compute a feature. How do you speed it up?"*

```python
import numpy as np
import pandas as pd
import time

np.random.seed(42)
n = 1_000_000
df = pd.DataFrame({
    "price":    np.random.uniform(10, 1000, n),
    "quantity": np.random.randint(1, 100, n),
    "discount": np.random.uniform(0, 0.3, n),
})

# ---- Method 1: Pure Python loop — NEVER do this ----
def compute_revenue_loop(df):
    revenues = []
    for _, row in df.iterrows():
        rev = row["price"] * row["quantity"] * (1 - row["discount"])
        revenues.append(rev)
    return revenues

# ---- Method 2: df.apply — better, but still row-by-row Python ----
def compute_revenue_apply(df):
    return df.apply(lambda r: r["price"] * r["quantity"] * (1 - r["discount"]), axis=1)

# ---- Method 3: Vectorized Pandas — FAST ----
def compute_revenue_vectorized(df):
    return df["price"] * df["quantity"] * (1 - df["discount"])

# ---- Method 4: NumPy — even faster (skips pandas overhead) ----
def compute_revenue_numpy(df):
    return df["price"].values * df["quantity"].values * (1 - df["discount"].values)

# Benchmark (approximate timings at n=1M):
# iterrows:   ~90 seconds
# apply:      ~3 seconds
# vectorized: ~0.05 seconds
# numpy:      ~0.015 seconds

# ---- Profiling tools ----
# 1. %timeit in Jupyter
# 2. cProfile for production scripts
import cProfile
cProfile.run("compute_revenue_vectorized(df)", sort="cumulative")

# 3. line_profiler (pip install line_profiler) — line-by-line timing
# %load_ext line_profiler
# %lprun -f compute_revenue_apply compute_revenue_apply(df)

# 4. memory_profiler
# from memory_profiler import memory_usage
# mem = memory_usage((compute_revenue_numpy, (df,)))
# print(f"Peak memory: {max(mem):.1f} MB")

# ---- Numba for custom math loops ----
from numba import njit

@njit(parallel=True)
def haversine_batch(lat1, lon1, lat2, lon2):
    """
    Compute distances between coordinate pairs.
    @njit compiles to machine code — approaches C speed.
    """
    n = len(lat1)
    out = np.empty(n)
    R = 6371.0
    for i in range(n):
        dlat = np.radians(lat2[i] - lat1[i])
        dlon = np.radians(lon2[i] - lon1[i])
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1[i])) * np.cos(np.radians(lat2[i])) * np.sin(dlon/2)**2
        out[i] = 2 * R * np.arcsin(np.sqrt(a))
    return out
```

> [!TIP]
> **Interview framework for optimization questions:**
> 1. **Profile first** — don't guess (`cProfile`, `%timeit`)
> 2. **Vectorize** — replace loops with NumPy/Pandas ops
> 3. **Reduce memory** — downcast dtypes, use chunking
> 4. **Parallelize** — `multiprocessing` or `Dask` for truly large data
> 5. **Compile** — `Numba @njit` for custom math, `Cython` for C extensions

---

## 11. Multiprocessing & Threading for Data Tasks

### Q19 — Parallel Feature Engineering with multiprocessing [Hard]

**Interview Prompt:** *"You have 100 large CSV files to process independently. How do you use all CPU cores?"*

```python
import multiprocessing as mp
import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial
from typing import List, Tuple

# ---- CPU-bound task: use multiprocessing (not threading — GIL blocks threads) ----
def process_single_file(filepath: str, target_col: str = "sales") -> dict:
    """
    Process one CSV file. Called in a child process — no GIL contention.
    Keep it pure (no shared state) to avoid race conditions.
    """
    df = pd.read_csv(filepath)
    return {
        "file":   Path(filepath).name,
        "mean":   df[target_col].mean(),
        "std":    df[target_col].std(),
        "max":    df[target_col].max(),
        "count":  len(df),
    }


def parallel_process(filepaths: List[str], n_workers: int = None) -> pd.DataFrame:
    """
    Fan out work across CPU cores using Pool.map.
    n_workers defaults to os.cpu_count().
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(process_single_file, filepaths)   # Blocks until all done

    return pd.DataFrame(results)


# ---- I/O-bound task: use threading (GIL released during I/O) ----
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

def fetch_price(symbol: str, session: requests.Session) -> Tuple[str, float]:
    """Fetch price via HTTP — I/O bound, threading works fine here."""
    r = session.get(f"https://api.example.com/price/{symbol}", timeout=5)
    return symbol, r.json()["price"]

def batch_fetch_prices(symbols: List[str], max_workers: int = 20) -> dict:
    prices = {}
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_price, s, session): s for s in symbols}
            for future in as_completed(futures):
                symbol, price = future.result()
                prices[symbol] = price
    return prices


# ---- ProcessPoolExecutor (modern API, matches ThreadPoolExecutor interface) ----
from concurrent.futures import ProcessPoolExecutor

def run_parallel_feature_engineering(datasets: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """ProcessPoolExecutor gives cleaner future-based API vs Pool."""
    worker = partial(engineer_features)   # pre-bind any extra args

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(worker, datasets))
    return results

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pure function — safe to run in subprocess."""
    df = df.copy()
    df["log_sales"]    = np.log1p(df["sales"])
    df["rolling_mean"] = df["sales"].rolling(7, min_periods=1).mean()
    return df
```

> [!WARNING]
> **GIL pitfall:** Python threads cannot truly parallelize CPU-bound code. Use `multiprocessing` for CPU-heavy transforms, `threading`/`asyncio` for I/O-bound tasks (API calls, DB queries).

---

## 12. Memory-Efficient Data Loading

### Q20 — Parquet vs CSV, Chunking, Lazy Loading [Medium]

**Interview Prompt:** *"You need to load a 50 GB dataset on a machine with 16 GB RAM. Walk me through your approach."*

```python
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# ---- Strategy 1: Read only needed columns ----
def load_relevant_columns(filepath: str, cols: list) -> pd.DataFrame:
    """Load only the columns you need — massive RAM saving."""
    return pd.read_csv(filepath, usecols=cols, dtype={
        "product_id": "category",
        "store_id":   "int32",
        "sales":      "float32",
    })

# ---- Strategy 2: Parquet (columnar, compressed, 5-10x smaller than CSV) ----
# Write once:
# df.to_parquet("sales.parquet", engine="pyarrow", compression="snappy")

# Read all:
# df = pd.read_parquet("sales.parquet", columns=["product_id", "sales"])

# Read row groups lazily (Parquet is natively chunked):
def stream_parquet(filepath: str, batch_size: int = 100_000):
    """Read Parquet in row-group batches without loading full file."""
    pf = pq.ParquetFile(filepath)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()

# ---- Strategy 3: Dask — lazy computation graph, out-of-core ----
# import dask.dataframe as dd
# ddf = dd.read_parquet("sales_dir/*.parquet")
# result = ddf.groupby("store_id")["sales"].sum().compute()  # Only executes here

# ---- Strategy 4: SQL for filtering before loading ----
import sqlite3

def query_subset(db_path: str, query: str) -> pd.DataFrame:
    """
    Use SQL to filter at the DB layer — load only what you need.
    Much faster than read_csv + filter.
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)

# result = query_subset("sales.db", "SELECT * FROM transactions WHERE amount > 500 AND year = 2024")

# ---- Memory profiling snippet ----
import tracemalloc

def profile_memory(func, *args, **kwargs):
    """Measure peak memory of a function call."""
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak memory: {peak / 1024**2:.1f} MB")
    return result
```

---

## 13. Google Colab Specific Patterns

### Q21 — Colab Setup: Drive Mounting, GPU, Large Datasets [Easy]

```python
# ---- Mount Google Drive ----
from google.colab import drive
drive.mount("/content/drive")

# Access files:
import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/datasets/sales_2024.csv")

# ---- GPU check and configuration ----
import torch

def check_gpu():
    """Verify GPU availability in Colab (T4/A100/L4 depending on plan)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} | VRAM: {total_vram:.1f} GB")
    else:
        device = torch.device("cpu")
        print("No GPU available — using CPU")
    return device

device = check_gpu()

# ---- Large dataset in Colab: use /content (fast SSD, ~100GB) ----
# Download directly to Colab's /content (faster than Drive):
import subprocess
# !wget -q "https://storage.googleapis.com/dataset/large_file.parquet" -O /content/data.parquet

# ---- Memory management in Colab (12 GB RAM limit on free tier) ----
import gc

def free_memory(*objects):
    """Delete objects and trigger garbage collection."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # Release GPU memory

# Example usage after processing a large batch:
# df_intermediate = process_step1(df_raw)
# free_memory(df_raw)   # Free 4 GB before step 2

# ---- Colab GPU training pattern ----
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_on_gpu(model, loader, epochs: int = 10, lr: float = 1e-3):
    """Move model and data to GPU if available."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)   # Critical: move batch to GPU
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss  = criterion(preds, y_batch.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

# ---- Colab: Save checkpoints to Drive (Colab resets on disconnect) ----
def save_checkpoint(model, path: str, epoch: int, optimizer=None):
    checkpoint = {
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict() if optimizer else None,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

# save_checkpoint(model, "/content/drive/MyDrive/checkpoints/model_ep10.pt", epoch=10)

# ---- Kaggle dataset download in Colab ----
# from google.colab import files
# files.upload()   # Upload kaggle.json API key
# !mkdir ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d dataset-owner/dataset-name -p /content/data/
```

> [!TIP]
> **Colab Pro tips for interviews:**
> - Use `%load_ext tensorboard` + TensorBoard for live training monitoring
> - `!nvidia-smi` to monitor GPU utilization
> - Runtime → Change runtime type → GPU to enable CUDA
> - Always save to Drive; Colab sessions disconnect after ~12 hours (free) or 24 hours (Pro)

---

## 14. Quick-Fire Interview Questions

### Python Fundamentals (Expect These in Any DS Interview)

| Question | Key Answer Points |
|---|---|
| What is the GIL? | Global Interpreter Lock — allows only one thread to execute Python bytecode at a time. Use `multiprocessing` to bypass for CPU-bound tasks. |
| `is` vs `==` | `is` checks object identity (same memory address). `==` checks value equality. |
| `*args` vs `**kwargs` | `*args` → positional arguments as tuple. `**kwargs` → keyword arguments as dict. |
| Mutable vs Immutable | Lists/dicts/sets are mutable. Tuples/strings/ints/frozensets are immutable. |
| `__init__` vs `__new__` | `__new__` creates the object; `__init__` initializes it. |
| Generator vs Iterator | Generators are iterators created with `yield`; they produce values lazily. |
| `@staticmethod` vs `@classmethod` | `@staticmethod` — no `self`/`cls`. `@classmethod` — receives `cls` (the class). |
| List vs Tuple performance | Tuple is slightly faster (fixed size, no resize overhead) and uses less memory. |
| `copy` vs `deepcopy` | Shallow copy → new container, same nested objects. Deep copy → recursive full clone. |
| Python memory model | Private heap + reference counting + generational GC for cycles. |

### DSA Complexity Cheat Sheet

| Algorithm / Structure | Time (avg) | Space | DS Use Case |
|---|---|---|---|
| Hash map lookup | O(1) | O(n) | Frequency counts, groupby |
| Sorting (Timsort) | O(n log n) | O(n) | `df.sort_values()` |
| Binary search | O(log n) | O(1) | Search in sorted feature arrays |
| Sliding window | O(n) | O(k) | Rolling metrics |
| Heap (top-k) | O(n log k) | O(k) | Top-K products/features |
| BFS/DFS | O(V+E) | O(V) | Rare in DS; graph-based features |

### Common Mistakes to Avoid in Interviews

```python
# ❌ 1. Using iterrows() on large DataFrames
for idx, row in df.iterrows():      # 100x slower than vectorized
    df.loc[idx, "new_col"] = row["a"] * 2

# ✅ Correct
df["new_col"] = df["a"] * 2

# ❌ 2. Chained indexing (SettingWithCopyWarning)
df[df["a"] > 0]["b"] = 1   # May not work — creates copy

# ✅ Correct
df.loc[df["a"] > 0, "b"] = 1

# ❌ 3. Reading entire CSV when only needing a few columns
df = pd.read_csv("huge.csv")   # Loads everything

# ✅ Correct
df = pd.read_csv("huge.csv", usecols=["date", "sales", "product_id"])

# ❌ 4. Storing categoricals as object dtype
df["region"] = df["region"].astype("object")   # 8 bytes per item

# ✅ Correct
df["region"] = df["region"].astype("category")  # 1-2 bytes per item

# ❌ 5. Forgetting to reset index after filtering
df_filtered = df[df["sales"] > 100]   # Index still has gaps
df_filtered.reset_index(drop=True, inplace=True)   # ✅
```

---

## Study Plan for General Mills & LSEG

```mermaid
flowchart LR
    A["Week 1\nPython Basics\n& OOP"] --> B["Week 2\nGenerators,\nDecorators,\nContext Managers"]
    B --> C["Week 3\nDSA: HashMap,\nSliding Window,\nSorting"]
    C --> D["Week 4\nPerformance:\nVectorization,\nMultiprocessing"]
    D --> E["Week 5\nMock Interviews\nCodility Practice"]
    E --> F["Interview\nReady 🎯"]

    style A fill:#4CAF50,color:#fff
    style F fill:#2196F3,color:#fff
```

### Resources
- **LeetCode:** Focus on Easy/Medium Array, HashMap, Sliding Window
- **Codility:** Practice on their platform (LSEG uses it directly)
- **Real Python:** Decorators, Generators, Context Managers deep-dives
- **Python Docs:** `functools`, `itertools`, `collections`, `multiprocessing`

---

*Prepared for Senior Data Scientist interviews — General Mills India & LSEG India | 2025–2026*
