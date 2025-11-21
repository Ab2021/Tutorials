# Day 3: Pandas & Data Manipulation at Scale

> **Phase**: 1 - Foundations
> **Week**: 1 - The ML Engineer's Toolkit
> **Focus**: Efficient Data Manipulation
> **Reading Time**: 60 mins

---

## 1. The DataFrame Abstraction

Pandas is the de facto standard for tabular data manipulation in Python. It is built on top of NumPy but adds two critical components: **Labels** (Index) and **Mixed Types**.

### 1.1 Architecture
- **Columnar Storage**: Conceptually, a DataFrame is a collection of Series (columns). Each Series is backed by a NumPy array. This means operations on a single column are fast (vectorized), but operations across rows (which might involve different data types) are generally slower.
- **The Index**: A hash map that allows O(1) lookups for row labels. However, maintaining the index during operations like filtering or sorting incurs overhead.

### 1.2 The "Apply" Trap
A common novice mistake is assuming `df.apply(func)` is vectorized.
- **Reality**: `apply` is often just a glorified `for` loop in Python. It passes each row/element to the Python function, incurring the overhead of Python function calls and type checking for every single element.
- **Optimization**: Always prefer built-in vectorized methods (e.g., `df['a'] + df['b']`) over `apply`. If you must use custom logic, look into `np.vectorize` or simply list comprehensions, which can sometimes be faster than `apply`.

---

## 2. Handling Large Data (The 100GB Problem)

Pandas is an "in-memory" tool. If your dataset is larger than your RAM, Pandas will crash (OOM - Out Of Memory).

### 2.1 Dtypes Matter
By default, Pandas uses 64-bit integers and floats.
- **Optimization**: Downcasting.
    - `int64` -> `int8` (if values are < 127).
    - `float64` -> `float32` (usually sufficient precision for ML).
    - **Impact**: Can reduce memory usage by 4x-8x.

### 2.2 Categoricals
Storing strings is expensive. A column of "Country" names repeated 1 million times stores the string "United States" 1 million times.
- **Solution**: Convert to `category` dtype.
- **Mechanism**: Pandas stores the unique strings once in a lookup table and uses lightweight integers (pointers) in the actual column.
- **Impact**: Massive memory reduction and faster operations (sorting/grouping by integer is faster than by string).

### 2.3 Chunking
When reading massive CSVs:
```python
chunk_size = 10000
for chunk in pd.read_csv('massive_data.csv', chunksize=chunk_size):
    process(chunk)
```
This processes the data in manageable blocks, keeping memory usage constant regardless of file size.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: The "SettingWithCopy" Warning
**Scenario**: You try `df[mask]['col'] = 0` and get a warning. Sometimes it works, sometimes it doesn't.
**Theory**: Pandas is unsure if `df[mask]` returned a **View** or a **Copy**. If it's a copy, your assignment is modifying a temporary object that is immediately thrown away—a bug.
**Solution**: Always use `.loc`: `df.loc[mask, 'col'] = 0`. This explicitly tells Pandas to modify the original dataframe.

### Challenge 2: Date Parsing Bottlenecks
**Scenario**: `pd.read_csv` takes 10 minutes to load. 9 minutes are spent on `parse_dates`.
**Theory**: The default parser tries to infer the format for every single row.
**Solution**: Explicitly specify the date format (e.g., `%Y-%m-%d`). This allows the parser to skip inference and jump straight to parsing.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: How does Pandas handle missing data internally?**
> **Answer**:
> *   **Floats**: Uses `NaN` (Not a Number), which is a special floating-point value defined by IEEE 754.
> *   **Objects**: Uses `None` or `NaN`.
> *   **Integers (Old)**: Historically, integer columns with missing values were forcibly cast to floats (because `NaN` is a float).
> *   **Integers (New)**: Modern Pandas introduces nullable integer types (`Int64`) that use a separate boolean mask to track missing values, preserving the integer dtype.

**Q2: Compare Pandas vs. Polars.**
> **Answer**:
> *   **Pandas**: Single-threaded, eager execution (runs immediately), built on NumPy. Great for exploration and small-medium data.
> *   **Polars**: Multi-threaded, lazy execution (builds a query plan and optimizes it before running), written in Rust. It can handle larger-than-RAM datasets via streaming and is often 10-50x faster for heavy operations.

**Q3: What is "Chained Indexing" and why is it bad?**
> **Answer**: Chained indexing is doing `df['A']['B']`. This involves two separate operations: `__getitem__` for 'A', then `__getitem__` for 'B'. It is inefficient (two calls) and ambiguous (Pandas can't guarantee if the first call returns a view or copy). `.loc['A', 'B']` is a single operation and is safe.

### System Design Challenge
**Scenario**: You need to build a feature engineering pipeline for a 2TB dataset of user logs. Pandas is crashing. What is your architecture?
**Approach**:
1.  **Vertical Scaling (Bad)**: Buying a 3TB RAM server is expensive and not scalable.
2.  **Horizontal Scaling (Better)**: Use a distributed computing framework like **Apache Spark** or **Dask**.
3.  **Modern Approach (2025)**: Use **Polars** with streaming mode on a decent single machine, or **Ray Data** for distributed processing.
4.  **Storage**: Convert the CSV/JSON logs to **Parquet**. Parquet is a columnar format that allows reading only specific columns (projection pushdown) and specific row groups (predicate pushdown), drastically reducing I/O.

---

## 5. Further Reading
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Polars vs Pandas Benchmark](https://pola.rs/posts/benchmarks/)
- [Apache Parquet Format](https://parquet.apache.org/)
