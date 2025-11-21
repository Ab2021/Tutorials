# Day 3 (Part 1): Advanced Pandas Internals & Optimization

> **Phase**: 6 - Deep Dive
> **Topic**: Data Manipulation at Scale
> **Focus**: The Block Manager, PyArrow, and Performance
> **Reading Time**: 60 mins

---

## 1. Pandas Internals: The Block Manager

Why is Pandas sometimes slow? Because of its memory layout.

### 1.1 The Layout
*   **DataFrame**: Not a single matrix. It is a collection of **Blocks**.
*   **Block**: A NumPy array containing columns of the *same dtype*.
    *   If you have 100 `float` columns and 1 `int` column, Pandas creates 1 huge Float Block and 1 small Int Block.
*   **Consolidation**: When you add a new column, it might fragment the blocks. Pandas occasionally runs "Consolidation" to merge blocks, which copies memory (slow).
*   **Optimization**: `df.copy(deep=False)` creates a new index/columns wrapper but points to the same blocks.

### 1.2 Pandas 2.0 & PyArrow
*   **The Shift**: Pandas 2.0 allows using Apache Arrow as the backend instead of NumPy.
*   **Benefits**:
    *   **Missing Data**: NumPy has no `NaN` for integers (it casts to float). Arrow has nullable integers.
    *   **Strings**: Arrow strings are zero-copy and much faster than Python object strings.
    *   **Usage**: `pd.read_csv("file.csv", dtype_backend="pyarrow")`.

---

## 2. Advanced GroupBy & Apply

### 2.1 Split-Apply-Combine Internals
1.  **Split**: Creates a mapping `Group Key -> [Row Indices]`.
2.  **Apply**:
    *   *Vectorized*: If using `sum`, `mean`, Pandas calls optimized Cython versions. Fast.
    *   *Generic*: If using `apply(lambda x: ...)`, Pandas loops in Python. Slow.
3.  **Combine**: Concatenates results.

### 2.2 The `transform` Trick
*   **Scenario**: Normalize each group by its mean.
*   **Bad**: `df.groupby('grp').apply(lambda x: x - x.mean())`.
*   **Good**: `df['mean'] = df.groupby('grp')['val'].transform('mean')`.
    *   `transform` returns a Series aligned with the original index. It broadcasts the group result to every row in the group.

---

## 3. MultiIndex (Hierarchical Indexing)

### 3.1 Slicing
*   **Structure**: `(Level 1, Level 2)`.
*   **Slicing**: `df.loc[(slice(None), 'Category_A'), :]`.
*   **IndexSlice**: `idx = pd.IndexSlice; df.loc[idx[:, 'Category_A'], :]`. Cleaner syntax.

### 3.2 Stacking & Unstacking
*   **Stack**: Moves columns to index (Wide to Long).
*   **Unstack**: Moves index to columns (Long to Wide).
*   **Pivot Table**: Essentially `groupby` + `unstack`.

---

## 4. Tricky Interview Questions

### Q1: What is the complexity of `pd.merge`?
> **Answer**:
> *   **Hash Join** (Default): Builds a hash map of the keys in the smaller table ($O(N)$). Probes with keys from larger table ($O(M)$). Total: $O(N+M)$.
> *   **Sort-Merge Join**: If data is sorted (or requested via `sort=True`), it sorts both ($O(N \log N)$) and merges ($O(N+M)$).
> *   **Memory**: Hash join requires RAM for the hash table.

### Q2: Why does `df.apply()` sometimes run twice on the first row?
> **Answer**: Pandas does a "trial run" on the first row to infer the output dtype of your function. If it fails or returns a complex type, it might fall back to a slower path. This can cause side effects if your function writes to a DB or prints logs.

### Q3: How do you handle a CSV larger than RAM?
> **Answer**:
> 1.  **Chunking**: `pd.read_csv(..., chunksize=10000)`. Returns an iterator.
> 2.  **Dtypes**: Specify `dtype={'id': 'int32'}` to save memory (default is int64).
> 3.  **Use Polars/Dask**: Libraries designed for out-of-core processing.

---

## 5. Practical Edge Case: SettingWithCopyWarning

```python
df[df['A'] > 0]['B'] = 10  # Warning!
```
*   **Reason**: `df[df['A'] > 0]` creates a copy. You are setting values on the *copy*, which is immediately thrown away. The original `df` is unchanged.
*   **Fix**: `df.loc[df['A'] > 0, 'B'] = 10`.

