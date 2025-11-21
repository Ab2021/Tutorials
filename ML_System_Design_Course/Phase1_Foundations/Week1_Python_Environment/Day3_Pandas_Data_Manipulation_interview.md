# Day 3: Pandas & Data Manipulation - Interview Questions

> **Topic**: Data Engineering with Pandas
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is the difference between `loc` and `iloc`?
**Answer:**
*   `loc`: **Label-based**. You pass the name of the index/column. `df.loc['row_name', 'col_name']`. Includes the end of the slice.
*   `iloc`: **Integer-position based**. You pass the index number. `df.iloc[0, 5]`. Excludes the end of the slice (Python style).

### 2. Explain the "SettingWithCopyWarning". How do you fix it?
**Answer:**
*   **Cause**: Occurs when you try to assign values to a slice of a DataFrame that might be a copy, not a view. `df[df['A']>0]['B'] = 1`. Pandas isn't sure if you want to modify the original `df` or the temporary copy.
*   **Fix**: Use `.loc` explicitly. `df.loc[df['A']>0, 'B'] = 1`. Or use `.copy()` if you intend to create a new object.

### 3. How does `groupby` work internally? Explain the Split-Apply-Combine paradigm.
**Answer:**
*   **Split**: Data is partitioned into groups based on keys.
*   **Apply**: A function (sum, mean, custom) is applied to each group independently.
*   **Combine**: The results are merged back into a single DataFrame.
*   **Optimization**: Pandas optimizes standard aggregations (sum, mean) using Cython to avoid Python loops.

### 4. What is the difference between `merge` and `join` in Pandas?
**Answer:**
*   `merge`: More general. Joins on **columns** (by default). Similar to SQL JOIN.
*   `join`: Joins on **index** (by default).
*   Usually, `merge` is preferred for clarity unless you are specifically working with indices.

### 5. How do you handle missing values in Pandas? Explain `fillna`, `dropna`, and `interpolate`.
**Answer:**
*   `dropna`: Removes rows/cols with nulls. Risk: Data loss.
*   `fillna`: Replaces nulls with a constant (0, "Unknown") or strategy (mean/median).
*   `interpolate`: Fills nulls by estimating values based on neighbors (linear, time-based). Good for time-series.

### 6. Why is iterating over rows (`iterrows`) considered an anti-pattern in Pandas? What should you use instead?
**Answer:**
*   `iterrows`: Returns a Series for each row. Creating a Series object has high overhead. It's extremely slow ($O(N)$ Python loop).
*   **Alternative**:
    1.  **Vectorization**: `df['A'] + df['B']` (Fastest).
    2.  `apply()`: Slower than vectorization but faster than iterrows.
    3.  `itertuples()`: Returns named tuples. Much faster than iterrows if you must loop.

### 7. What is the difference between a Pandas Series and a DataFrame?
**Answer:**
*   **Series**: 1D labeled array. Homogeneous type (mostly).
*   **DataFrame**: 2D labeled data structure. Collection of Series (columns). Heterogeneous types (col A can be int, col B can be float).

### 8. How does Pandas handle memory management? What is the "Block Manager"?
**Answer:**
*   **Block Manager**: Pandas groups columns of the same dtype into "Blocks". E.g., all `int64` columns are stored in one contiguous NumPy array (Block).
*   **Impact**: Adding a column of a new type triggers block consolidation, which can be expensive.

### 9. Explain the difference between `map`, `apply`, and `applymap`.
**Answer:**
*   `map`: Defined on **Series** only. Used for element-wise mapping (dict or function).
*   `apply`: Defined on **Series** (element-wise) and **DataFrame** (axis-wise). Can apply function to rows or columns.
*   `applymap`: Defined on **DataFrame**. Applies function element-wise to *every cell*.

### 10. How do you optimize memory usage for a DataFrame with many string columns? (Hint: Categoricals).
**Answer:**
*   **Problem**: Strings are stored as Python objects (expensive).
*   **Solution**: Convert low-cardinality string columns to `category` dtype.
*   **Mechanism**: Pandas stores unique strings once and uses integer codes in the column. Reduces memory by 10x-100x.

### 11. What is a MultiIndex? How do you slice a DataFrame with a MultiIndex?
**Answer:**
*   **MultiIndex**: Hierarchical indexing (multiple levels of index).
*   **Slicing**: Use `df.loc[(level1_val, level2_val), :]`. Or `df.xs()` (Cross-section) to select data at a particular level.

### 12. How would you read a 50GB CSV file on a 16GB RAM machine using Pandas?
**Answer:**
*   **Chunking**: `pd.read_csv('file.csv', chunksize=10000)`. Returns an iterator. Process chunks and aggregate results.
*   **Dtypes**: Specify `dtype` to use smaller types (`int32` instead of `int64`).
*   **Library Switch**: Use **Polars** or **Dask** which handle out-of-core processing better.

### 13. What is the difference between `pivot` and `pivot_table`?
**Answer:**
*   `pivot`: Reshapes data. Does **not** support aggregation. Raises error if duplicate entries exist for index/column pair.
*   `pivot_table`: Generalization of pivot. Supports **aggregation** (e.g., `aggfunc='mean'`). Handles duplicates.

### 14. How do you perform a "Cross Join" (Cartesian Product) in Pandas?
**Answer:**
*   `pd.merge(df1, df2, how='cross')` (Pandas 1.2+).
*   Old way: Add a dummy key column to both, merge on key, drop key.

### 15. What is the `pipe` method in Pandas? Why is it useful for code readability?
**Answer:**
*   `pipe`: Allows method chaining with custom functions.
*   `df.pipe(func1).pipe(func2, arg1=1)`.
*   **Readability**: Avoids nested function calls `func2(func1(df), arg1=1)`.

### 16. Explain the difference between `rank` and `sort_values`.
**Answer:**
*   `sort_values`: Reorders the rows based on values.
*   `rank`: Assigns a numerical rank (1, 2, 3...) to each row based on values, preserving the original order. Useful for statistical tests.

### 17. How do you handle time zones in Pandas datetime objects?
**Answer:**
*   `dt.tz_localize('UTC')`: Sets the timezone for naive datetimes.
*   `dt.tz_convert('US/Eastern')`: Converts to a different timezone.

### 18. What is the performance impact of using `object` dtype vs specific types (like `int64` or `string[pyarrow]`)?
**Answer:**
*   `object`: Pointers to Python objects. Slow, high memory, no SIMD.
*   `int64`/`float`: Native C arrays. Fast.
*   `string[pyarrow]`: Newer dtype backed by Arrow. Much faster and memory efficient than `object` for text.

### 19. How do you use `pd.cut` vs `pd.qcut` for binning continuous variables?
**Answer:**
*   `pd.cut`: Bins by **value** (Equal width bins). E.g., 0-10, 10-20.
*   `pd.qcut`: Bins by **quantile** (Equal frequency bins). E.g., Top 10%, Next 10%. Ensures each bin has same number of items.

### 20. What is the difference between `concat` and `append` (deprecated)?
**Answer:**
*   `append`: Was a shortcut for concat. Inefficient because it created a new object every time. Deprecated.
*   `concat`: Designed to join a *list* of DataFrames at once. `pd.concat([df1, df2, df3])`. Much more efficient than appending in a loop.
