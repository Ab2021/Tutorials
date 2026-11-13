# Pandas & NumPy — Senior Data Scientist Interview Questions
### Targeted: General Mills (India Hub) · LSEG (London Stock Exchange Group, India)

> [!IMPORTANT]
> All code is **complete and runnable**. Each question reflects real interview patterns sourced from
> Glassdoor, AmbitionBox, InterviewQuery, and recruiter reports for Senior/Lead DS roles at General Mills
> (Mumbai/Bengaluru CPG Analytics Hub) and LSEG (Bengaluru/Hyderabad Fintech Hub).

---

## Table of Contents
1. [DataFrame Basics & Memory Optimization](#1-dataframe-basics--memory-optimization)
2. [Missing Data Handling](#2-missing-data-handling)
3. [GroupBy & Aggregations](#3-groupby--aggregations)
4. [Merge / Join Operations](#4-merge--join-operations)
5. [Pivot Tables & Reshaping](#5-pivot-tables--reshaping)
6. [Time Series with Pandas](#6-time-series-with-pandas)
7. [NumPy Vectorized Operations](#7-numpy-vectorized-operations)
8. [NumPy Broadcasting](#8-numpy-broadcasting)
9. [Advanced / Hard Problems](#9-advanced--hard-problems)
10. [Common Gotchas & Interview Tips](#10-common-gotchas--interview-tips)

---

## 1. DataFrame Basics & Memory Optimization

### Q1 · `Easy` — Identify and reduce DataFrame memory footprint

**Interview context (General Mills):** *"We receive daily store-level sales CSVs with ~10 million rows.
Our pipeline is hitting OOM errors. Walk me through how you'd optimize memory usage in Pandas."*

```python
import pandas as pd
import numpy as np

# ── Simulate a large-ish sales DataFrame ──────────────────────────────────────
np.random.seed(42)
n = 500_000

df = pd.DataFrame({
    "store_id":    np.random.randint(1, 500, n).astype("int64"),
    "product":     np.random.choice(["Cheerios", "Yoplait", "Nature Valley", "Pillsbury"], n),
    "region":      np.random.choice(["North", "South", "East", "West"], n),
    "units_sold":  np.random.randint(0, 200, n).astype("int64"),
    "price":       np.random.uniform(1.5, 15.0, n).astype("float64"),
    "promo_flag":  np.random.choice([0, 1], n).astype("int64"),
})

def memory_report(df, label=""):
    mb = df.memory_usage(deep=True).sum() / 1_024**2
    print(f"{label}: {mb:.2f} MB")

memory_report(df, "Before optimization")

# ── Step 1: Downcast integers ──────────────────────────────────────────────────
# store_id fits in int16 (max 500), units_sold fits in int16 (max 200)
df["store_id"]   = df["store_id"].astype("int16")
df["units_sold"] = df["units_sold"].astype("int16")
df["promo_flag"] = df["promo_flag"].astype("int8")   # only 0/1

# ── Step 2: Downcast floats ────────────────────────────────────────────────────
df["price"] = df["price"].astype("float32")

# ── Step 3: Convert low-cardinality strings to category ───────────────────────
# Rule of thumb: cardinality < 50% of rows → category dtype is worth it
for col in ["product", "region"]:
    df[col] = df[col].astype("category")

memory_report(df, "After optimization")
# Typical result: ~70-80% reduction

# ── Bonus: read_csv with dtype hints to avoid loading wrong types ──────────────
# dtype_map = {"store_id": "int16", "product": "category", "promo_flag": "int8"}
# df = pd.read_csv("sales.csv", dtype=dtype_map)
```

> **Gotcha:** `category` dtype breaks some Pandas operations (e.g., `pd.concat` can fail if
> category levels differ). Always call `.astype(str)` before concatenating DataFrames from
> different sources.

---

### Q2 · `Easy` — Remove duplicates safely on a subset of columns

**Interview context (General Mills):** *"Our ETL sometimes double-loads records. How would you
deduplicate the sales table keeping only the latest record per (store_id, product, date)?"*

```python
import pandas as pd

data = {
    "store_id": [1, 1, 2, 2, 3],
    "product":  ["Cheerios", "Cheerios", "Yoplait", "Yoplait", "Pillsbury"],
    "date":     ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-01"],
    "units_sold": [100, 110, 50, 50, 80],   # row 0 and 1 are duplicates with different units
    "load_ts":  ["2024-01-02 06:00", "2024-01-02 09:00", "2024-01-02 06:00",
                 "2024-01-02 06:00", "2024-01-02 06:00"],
}
df = pd.DataFrame(data)
df["load_ts"] = pd.to_datetime(df["load_ts"])

# Sort descending by load_ts so .drop_duplicates keep='first' keeps the latest
df_clean = (
    df.sort_values("load_ts", ascending=False)
      .drop_duplicates(subset=["store_id", "product", "date"], keep="first")
      .sort_index()   # restore original index order if needed
)

print(df_clean)
# Result: row with units_sold=110 is kept (later load_ts)
```

---

## 2. Missing Data Handling

### Q3 · `Medium` — Multi-strategy imputation on a sales DataFrame

**Interview context (General Mills):** *"Given a weekly sales DataFrame with missing values in
price and units columns, impute prices using the product-level median and units using
forward-fill within each store."*

```python
import pandas as pd
import numpy as np

np.random.seed(0)
dates = pd.date_range("2024-01-01", periods=12, freq="W")

df = pd.DataFrame({
    "date":       dates.tolist() * 2,
    "store_id":   [1]*12 + [2]*12,
    "product":    (["Cheerios"]*6 + ["Yoplait"]*6) * 2,
    "units_sold": np.where(np.random.rand(24) < 0.2, np.nan,
                           np.random.randint(50, 200, 24).astype(float)),
    "price":      np.where(np.random.rand(24) < 0.15, np.nan,
                           np.random.uniform(3.0, 8.0, 24)),
})

print("Missing before:\n", df.isnull().sum())

# ── Strategy A: Price → product-level median imputation ───────────────────────
# groupby transform broadcasts the group median back to original shape
df["price"] = df.groupby("product")["price"].transform(
    lambda x: x.fillna(x.median())
)

# ── Strategy B: Units → forward-fill within each store (time-ordered) ─────────
df = df.sort_values(["store_id", "date"])
df["units_sold"] = df.groupby("store_id")["units_sold"].transform(
    lambda x: x.ffill()
)

# ── Strategy C: Any remaining NaNs → fill with 0 (safe fallback) ──────────────
df.fillna(0, inplace=True)

print("\nMissing after:\n", df.isnull().sum())
print(df.head(8))
```

> **Interview tip:** Always explain *why* you chose a strategy:
> - Median is robust to outliers for price data.
> - Forward-fill preserves temporal continuity for time series.
> - Never just `fillna(mean)` without justifying it — interviewers will push back.

---

### Q4 · `Medium` — Detect and flag outliers using IQR + mark-not-impute

**Interview context (LSEG):** *"In financial data, you must NOT impute extreme values silently.
Show how to flag them and then decide what to do downstream."*

```python
import pandas as pd
import numpy as np

np.random.seed(7)
df = pd.DataFrame({
    "ticker": np.random.choice(["RELIANCE", "TCS", "INFY", "HDFC"], 200),
    "daily_return": np.concatenate([
        np.random.normal(0.001, 0.02, 190),   # normal returns
        np.random.uniform(0.5, 1.0, 10),       # extreme outliers (fat tails / bad data)
    ])
})

# ── IQR-based outlier flagging per ticker ─────────────────────────────────────
def flag_iqr_outliers(group):
    q1 = group.quantile(0.25)
    q3 = group.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (group < lower) | (group > upper)

df["is_outlier"] = df.groupby("ticker")["daily_return"].transform(flag_iqr_outliers)

print(f"Outlier count: {df['is_outlier'].sum()}")
print(df[df["is_outlier"]].head())

# ── Options after flagging (discuss with interviewer) ─────────────────────────
# 1. df = df[~df["is_outlier"]]  → drop outliers
# 2. df.loc[df["is_outlier"], "daily_return"] = np.nan  → nullify then impute
# 3. Keep them but create a separate model for extreme-event detection
```

---

## 3. GroupBy & Aggregations

### Q5 · `Medium` — Top-N products by revenue per region

**Interview context (General Mills):** *"Given a transactions dataset, find the top 3 products
by total revenue for each region. Return a clean DataFrame."*

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 10_000

df = pd.DataFrame({
    "region":  np.random.choice(["North", "South", "East", "West"], n),
    "product": np.random.choice(["Cheerios", "Yoplait", "Nature Valley",
                                  "Pillsbury", "Häagen-Dazs", "Old El Paso"], n),
    "revenue": np.random.exponential(scale=50, size=n),
})

# ── Method 1: groupby + nlargest ──────────────────────────────────────────────
top3 = (
    df.groupby(["region", "product"])["revenue"]
      .sum()
      .reset_index(name="total_revenue")
      .sort_values(["region", "total_revenue"], ascending=[True, False])
      .groupby("region")
      .head(3)
      .reset_index(drop=True)
)

# Add rank column within each region
top3["rank"] = top3.groupby("region")["total_revenue"].rank(ascending=False, method="first").astype(int)

print(top3.sort_values(["region", "rank"]))
```

> **Follow-up question they will ask:** *"What if you need top-N per TWO grouping keys,
> like region × channel?"* → Same approach, just add more keys to `groupby`.

---

### Q6 · `Hard` — Conditional aggregation (promo vs non-promo sales in one pass)

**Interview context (General Mills):** *"Calculate total sales, promo sales, and promo uplift %
per category — in a single groupby pass, no filtering."*

```python
import pandas as pd
import numpy as np

np.random.seed(1)
n = 50_000
df = pd.DataFrame({
    "category":   np.random.choice(["Cereal", "Dairy", "Snacks", "Baking"], n),
    "sales":      np.random.uniform(10, 500, n),
    "promo_flag": np.random.choice([0, 1], n, p=[0.7, 0.3]),
})

result = df.groupby("category").agg(
    total_sales=("sales", "sum"),
    # Conditional aggregation using lambda that accesses the flag via aligned index
    promo_sales=("sales", lambda x: x[df.loc[x.index, "promo_flag"] == 1].sum()),
    non_promo_sales=("sales", lambda x: x[df.loc[x.index, "promo_flag"] == 0].sum()),
    avg_promo_basket=("sales", lambda x: x[df.loc[x.index, "promo_flag"] == 1].mean()),
)

result["promo_uplift_pct"] = (
    (result["promo_sales"] / result["non_promo_sales"] - 1) * 100
).round(2)

print(result)
```

---

### Q7 · `Medium` — GroupBy + transform vs agg (a classic trap question)

**Interview context (LSEG):** *"What is the difference between `.agg()` and `.transform()` in a
groupby? Give a use case where you must use transform."*

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "ticker":        ["RELIANCE", "RELIANCE", "TCS", "TCS", "TCS", "INFY"],
    "daily_return":  [0.01, -0.02, 0.03, 0.01, -0.01, 0.05],
})

# .agg() → reduces each group to a single value (shape: n_groups × cols)
group_means = df.groupby("ticker")["daily_return"].agg("mean")
print("agg result:\n", group_means)
# INFY: 0.05, RELIANCE: -0.005, TCS: 0.01

# .transform() → returns a Series with the SAME length/index as original df
# Use case: add a group-level metric as a new column without losing original rows
df["ticker_avg_return"] = df.groupby("ticker")["daily_return"].transform("mean")

# This enables vectorized row-level comparisons:
df["above_ticker_avg"] = df["daily_return"] > df["ticker_avg_return"]

print("\nWith transform:\n", df)
```

---

## 4. Merge / Join Operations

### Q8 · `Medium` — Diagnosing and preventing join cardinality explosion

**Interview context (General Mills/LSEG):** *"You merged two DataFrames and the row count
tripled unexpectedly. How do you diagnose and fix it?"*

```python
import pandas as pd

# ── Simulate a many-to-many scenario ─────────────────────────────────────────
transactions = pd.DataFrame({
    "customer_id": [1, 1, 2, 3, 3],
    "amount":      [100, 200, 150, 300, 50],
})

# BUG: customer 1 appears twice in the reference table (data quality issue)
customers = pd.DataFrame({
    "customer_id": [1, 1, 2, 3],      # duplicate key!
    "segment":     ["Gold", "Gold", "Silver", "Bronze"],
})

# ── Bad merge: row count explodes ─────────────────────────────────────────────
bad_merge = pd.merge(transactions, customers, on="customer_id", how="left")
print(f"Original rows: {len(transactions)}, After merge: {len(bad_merge)}")  # 7 rows!

# ── Fix 1: Validate the join to detect problems early ─────────────────────────
try:
    safe_merge = pd.merge(
        transactions, customers,
        on="customer_id", how="left",
        validate="many_to_one"   # Raises MergeError if customer_id not unique in customers
    )
except pd.errors.MergeError as e:
    print(f"\nMerge validation failed: {e}")

# ── Fix 2: Deduplicate the reference table before merging ─────────────────────
customers_deduped = customers.drop_duplicates(subset=["customer_id"], keep="first")
clean_merge = pd.merge(transactions, customers_deduped, on="customer_id", how="left",
                       validate="many_to_one")
print(f"\nClean merge rows: {len(clean_merge)}")  # 5 rows ✓

# ── Fix 3: Post-merge sanity check ────────────────────────────────────────────
assert len(clean_merge) == len(transactions), "Row count mismatch after left join!"
print(clean_merge)
```

---

### Q9 · `Hard` — As-of Merge for financial time-series data

**Interview context (LSEG):** *"You have tick-level quote data and trade execution logs.
Join them such that each trade gets the last available quote before the trade time."*

```python
import pandas as pd

# ── Quote feed (bid/ask prices at irregular intervals) ────────────────────────
quotes = pd.DataFrame({
    "timestamp": pd.to_datetime([
        "2024-01-15 09:15:00", "2024-01-15 09:15:30", "2024-01-15 09:16:00",
        "2024-01-15 09:16:45", "2024-01-15 09:17:00",
    ]),
    "ticker": ["TCS", "TCS", "TCS", "TCS", "TCS"],
    "bid":    [3800.5, 3801.0, 3799.5, 3802.0, 3801.5],
    "ask":    [3801.0, 3801.5, 3800.0, 3802.5, 3802.0],
})

# ── Trade executions (happens at specific times) ──────────────────────────────
trades = pd.DataFrame({
    "timestamp": pd.to_datetime([
        "2024-01-15 09:15:20",   # between first and second quote
        "2024-01-15 09:16:30",   # between third and fourth quote
        "2024-01-15 09:17:10",   # after last quote
    ]),
    "ticker":       ["TCS", "TCS", "TCS"],
    "trade_price":  [3800.8, 3800.0, 3801.6],
    "quantity":     [100, 200, 150],
})

# ── Both must be sorted by timestamp for merge_asof ──────────────────────────
quotes = quotes.sort_values("timestamp")
trades = trades.sort_values("timestamp")

# merge_asof finds the last quote with timestamp <= trade timestamp
enriched = pd.merge_asof(
    trades, quotes,
    on="timestamp",
    by="ticker",           # match within the same ticker
    direction="backward",  # last available quote before the trade
    suffixes=("_trade", "_quote"),
)

# Calculate slippage: trade price vs mid-quote at time of trade
enriched["mid_quote"] = (enriched["bid"] + enriched["ask"]) / 2
enriched["slippage_bps"] = (
    (enriched["trade_price"] - enriched["mid_quote"]) / enriched["mid_quote"] * 10_000
).round(2)

print(enriched[["timestamp", "ticker", "trade_price", "mid_quote", "slippage_bps"]])
```

---

### Q10 · `Medium` — Multi-key merge with indicator and coverage analysis

```python
import pandas as pd

# Two DataFrames from different data sources
sales_df = pd.DataFrame({
    "store_id": [1, 2, 3, 4],
    "week":     ["W1", "W1", "W1", "W2"],
    "sales":    [1000, 2000, 1500, 800],
})

promo_df = pd.DataFrame({
    "store_id": [1, 2, 5],       # store 5 has promo but no sales data; store 3 has no promo
    "week":     ["W1", "W1", "W1"],
    "promo":    [1, 1, 1],
})

# Full outer join with indicator to find coverage gaps
merged = pd.merge(
    sales_df, promo_df,
    on=["store_id", "week"],
    how="outer",
    indicator=True,   # adds _merge column: left_only / right_only / both
)

# Analyze coverage
print("Merge breakdown:\n", merged["_merge"].value_counts())
print("\nStores with sales but no promo data (left_only):")
print(merged[merged["_merge"] == "left_only"])
print("\nStores with promo but no sales (right_only):")
print(merged[merged["_merge"] == "right_only"])
```

---

## 5. Pivot Tables & Reshaping

### Q11 · `Medium` — Pivot, flatten MultiIndex, feed to ML model

**Interview context (General Mills):** *"Reshape weekly store-product sales into a wide feature
matrix where each row is a store and each column is a product's weekly sales — suitable for
sklearn."*

```python
import pandas as pd
import numpy as np

np.random.seed(99)
dates = pd.date_range("2024-01-01", periods=4, freq="W")

records = []
for store in [1, 2, 3]:
    for product in ["Cheerios", "Yoplait", "Pillsbury"]:
        for d in dates:
            records.append({
                "store_id": store, "product": product,
                "date": d.strftime("W%U"),
                "sales": np.random.randint(50, 300),
            })

df = pd.DataFrame(records)

# ── Pivot: rows = store_id, columns = (product × week), values = sales ────────
pivot = df.pivot_table(
    index="store_id",
    columns=["product", "date"],
    values="sales",
    aggfunc="sum",
    fill_value=0,       # fill missing combinations with 0
)

# ── Flatten MultiIndex columns for sklearn compatibility ──────────────────────
pivot.columns = ["_".join(col).strip() for col in pivot.columns.values]
pivot = pivot.reset_index()

print("Pivoted shape:", pivot.shape)
print(pivot.columns.tolist())
print(pivot.head())
```

---

### Q12 · `Medium` — Melt (wide → long) and back again

**Interview context (General Mills):** *"Our reporting team stores monthly data in wide format
(one column per month). Convert it to long format for modeling, then reshape it back."*

```python
import pandas as pd

wide_df = pd.DataFrame({
    "product":   ["Cheerios", "Yoplait", "Pillsbury"],
    "Jan_2024":  [12000, 8000, 5000],
    "Feb_2024":  [11000, 8500, 5200],
    "Mar_2024":  [13000, 7800, 5100],
})

# ── Wide → Long using pd.melt ─────────────────────────────────────────────────
long_df = pd.melt(
    wide_df,
    id_vars=["product"],        # columns to keep as-is
    var_name="month",           # name for the new 'variable' column
    value_name="sales",         # name for the new 'value' column
)
long_df["month"] = pd.to_datetime(long_df["month"], format="%b_%Y")
long_df = long_df.sort_values(["product", "month"]).reset_index(drop=True)
print("Long format:\n", long_df.head(6))

# ── Long → Wide using pivot ───────────────────────────────────────────────────
wide_again = long_df.pivot(index="product", columns="month", values="sales")
wide_again.columns = [d.strftime("%b_%Y") for d in wide_again.columns]
wide_again = wide_again.reset_index()
print("\nBack to wide:\n", wide_again)
```

---

### Q13 · `Hard` — Stack / Unstack with MultiIndex

**Interview context (LSEG):** *"You have a MultiIndex DataFrame of financial metrics per
(ticker, date). Reshape it so each metric becomes a separate column."*

```python
import pandas as pd
import numpy as np

tickers = ["TCS", "RELIANCE"]
dates = pd.date_range("2024-01-01", periods=3)
metrics = ["close", "volume", "volatility"]

# Create MultiIndex DataFrame
idx = pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"])
data = np.random.rand(len(idx), len(metrics))
df = pd.DataFrame(data, index=idx, columns=metrics)
print("Original (MultiIndex):\n", df)

# ── Unstack: move 'ticker' level from index into columns ─────────────────────
unstacked = df.unstack(level="ticker")
print("\nUnstacked:\n", unstacked)

# ── Flatten for downstream use ────────────────────────────────────────────────
unstacked.columns = ["_".join(col) for col in unstacked.columns]
print("\nFlattened:\n", unstacked)

# ── Stack back: move ticker columns back into index ───────────────────────────
# Original = unstacked.stack(level="ticker")
```

---

## 6. Time Series with Pandas

### Q14 · `Medium` — Resampling & rolling window without data leakage

**Interview context (General Mills):** *"Calculate a 4-week rolling average of sales per store.
Ensure no future data leaks into the rolling window (closed='left')."*

```python
import pandas as pd
import numpy as np

np.random.seed(5)
dates = pd.date_range("2023-01-01", periods=52, freq="W")

df = pd.DataFrame({
    "date":     dates.tolist() * 2,
    "store_id": [101]*52 + [102]*52,
    "sales":    np.random.randint(500, 2000, 104).astype(float),
})
df.loc[[10, 20, 55], "sales"] = np.nan   # inject some missing values

df = df.sort_values(["store_id", "date"]).set_index("date")

# ── Forward-fill missing sales within each store ──────────────────────────────
df["sales"] = df.groupby("store_id")["sales"].transform(lambda x: x.ffill())

# ── Rolling 4-week average per store (leak-free: min_periods=1) ───────────────
# Groupby rolling: each store gets its own rolling window
df["rolling_4w_avg"] = (
    df.groupby("store_id")["sales"]
      .rolling(window=4, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)   # align index after groupby rolling
)

# ── Resample to monthly (for dashboarding) ────────────────────────────────────
monthly = (
    df.groupby("store_id")
      .resample("ME")["sales"]    # ME = Month End frequency
      .sum()
      .reset_index()
)

print(df.tail(10))
print("\nMonthly aggregation:\n", monthly.head(6))
```

---

### Q15 · `Hard` — Feature engineering for demand forecasting (lag + rolling features)

**Interview context (General Mills):** *"Build a lag-feature dataset for a weekly demand
forecasting model. Explain how you prevent data leakage."*

```python
import pandas as pd
import numpy as np

np.random.seed(0)
df = pd.DataFrame({
    "date":    pd.date_range("2022-01-03", periods=104, freq="W"),
    "product": "Cheerios",
    "sales":   np.random.randint(1000, 5000, 104).astype(float),
})
# Add seasonality spike
df.loc[df["date"].dt.month.isin([11, 12]), "sales"] *= 1.4

df = df.set_index("date")

# ── Lag features (always shift FORWARD in time so lag-1 means last week) ──────
# shift(1) → lag of 1 week (previous week's sales)
df["lag_1w"]  = df["sales"].shift(1)
df["lag_4w"]  = df["sales"].shift(4)
df["lag_52w"] = df["sales"].shift(52)   # same week last year

# ── Rolling features (use closed='left' or min_periods to avoid leakage) ──────
df["roll_4w_mean"]  = df["sales"].shift(1).rolling(4).mean()   # shift first!
df["roll_12w_std"]  = df["sales"].shift(1).rolling(12).std()

# ── Calendar features ─────────────────────────────────────────────────────────
df["week_of_year"] = df.index.isocalendar().week.astype(int)
df["month"]        = df.index.month
df["is_holiday_season"] = df.index.month.isin([11, 12]).astype(int)

# Drop rows where lags are NaN (first 52 weeks will have NaN lag_52w)
df_model_ready = df.dropna()

print(f"Model-ready shape: {df_model_ready.shape}")
print(df_model_ready.tail(5))
```

> **Critical interview point:** Always `.shift(1)` *before* computing rolling stats.
> Computing `rolling().mean()` on unshifted sales leaks current-week data into the feature.

---

### Q16 · `Medium` — Detecting and handling irregular / duplicate timestamps

**Interview context (LSEG):** *"Market data feeds sometimes produce duplicate or out-of-order
timestamps. How do you clean this before modeling?"*

```python
import pandas as pd
import numpy as np

# Simulate messy financial tick data
np.random.seed(3)
timestamps = pd.date_range("2024-01-15 09:00", periods=20, freq="30s")
# Inject duplicates and out-of-order records
bad_ts = pd.DatetimeIndex(list(timestamps) + [timestamps[5], timestamps[3]])
prices = np.random.uniform(3800, 3900, len(bad_ts))

df = pd.DataFrame({"timestamp": bad_ts, "price": prices})
df = df.sample(frac=1, random_state=42)  # shuffle to simulate out-of-order

print(f"Raw records: {len(df)}")

# ── Step 1: Sort by timestamp ─────────────────────────────────────────────────
df = df.sort_values("timestamp")

# ── Step 2: Detect and remove exact duplicate timestamps ──────────────────────
# Keep last (most recently received) record for each timestamp
df = df[~df["timestamp"].duplicated(keep="last")].reset_index(drop=True)
print(f"After dedup: {len(df)}")

# ── Step 3: Set DatetimeIndex and check for gaps ──────────────────────────────
df = df.set_index("timestamp")
expected_freq = "30s"
full_range = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
missing_ts = full_range.difference(df.index)
print(f"Missing timestamps: {missing_ts.tolist()}")

# ── Step 4: Reindex to fill gaps, forward-fill price ─────────────────────────
df = df.reindex(full_range)
df["price"] = df["price"].ffill()
print(df.tail(5))
```

---

## 7. NumPy Vectorized Operations

### Q17 · `Easy` — Replace a loop with vectorized NumPy

**Interview context (LSEG):** *"This function uses a for loop. Rewrite it using NumPy
vectorization and explain the speedup."*

```python
import numpy as np
import time

n = 1_000_000
returns = np.random.normal(0.001, 0.02, n)

# ── Slow: Python for loop ─────────────────────────────────────────────────────
def slow_sharpe(returns, rf=0.0):
    excess = []
    for r in returns:
        excess.append(r - rf)
    mean_excess = sum(excess) / len(excess)
    variance = sum((e - mean_excess)**2 for e in excess) / len(excess)
    return mean_excess / (variance**0.5)

start = time.time()
result_slow = slow_sharpe(returns)
print(f"Loop: {time.time()-start:.3f}s → Sharpe ≈ {result_slow:.4f}")

# ── Fast: NumPy vectorized ────────────────────────────────────────────────────
def fast_sharpe(returns, rf=0.0):
    excess = returns - rf               # vectorized subtraction
    return excess.mean() / excess.std()  # vectorized reduction

start = time.time()
result_fast = fast_sharpe(returns)
print(f"NumPy: {time.time()-start:.3f}s → Sharpe ≈ {result_fast:.4f}")
# Typical speedup: 100-300x
```

---

### Q18 · `Medium` — Pairwise Euclidean distance matrix (vectorized, no loops)

**Interview context (LSEG):** *"Compute the N×M distance matrix between two sets of feature
vectors without any Python loops. What are the memory implications?"*

```python
import numpy as np

np.random.seed(42)
N, M, D = 500, 300, 10   # 500 reference points, 300 query points, 10 features

A = np.random.rand(N, D)   # shape (N, D)
B = np.random.rand(M, D)   # shape (M, D)

# ── Method 1: Broadcasting expansion ─────────────────────────────────────────
# A[:, np.newaxis, :] → (N, 1, D)
# B[np.newaxis, :, :] → (1, M, D)
# diff → (N, M, D)  — intermediate array!
diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]     # shape (N, M, D)
dist_matrix = np.sqrt((diff**2).sum(axis=-1))          # shape (N, M)

# ── Method 2: Memory-efficient via algebraic identity ─────────────────────────
# ||a - b||² = ||a||² + ||b||² - 2·a·bᵀ
A_sq = (A**2).sum(axis=1, keepdims=True)   # (N, 1)
B_sq = (B**2).sum(axis=1, keepdims=True).T  # (1, M)
dist_matrix_v2 = np.sqrt(np.maximum(A_sq + B_sq - 2 * A @ B.T, 0))

print("Distance matrix shape:", dist_matrix.shape)
print("Max diff between methods:", np.abs(dist_matrix - dist_matrix_v2).max())

# ── Memory warning discussion point ───────────────────────────────────────────
# Method 1 creates (N, M, D) = (500, 300, 10) = 1.5M floats = ~12 MB here.
# If N=M=10,000, D=100 → 10,000×10,000×100×8 bytes = 80 GB! → Use scipy.cdist
from scipy.spatial.distance import cdist
dist_scipy = cdist(A, B, metric="euclidean")
print("scipy dist shape:", dist_scipy.shape)
```

---

### Q19 · `Medium` — Z-score normalization, row-wise vs column-wise

**Interview context (General Mills/LSEG):** *"Normalize a feature matrix column-wise for model
training. Then explain how to normalize row-wise (per-sample) and why you'd use it."*

```python
import numpy as np

np.random.seed(0)
X = np.random.rand(100, 5) * np.array([1, 100, 0.5, 1000, 10])  # different scales

# ── Column-wise (feature) normalization — standard for supervised learning ────
col_mean = X.mean(axis=0)         # shape (5,)  — mean per feature
col_std  = X.std(axis=0)          # shape (5,)
X_col_scaled = (X - col_mean) / col_std   # broadcasting: (100,5) - (5,) ✓

print("Column-wise scaled mean ≈ 0:", X_col_scaled.mean(axis=0).round(5))
print("Column-wise scaled std  ≈ 1:", X_col_scaled.std(axis=0).round(5))

# ── Row-wise (sample) normalization — used in NLP (TF-IDF), image embeddings ──
row_mean = X.mean(axis=1, keepdims=True)   # shape (100, 1)
row_std  = X.std(axis=1, keepdims=True)    # shape (100, 1)
X_row_scaled = (X - row_mean) / row_std    # broadcasting: (100,5) - (100,1) ✓

print("\nRow-wise scaled mean per sample ≈ 0:", X_row_scaled.mean(axis=1)[:5].round(5))
```

---

## 8. NumPy Broadcasting

### Q20 · `Medium` — Broadcasting rules & shape prediction (classic trap question)

**Interview context (LSEG):** *"Predict the output shape for each operation. This is asked
verbatim in live coding rounds at LSEG Bangalore."*

```python
import numpy as np

A = np.ones((4, 3))          # shape (4, 3)
B = np.array([3, 2, 1])      # shape (3,)
C = np.array([[1],[2],[3],[4]])  # shape (4, 1)
D = np.ones((1, 1, 3))       # shape (1, 1, 3)
E = np.ones((5, 4, 1))       # shape (5, 4, 1)

# ── Predict before running! ───────────────────────────────────────────────────
print("A + B →", (A + B).shape)        # (4,3) + (3,) → (4,3)  B broadcast rows
print("A + C →", (A + C).shape)        # (4,3) + (4,1) → (4,3)  C broadcast cols
print("B + C →", (B + C).shape)        # (3,) + (4,1) → (4,3)  outer-product style
print("E + D →", (E + D).shape)        # (5,4,1) + (1,1,3) → (5,4,3)

# ── A common broadcasting gotcha: ────────────────────────────────────────────
row_sums = A.sum(axis=1)               # shape (4,) — NOT (4,1)
# A / row_sums  ← this FAILS! (4,3) / (4,) is ambiguous
# Fix: add newaxis to make it (4,1) so it broadcasts across columns
A_row_normalized = A / row_sums[:, np.newaxis]   # (4,3) / (4,1) → (4,3) ✓
print("\nRow-normalized A:\n", A_row_normalized)
```

---

### Q21 · `Hard` — Vectorized rolling z-score using stride tricks (advanced)

**Interview context (LSEG):** *"Compute a rolling 20-day z-score for a price series using
NumPy only — no Pandas. Explain the strides approach."*

```python
import numpy as np

np.random.seed(42)
prices = np.cumsum(np.random.randn(252)) + 100  # 252 trading days

window = 20

# ── Method 1: Explicit loop (baseline, slow) ──────────────────────────────────
rolling_z_slow = np.full(len(prices), np.nan)
for i in range(window - 1, len(prices)):
    w = prices[i - window + 1 : i + 1]
    rolling_z_slow[i] = (prices[i] - w.mean()) / w.std()

# ── Method 2: NumPy stride tricks (memory-efficient, vectorized) ──────────────
def rolling_window(arr, window):
    """Create a 2D view where each row is a rolling window slice."""
    shape = (arr.size - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])  # step by 1 element for each row
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

windows = rolling_window(prices, window)       # shape: (233, 20)
means  = windows.mean(axis=1)                  # shape: (233,)
stds   = windows.std(axis=1)                   # shape: (233,)
rolling_z_fast = np.full(len(prices), np.nan)
rolling_z_fast[window - 1:] = (prices[window - 1:] - means) / stds

# Verify equivalence
print("Max diff (slow vs fast):", np.nanmax(np.abs(rolling_z_slow - rolling_z_fast)))

# ── Pandas equivalent for comparison ─────────────────────────────────────────
import pandas as pd
s = pd.Series(prices)
rolling_z_pd = (s - s.rolling(window).mean()) / s.rolling(window).std()
print("Max diff (numpy vs pandas):", np.nanmax(np.abs(rolling_z_fast - rolling_z_pd.values)))
```

---

### Q22 · `Hard` — Custom weighted portfolio returns (NumPy broadcasting + einsum)

**Interview context (LSEG):** *"You have daily returns for N assets and M different weight
vectors. Compute portfolio returns for all M portfolios simultaneously without loops."*

```python
import numpy as np

np.random.seed(7)
T, N, M = 252, 10, 5   # 252 days, 10 assets, 5 portfolios

# Daily asset returns: shape (T, N)
asset_returns = np.random.normal(0.0005, 0.02, (T, N))

# Portfolio weights: shape (M, N), each row sums to 1
raw_weights = np.random.dirichlet(np.ones(N), size=M)  # shape (M, N)

# ── Method 1: Broadcasting ────────────────────────────────────────────────────
# asset_returns: (T, N) → (T, 1, N)
# raw_weights:   (M, N) → (1, M, N)
# product:                (T, M, N)
# sum over assets (axis=2) → (T, M) = daily portfolio returns
portfolio_returns = (asset_returns[:, np.newaxis, :] * raw_weights[np.newaxis, :, :]).sum(axis=2)

# ── Method 2: Matrix multiplication (more readable, same result) ──────────────
# asset_returns (T, N) @ weights.T (N, M) → (T, M)
portfolio_returns_v2 = asset_returns @ raw_weights.T

print("Shape of portfolio returns:", portfolio_returns.shape)   # (252, 5)
print("Max diff between methods:", np.abs(portfolio_returns - portfolio_returns_v2).max())

# Annualized Sharpe ratio for each portfolio (vectorized)
ann_factor = np.sqrt(252)
sharpe_ratios = ann_factor * portfolio_returns.mean(axis=0) / portfolio_returns.std(axis=0)
print("\nAnnualized Sharpe per portfolio:", sharpe_ratios.round(3))
```

---

## 9. Advanced / Hard Problems

### Q23 · `Hard` — General Mills: Weekly store × product revenue summary report

**Interview context:** *"Given raw transaction data, produce a full summary: top products per
region, week-over-week growth, % contribution to region total."*

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 20_000

df = pd.DataFrame({
    "transaction_id": range(n),
    "date":    pd.to_datetime(
        np.random.choice(pd.date_range("2024-01-01", "2024-03-31"), n)
    ),
    "store_id":   np.random.randint(1, 51, n),
    "region":     np.random.choice(["North", "South", "East", "West"], n),
    "product":    np.random.choice(
        ["Cheerios", "Yoplait", "Nature Valley", "Pillsbury", "Häagen-Dazs"], n
    ),
    "units":    np.random.randint(1, 10, n),
    "price":    np.random.uniform(2.0, 15.0, n),
})
df["revenue"] = df["units"] * df["price"]
df["week"] = df["date"].dt.to_period("W")

# ── Step 1: Weekly revenue per region × product ───────────────────────────────
weekly = (
    df.groupby(["week", "region", "product"])["revenue"]
      .sum()
      .reset_index(name="revenue")
)

# ── Step 2: Week-over-week growth per (region, product) ───────────────────────
weekly = weekly.sort_values(["region", "product", "week"])
weekly["prev_revenue"] = weekly.groupby(["region", "product"])["revenue"].shift(1)
weekly["wow_growth_pct"] = (
    (weekly["revenue"] - weekly["prev_revenue"]) / weekly["prev_revenue"] * 100
).round(2)

# ── Step 3: % contribution to region-week total ───────────────────────────────
region_week_total = (
    weekly.groupby(["week", "region"])["revenue"]
          .transform("sum")
)
weekly["pct_of_region"] = (weekly["revenue"] / region_week_total * 100).round(2)

# ── Step 4: Latest week's top 3 products per region ──────────────────────────
latest_week = weekly["week"].max()
top3 = (
    weekly[weekly["week"] == latest_week]
      .sort_values(["region", "revenue"], ascending=[True, False])
      .groupby("region")
      .head(3)
      [["region", "product", "revenue", "wow_growth_pct", "pct_of_region"]]
)

print(f"Latest week: {latest_week}")
print(top3.to_string(index=False))
```

---

### Q24 · `Hard` — LSEG: Rolling Sharpe, Max Drawdown, VaR for a portfolio

**Interview context (LSEG):** *"Implement risk metrics for a portfolio — rolling Sharpe,
maximum drawdown, and Value-at-Risk — using vectorized Pandas/NumPy only."*

```python
import pandas as pd
import numpy as np

np.random.seed(0)
T = 504   # 2 years of trading days
returns = pd.Series(
    np.random.normal(0.0006, 0.018, T),
    index=pd.bdate_range("2022-01-03", periods=T),
    name="portfolio_return"
)

# ── Cumulative returns ─────────────────────────────────────────────────────────
cum_returns = (1 + returns).cumprod()

# ── Rolling 63-day (≈ 1 quarter) Sharpe Ratio ─────────────────────────────────
window = 63
ann = np.sqrt(252)
rolling_sharpe = (
    returns.rolling(window).mean() / returns.rolling(window).std() * ann
)

# ── Maximum Drawdown ───────────────────────────────────────────────────────────
rolling_peak = cum_returns.cummax()   # running maximum
drawdown = (cum_returns - rolling_peak) / rolling_peak
max_drawdown = drawdown.min()
print(f"Max Drawdown: {max_drawdown:.2%}")

# ── Value-at-Risk (95% VaR — historical simulation) ──────────────────────────
var_95 = np.percentile(returns, 5)
print(f"95% Daily VaR: {var_95:.4f} ({var_95:.2%})")

# ── Conditional VaR (CVaR / Expected Shortfall) ───────────────────────────────
cvar_95 = returns[returns <= var_95].mean()
print(f"95% CVaR (Expected Shortfall): {cvar_95:.4f} ({cvar_95:.2%})")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = pd.DataFrame({
    "Metric": ["Ann. Return", "Ann. Volatility", "Sharpe (full period)",
               "Max Drawdown", "95% VaR (daily)", "95% CVaR (daily)"],
    "Value": [
        f"{returns.mean() * 252:.2%}",
        f"{returns.std() * ann:.2%}",
        f"{returns.mean() / returns.std() * ann:.2f}",
        f"{max_drawdown:.2%}",
        f"{var_95:.2%}",
        f"{cvar_95:.2%}",
    ]
})
print("\n", summary.to_string(index=False))
```

---

### Q25 · `Hard` — Chunked processing of large CSV (OOM-safe pipeline)

**Interview context (General Mills):** *"Our daily transaction file is 8 GB. Write a memory-safe
pipeline to compute total revenue per product category without loading it all at once."*

```python
import pandas as pd
import numpy as np
import io

# ── Simulate writing a large-ish CSV to memory (in real life: actual file) ────
np.random.seed(0)
n_total = 1_000_000
fake_csv_data = pd.DataFrame({
    "category": np.random.choice(["Cereal", "Dairy", "Snacks", "Baking"], n_total),
    "product":  np.random.choice(["Cheerios", "Yoplait", "Pillsbury", "Nature Valley"], n_total),
    "units":    np.random.randint(1, 20, n_total),
    "price":    np.random.uniform(1.5, 15.0, n_total).round(2),
})
fake_csv_data["revenue"] = fake_csv_data["units"] * fake_csv_data["price"]
csv_buffer = io.StringIO()
fake_csv_data.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

# ── Chunked read and aggregation ──────────────────────────────────────────────
CHUNK_SIZE = 100_000
accumulator = {}   # {category: total_revenue}

for chunk in pd.read_csv(csv_buffer, chunksize=CHUNK_SIZE,
                         dtype={"category": "category", "product": "category",
                                "units": "int16", "price": "float32"}):
    chunk["revenue"] = chunk["units"] * chunk["price"]
    chunk_agg = chunk.groupby("category", observed=True)["revenue"].sum()
    for cat, rev in chunk_agg.items():
        accumulator[cat] = accumulator.get(cat, 0.0) + rev

result = pd.Series(accumulator, name="total_revenue").sort_values(ascending=False)
print("Revenue by category (chunked):\n", result)

# ── Verify against full in-memory result ─────────────────────────────────────
verify = fake_csv_data.groupby("category")["revenue"].sum().sort_values(ascending=False)
print("\nRevenue by category (full load):\n", verify)
print("\nDiff:", (result.sort_index() - verify.sort_index()).abs().max())
```

---

## 10. Common Gotchas & Interview Tips

### 🚨 Top Gotchas

| Gotcha | Symptom | Fix |
|---|---|---|
| `SettingWithCopyWarning` | Silent NaN or unexpected mutation | Use `.loc[]` for all in-place writes; Pandas 3.0 CoW resolves this |
| `iterrows()` in production | 100-1000× slower than vectorized | Replace with `.apply()`, vectorized ops, or NumPy |
| Rolling without shift | Feature leakage in ML models | Always `series.shift(1).rolling(n).mean()` |
| Missing `validate=` in merge | Silent row-count explosion | Add `validate='many_to_one'` or `'one_to_one'` |
| `category` dtype in `pd.concat` | ValueError on incompatible categories | Cast to `str` before concat, then back to `category` |
| `as_strided` out of bounds | Memory corruption / crash | Only use with read-only result; never write to stride views |
| `groupby(...).apply()` on large data | 10-100× slower than `.transform()` | Use `.transform()` for broadcast aggregations |
| Timezone-naive vs timezone-aware | TypeError on merge/compare | Localize early: `df["ts"] = df["ts"].dt.tz_localize("UTC")` |
| `inplace=True` | Slower; incompatible with method chaining | Assign back: `df = df.dropna()` |
| NumPy broadcasting `(N,)` vs `(N,1)` | Wrong axis broadcast | Use `keepdims=True` or `[:, np.newaxis]` deliberately |

---

### 💡 General Mills Interview Tips
1. **CPG framing:** Tie every answer to real business outcomes — *"This reduces demand-forecast MAPE by X%"* lands better than pure technical explanations.
2. **EDA first:** They like structured thinkers. Start with `.describe()`, `.info()`, `.value_counts()` before jumping to complex operations.
3. **SQL + Pandas synergy:** Expect SQL alongside Python in the same round. GroupBy = `GROUP BY`, merge = `JOIN`, pivot = pivot tables.
4. **Seasonality awareness:** Mention Diwali/Navratri/festive season effects in Indian retail data when discussing time-series features.

### 💡 LSEG Interview Tips
1. **Write without IDE autocomplete:** Codility and live rounds use bare editors. Practice with `vim` or plain notepad.
2. **Justify complexity:** For Lead roles, always explain *why* — `merge_asof` vs exact join, `stride_tricks` vs rolling, `cdist` vs broadcasting.
3. **Financial domain vocabulary:** Use terms like slippage, drawdown, VaR, tick data, market microstructure — shows domain fluency.
4. **Production-grade instincts:** Mention `validate=` on merges, `min_periods=` on rolling, `chunksize=` for large files — signals seniority.
5. **LSEG-specific:** They heavily test Python string parsing and log file processing (Codility rounds). Practice `str.split()`, regex, and `datetime.strptime()` without a library.

---

### 📋 Quick Reference: Time Complexity Cheatsheet

| Operation | Time Complexity | Notes |
|---|---|---|
| `df.groupby().agg()` | O(N log N) | Cython-optimized |
| `df.groupby().apply()` | O(N × fn_cost) | Python loop — use sparingly |
| `pd.merge()` (hash join) | O(N + M) avg | Hash-based for equality joins |
| `pd.merge_asof()` | O(N + M) | Requires both sorted by key |
| `df.pivot_table()` | O(N log N) | Internally groupby + reshape |
| `np.dot()` / `@` | O(N²·M) BLAS | BLAS-optimized, very fast |
| NumPy broadcasting | O(output size) | No Python loops; memory beware |
| `iterrows()` | O(N) Python iters | Avoid at all costs in DS work |

---

*Last updated: November 2026 · Sources: Glassdoor, AmbitionBox, InterviewQuery, LSEG Codility reports, General Mills recruiter feedback*
