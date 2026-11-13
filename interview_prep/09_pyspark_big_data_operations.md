# PySpark, Databricks & Distributed Data Operations — Senior & Lead Data Scientist Interview Prep
### Target Companies: General Mills India (Azure Databricks Lakehouse) | LSEG India (AWS/Azure Financial Big Data)
### Role Focus: Senior Data Scientist / Lead Data Scientist / Big Data ML Engineer
### Last Updated: November 2026

---

> [!IMPORTANT]
> **Why PySpark & Databricks are Mandatory for Senior / Lead Roles in India:**
> Both **General Mills (Powai, Mumbai)** and **LSEG (Bangalore & Mumbai)** run global analytical workloads that far exceed single-node RAM:
> * **General Mills:** Ingests daily point-of-sale (POS) scanner data from Walmart, Target, Kroger, and global grocers across 100,000+ retail stores and 50,000+ SKUs. Their entire global Demand Sensing and Trade Promotion engine runs on **Azure Databricks and Delta Lake**.
> * **LSEG:** Ingests hundreds of millions of daily tick updates, order book state transitions, and regulatory trade records. Feature engineering and factor backtesting rely heavily on distributed Spark clusters.
>
> In technical rounds, interviewers evaluate whether you know how to **avoid shuffle bottlenecks**, **mitigate data skew via salting**, leverage **`pandas_udf` with Apache Arrow**, write complex **PySpark Window functions**, and perform **Delta Lake ACID merges**.

---

## Table of Contents

1. [Spark Core Architecture & Execution Model](#1-spark-core-architecture--execution-model)
   - [Q1 — Spark Execution Engine: Driver, Executors, Py4J, Catalyst, and Tungsten [Medium]](#q1--spark-execution-engine-driver-executors-py4j-catalyst-and-tungsten-medium)
   - [Q2 — Narrow vs Wide Transformations & Shuffle Mechanics [Easy]](#q2--narrow-vs-wide-transformations--shuffle-mechanics-easy)
2. [Data Skew Mitigation & Join Strategies](#2-data-skew-mitigation--join-strategies)
   - [Q3 — Data Skew in Joins: Salting Technique Implementation [Hard]](#q3--data-skew-in-joins-salting-technique-implementation-hard)
   - [Q4 — Join Strategies: Broadcast Hash Join (BHJ) vs Sort-Merge Join (SMJ) [Medium]](#q4--join-strategies-broadcast-hash-join-bhj-vs-sort-merge-join-smj-medium)
   - [Q5 — Adaptive Query Execution (AQE) in Spark 3.x+: Auto-Skew & Dynamic Coalesce [Medium]](#q5--adaptive-query-execution-aqe-in-spark-3x-auto-skew--dynamic-coalesce-medium)
3. [PySpark Window Functions & Analytical Operations](#3-pyspark-window-functions--analytical-operations)
   - [Q6 — Rolling 7-Day Average & Cumulative Revenue by Store-SKU [Medium]](#q6--rolling-7-day-average--cumulative-revenue-by-store-sku-medium)
   - [Q7 — Gaps & Islands in PySpark: Stockout Streak Detection [Hard]](#q7--gaps--islands-in-pyspark-stockout-streak-detection-hard)
   - [Q8 — Lag / Lead with Date Difference for Inter-Purchase Time [Easy]](#q8--lag--lead-with-date-difference-for-inter-purchase-time-easy)
4. [User-Defined Functions (UDFs) & Apache Arrow](#4-user-defined-functions-udfs--apache-arrow)
   - [Q9 — Standard Python UDF vs `pandas_udf` (Vectorized UDF): The 10x Speedup [Hard]](#q9--standard-python-udf-vs-pandas_udf-vectorized-udf-the-10x-speedup-hard)
   - [Q10 — Grouped Map Pandas UDF: Distributed SKU-Level Forecasting [Hard]](#q10--grouped-map-pandas-udf-distributed-sku-level-forecasting-hard)
5. [Delta Lake on Databricks & ACID Transactions](#5-delta-lake-on-databricks--acid-transactions)
   - [Q11 — Delta Lake `MERGE INTO` (Upserting Daily POS Data) [Medium]](#q11--delta-lake-merge-into-upserting-daily-pos-data-medium)
   - [Q12 — Time Travel & Audit Logging in Financial Data [Easy]](#q12--time-travel--audit-logging-in-financial-data-easy)
   - [Q13 — Partitioning vs Z-Ordering for High-Cardinality Queries [Medium]](#q13--partitioning-vs-z-ordering-for-high-cardinality-queries-medium)
6. [Memory Management & OOM Debugging](#6-memory-management--oom-debugging)
   - [Q14 — Diagnosing Driver OOM vs Executor OOM in Production [Hard]](#q14--diagnosing-driver-oom-vs-executor-oom-in-production-hard)
   - [Q15 — Repartition vs Coalesce: When to Use Which [Easy]](#q15--repartition-vs-coalesce-when-to-use-which-easy)
   - [Q16 — Cache vs Persist with Storage Levels [Easy]](#q16--cache-vs-persist-with-storage-levels-easy)
7. [Domain-Specific Big Data Scenarios](#7-domain-specific-big-data-scenarios)
   - [Q17 — [General Mills] Hierarchical SKU Demand Aggregation & Outlier Winsorization [Medium]](#q17--general-mills-hierarchical-sku-demand-aggregation--outlier-winsorization-medium)
   - [Q18 — [LSEG] Real-Time Tick Resampling & VWAP Calculation from Order Stream [Hard]](#q18--lseg-real-time-tick-resampling--vwap-calculation-from-order-stream-hard)
   - [Q19 — [LSEG] Wash-Trading Anomaly Detection via PySpark Self-Join [Hard]](#q19--lseg-wash-trading-anomaly-detection-via-pyspark-self-join-hard)
   - [Q20 — Distributed ML: Hyperparameter Tuning with SparkTrials & Hyperopt [Medium]](#q20--distributed-ml-hyperparameter-tuning-with-sparktrials--hyperopt-medium)

---

## 1. Spark Core Architecture & Execution Model

### Q1 — Spark Execution Engine: Driver, Executors, Py4J, Catalyst, and Tungsten [Medium]

**Interview Context:**
Common in the first 15 minutes of senior technical rounds.
*"Explain the lifecycle of a PySpark job. How does Python communicate with the JVM? What are Catalyst and Project Tungsten?"*

**Key Architectural Insights:**
1. **Driver & Executors:** The **Driver** converts user code into a Directed Acyclic Graph (DAG) of stages, schedules tasks, and coordinates execution. **Executors** are worker JVM processes running tasks on cluster nodes.
2. **Py4J Bridge:** PySpark driver runs Python code that calls JVM objects via a Py4J gateway. Worker nodes launch a Python daemon subprocess per executor task; data is serialized between JVM memory and Python worker memory (unless using Apache Arrow).
3. **Catalyst Optimizer:** Performs rule-based and cost-based optimizations:
   - Analysis (resolves column names against catalog).
   - Logical Optimization (predicate pushdown, constant folding, projection pruning).
   - Physical Planning (selects join strategies: BHJ vs SMJ).
   - Code Generation (generates Java bytecode on the fly).
4. **Project Tungsten:** Bypasses JVM garbage collection by managing off-heap binary memory (`sun.misc.Unsafe`), leveraging CPU L1/L2 caches and cache-aware algorithms.

```python
# Demonstrating Catalyst Predicate Pushdown and Projection Pruning
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .appName("CatalystOptimizationDemo") \
    .master("local[*]") \
    .getOrCreate()

# Create sample dataset
data = [(i, f"Store_{i%10}", f"SKU_{i%100}", float(i * 1.5)) for i in range(1000)]
df = spark.createDataFrame(data, ["order_id", "store_id", "sku_id", "sales"])

# Query: Filter before or after transformation
# Catalyst optimizer pushes the predicate 'sales > 500' down to the data read phase
optimized_query = df.filter(F.col("sales") > 500.0) \
                    .select("store_id", "sales") \
                    .groupBy("store_id") \
                    .agg(F.sum("sales").alias("total_sales"))

# Inspect the Catalyst Logical vs Physical Plan
print("=== Catalyst Physical Plan ===")
optimized_query.explain(extended=False)
```

---

### Q2 — Narrow vs Wide Transformations & Shuffle Mechanics [Easy]

**Interview Context:**
*"What is the difference between a narrow and a wide transformation in Spark? Why are wide transformations expensive, and how do they determine stage boundaries?"*

* **Narrow Transformations (`map`, `filter`, `withColumn`, `union`):** Each input partition contributes to at most one output partition. No data movement across cluster nodes. Pipelined within a single stage.
* **Wide Transformations (`groupBy`, `join`, `distinct`, `repartition`, `orderBy`):** Input partitions contribute to multiple output partitions. Requires a **Shuffle**—serializing data, writing shuffle spill files to local disk, transmitting data over the network, and deserializing on receiving executors.
* **Stage Boundary:** A shuffle operation creates a hard boundary between two stages in the DAG.

```python
# Narrow Transformation (Pipelined within Stage 1)
df_narrow = df.withColumn("tax", F.col("sales") * 0.18).filter(F.col("sales") > 100)

# Wide Transformation (Causes Shuffle, Splits into Stage 2)
df_wide = df_narrow.groupBy("store_id").agg(F.avg("sales").alias("avg_sales"))

# Inspect RDD Stage Dependencies (ShuffleRDD marks wide transformation)
print("RDD Lineage:")
print(df_wide.rdd.toDebugString().decode())
```

---

## 2. Data Skew Mitigation & Join Strategies

### Q3 — Data Skew in Joins: Salting Technique Implementation [Hard]

**Interview Context:**
A classic scenario in General Mills and LSEG interviews.
*"You are joining a 100-million row transaction table with a dimension table on `product_id`. One product (e.g., Cheerios) represents 35% of all transactions. 199 tasks finish in 10 seconds, but 1 task hangs for 45 minutes and fails with `FetchFailedException` or executor OOM. How do you resolve this using key salting?"*

**The Salting Algorithm:**
1. Append a random integer (`0` to `K-1`) to the skewed join key in the large table (e.g., `Cheerios_3`).
2. Replicate (explode) the small dimension table $K$ times, creating all possible salt suffix matches (`Cheerios_0`, `Cheerios_1`, ..., `Cheerios_K-1`).
3. Join on the composite salted key. The massive partition is evenly distributed across $K$ executors!

```python
from pyspark.sql import functions as F

def salted_join(df_large, df_dim, join_key: str, salt_bins: int = 10):
    """
    Mitigates severe join skew by salting the join key.
    """
    # 1. Salt the large transaction dataframe
    df_large_salted = df_large.withColumn(
        "salted_key",
        F.concat(F.col(join_key), F.lit("_"), F.floor(F.rand() * salt_bins))
    )

    # 2. Explode the dimension dataframe across all salt bins
    salt_array = F.array([F.lit(i) for i in range(salt_bins)])
    df_dim_exploded = df_dim.withColumn("salt_id", F.explode(salt_array)) \
                            .withColumn("salted_key", F.concat(F.col(join_key), F.lit("_"), F.col("salt_id"))) \
                            .drop("salt_id")

    # 3. Perform join on salted_key (eliminates data skew!)
    joined = df_large_salted.join(df_dim_exploded, on="salted_key", how="inner") \
                            .drop("salted_key")
    return joined

# --- Verification ---
sample_large = spark.createDataFrame([
    ("Cheerios", 4.99), ("Cheerios", 4.99), ("Cheerios", 4.99),
    ("NatureValley", 3.29), ("Yoplait", 1.19)
], ["product_id", "price"])

sample_dim = spark.createDataFrame([
    ("Cheerios", "Breakfast Cereal"),
    ("NatureValley", "Healthy Snacks"),
    ("Yoplait", "Dairy & Yogurt")
], ["product_id", "category"])

salted_result = salted_join(sample_large, sample_dim, join_key="product_id", salt_bins=4)
salted_result.show()
```

---

### Q4 — Join Strategies: Broadcast Hash Join (BHJ) vs Sort-Merge Join (SMJ) [Medium]

**Interview Context:**
*"Explain how Spark decides between Broadcast Hash Join, Sort-Merge Join, and Shuffle Hash Join. When should you explicitly use `broadcast()`?"*

| Strategy | Pre-conditions | Network Shuffle | Memory Footprint | When to Use |
|---|---|---|---|---|
| **Broadcast Hash Join (BHJ)** | One table $\le$ 10MB (configurable via `autoBroadcastJoinThreshold`) | **Zero** (driver broadcasts small table to all workers) | Fits into executor RAM | Dimension table $\le$ 100MB joined with billion-row fact table |
| **Sort-Merge Join (SMJ)** | Large tables, sortable keys | **High** (both tables partitioned by hash of key and sorted) | Disk/spill friendly | Default for joining two massive datasets |
| **Shuffle Hash Join** | One table moderately smaller | **High** (builds in-memory hash table on partitioned chunk) | High executor memory needed | When sorting is expensive and memory is ample |

```python
from pyspark.sql.functions import broadcast

# Explicitly forcing Broadcast Hash Join to avoid shuffle
df_transactions = spark.createDataFrame([(1, "SKU101", 10.0), (2, "SKU102", 20.0)], ["tx_id", "sku_id", "amount"])
df_sku_dim = spark.createDataFrame([("SKU101", "Cheerios"), ("SKU102", "Pillsbury")], ["sku_id", "sku_name"])

# Wrapping small table with broadcast() hint
df_bhj = df_transactions.join(broadcast(df_sku_dim), on="sku_id", how="inner")
print("BHJ Execution Plan:")
df_bhj.explain()  # Verify 'BroadcastHashJoin' in plan
```

---

### Q5 — Adaptive Query Execution (AQE) in Spark 3.x+: Auto-Skew & Dynamic Coalesce [Medium]

**Interview Context:**
Probed heavily in Databricks and modern PySpark rounds.
*"What is AQE (Adaptive Query Execution)? Name the three major runtime optimizations it performs."*

**The 3 AQE Pillars (Runtime Re-planning):**
1. **Dynamically Coalescing Shuffle Partitions:** Automatically merges tiny post-shuffle partitions into sensible sizes (e.g., 64MB) to prevent the "200 small partitions" performance drain.
2. **Dynamically Switching Join Strategies:** If runtime filtering reduces a table's size below `autoBroadcastJoinThreshold`, AQE converts a Sort-Merge Join into a Broadcast Join at runtime!
3. **Dynamically Handling Skew Joins:** Automatically detects partitions that are significantly larger than the median and splits them into smaller sub-partitions using salting behind the scenes.

```python
# Enabling AQE in Spark configuration
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", 5)  # 5x larger than median
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "64MB")
```

---

## 3. PySpark Window Functions & Analytical Operations

### Q6 — Rolling 7-Day Average & Cumulative Revenue by Store-SKU [Medium]

**Interview Context:**
Standard General Mills live coding challenge:
*"Given a daily sales DataFrame with columns `store_id`, `sku_id`, `date`, `sales`, calculate: (1) 7-day rolling average sales, (2) cumulative year-to-date sales, and (3) percentage contribution to total store sales for that day."*

```python
from pyspark.sql import Window
import pyspark.sql.functions as F

# Sample daily sales data
sales_data = [
    ("Store_01", "SKU_A", "2026-11-01", 100.0),
    ("Store_01", "SKU_A", "2026-11-02", 120.0),
    ("Store_01", "SKU_A", "2026-11-03", 90.0),
    ("Store_01", "SKU_A", "2026-11-04", 110.0),
    ("Store_01", "SKU_B", "2026-11-01", 50.0),
    ("Store_01", "SKU_B", "2026-11-02", 70.0),
]
df_sales = spark.createDataFrame(sales_data, ["store_id", "sku_id", "date", "sales"])

# Window 1: 7-day rolling window by Store-SKU (last 6 days + current row)
# Note: rowsBetween(-6, 0) specifies physical row offset
rolling_7d_window = Window.partitionBy("store_id", "sku_id") \
                          .orderBy("date") \
                          .rowsBetween(-6, 0)

# Window 2: Cumulative sales from beginning of history to current date
cumulative_window = Window.partitionBy("store_id", "sku_id") \
                          .orderBy("date") \
                          .rowsBetween(Window.unboundedPreceding, Window.currentRow)

# Window 3: Total store sales on that day across all SKUs
daily_store_window = Window.partitionBy("store_id", "date")

df_features = df_sales \
    .withColumn("rolling_7d_avg", F.avg("sales").over(rolling_7d_window)) \
    .withColumn("cum_ytd_sales", F.sum("sales").over(cumulative_window)) \
    .withColumn("daily_store_total", F.sum("sales").over(daily_store_window)) \
    .withColumn("sku_pct_of_store", (F.col("sales") / F.col("daily_store_total")) * 100.0)

df_features.show()
```

---

### Q7 — Gaps & Islands in PySpark: Stockout Streak Detection [Hard]

**Interview Context:**
Critical for General Mills supply chain resilience analytics.
*"Detect consecutive stockout periods (islands where inventory == 0) for each store-SKU. Return the start date, end date, and duration (in days) of each out-of-stock streak."*

**The PySpark Gaps & Islands Solution:**
1. Generate row numbers ordered by date (`rn`).
2. Generate row numbers ordered by date *only for zero-inventory rows* (`stockout_rn`).
3. The difference `rn - stockout_rn` forms a unique invariant group ID for contiguous consecutive dates!

```python
from pyspark.sql import Window
import pyspark.sql.functions as F

stock_data = [
    ("Store1", "Cheerios", "2026-11-01", 50),
    ("Store1", "Cheerios", "2026-11-02", 0),  # Streak 1 Start
    ("Store1", "Cheerios", "2026-11-03", 0),
    ("Store1", "Cheerios", "2026-11-04", 0),  # Streak 1 End (3 days)
    ("Store1", "Cheerios", "2026-11-05", 25),
    ("Store1", "Cheerios", "2026-11-06", 0),  # Streak 2 Start
    ("Store1", "Cheerios", "2026-11-07", 0),  # Streak 2 End (2 days)
    ("Store1", "Cheerios", "2026-11-08", 40),
]
df_stock = spark.createDataFrame(stock_data, ["store", "sku", "date", "inventory"])

# Filter to stockout days
df_zero = df_stock.filter(F.col("inventory") == 0)

# Window to calculate row numbers
w_full = Window.partitionBy("store", "sku").orderBy("date")

# Difference between actual date offset and row_number creates invariant group
df_grouped = df_zero.withColumn("date_ts", F.to_date("date")) \
                    .withColumn("rn", F.row_number().over(w_full)) \
                    .withColumn("streak_id", F.date_sub(F.col("date_ts"), F.col("rn")))

# Aggregate by streak_id
df_streaks = df_grouped.groupBy("store", "sku", "streak_id") \
                       .agg(
                           F.min("date").alias("streak_start"),
                           F.max("date").alias("streak_end"),
                           F.count("*").alias("duration_days")
                       ).drop("streak_id")

df_streaks.show()
```

---

### Q8 — Lag / Lead with Date Difference for Inter-Purchase Time [Easy]

**Interview Context:**
*"Calculate the inter-purchase time (days elapsed since last order) for each retail customer in PySpark."*

```python
from pyspark.sql import Window
import pyspark.sql.functions as F

orders = [
    ("Cust_A", "2026-10-01", 120.0),
    ("Cust_A", "2026-10-15", 85.0),
    ("Cust_A", "2026-11-05", 210.0),
    ("Cust_B", "2026-10-10", 45.0),
    ("Cust_B", "2026-10-12", 60.0),
]
df_orders = spark.createDataFrame(orders, ["customer_id", "order_date", "amount"])

w_cust = Window.partitionBy("customer_id").orderBy("order_date")

df_inter_purchase = df_orders \
    .withColumn("prev_order_date", F.lag("order_date", 1).over(w_cust)) \
    .withColumn("days_since_prev_order", F.datediff(F.col("order_date"), F.col("prev_order_date")))

df_inter_purchase.show()
```

---

## 4. User-Defined Functions (UDFs) & Apache Arrow

### Q9 — Standard Python UDF vs `pandas_udf` (Vectorized UDF): The 10x Speedup [Hard]

**Interview Context:**
Senior-level evaluation:
*"Why are standard PySpark Python UDFs considered an anti-pattern in production pipelines? How does `pandas_udf` use Apache Arrow to solve serialization bottlenecks? Write a vectorized Pandas UDF to compute z-scores."*

* **Standard Python UDF:** Rows are serialized to bytes, sent via socket to Python worker, evaluated one-by-one as Python objects, and serialized back to JVM. High GC pressure and CPU serialization overhead.
* **`pandas_udf` (Vectorized):** Leverages **Apache Arrow** shared memory format. Data batches are transferred in columnar chunks with zero copy overhead. The Python worker executes vectorized C/NumPy operations on entire Pandas Series.

```python
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd
import numpy as np

# Vectorized Series-to-Series Pandas UDF using Apache Arrow
@pandas_udf(DoubleType())
def calculate_zscore(values: pd.Series) -> pd.Series:
    """Vectorized calculation executed at C/NumPy speed on chunked Arrow batches."""
    mean = values.mean()
    std = values.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=values.index)
    return (values - mean) / std

df_prices = spark.createDataFrame([(1, 100.0), (2, 105.0), (3, 98.0), (4, 150.0)], ["id", "price"])
df_z = df_prices.withColumn("price_zscore", calculate_zscore(F.col("price")))
df_z.show()
```

---

### Q10 — Grouped Map Pandas UDF: Distributed SKU-Level Forecasting [Hard]

**Interview Context:**
Common in General Mills demand modeling:
*"We have 10,000 distinct SKUs. We want to fit an independent Scikit-Learn linear regression model for EACH SKU in parallel across the cluster using PySpark. How do you implement this using `applyInPandas`?"*

```python
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import pandas as pd
from sklearn.linear_model import LinearRegression

# Schema of output DataFrame returned by applyInPandas
output_schema = StructType([
    StructField("sku_id", StringType(), False),
    StructField("price_elasticity", DoubleType(), False),
    StructField("intercept", DoubleType(), False),
])

def fit_sku_elasticity(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Executed in parallel for each unique SKU partition on worker nodes.
    Input: Pandas DataFrame for one SKU.
    Output: Pandas DataFrame matching output_schema.
    """
    sku = pdf["sku_id"].iloc[0]
    
    # Simple log-log regression: ln(demand) = beta * ln(price) + alpha
    X = np.log(pdf[["price"]].values)
    y = np.log(pdf["demand"].values)

    model = LinearRegression().fit(X, y)
    elasticity = float(model.coef_[0])
    intercept = float(model.intercept_)

    return pd.DataFrame([{
        "sku_id": sku,
        "price_elasticity": elasticity,
        "intercept": intercept
    }])

# Sample distributed data across 2 SKUs
sku_data = [
    ("Cheerios", 4.0, 1000.0), ("Cheerios", 4.5, 880.0), ("Cheerios", 5.0, 750.0),
    ("Yoplait", 1.0, 5000.0), ("Yoplait", 1.2, 4200.0), ("Yoplait", 1.5, 3100.0)
]
df_elasticity_input = spark.createDataFrame(sku_data, ["sku_id", "price", "demand"])

# Execute distributed modeling per SKU across Spark partitions
df_sku_models = df_elasticity_input.groupBy("sku_id").applyInPandas(fit_sku_elasticity, schema=output_schema)
df_sku_models.show()
```

---

## 5. Delta Lake on Databricks & ACID Transactions

### Q11 — Delta Lake `MERGE INTO` (Upserting Daily POS Data) [Medium]

**Interview Context:**
General Mills relies on Delta Lake for reliable batch/streaming ingestion.
*"Explain how Delta Lake achieves ACID transactions on top of Parquet files. Write a PySpark Delta Lake `MERGE` statement to upsert daily sales updates (updating existing orders if price changed, inserting new orders)."*

**The Mechanism:**
Delta Lake uses a **transaction log (`_delta_log/`)** containing ordered JSON commit records. It uses optimistic concurrency control: reading is snapshot-isolated, and writes commit atomically.

```python
# Pure PySpark simulation of DeltaTable MERGE semantics
from pyspark.sql import functions as F

# Base Delta Table (Current state in Lakehouse)
existing_orders = spark.createDataFrame([
    (101, "SKU_A", 100.0, "CONFIRMED"),
    (102, "SKU_B", 250.0, "PENDING")
], ["order_id", "sku_id", "amount", "status"])

# Incremental daily batch (Updates order 102, adds order 103)
daily_updates = spark.createDataFrame([
    (102, "SKU_B", 250.0, "CONFIRMED"),  # Update status
    (103, "SKU_C", 180.0, "PENDING")     # New insert
], ["order_id", "sku_id", "amount", "status"])

# --- Delta Lake SQL Syntax used in Databricks ---
"""
MERGE INTO target_table AS target
USING daily_updates AS source
ON target.order_id = source.order_id
WHEN MATCHED THEN
  UPDATE SET target.amount = source.amount, target.status = source.status
WHEN NOT MATCHED THEN
  INSERT (order_id, sku_id, amount, status)
  VALUES (source.order_id, source.sku_id, source.amount, source.status)
"""

# Replicating MERGE in standard PySpark: Full Outer Join + Coalesce
merged_df = existing_orders.alias("t").join(
    daily_updates.alias("s"),
    on="order_id",
    how="full_outer"
).select(
    F.col("order_id"),
    F.coalesce(F.col("s.sku_id"), F.col("t.sku_id")).alias("sku_id"),
    F.coalesce(F.col("s.amount"), F.col("t.amount")).alias("amount"),
    F.coalesce(F.col("s.status"), F.col("t.status")).alias("status")
)
merged_df.show()
```

---

### Q12 — Time Travel & Audit Logging in Financial Data [Easy]

**Interview Context:**
Crucial for LSEG regulatory reporting (MiFID II / SEC compliance).
*"How does Delta Lake time travel work? Why is it essential for financial auditability and reproducible machine learning models?"*

```sql
-- Querying historical snapshot of financial risk table as of 30 days ago
SELECT * FROM lseg_risk_scores TIMESTAMP AS OF '2026-10-14 00:00:00';

-- Querying exact table version used to train production ML model v1.4
SELECT * FROM lseg_risk_scores VERSION AS OF 42;

-- Rollback corrupted table state after bad data ingestion pipeline run
RESTORE TABLE lseg_risk_scores TO VERSION AS OF 41;
```

---

### Q13 — Partitioning vs Z-Ordering for High-Cardinality Queries [Medium]

**Interview Context:**
*"When should you partition a Delta Lake table versus using Z-Order clustering? Why is partitioning by high-cardinality keys like `customer_id` or `timestamp` a disastrous mistake?"*

* **Partitioning (Directory level):** Creates separate subdirectories (e.g., `date=2026-11-13/`).
  - **Best for:** Low-cardinality columns (e.g., `year`, `month`, `region`).
  - **Anti-pattern:** High-cardinality columns (e.g., `timestamp`, `user_id`) generate millions of tiny subdirectories, crashing the cloud storage metastore (the "small file problem").
* **Z-Order Clustering (File-internal multidimensional sorting):** Groups related information along a space-filling Peano curve inside Parquet files.
  - **Best for:** High-cardinality columns frequently queried in `WHERE` clauses (e.g., `store_id`, `sku_id`, `instrument_ticker`). Enables **data skipping** (skipping whole Parquet files using min/max metadata stats).

```sql
-- Optimizing a Delta table on Databricks with Z-ORDER
OPTIMIZE general_mills_pos_lakehouse
ZORDER BY (store_id, sku_id);
```

---

## 6. Memory Management & OOM Debugging

### Q14 — Diagnosing Driver OOM vs Executor OOM in Production [Hard]

**Interview Context:**
Standard troubleshooting question for Lead Data Scientists.
*"Your Spark job failed with `OutOfMemoryError: Java heap space`. How do you immediately identify whether it was a Driver OOM or an Executor OOM, and what are the exact remedies?"*

```
                 ┌──────────────────────────────────────────────────────────┐
                 │                SPARK OUT-OF-MEMORY (OOM)                │
                 └────────────────────────────┬─────────────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
         DRIVER OUT-OF-MEMORY                                EXECUTOR OUT-OF-MEMORY
  • Causes:                                           • Causes:
    1. Calling df.collect() on 50M rows                 1. Severe Data Skew (one task gets 10GB)
    2. broadcast(large_df) where df > driver RAM        2. Excessive memory overhead in Python worker
    3. Broadcast join collecting too much data          3. Shuffle spill to disk failing
  • Fixes:                                            • Fixes:
    1. Replace collect() with take(N) or write          1. Salt skewed keys or enable AQE skew join
    2. Increase spark.driver.memory                     2. Increase spark.executor.memoryOverhead
    3. Disable autoBroadcastJoin                        3. Re-partition (increase shuffle partitions)
```

---

### Q15 — Repartition vs Coalesce: When to Use Which [Easy]

**Interview Context:**
*"Explain the difference between `repartition(n)` and `coalesce(n)`. When should you NEVER use `repartition`?"*

* **`repartition(n)`:** Full shuffle across all nodes. Can **increase or decrease** the number of partitions. Produces uniformly sized partitions. Essential for fixing data skew.
* **`coalesce(n)`:** Avoids a full shuffle by merging existing partitions locally on the same worker. Can **only decrease** the number of partitions. Highly efficient before writing final outputs to storage (e.g., collapsing 200 partitions into 5 output files).

```python
# Before writing to S3 / ADLS: reduce from 200 shuffle partitions to 4 files WITHOUT shuffle
df_final = df_features.coalesce(4)
# df_final.write.mode("overwrite").parquet("abfss://lakehouse@generalmills.dfs.core.windows.net/curated/")
```

---

## 7. Domain-Specific Big Data Scenarios

### Q16 — [General Mills] Hierarchical SKU Demand Aggregation & Outlier Winsorization [Medium]

**Interview Context:**
Real-world General Mills problem: Aggregating store-level sales up to Division and National levels, while winsorizing demand spikes caused by panic buying.

```python
from pyspark.sql import Window
import pyspark.sql.functions as F

# Compute 1st and 99th percentiles per category across millions of rows
w_cat = Window.partitionBy("category")

df_winsorized = df_sales.join(sample_dim, on="sku_id") \
    .withColumn("p01", F.expr("percentile_approx(sales, 0.01)").over(w_cat)) \
    .withColumn("p99", F.expr("percentile_approx(sales, 0.99)").over(w_cat)) \
    .withColumn(
        "sales_capped",
        F.when(F.col("sales") > F.col("p99"), F.col("p99"))
         .when(F.col("sales") < F.col("p01"), F.col("p01"))
         .otherwise(F.col("sales"))
    )

df_winsorized.select("sku_id", "category", "sales", "sales_capped").show(5)
```

---

### Q17 — [LSEG] Real-Time Tick Resampling & VWAP Calculation from Order Stream [Hard]

**Interview Context:**
LSEG interview challenge:
*"Given a high-frequency stream of financial trades (millisecond granularity), resample the data into 5-minute bars and compute the Volume Weighted Average Price (VWAP) per ticker."*

```python
import pyspark.sql.functions as F

# Sample financial ticks
ticks = [
    ("LSEG.L", "2026-11-13 09:00:01.120", 102.50, 500),
    ("LSEG.L", "2026-11-13 09:02:15.450", 102.60, 1000),
    ("LSEG.L", "2026-11-13 09:06:30.010", 102.80, 2000),  # In next 5-min bucket
    ("GIS", "2026-11-13 09:01:00.000", 68.20, 300),
]
df_ticks = spark.createDataFrame(ticks, ["symbol", "tick_time", "price", "volume"]) \
                .withColumn("ts", F.to_timestamp("tick_time"))

# 5-minute window aggregation using PySpark tumbling window
df_5min_bars = df_ticks \
    .groupBy(
        F.col("symbol"),
        F.window(F.col("ts"), "5 minutes").alias("time_bucket")
    ) \
    .agg(
        F.first("price").alias("open"),
        F.max("price").alias("high"),
        F.min("price").alias("low"),
        F.last("price").alias("close"),
        F.sum("volume").alias("total_volume"),
        (F.sum(F.col("price") * F.col("volume")) / F.sum("volume")).alias("vwap")
    ) \
    .select(
        "symbol",
        F.col("time_bucket.start").alias("bar_start"),
        F.col("time_bucket.end").alias("bar_end"),
        "open", "high", "low", "close", "total_volume",
        F.round("vwap", 2).alias("vwap")
    )

df_5min_bars.show(truncate=False)
```

---

### Q18 — [LSEG] Wash-Trading Anomaly Detection via PySpark Self-Join [Hard]

**Interview Context:**
Market surveillance compliance test:
*"Wash-trading occurs when the same financial institution or trader buys and sells the same financial asset within a 30-second window at essentially the same price to artificially inflate volume. Detect wash trades in a 500-million row trade log."*

```python
import pyspark.sql.functions as F

trade_records = [
    ("T1", "Acct_101", "VOD.L", "BUY", 100.0, 5000, "2026-11-13 10:00:00"),
    ("T2", "Acct_101", "VOD.L", "SELL", 100.0, 5000, "2026-11-13 10:00:15"),  # Wash trade with T1! (15s diff)
    ("T3", "Acct_202", "VOD.L", "BUY", 100.0, 1000, "2026-11-13 10:00:10"),
    ("T4", "Acct_101", "VOD.L", "SELL", 100.0, 5000, "2026-11-13 10:05:00"),  # > 30s diff (Normal trade)
]
df_trades = spark.createDataFrame(trade_records, ["trade_id", "trader_id", "symbol", "side", "price", "qty", "ts_str"]) \
                 .withColumn("ts", F.to_timestamp("ts_str"))

# Self-join on matching trader, symbol, and opposite side
df_buys = df_trades.filter(F.col("side") == "BUY").alias("b")
df_sells = df_trades.filter(F.col("side") == "SELL").alias("s")

df_wash_trades = df_buys.join(
    df_sells,
    (F.col("b.trader_id") == F.col("s.trader_id")) &
    (F.col("b.symbol") == F.col("s.symbol")) &
    (F.col("b.price") == F.col("s.price")) &
    (F.col("b.qty") == F.col("s.qty")) &
    # Time window constraint: within 30 seconds
    (F.abs(F.col("b.ts").cast("long") - F.col("s.ts").cast("long")) <= 30)
).select(
    F.col("b.trader_id"),
    F.col("b.symbol"),
    F.col("b.trade_id").alias("buy_trade_id"),
    F.col("s.trade_id").alias("sell_trade_id"),
    F.col("b.price"),
    F.col("b.qty"),
    F.abs(F.col("b.ts").cast("long") - F.col("s.ts").cast("long")).alias("time_difference_sec")
)

print("Detected Wash Trading Violations:")
df_wash_trades.show()
```

---

## 8. Quick Reference: PySpark Optimization Checklist

| Problem | Root Cause | Production Remedy |
|---|---|---|
| **Straggling Tasks** | Data Skew (hot keys) | Apply **Key Salting** or enable `spark.sql.adaptive.skewJoin.enabled`. |
| **Small File Problem** | Thousands of tiny shuffle partitions | Run `.coalesce(N)` before write or execute Delta Lake `OPTIMIZE`. |
| **Driver OOM** | `df.collect()` or oversized broadcast | Use `df.take(100)` or write partitioned outputs directly to cloud storage. |
| **Executor OOM** | Insufficient off-heap memory | Increase `spark.executor.memoryOverhead` to 15–20% of executor memory. |
| **Slow Python UDFs** | Row-by-row Py4J serialization | Rewrite using **`pandas_udf`** with Apache Arrow vectorized batching. |
| **Shuffle Spills to Disk** | Partitions larger than executor buffer | Increase `spark.sql.shuffle.partitions` (e.g., from 200 to 2000). |

---

*Companion Modules:*
* [00_master_guide.md](file:///D:/Int/General_Mills/interview_prep/00_master_guide.md) — Master prep plan, company profiles, and salary benchmarks.
* [01_pandas_numpy_operations.md](file:///D:/Int/General_Mills/interview_prep/01_pandas_numpy_operations.md) — Pandas & NumPy live coding questions.
* [02_sql_database_operations.md](file:///D:/Int/General_Mills/interview_prep/02_sql_database_operations.md) — Advanced SQL & database operations.
* [06_domain_case_studies.md](file:///D:/Int/General_Mills/interview_prep/06_domain_case_studies.md) — FMCG & Financial Services business case studies.
* [08_pythonic_programming_fundamentals.md](file:///D:/Int/General_Mills/interview_prep/08_pythonic_programming_fundamentals.md) — Advanced Python internals, dunder methods, and memory management.
