# Day 151: The Skew Slayer: Feature Stores & Feast
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 22: Data & Model Quality

---

> **🎯 Focus Area:** You calculate "Average User Spend" in Python for Training. But in Production, a Java service calculates it slightly differently. Result: **Training-Serving Skew**. The **Feature Store** ensures the exact same logic serves both.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** Feature Views and Entities using the `feast` Python SDK.
2.  **Materialize** features from an Offline Store (Parquet/File) to an Online Store (Redis/Sqlite).
3.  **Retrieve** Point-in-Time correct features for historical training (Time Travel).
4.  **Detect** feature leakage using timestamp-aware joins.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install feast`.

---

## 📖 Theoretical Foundation

### 1. The Skew Problem
*   **Logic Skew:** Defining `age = now - dob` in SQL (Train) vs `age = floor(days / 365)` in App (Serve).
*   **Timing Skew:** Using "Current Account Balance" to train on data from last year. You must use "Balance AS OF the transaction time".

### 2. Feast Architecture
*   **Registry:** Protocol Buffers definition of features. (Git).
*   **Offline Store:** Data Warehouse (BigQuery/Snowflake/Parquet). Source of truth.
*   **Online Store:** Low latency KV Store (Redis/DynamoDB). Latest values only.
*   **SDK:** `get_historical_features()` (Train) and `get_online_features()` (Serve).

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Feast Repository Setup

```bash
feast init my_feature_repo
cd my_feature_repo
```

#### 📁 `my_feature_repo/feature_store.yaml`
```yaml
project: my_project
registry: data/registry.db
provider: local
online_store:
  type: sqlite
  path: data/online_store.db
offline_store:
  type: file
entity_key_serialization_version: 2
```

### 👨‍💻 Core Implementation: Defining Features

#### 📁 `my_feature_repo/features.py`
```python
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# 1. Define Entity (Primary Key)
driver = Entity(name="driver", join_keys=["driver_id"])

# 2. Define Source (Parquet File)
driver_stats_source = FileSource(
    name="driver_stats_source",
    path="data/driver_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# 3. Define Feature View (Group of features)
driver_stats_view = FeatureView(
    name="driver_stats",
    entities=[driver],
    ttl=timedelta(days=1), # Look back 1 day max
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Int64),
    ],
    online=True,
    source=driver_stats_source,
    tags={"team": "driver_performance"},
)
```

### 👨‍💻 Core Implementation: Training (Historical Retrieval)

This performs the "Point-in-time Join" automatically.

#### 📁 `src/train_with_feast.py`
```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="my_feature_repo")

# 1. Entity DataFrame (The "Trigger" events)
# We want to know features for these specific drivers at these specific times
entity_df = pd.DataFrame.from_dict({
    "driver_id": [1001, 1002, 1003],
    "event_timestamp": [
        pd.Timestamp("2023-01-01 10:00:00"),
        pd.Timestamp("2023-01-01 10:00:00"),
        pd.Timestamp("2023-01-01 10:00:00"),
    ]
})

# 2. Fetch Features
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "driver_stats:conv_rate",
        "driver_stats:avg_daily_trips"
    ]
).to_df()

print(training_df.head())
# Result: driver_id | event_timestamp | conv_rate | avg_daily_trips
```

### 👨‍💻 Core Implementation: Serving (Online Retrieval)

#### 📁 `src/serve_with_feast.py`
```python
from feast import FeatureStore

store = FeatureStore(repo_path="my_feature_repo")

# 1. Materialize (Load latest data into Online Store)
# Usually run by a Cron Job
from datetime import datetime
store.materialize_incremental(end_date=datetime.now())

# 2. Fetch Vector (Low Latency)
def predict_handler(driver_id):
    features = store.get_online_features(
        features=[
            "driver_stats:conv_rate",
            "driver_stats:avg_daily_trips"
        ],
        entity_rows=[{"driver_id": driver_id}]
    ).to_dict()
    
    print(features)
    # {'driver_id': [1001], 'conv_rate': [0.5], ...}

predict_handler(1001)
```

---

## 🔬 Lab Exercise: "Time Travel"

### Task
Verify Point-in-Time Correctness.
1.  **Data:**
    *   Jan 1: Driver 1 has `rate=0.5`.
    *   Jan 2: Driver 1 has `rate=0.8`.
2.  **Query:** `get_historical_features` for Jan 1 12:00.
3.  **Result:** Should return 0.5. (Even though current state is 0.8).
4.  **Why?** If you train using 0.8 (Leakage), your model thinks the driver was better than they actually were at the time of the event.

---

## 📖 Advanced Theory: On-Demand Transforms
What if you need `rate * 100`?
Feast supports **On-Demand Feature Views** (Pandas transforms) that run at request time.
```python
@on_demand_feature_view(
    sources=[driver_stats_view],
    schema=[Field(name="rate_percent", dtype=Float32)]
)
def calculate_percent(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["rate_percent"] = inputs["conv_rate"] * 100
    return df
```

---

## 📝 Daily Summary

### Key Takeaways
1.  **Decoupling:** Data Engineers own the SQL/Parquet. ML Engineers own the Feature Definitions. Production engineers own the Redis.
2.  **Consistency:** The same `driver_stats:conv_rate` string is used in Training code and Serving code. Zero possibility of Logic Skew.
3.  **Latency:** Offline retrieval is slow (Batch). Online retrieval is fast (<10ms). Materialization bridges the gap.

### API Summary
```python
store.get_historical_features(entity_df, features)
store.get_online_features(features, entity_rows)
```

---

**Day 151 Complete** ✅

*Next: Day 152 - A/B Testing Infrastructure.*
