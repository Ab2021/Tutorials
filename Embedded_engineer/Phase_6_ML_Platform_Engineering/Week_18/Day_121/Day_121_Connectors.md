# Day 121: Universal Adapter: Connectors
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 18: Ray Data & Pipelines

---

> **🎯 Focus Area:** Data lives in silos. S3 buckets, SQL databases, HDFS clusters. **Ray Data Connectors** provide high-performance, parallel ingestion from these sources.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Read** partitioned Parquet files from S3.
2.  **Execute** a parallel `read_sql` query against a database.
3.  **Ingest** a folder of images into a dataset of Tensors.
4.  **Configure** per-task parallelism to avoid overwhelming the source DB.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install "ray[data]" s3fs sqlite3`.

---

## 📖 Theoretical Foundation

### 1. The File System Interface
Ray uses `pyarrow.fs`.
*   S3: `s3://bucket/path`
*   HDFS: `hdfs://namenode:port/path`
*   **Parallelism:** Driver lists files. Ray distributes file paths to workers. Workers read chunks.

### 2. Database partitioning
To read a 100GB Table from SQL:
*   Ray needs a `partition_column` (e.g., `id`).
*   Ray issues queries: `SELECT * FROM table WHERE id < 1000`, `WHERE id >= 1000 AND id < 2000`, etc.
*   Runs queries in parallel.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: S3 and Images

#### 📁 `src/02_connectors.py`
```python
import ray

# 1. Read S3 (Public Bucket)
# Ray detects .parquet extension and uses ParquetDatasource
ds_parquet = ray.data.read_parquet(
    "s3://anonymous@air-example-data/uria/100m_rows_parquet/",
    files_per_tasks=2 # Control granularity
)
print("S3 Columns:", ds_parquet.schema().names)

# 2. Read Images (Public Bucket)
# Returns a dataset where column 'image' is numpy ndarray
ds_images = ray.data.read_images("s3://anonymous@air-example-data/cifar-10/images")
print("Image Schema:", ds_images.schema())
```

### 👨‍💻 Core Implementation: Parallel SQL

```python
import sqlite3

# Create dummy DB
conn = sqlite3.connect("test.db")
conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER, name TEXT)")
for i in range(100):
    conn.execute(f"INSERT INTO users VALUES ({i}, 'User_{i}')")
conn.commit()
conn.close()

# Read in parallel
# Note: SQLite concurrency is bad, better with Postgres.
# But logic holds.
ds_sql = ray.data.read_sql(
    sql="SELECT * FROM users",
    connection_factory=lambda: sqlite3.connect("test.db"),
    parallelism=4
)

print(ds_sql.take(5))
```

---

## 🔬 Lab Exercise: "Small File Problem"

### Task
Performance Killer.
1.  Read a bucket with 1,000,000 small (1KB) JSON files.
2.  `ray.data.read_json("s3://...")`.
3.  **Observation:** Listing files takes forever. Reading creates 1M tasks (Overhead explodes).
4.  **Fix:** `ray.data.read_json(..., meta_provider=FastFileMetadataProvider())`. Or pre-generate a manifest.csv listing the files and feed that to Ray.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Credentials:** Ray picks up `~/.aws/credentials`. On K8s, use IRSA (IAM Roles for Service Accounts) so pods inherit permissions.
2.  **Concurrency:** `parallelism=N` controls how many read tasks launch. Don't DDoS your production Postgres DB.
3.  **Pushdown:** Ray Data (currently) does limited predicate pushdown to Parquet. It often reads the column chunk and filters in memory.

### API Summary
```python
ray.data.read_parquet()
ray.data.read_images()
ray.data.read_sql()
```

---

**Day 121 Complete** ✅

*Next: Day 122 - Transformations - Modeling the ETL pipeline.*
