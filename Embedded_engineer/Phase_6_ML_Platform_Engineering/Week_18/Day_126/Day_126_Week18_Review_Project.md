# Day 126: Week 18 Review & Project - The Petabyte Sorter
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 18: Ray Data & Pipelines

---

> **🎯 Focus Area:** We have built the pipes. Now we flow the water. We will build a complete ETL pipeline that ingests raw logs, enriches them with AI-based classification, and reduces them for analytics.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Construct** an end-to-end Ray Data pipeline (Read -> Enrich -> Shuffle -> Write).
2.  **Use** Actor Pool Strategy to amortize the cost of loading the Enrichment Model.
3.  **Verify** the output partitions in the destination folder.

---

## 📚 Week 18 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 120 | Ray Data Arch | "Streaming execution hides I/O latency." |
| 121 | Connectors | "I can read S3 and SQL with one API." |
| 122 | Transforms | "ActorPool keeps my heavy model loaded." |
| 123 | Shuffle | "Global sort is expensive. Avoid if possible." |
| 124 | Custom Sources | "I can parse my proprietary binary format." |
| 125 | Tuning | "Limiting CPU tasks prevents OOM." |

---

## 🏗️ Final Project: "LogAnalyzer"

### Scenario
We have raw server logs.
1.  **Parse:** Extract `IP` and `Message`.
2.  **Enrich:** Use a ML model to classify `Message` sentiment (Error/Info). use GeoIP for `IP`.
3.  **Group:** Count Sentiment per Country.
4.  **Output:** Parquet report.

### Step 1: Data Generation

```python
# project/gen_logs.py
import csv
import random

ips = ["1.1.1.1", "2.2.2.2", "8.8.8.8"]
msgs = ["Server started", "Kernel panic", "User logged in", "Segfault"]

with open("logs.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["raw_log"])
    for _ in range(10000):
        writer.writerow([f"{random.choice(ips)} - {random.choice(msgs)}"])
```

### Step 2: The ETL Pipeline

#### 📁 `project/etl.py`
```python
import ray
import pandas as pd

# 1. Read
# Ray infers CSV header
ds = ray.data.read_csv("logs.csv")

# 2. Parse (Map)
def parse_log(row):
    # row = {"raw_log": "1.1.1.1 - Msg"}
    parts = row["raw_log"].split(" - ")
    return {"ip": parts[0], "msg": parts[1]}

ds_parsed = ds.map(parse_log)

# 3. Enrich (Actor Map)
# Simulate GeoIP DB and Sentiment Model
class Enricher:
    def __init__(self):
        print("Loading GeoDB...")
        self.geo = {"1.1.1.1": "US", "2.2.2.2": "FR", "8.8.8.8": "US"}
    
    def __call__(self, batch):
        # batch is Dict[str, np.array]
        ips = batch["ip"]
        msgs = batch["msg"]
        
        countries = [self.geo.get(ip, "UNKNOWN") for ip in ips]
        sentiments = ["BAD" if "panic" in m or "Segfault" in m else "GOOD" for m in msgs]
        
        return {"country": countries, "sentiment": sentiments}

ds_enriched = ds_parsed.map_batches(
    Enricher,
    compute=ray.data.ActorPoolStrategy(size=2),
    batch_size=128
)

# 4. Group & Agg (Shuffle)
# We want Count of (Country, Sentiment)
# Ray Data groupby supports single key. For composite, create tuple key?
# Easier: Group by Country, then map_groups to count sentiments locally.
grouped = ds_enriched.groupby("country")

def count_sentiments(group):
    # This runs on the reducer node
    # group is a Dict[str, np.array]
    df = pd.DataFrame(group)
    counts = df["sentiment"].value_counts()
    return {"sentiment_counts": str(counts.to_dict())}

ds_report = grouped.map_groups(count_sentiments)

# 5. Write
ds_report.write_parquet("output_report")
```

---

## 🔬 Lab Exercise: "Repartitioning"

### Task
Optimize Write.
1.  Run the pipeline.
2.  Check `output_report` folder.
3.  If you see 1 file, it means `groupby` reduced everything to 1 block (unlikely for 10k rows, but possible if parallelism low).
4.  If you see 200 files (1 per CPU), it might be too fragmeted.
5.  Use `ds_report.repartition(10).write_parquet(...)`.
6.  **Verify:** Exactly 10 files created.

---

## 📝 Success Criteria
1.  **Correctness:** US should have BAD and GOOD counts.
2.  **Resources:** 2 `Enricher` actors are created and reused.
3.  **Output:** Parquet files are readable by Pandas.

---

**Week 18 Complete** ✅
**Phase 6C Complete** ✅

*Next Phase: Phase 6D - MLOps & Production Systems.*
