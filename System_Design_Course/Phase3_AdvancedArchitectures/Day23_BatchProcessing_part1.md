# Day 23 Deep Dive: Apache Spark Internals

## 1. Architecture
*   **Driver:** The "Main" program. Creates SparkContext. Converts code to DAG.
*   **Cluster Manager:** (YARN/K8s). Allocates resources.
*   **Executors:** Worker processes. Run tasks. Store data in RAM.

## 2. Transformations vs Actions
*   **Lazy Evaluation:** Spark does nothing until an Action is called.
*   **Transformation:** `map()`, `filter()`, `join()`. Returns new RDD.
*   **Action:** `count()`, `collect()`, `saveAsTextFile()`. Triggers execution.

## 3. Wide vs Narrow Dependencies
*   **Narrow:** Data stays in same partition. (e.g., `map`, `filter`). Fast.
*   **Wide:** Data moves across network (Shuffle). (e.g., `groupByKey`, `join`). Slow.
*   **Optimization:** Minimize Shuffles. Filter early.

## 4. Data Skew
*   **Problem:** One key has 90% of data (e.g., "Null" key). One executor works forever, others idle.
*   **Solution:**
    *   **Salting:** Add random suffix to key (`key_1`, `key_2`). Distributes data.
    *   **Broadcast Join:** If one table is small, send copy to all nodes. Avoid shuffle.

## 5. Code: PySpark Word Count
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("WordCount").getOrCreate()

text_file = spark.read.text("hdfs://path/to/file.txt")
counts = text_file.rdd \
    .flatMap(lambda line: line.value.split(" ")) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a + b)

counts.saveAsTextFile("hdfs://path/to/output")
```
