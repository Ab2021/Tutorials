import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase2_Stream_Processing_Flink\Week5_Advanced_Flink\labs"

labs_content = {
    "lab_01.md": """# Lab 01: Flink SQL Basics

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Create a Table Environment.
-   Execute a simple SQL query.

## Problem Statement
1.  Create a DataStream of `(name, age)`.
2.  Register it as a view `People`.
3.  Run SQL: `SELECT name, age + 1 FROM People`.
4.  Print results.

## Starter Code
```python
t_env.create_temporary_view("People", ds)
result = t_env.sql_query("...")
```

## Hints
<details>
<summary>Hint 1</summary>
Use `StreamTableEnvironment`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.table import StreamTableEnvironment

def run():
    env = StreamExecutionEnvironment.get_execution_environment()
    t_env = StreamTableEnvironment.create(env)

    ds = env.from_collection([("Alice", 25), ("Bob", 30)], 
                             type_info=Types.ROW([Types.STRING(), Types.INT()]))
    
    t_env.create_temporary_view("People", ds, ["name", "age"])
    
    result = t_env.sql_query("SELECT name, age + 1 FROM People")
    result.execute().print()
```
</details>
""",
    "lab_02.md": """# Lab 02: SQL Window Aggregation

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Use `TUMBLE` window in SQL.

## Problem Statement
Input: `(user, amount, rowtime)`.
Query: Calculate sum of amount per user every 10 seconds (Tumbling Window).

## Starter Code
```sql
SELECT user, SUM(amount)
FROM Orders
GROUP BY user, TUMBLE(rowtime, INTERVAL '10' SECOND)
```

## Hints
<details>
<summary>Hint 1</summary>
You need to define a watermark strategy on the table for `rowtime` to work.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# DDL
t_env.execute_sql(\"\"\"
    CREATE TABLE Orders (
        user_name STRING,
        amount INT,
        ts TIMESTAMP(3),
        WATERMARK FOR ts AS ts - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'datagen'
    )
\"\"\")

# Query
t_env.execute_sql(\"\"\"
    SELECT user_name, SUM(amount), TUMBLE_END(ts, INTERVAL '10' SECOND)
    FROM Orders
    GROUP BY user_name, TUMBLE(ts, INTERVAL '10' SECOND)
\"\"\").print()
```
</details>
""",
    "lab_03.md": """# Lab 03: SQL Join (Interval)

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Perform an Interval Join in SQL.

## Problem Statement
Join `Orders` and `Shipments` where shipment happens within 1 hour of order.
`Orders(id, ts)`, `Shipments(order_id, ts)`.

## Starter Code
```sql
SELECT *
FROM Orders o, Shipments s
WHERE o.id = s.order_id
AND s.ts BETWEEN o.ts AND o.ts + INTERVAL '1' HOUR
```

## Hints
<details>
<summary>Hint 1</summary>
This is a standard SQL join with time bounds.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# Assuming tables exist
sql = \"\"\"
SELECT o.id, o.ts, s.ts
FROM Orders o
JOIN Shipments s ON o.id = s.order_id
WHERE s.ts BETWEEN o.ts AND o.ts + INTERVAL '1' HOUR
\"\"\"
t_env.execute_sql(sql).print()
```
</details>
""",
    "lab_04.md": """# Lab 04: CEP Pattern (Next)

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Define a strict sequence pattern.

## Problem Statement
Detect pattern: `Start` event followed immediately by `End` event for the same ID.
Input: `(id, type)`.

## Starter Code
```python
pattern = Pattern.begin("start").where(...) \
    .next("end").where(...)
```

## Hints
<details>
<summary>Hint 1</summary>
Use `SimpleCondition`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
pattern = Pattern.begin("start").where(
    SimpleCondition(lambda x: x['type'] == 'Start')
).next("end").where(
    SimpleCondition(lambda x: x['type'] == 'End')
)

CEP.pattern(ds.key_by(lambda x: x['id']), pattern) \
   .select(lambda map: f"Matched: {map['start'][0]} -> {map['end'][0]}") \
   .print()
```
</details>
""",
    "lab_05.md": """# Lab 05: CEP Pattern (FollowedBy)

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Define a relaxed sequence pattern.

## Problem Statement
Detect: `Login` followed by `Purchase` within 1 hour. (Other events can happen in between).

## Starter Code
```python
pattern = Pattern.begin("login")...followed_by("purchase")...within(Time.hours(1))
```

## Hints
<details>
<summary>Hint 1</summary>
Don't forget `.within()`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
pattern = Pattern.begin("login").where(
    SimpleCondition(lambda x: x['type'] == 'Login')
).followed_by("purchase").where(
    SimpleCondition(lambda x: x['type'] == 'Purchase')
).within(Time.hours(1))
```
</details>
""",
    "lab_06.md": """# Lab 06: CEP Looping Pattern

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Use `.times()` or `.oneOrMore()`.

## Problem Statement
Detect: 3 consecutive failed logins.

## Starter Code
```python
pattern = Pattern.begin("fail").where(...).times(3).consecutive()
```

## Hints
<details>
<summary>Hint 1</summary>
`consecutive()` ensures they are adjacent.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
pattern = Pattern.begin("fail").where(
    SimpleCondition(lambda x: x['type'] == 'Fail')
).times(3).consecutive()
```
</details>
""",
    "lab_07.md": """# Lab 07: Kafka SQL Connector

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Define a Kafka table using DDL.

## Problem Statement
Create a table `KafkaTable` backed by topic `input`. Read from it using SQL.

## Starter Code
```sql
CREATE TABLE KafkaTable (
  `user` STRING,
  `age` INT
) WITH (
  'connector' = 'kafka',
  'topic' = 'input',
  'properties.bootstrap.servers' = 'localhost:9092',
  'format' = 'json'
)
```

## Hints
<details>
<summary>Hint 1</summary>
Ensure you have the kafka-sql-connector JAR.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
t_env.execute_sql(\"\"\"
    CREATE TABLE KafkaTable (
      `user` STRING,
      `age` INT
    ) WITH (
      'connector' = 'kafka',
      'topic' = 'input',
      'properties.bootstrap.servers' = 'localhost:9092',
      'properties.group.id' = 'testGroup',
      'scan.startup.mode' = 'earliest-offset',
      'format' = 'json'
    )
\"\"\")

t_env.execute_sql("SELECT * FROM KafkaTable").print()
```
</details>
""",
    "lab_08.md": """# Lab 08: Upsert Kafka SQL

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Use `upsert-kafka` connector.
-   Write a changelog stream.

## Problem Statement
Read from `Orders`, aggregate `SUM(amount)` per user, and write to `UserStats` (Kafka Upsert topic).

## Starter Code
```sql
CREATE TABLE UserStats (
  user_name STRING PRIMARY KEY NOT ENFORCED,
  total_amount INT
) WITH (
  'connector' = 'upsert-kafka',
  ...
)
```

## Hints
<details>
<summary>Hint 1</summary>
Upsert Kafka requires a Primary Key.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
t_env.execute_sql(\"\"\"
    CREATE TABLE UserStats (
      user_name STRING PRIMARY KEY NOT ENFORCED,
      total_amount INT
    ) WITH (
      'connector' = 'upsert-kafka',
      'topic' = 'user_stats',
      'properties.bootstrap.servers' = 'localhost:9092',
      'key.format' = 'json',
      'value.format' = 'json'
    )
\"\"\")

t_env.execute_sql(\"\"\"
    INSERT INTO UserStats
    SELECT user_name, SUM(amount)
    FROM Orders
    GROUP BY user_name
\"\"\")
```
</details>
""",
    "lab_09.md": """# Lab 09: PyFlink UDF (Scalar)

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Write a Python UDF.
-   Register it in SQL.

## Problem Statement
Write a UDF `to_upper(s)` that converts string to uppercase. Use it in SQL.

## Starter Code
```python
@udf(result_type=Types.STRING())
def to_upper(s):
    return s.upper()
```

## Hints
<details>
<summary>Hint 1</summary>
Register with `t_env.create_temporary_system_function`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.table.udf import udf

@udf(result_type=Types.STRING())
def to_upper(s):
    return s.upper()

t_env.create_temporary_system_function("to_upper", to_upper)

t_env.sql_query("SELECT to_upper(name) FROM People").execute().print()
```
</details>
""",
    "lab_10.md": """# Lab 10: Pandas UDF (Vectorized)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Write a Vectorized UDF for performance.

## Problem Statement
Implement `add_one(series)` using Pandas. It should take a `pd.Series` and return a `pd.Series`.

## Starter Code
```python
@udf(result_type=Types.INT(), func_type="pandas")
def add_one(i):
    return i + 1
```

## Hints
<details>
<summary>Hint 1</summary>
Requires `pandas` and `pyarrow` installed.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
@udf(result_type=Types.INT(), func_type="pandas")
def add_one(i):
    return i + 1

# Usage is same as scalar UDF
```
</details>
""",
    "lab_11.md": """# Lab 11: MiniCluster Test

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Write a JUnit test using MiniCluster.

## Problem Statement
*Java Lab*. Write a JUnit test that starts a `MiniClusterWithClientResource`, submits a job, and verifies output.

## Starter Code
```java
@ClassRule
public static MiniClusterWithClientResource flinkCluster =
    new MiniClusterWithClientResource(...);
```

## Hints
<details>
<summary>Hint 1</summary>
Use `Sink.collect()` (test utility) to capture output.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```java
// Conceptual
@Test
public void testJob() throws Exception {
    StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
    // ... build job ...
    env.execute();
    // Verify results
}
```
</details>
""",
    "lab_12.md": """# Lab 12: TestHarness (ProcessFunction)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Use `KeyedOneInputStreamOperatorTestHarness`.
-   Test time-dependent logic.

## Problem Statement
*Java Lab*. Test a ProcessFunction that sets a timer for 1 minute.
1.  Push element.
2.  Advance time by 1 minute.
3.  Verify output.

## Starter Code
```java
harness.processElement("A", 1000);
harness.setProcessingTime(1000 + 60000);
```

## Hints
<details>
<summary>Hint 1</summary>
Harness allows full control over time.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```java
// Conceptual
OneInputStreamOperatorTestHarness<String, String, String> harness = 
    new KeyedOneInputStreamOperatorTestHarness<>(operator, ...);

harness.open();
harness.processElement("key", 100);
harness.setProcessingTime(60100); // Trigger timer
// Assert output contains alert
```
</details>
""",
    "lab_13.md": """# Lab 13: K8s Deployment (Session)

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Deploy Flink Session Cluster on K8s.

## Problem Statement
Use `kubectl` to deploy a JobManager and TaskManager. Submit a job to it.

## Starter Code
```yaml
# jobmanager-deployment.yaml
# taskmanager-deployment.yaml
```

## Hints
<details>
<summary>Hint 1</summary>
Use the official Flink K8s docs YAMLs.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```bash
kubectl create -f jobmanager-service.yaml
kubectl create -f jobmanager-deployment.yaml
kubectl create -f taskmanager-deployment.yaml

# Forward port
kubectl port-forward service/flink-jobmanager 8081:8081

# Submit
flink run -m localhost:8081 examples/streaming/WordCount.jar
```
</details>
""",
    "lab_14.md": """# Lab 14: K8s Application Mode

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Build a custom Docker image.
-   Deploy in Application Mode.

## Problem Statement
1.  Create Dockerfile: `FROM flink`, `COPY my-job.jar`.
2.  Build image.
3.  Deploy using `flink run-application -t kubernetes-application ...`.

## Starter Code
```bash
flink run-application -t kubernetes-application \
  -Dkubernetes.cluster-id=my-app \
  -Dkubernetes.container.image=my-image:latest \
  local:///opt/flink/usrlib/my-job.jar
```

## Hints
<details>
<summary>Hint 1</summary>
The JAR path must be local to the container (`local:///...`).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Dockerfile
```dockerfile
FROM flink:1.18
RUN mkdir -p /opt/flink/usrlib
COPY target/my-job.jar /opt/flink/usrlib/my-job.jar
```

### Command
```bash
flink run-application \
    --target kubernetes-application \
    -Dkubernetes.cluster-id=my-first-app-cluster \
    -Dkubernetes.container.image=my-custom-flink-image \
    local:///opt/flink/usrlib/my-job.jar
```
</details>
""",
    "lab_15.md": """# Lab 15: Reactive Mode

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Configure Reactive Mode.
-   Scale TaskManagers.

## Problem Statement
1.  Configure `scheduler-mode: reactive`.
2.  Deploy cluster.
3.  Scale TaskManagers from 1 to 2.
4.  Verify job rescales automatically.

## Starter Code
```yaml
jobmanager.scheduler: reactive
```

## Hints
<details>
<summary>Hint 1</summary>
Reactive mode requires Application Mode.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Config
```yaml
jobmanager.scheduler: reactive
```

### Scaling
```bash
kubectl scale deployment/flink-taskmanager --replicas=2
```
Watch the logs. The job should restart with parallelism 2.
</details>
"""
}

print("🚀 Upgrading Week 5 Labs with Comprehensive Solutions...")

for filename, content in labs_content.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Upgraded {filename}")

print("✅ Week 5 Labs Upgrade Complete!")
