import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase2_Stream_Processing_Flink\Week3_Flink_Fundamentals\labs"

labs_content = {
    "lab_01.md": """# Lab 01: Local Flink Cluster

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Start a local Flink cluster.
-   Access the Flink Web UI.
-   Submit a job via CLI.

## Problem Statement
1.  Download Flink (or use Docker).
2.  Start the cluster (`start-cluster.sh`).
3.  Access the Dashboard at `localhost:8081`.
4.  Run the built-in `WordCount` example.

## Starter Code
```bash
# Docker command
docker run -d -p 8081:8081 flink:latest jobmanager
docker run -d flink:latest taskmanager
```

## Hints
<details>
<summary>Hint 1</summary>
If using Docker Compose, you need a JobManager and a TaskManager service.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Docker Compose
```yaml
version: "2.2"
services:
  jobmanager:
    image: flink:latest
    ports:
      - "8081:8081"
    command: jobmanager
    environment:
      - |
        FLINK_PROPERTIES=
        jobmanager.rpc.address: jobmanager

  taskmanager:
    image: flink:latest
    depends_on:
      - jobmanager
    command: taskmanager
    scale: 1
    environment:
      - |
        FLINK_PROPERTIES=
        jobmanager.rpc.address: jobmanager
        taskmanager.numberOfTaskSlots: 2
```

### Run Example
```bash
docker exec -it <jobmanager_container> flink run examples/streaming/WordCount.jar
```
</details>
""",
    "lab_02.md": """# Lab 02: DataStream WordCount

## Difficulty
🟢 Easy

## Estimated Time
45 mins

## Learning Objectives
-   Write a basic Flink DataStream application in Java/Python.
-   Use `socketTextStream`.
-   Use `flatMap` and `keyBy`.

## Problem Statement
Write a Flink job that reads text from a socket (port 9999), splits lines into words, counts them, and prints the result.

## Starter Code
```python
from pyflink.datastream import StreamExecutionEnvironment

env = StreamExecutionEnvironment.get_execution_environment()
ds = env.socket_text_stream("localhost", 9999)

# ds.flat_map(...).key_by(...).sum(...)

env.execute("WordCount")
```

## Hints
<details>
<summary>Hint 1</summary>
You need `nc -lk 9999` running in a terminal to provide input.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment

def split(line):
    return line.split()

def run():
    env = StreamExecutionEnvironment.get_execution_environment()
    
    # Source
    text = env.socket_text_stream("host.docker.internal", 9999)
    
    # Transformation
    counts = text \
        .flat_map(lambda x: x.split(), output_type=Types.STRING()) \
        .map(lambda i: (i, 1), output_type=Types.TUPLE([Types.STRING(), Types.INT()])) \
        .key_by(lambda i: i[0]) \
        .sum(1)
        
    # Sink
    counts.print()
    
    env.execute("Socket WordCount")

if __name__ == '__main__':
    run()
```
</details>
""",
    "lab_03.md": """# Lab 03: Kafka Source

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Connect Flink to Kafka.
-   Use `KafkaSource`.

## Problem Statement
1.  Start Kafka.
2.  Create a topic `input-topic`.
3.  Write a Flink job that reads from `input-topic` and prints the messages.

## Starter Code
```python
source = KafkaSource.builder() \
    .set_bootstrap_servers("localhost:9092") \
    .set_topics("input-topic") \
    .set_group_id("my-group") \
    .set_starting_offsets(OffsetsInitializer.earliest()) \
    .set_value_only_deserializer(SimpleStringSchema()) \
    .build()
```

## Hints
<details>
<summary>Hint 1</summary>
You need the `flink-connector-kafka` JAR dependency. In PyFlink, you add it via `env.add_jars()`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, OffsetsInitializer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common import WatermarkStrategy

def run():
    env = StreamExecutionEnvironment.get_execution_environment()
    # Add JAR (ensure path is correct)
    env.add_jars("file:///path/to/flink-sql-connector-kafka-3.0.0-1.18.jar")

    source = KafkaSource.builder() \
        .set_bootstrap_servers("localhost:9092") \
        .set_topics("input-topic") \
        .set_group_id("my-group") \
        .set_starting_offsets(OffsetsInitializer.earliest()) \
        .set_value_only_deserializer(SimpleStringSchema()) \
        .build()

    ds = env.from_source(source, WatermarkStrategy.no_watermarks(), "Kafka Source")
    ds.print()
    
    env.execute("Kafka Reader")
```
</details>
""",
    "lab_04.md": """# Lab 04: Filter & Map

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Use basic transformations.
-   Parse JSON strings.

## Problem Statement
Read a stream of JSON strings `{"user": "A", "age": 25}`.
1.  Parse JSON.
2.  Filter out users under 18.
3.  Map to `Name: A`.

## Starter Code
```python
import json
# ds.map(json.loads).filter(...)
```

## Hints
<details>
<summary>Hint 1</summary>
Handle JSON parsing errors gracefully (try/except) or the job will fail.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
import json
from pyflink.common import Types

def parse_and_filter(ds):
    return ds \
        .map(lambda x: json.loads(x), output_type=Types.MAP(Types.STRING(), Types.STRING())) \
        .filter(lambda x: int(x['age']) >= 18) \
        .map(lambda x: f"Name: {x['user']}", output_type=Types.STRING())
```
</details>
""",
    "lab_05.md": """# Lab 05: Tumbling Windows

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Implement a Tumbling Processing Time Window.
-   Aggregate data per window.

## Problem Statement
Count words arriving from a socket, but aggregate them in **10-second tumbling windows**.
Output: `(word, count)` every 10 seconds.

## Starter Code
```python
from pyflink.datastream.window import TumblingProcessingTimeWindows
from pyflink.common import Time

# ds.key_by(...).window(TumblingProcessingTimeWindows.of(Time.seconds(10))).sum(...)
```

## Hints
<details>
<summary>Hint 1</summary>
Processing Time is easier for testing than Event Time (no watermarks needed).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
    ds.key_by(lambda x: x[0]) \
      .window(TumblingProcessingTimeWindows.of(Time.seconds(10))) \
      .sum(1) \
      .print()
```
</details>
""",
    "lab_06.md": """# Lab 06: Sliding Windows

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Implement Sliding Windows.
-   Understand overlap.

## Problem Statement
Calculate the moving average of numbers over the last **1 minute**, updated every **10 seconds**.
Input: Stream of integers.

## Starter Code
```python
from pyflink.datastream.window import SlidingProcessingTimeWindows

# window(SlidingProcessingTimeWindows.of(Time.minutes(1), Time.seconds(10)))
```

## Hints
<details>
<summary>Hint 1</summary>
You might need a custom `AggregateFunction` or `ProcessWindowFunction` to calculate the average (sum/count).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# Using Reduce for simplicity (Sum)
ds.key_by(lambda x: "key") \
  .window(SlidingProcessingTimeWindows.of(Time.minutes(1), Time.seconds(10))) \
  .reduce(lambda a, b: a + b) \
  .print()
```
</details>
""",
    "lab_07.md": """# Lab 07: Event Time & Watermarks

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Assign timestamps and watermarks.
-   Use Event Time windows.

## Problem Statement
Read a stream of CSVs: `timestamp, value`.
1.  Assign timestamps from the first column.
2.  Generate watermarks (BoundedOutOfOrderness = 5 seconds).
3.  Window by Event Time (10s tumbling).

## Starter Code
```python
class MyTimestampAssigner(TimestampAssigner):
    def extract_timestamp(self, value, record_timestamp):
        return int(value[0])

watermark_strategy = WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(5)) \
    .with_timestamp_assigner(MyTimestampAssigner())
```

## Hints
<details>
<summary>Hint 1</summary>
Timestamps must be in milliseconds.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.common import WatermarkStrategy, Duration
from pyflink.datastream.window import TumblingEventTimeWindows

# Define Strategy
watermark_strategy = WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(5)) \
    .with_timestamp_assigner(lambda event, timestamp: int(event.split(',')[0]))

# Apply
ds = env.from_collection([...]) \
    .assign_timestamps_and_watermarks(watermark_strategy) \
    .key_by(...) \
    .window(TumblingEventTimeWindows.of(Time.seconds(10))) \
    .sum(...)
```
</details>
""",
    "lab_08.md": """# Lab 08: Session Windows

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Implement Session Windows.
-   Understand the "Gap".

## Problem Statement
Group user clicks into sessions. A session ends if the user is idle for **5 seconds**.
Count clicks per session.

## Starter Code
```python
from pyflink.datastream.window import ProcessingTimeSessionWindows

# window(ProcessingTimeSessionWindows.with_gap(Time.seconds(5)))
```

## Hints
<details>
<summary>Hint 1</summary>
Session windows merge. The key is the user ID.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
ds.key_by(lambda x: x['user_id']) \
  .window(ProcessingTimeSessionWindows.with_gap(Time.seconds(5))) \
  .sum('clicks') \
  .print()
```
</details>
""",
    "lab_09.md": """# Lab 09: ProcessWindowFunction

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Use `ProcessWindowFunction` to access window metadata (start/end time).

## Problem Statement
For a 10s tumbling window, output a string: `"Window [Start-End]: Sum = X"`.
You need `ProcessWindowFunction` to get the `Context`.

## Starter Code
```python
class MyProcessWindowFunction(ProcessWindowFunction):
    def process(self, key, context, elements):
        # context.window().start
        # context.window().end
        pass
```

## Hints
<details>
<summary>Hint 1</summary>
In PyFlink, this is often a simple python function if using the functional API, or a class inheriting from `ProcessWindowFunction`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream.functions import ProcessWindowFunction

class WindowLogger(ProcessWindowFunction):
    def process(self, key, context, elements):
        total = sum([e[1] for e in elements])
        start = context.window().start
        end = context.window().end
        yield f"Window [{start}-{end}]: Sum = {total}"

ds.key_by(...) \
  .window(...) \
  .process(WindowLogger()) \
  .print()
```
</details>
""",
    "lab_10.md": """# Lab 10: Side Outputs (Late Data)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Handle late data using Side Outputs.

## Problem Statement
1.  Use Event Time window (10s).
2.  Set Allowed Lateness to 0.
3.  Send late data to a Side Output tag `late-data`.
4.  Print the main stream and the side output stream separately.

## Starter Code
```python
late_tag = OutputTag("late-data")

result = ds.window(...) \
    .side_output_late_data(late_tag) \
    .sum(...)

late_stream = result.get_side_output(late_tag)
```

## Hints
<details>
<summary>Hint 1</summary>
You need to simulate late data by sending a timestamp older than the current watermark.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream import OutputTag

late_tag = OutputTag("late-data", Types.TUPLE([Types.STRING(), Types.INT()]))

main_stream = ds.window(...) \
    .side_output_late_data(late_tag) \
    .sum(1)

main_stream.print()
main_stream.get_side_output(late_tag).print_to_err() # Print late data to stderr
```
</details>
""",
    "lab_11.md": """# Lab 11: Kafka Sink

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Write data back to Kafka.
-   Configure `KafkaSink`.

## Problem Statement
Read from `input-topic`, transform (uppercase), and write to `output-topic` with `at-least-once` semantics.

## Starter Code
```python
sink = KafkaSink.builder() \
    .set_bootstrap_servers("localhost:9092") \
    .set_record_serializer(...) \
    .build()
```

## Hints
<details>
<summary>Hint 1</summary>
Use `KafkaRecordSerializationSchema`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream.connectors.kafka import KafkaSink, KafkaRecordSerializationSchema

sink = KafkaSink.builder() \
    .set_bootstrap_servers("localhost:9092") \
    .set_record_serializer(
        KafkaRecordSerializationSchema.builder()
            .set_topic("output-topic")
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
    ) \
    .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE) \
    .build()

ds.sink_to(sink)
```
</details>
""",
    "lab_12.md": """# Lab 12: Rich Functions (Lifecycle)

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Use `open()` and `close()` in a RichMapFunction.
-   Simulate a database connection.

## Problem Statement
Create a `RichMapFunction` that:
1.  In `open()`, prints "Opening DB Connection".
2.  In `map()`, appends " processed" to the input.
3.  In `close()`, prints "Closing DB Connection".

## Starter Code
```python
class MyMapper(RichMapFunction):
    def open(self, ctx):
        pass
    def map(self, value):
        pass
```

## Hints
<details>
<summary>Hint 1</summary>
`open()` is called once per parallel task, not per element.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream.functions import RichMapFunction

class DBMapper(RichMapFunction):
    def open(self, runtime_context):
        print("Opening DB Connection...")

    def map(self, value):
        return value + " processed"

    def close(self):
        print("Closing DB Connection...")

ds.map(DBMapper()).print()
```
</details>
""",
    "lab_13.md": """# Lab 13: CoProcessFunction (Connect)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Connect two streams.
-   Share state between streams.

## Problem Statement
Stream A: `ControlStream` (Switch: ON/OFF).
Stream B: `DataStream` (Words).
Requirement: Only print words from Stream B if the Switch (Stream A) is ON.

## Starter Code
```python
connected = control_stream.connect(data_stream)
connected.process(MyCoProcessFunction())
```

## Hints
<details>
<summary>Hint 1</summary>
You need a `ValueState` to store the current switch status.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream.functions import CoProcessFunction

class Switch(CoProcessFunction):
    def process_element1(self, value, ctx, out):
        # Control Stream
        self.enabled = (value == "ON")

    def process_element2(self, value, ctx, out):
        # Data Stream
        if getattr(self, 'enabled', False):
            out.collect(value)

# Note: In real Flink, you'd use Flink State (ValueState) to persist 'enabled' across restarts.
# For this simple lab, a python attribute works if no failure.
```
</details>
""",
    "lab_14.md": """# Lab 14: Async I/O

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Use Async I/O to call external APIs without blocking.

## Problem Statement
*Note: PyFlink support for Async I/O is limited compared to Java. We will simulate the concept or use a ThreadPool.*
Simulate an external API call that takes 1s. Use `map` vs `async_wait` (conceptual).

## Starter Code
```python
# PyFlink Async I/O is complex. 
# We will focus on the concept:
# OrderedWait vs UnorderedWait
```

## Hints
<details>
<summary>Hint 1</summary>
If you block in a `map` function, you block the checkpointing barrier too!
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

*Conceptual Solution (Java syntax is standard for Async I/O)*:
```java
AsyncDataStream.unorderedWait(
    stream,
    new AsyncDatabaseRequest(),
    1000, TimeUnit.MILLISECONDS,
    100);
```
In PyFlink, ensure you use a thread pool inside your map function if you must do blocking I/O, but true Async I/O requires the Async operator.
</details>
""",
    "lab_15.md": """# Lab 15: Flink SQL Basics

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Use the Table API / SQL.
-   Convert DataStream to Table.

## Problem Statement
1.  Create a DataStream of `(name, age)`.
2.  Convert to Table.
3.  Run SQL: `SELECT name FROM table WHERE age > 18`.
4.  Print result.

## Starter Code
```python
t_env = StreamTableEnvironment.create(env)
table = t_env.from_data_stream(ds)
result = t_env.sql_query("...")
```

## Hints
<details>
<summary>Hint 1</summary>
You need `flink-table` dependencies.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.table import StreamTableEnvironment, EnvironmentSettings

def run():
    env = StreamExecutionEnvironment.get_execution_environment()
    t_env = StreamTableEnvironment.create(env)

    ds = env.from_collection([("Alice", 25), ("Bob", 10)], 
                             type_info=Types.ROW([Types.STRING(), Types.INT()]))

    table = t_env.from_data_stream(ds, ["name", "age"])
    
    result = t_env.sql_query("SELECT name FROM %s WHERE age > 18" % table)
    
    result.execute().print()

if __name__ == '__main__':
    run()
```
</details>
"""
}

print("🚀 Upgrading Week 3 Labs with Comprehensive Solutions...")

for filename, content in labs_content.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Upgraded {filename}")

print("✅ Week 3 Labs Upgrade Complete!")
