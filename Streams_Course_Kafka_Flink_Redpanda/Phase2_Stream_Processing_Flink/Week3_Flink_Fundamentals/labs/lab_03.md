# Lab 03: Kafka Source

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
source = KafkaSource.builder()     .set_bootstrap_servers("localhost:9092")     .set_topics("input-topic")     .set_group_id("my-group")     .set_starting_offsets(OffsetsInitializer.earliest())     .set_value_only_deserializer(SimpleStringSchema())     .build()
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

    source = KafkaSource.builder()         .set_bootstrap_servers("localhost:9092")         .set_topics("input-topic")         .set_group_id("my-group")         .set_starting_offsets(OffsetsInitializer.earliest())         .set_value_only_deserializer(SimpleStringSchema())         .build()

    ds = env.from_source(source, WatermarkStrategy.no_watermarks(), "Kafka Source")
    ds.print()
    
    env.execute("Kafka Reader")
```
</details>
