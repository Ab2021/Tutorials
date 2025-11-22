# Lab 11: Kafka Sink

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
sink = KafkaSink.builder()     .set_bootstrap_servers("localhost:9092")     .set_record_serializer(...)     .build()
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

sink = KafkaSink.builder()     .set_bootstrap_servers("localhost:9092")     .set_record_serializer(
        KafkaRecordSerializationSchema.builder()
            .set_topic("output-topic")
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
    )     .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)     .build()

ds.sink_to(sink)
```
</details>
