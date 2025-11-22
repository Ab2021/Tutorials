# Lab 10: Redpanda Connect (Benthos)

## Difficulty
🟢 Easy

## Estimated Time
45 mins

## Learning Objectives
-   Use Redpanda Connect (formerly Benthos) for simple ETL.
-   Ingest data from an HTTP API into a topic.

## Problem Statement
Create a Benthos config `pipeline.yaml` that:
1.  Polls `https://randomuser.me/api/` every 1 second.
2.  Writes the JSON to topic `random-users`.
Run it using `rpk connect run`.

## Starter Code
```yaml
input:
  http_client:
    url: https://randomuser.me/api/
    verb: GET

output:
  kafka_franz:
    topic: random-users
    seed_brokers: ["localhost:9092"]
```

## Hints
<details>
<summary>Hint 1</summary>
Redpanda Connect is bundled in `rpk`. Use `rpk connect run pipeline.yaml`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### pipeline.yaml
```yaml
input:
  generate:
    interval: 1s
    mapping: |
      root = http_get("https://randomuser.me/api/")

output:
  kafka_franz:
    seed_brokers: ["localhost:9092"]
    topic: random-users
```

### Run
```bash
rpk connect run pipeline.yaml
```
</details>
