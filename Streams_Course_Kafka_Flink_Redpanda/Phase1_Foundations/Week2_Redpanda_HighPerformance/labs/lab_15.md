# Lab 15: HTTP Proxy (PandaProxy)

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Use the REST API to produce/consume.

## Problem Statement
Redpanda has a built-in HTTP proxy on port 8082.
1.  Create a topic `http-test`.
2.  POST a message to it using `curl`.
3.  GET messages from it using `curl`.

## Starter Code
```bash
curl -X POST http://localhost:8082/topics/http-test ...
```

## Hints
<details>
<summary>Hint 1</summary>
Content-Type must be `application/vnd.kafka.json.v2+json`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Produce
```bash
curl -X POST "http://localhost:8082/topics/http-test"   -H "Content-Type: application/vnd.kafka.json.v2+json"   -d '{"records":[{"value":"hello http"}]}'
```

### Consume
First, create a consumer:
```bash
curl -X POST "http://localhost:8082/consumers/my-group"   -H "Content-Type: application/vnd.kafka.v2+json"   -d '{"name": "my-consumer", "format": "json", "auto.offset.reset": "earliest"}'
```

Subscribe:
```bash
curl -X POST "http://localhost:8082/consumers/my-group/instances/my-consumer/subscription"   -H "Content-Type: application/vnd.kafka.v2+json"   -d '{"topics":["http-test"]}'
```

Fetch:
```bash
curl "http://localhost:8082/consumers/my-group/instances/my-consumer/records"   -H "Accept: application/vnd.kafka.json.v2+json"
```
</details>
