# Lab 01: Redpanda Docker Setup

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Run Redpanda using Docker.
-   Understand the single-binary architecture.
-   Access the `rpk` CLI inside the container.

## Problem Statement
Start a single-node Redpanda cluster. Expose port 9092. Use `rpk` inside the container to check the cluster status.

## Starter Code
```bash
docker run -d --name redpanda -p 9092:9092 ...
```

## Hints
<details>
<summary>Hint 1</summary>
Use the official image `docker.redpanda.com/redpandadata/redpanda:latest`. You need to set `redpanda start --overprovisioned --smp 1 --memory 1G` for local dev.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Command
```bash
docker run -d --name redpanda --rm     -p 9092:9092     -p 9644:9644     docker.redpanda.com/redpandadata/redpanda:latest     redpanda start     --overprovisioned     --smp 1      --memory 1G     --kafka-addr internal://0.0.0.0:9092,external://0.0.0.0:19092     --advertise-kafka-addr internal://redpanda:9092,external://localhost:9092     --pandaproxy-addr internal://0.0.0.0:8082,external://0.0.0.0:18082     --advertise-pandaproxy-addr internal://redpanda:8082,external://localhost:18082     --schema-registry-addr internal://0.0.0.0:8081,external://0.0.0.0:18081     --rpc-addr redpanda:33145     --advertise-rpc-addr redpanda:33145
```

### Verification
```bash
docker exec -it redpanda rpk cluster info
```
</details>
