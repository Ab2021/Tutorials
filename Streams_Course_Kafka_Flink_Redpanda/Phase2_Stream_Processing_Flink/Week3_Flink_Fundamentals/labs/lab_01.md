# Lab 01: Local Flink Cluster

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
