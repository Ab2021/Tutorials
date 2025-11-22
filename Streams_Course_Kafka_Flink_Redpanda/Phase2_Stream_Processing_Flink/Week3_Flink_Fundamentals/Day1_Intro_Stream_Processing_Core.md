# Day 1: Introduction to Stream Processing

## Core Concepts & Theory

### The Dataflow Model
Stream processing is based on the **Dataflow Model** (pioneered by Google).
-   **DAG (Directed Acyclic Graph)**: A job is represented as a graph where nodes are operators (Map, Filter, KeyBy) and edges are data streams.
-   **Parallelism**: Each operator can have multiple parallel instances running on different machines.

### JobManager & TaskManager
-   **JobManager (Master)**: Coordinates the execution, checkpoints, and recovery. It turns the JobGraph into an ExecutionGraph.
-   **TaskManager (Worker)**: Executes the actual tasks (sub-tasks of operators) in slots.

### Architectural Reasoning
**Why Flink?**
Flink is a **True Streaming** engine.
-   **Spark Streaming**: Micro-batch (simulates streaming by chopping data into small batches). High latency.
-   **Flink**: Row-at-a-time processing. Ultra-low latency.
Flink treats batch processing as a special case of streaming (bounded stream).

### Key Components
-   **DataStream**: The core abstraction for unbounded data.
-   **Operator**: A transformation function.
-   **Slot**: The unit of resource allocation in a TaskManager.
