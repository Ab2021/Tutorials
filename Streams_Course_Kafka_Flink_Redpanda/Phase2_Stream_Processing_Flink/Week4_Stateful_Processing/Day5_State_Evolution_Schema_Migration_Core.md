# Day 5: State Evolution & Schema Migration

## Core Concepts & Theory

### The Problem
You have a running job with State `User(name, age)`. You want to upgrade code to `User(name, age, email)`.
-   If you just deploy, deserialization fails.

### Evolution Strategies
1.  **POJO / Avro Evolution**: Flink supports adding/removing fields if using supported serializers.
2.  **State Processor API**: Offline tool to read a Savepoint, transform it (ETL for State), and write a new Savepoint.
3.  **Drop State**: Start from scratch (reprocess from Kafka).

### Architectural Reasoning
**State Processor API**
Think of it as "MapReduce for Savepoints". It allows you to read the binary snapshot as a DataSet/DataStream, modify it using Flink code, and write it back.
-   Use cases: Schema migration, changing window definitions, bootstrapping state from a DB.

### Key Components
-   `SavepointReader`: Reads savepoint.
-   `SavepointWriter`: Writes savepoint.
-   `BootstrapTransformation`: Defines how to write new state.
