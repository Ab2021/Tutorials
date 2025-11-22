import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase2_Stream_Processing_Flink\Week5_Advanced_Flink"

content_map = {
    # --- Day 1: Flink SQL ---
    "Day1_Flink_SQL_Table_API_Core.md": """# Day 1: Flink SQL & Table API

## Core Concepts & Theory

### Declarative Stream Processing
Instead of writing Java/Python code (DataStream API), you write SQL.
-   **Table API**: Fluent API in Java/Python (`table.select(...)`).
-   **SQL**: Standard ANSI SQL (`SELECT * FROM ...`).

### Dynamic Tables
A stream is a table that is constantly changing.
-   **Stream -> Table**: The stream is interpreted as a changelog.
-   **Continuous Query**: The query runs forever, updating the result table as new rows arrive.
-   **Table -> Stream**: The result table is converted back to a stream (Append-only, Retract, or Upsert).

### Architectural Reasoning
**Why SQL?**
-   **Accessibility**: Analysts can write streaming jobs.
-   **Optimization**: The Catalyst-like optimizer (Calcite) can reorder joins, push down filters, and choose efficient state backends automatically.

### Key Components
-   `StreamTableEnvironment`: The entry point.
-   `CREATE TABLE`: Defines sources/sinks (Kafka, JDBC, Files).
-   `INSERT INTO`: Submits a job.
""",
    "Day1_Flink_SQL_Table_API_DeepDive.md": """# Day 1: Flink SQL - Deep Dive

## Deep Dive & Internals

### Changelog Stream Types
1.  **Append-Only**: Only INSERTs. (e.g., Log files).
2.  **Retract**: INSERT and DELETE. (e.g., `COUNT` decreases when a record leaves a window).
3.  **Upsert**: INSERT and UPDATE. (e.g., Database CDC).

### Joins in SQL
-   **Regular Join**: Keeps ALL history of both sides in state. Infinite state growth!
-   **Interval Join**: `A.time BETWEEN B.time - 1h AND B.time`. State is cleaned up after 1h.
-   **Temporal Table Join**: Join with a "versioned table" (e.g., Currency Rates at time T).

### Advanced Reasoning
**Mini-Batch Aggregation**
By default, Flink SQL updates the result for *every* input row. This causes high I/O.
-   **Mini-Batch**: Buffer inputs for 500ms, compute aggregate in memory, send 1 update. Drastically reduces state access.

### Performance Implications
-   **State Size**: `GROUP BY` on high-cardinality keys (User ID) without windows will grow state forever. Use `Idle State Retention` config to clean up old keys.
""",
    "Day1_Flink_SQL_Table_API_Interview.md": """# Day 1: Flink SQL - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between the Table API and SQL?**
    -   *A*: They are equivalent. Table API is embedded in Java/Python (compile-time checks). SQL is text-based (runtime checks). Both use the same optimizer.

2.  **Q: How do you handle infinite state in Flink SQL?**
    -   *A*: Use Window Aggregations, Interval Joins, or configure `table.exec.state.ttl` to expire old state.

3.  **Q: What is a Temporal Table Join?**
    -   *A*: Joining a stream with a slowly changing dimension table (e.g., Orders JOIN Products) at a specific point in time.

### Production Challenges
-   **Challenge**: **Retraction Storm**.
    -   *Scenario*: A multi-level aggregation (Count -> Max). An update upstream causes a Retract(-1) and Accumulate(+1) downstream. This doubles the load.
    -   *Fix*: Use Mini-Batch aggregation or optimize the query plan.

### Troubleshooting Scenarios
**Scenario**: `TableException: Table is not an append-only table. Use the toChangelogStream()`.
-   *Cause*: You tried to convert a dynamic table with updates/deletes into a simple DataStream.
-   *Fix*: Use `to_changelog_stream` to handle the update flags.
""",

    # --- Day 2: CEP ---
    "Day2_CEP_Pattern_Matching_Core.md": """# Day 2: Complex Event Processing (CEP)

## Core Concepts & Theory

### What is CEP?
Detecting **patterns** across a stream of events.
-   "If Event A happens, followed by Event B within 10 minutes, trigger Alert."

### Pattern API
-   `begin("start")`: Define start state.
-   `where(condition)`: Filter.
-   `next("middle")`: Strict contiguity (A immediately followed by B).
-   `followedBy("middle")`: Relaxed contiguity (A ... B).
-   `within(Time)`: Time constraint.

### Architectural Reasoning
**NFA (Nondeterministic Finite Automaton)**
Flink compiles the pattern into an NFA.
-   State is stored for every partial match.
-   If you have a pattern "A followed by B", and you get "A", Flink stores "A" in state waiting for "B".

### Key Components
-   `CEP.pattern(stream, pattern)`: Applies the pattern.
-   `PatternSelectFunction`: Extracts the result when a match is found.
""",
    "Day2_CEP_Pattern_Matching_DeepDive.md": """# Day 2: CEP - Deep Dive

## Deep Dive & Internals

### Contiguity Types
1.  **Strict (`next`)**: A, B. Input: A, C, B. Match: No.
2.  **Relaxed (`followedBy`)**: A, B. Input: A, C, B. Match: Yes (skips C).
3.  **Non-Deterministic Relaxed (`followedByAny`)**: A, B. Input: A, C, B1, B2. Matches: (A, B1) AND (A, B2).

### After Match Skip Strategy
What happens after a match?
-   **NO_SKIP**: All possible matches. (Expensive).
-   **SKIP_PAST_LAST_EVENT**: Discard partial matches that overlap.
-   **SKIP_TO_NEXT**: Jump to the next start.

### Advanced Reasoning
**CEP vs SQL MATCH_RECOGNIZE**
Flink SQL supports `MATCH_RECOGNIZE` (standard SQL CEP).
-   **SQL**: Standard, declarative, easier to optimize.
-   **CEP Library**: More flexible (imperative conditions), allows complex Java/Python logic in conditions.

### Performance Implications
-   **State Explosion**: A pattern like "A followed by B" with no time limit will store "A" forever. ALWAYS use `.within(Time)`.
""",
    "Day2_CEP_Pattern_Matching_Interview.md": """# Day 2: CEP - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between `next()` and `followedBy()`?**
    -   *A*: `next` requires strict sequence (no events in between). `followedBy` allows non-matching events in between.

2.  **Q: How does CEP handle state?**
    -   *A*: It stores partial matches in the State Backend (RocksDB).

3.  **Q: Why is `.within()` important?**
    -   *A*: To prune state. Without it, partial matches accumulate forever, leading to OOM.

### Production Challenges
-   **Challenge**: **High Latency with Complex Patterns**.
    -   *Scenario*: Pattern `A followedBy B`. Stream has 1M "A"s and no "B".
    -   *Cause*: Flink checks every "A" against every incoming event.
    -   *Fix*: Use stricter conditions or shorter windows.

### Troubleshooting Scenarios
**Scenario**: Pattern not matching.
-   *Cause*: Time characteristic. Are you using Event Time? Are watermarks progressing? CEP relies heavily on correct time.
""",

    # --- Day 3: Deployment ---
    "Day3_Deployment_K8s_HA_Core.md": """# Day 3: Deployment & High Availability

## Core Concepts & Theory

### Deployment Modes
1.  **Session Mode**: A long-running cluster. You submit multiple jobs to it. (Resources shared).
2.  **Application Mode**: The cluster is created *for* the job. The `main()` runs on the JobManager. (Better isolation).
3.  **Per-Job Mode** (Deprecated): Client runs `main()`, creates JobGraph, submits to cluster.

### High Availability (HA)
-   **Zookeeper / Kubernetes HA**:
    -   Stores metadata (JobGraph, Checkpoint pointers) in ZK/K8s ConfigMaps.
    -   If JobManager fails, a standby takes over and recovers from the metadata.

### Architectural Reasoning
**Why Application Mode on K8s?**
-   **Isolation**: If one job crashes the cluster, others are safe.
-   **GitOps**: The container image contains the JAR. `kubectl apply` deploys the job. No external client needed to submit.

### Key Components
-   **JobManager**: Coordinator.
-   **TaskManager**: Worker.
-   **BlobServer**: Distributes JARs.
""",
    "Day3_Deployment_K8s_HA_DeepDive.md": """# Day 3: Deployment - Deep Dive

## Deep Dive & Internals

### Kubernetes Native Integration
Flink talks directly to the K8s API Server.
-   **Dynamic Resource Allocation**: If a job needs more slots, the JobManager asks K8s to spin up a new TaskManager Pod.
-   **Pod Templates**: Customize sidecars, volumes, and init containers.

### Reactive Mode
Allows Flink to scale automatically based on available resources.
-   If you add a TaskManager, Flink detects it and rescales the job (restarts with higher parallelism).
-   If a TaskManager dies, Flink rescales down instead of failing.
-   **Use Case**: Autoscaling on K8s (HPA).

### Advanced Reasoning
**Classloading**
-   **Parent-First**: Java default.
-   **Child-First**: Flink default. Allows user code to use different library versions than Flink core (avoids "Dependency Hell").

### Performance Implications
-   **Network Buffers**: In K8s, ensure `taskmanager.memory.network.fraction` is sufficient if pods are on different nodes.
""",
    "Day3_Deployment_K8s_HA_Interview.md": """# Day 3: Deployment - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the benefit of Application Mode over Session Mode?**
    -   *A*: Isolation (one cluster per job) and the `main()` method runs on the cluster (saving bandwidth/client resources).

2.  **Q: How does Flink HA work in Kubernetes?**
    -   *A*: It uses K8s ConfigMaps to store leader information and checkpoint pointers. No Zookeeper needed.

3.  **Q: What is Reactive Mode?**
    -   *A*: A mode where Flink adjusts parallelism based on available TaskManagers. Enables autoscaling.

### Production Challenges
-   **Challenge**: **"No Resource Available"**.
    -   *Scenario*: JobManager requests pods, but K8s is full.
    -   *Fix*: Cluster Autoscaler or priority classes.

-   **Challenge**: **Slow Classloading**.
    -   *Cause*: Huge Uber-JARs.
    -   *Fix*: Shade dependencies properly.

### Troubleshooting Scenarios
**Scenario**: JobManager keeps restarting (CrashLoopBackOff).
-   *Cause*: OOM (Heap) or MetaSpace OOM.
-   *Fix*: Increase `jobmanager.memory.jvm-overhead` or heap size.
""",

    # --- Day 4: Testing ---
    "Day4_Testing_Debugging_Monitoring_Core.md": """# Day 4: Testing & Monitoring

## Core Concepts & Theory

### Testing Levels
1.  **Unit Tests**: Test individual `MapFunction` or `ProcessFunction`.
2.  **Integration Tests**: Test the pipeline using `MiniCluster`.
3.  **E2E Tests**: Test with real Kafka/Docker.

### Test Harness
Flink provides `KeyedOneInputStreamOperatorTestHarness`.
-   Allows you to push elements, set watermarks, and inspect state/output *without* starting a full cluster.
-   Crucial for testing ProcessFunctions with time logic.

### Monitoring
-   **Metrics**: Throughput, Latency, Checkpoint Size, GC time.
-   **Reporters**: Prometheus, JMX, Datadog.
-   **Backpressure**: The Web UI shows "High/Low" backpressure.

### Architectural Reasoning
**Why TestHarness?**
Mocking time is hard. `TestHarness` allows you to say "Advance processing time by 10 seconds" and verify that your timer fired.

### Key Components
-   `MiniClusterWithClientResource`: JUnit rule to start a local Flink.
-   `TestHarness`: For operator testing.
""",
    "Day4_Testing_Debugging_Monitoring_DeepDive.md": """# Day 4: Testing - Deep Dive

## Deep Dive & Internals

### Backpressure Monitoring
Flink 1.13+ uses **Task Sampling**.
-   The JobManager periodically samples the stack traces of TaskManagers.
-   If a task is stuck in `requestMemoryBuffer` (waiting for network), it is backpressured.
-   This is zero-overhead compared to the old method (injecting metrics).

### Flame Graphs
Flink Web UI can generate Flame Graphs (CPU profiles) on the fly.
-   Identify hot methods (e.g., Regex parsing, serialization).

### Advanced Reasoning
**Unit Testing Stateful Functions**
You cannot just instantiate `MyProcessFunction` and call `processElement`. You need the `RuntimeContext` (for state access). The `TestHarness` mocks this context, the state backend, and the timer service.

### Performance Implications
-   **Metric Overhead**: Too many metrics (e.g., per-key metrics) can degrade performance. Use `metrics.latency.interval` sparingly.
""",
    "Day4_Testing_Debugging_Monitoring_Interview.md": """# Day 4: Testing - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you test a function that depends on processing time?**
    -   *A*: Use `TestHarness` to manually advance processing time in a deterministic way.

2.  **Q: How do you detect backpressure?**
    -   *A*: Check the Flink Web UI (Backpressure tab) or monitor `outPoolUsage` metrics.

3.  **Q: What is a Flame Graph?**
    -   *A*: A visualization of stack traces to identify CPU hotspots.

### Production Challenges
-   **Challenge**: **Silent Failure**.
    -   *Scenario*: Job is running but producing no data.
    -   *Cause*: Watermark stalled, or logic error filtering everything.
    -   *Fix*: Monitor `numRecordsOut` and `lastCheckpointDuration`.

### Troubleshooting Scenarios
**Scenario**: Checkpoint size is growing linearly.
-   *Cause*: State leak.
-   *Fix*: Analyze checkpoint metrics. Use State Processor API to inspect the checkpoint content.
""",

    # --- Day 5: Advanced Topics ---
    "Day5_Advanced_Topics_PyFlink_Core.md": """# Day 5: Advanced Topics & PyFlink

## Core Concepts & Theory

### PyFlink Architecture
Python API for Flink.
-   **Architecture**: Python runs in a separate process (Py4J / gRPC).
-   **Data Exchange**: Data is serialized from JVM -> Python Process -> JVM.
-   **Vectorized UDFs**: Uses Apache Arrow to transfer batches of data (much faster than row-by-row).

### Async I/O
Calling external APIs (REST/DB) from a stream.
-   **OrderedWait**: Preserves order (Head-of-line blocking).
-   **UnorderedWait**: Faster, order not guaranteed.

### Architectural Reasoning
**Why Vectorized Python UDFs?**
Python loop overhead is high. Vectorized UDFs (Pandas UDFs) allow executing logic on a *batch* of rows using optimized C libraries (NumPy/Pandas), reducing the serialization/invocation overhead.

### Key Components
-   `@udf`: Decorator for Python functions.
-   `AsyncDataStream`: Helper for Async I/O.
""",
    "Day5_Advanced_Topics_PyFlink_DeepDive.md": """# Day 5: Advanced Topics - Deep Dive

## Deep Dive & Internals

### Loop Unrolling & Code Gen
Flink's SQL engine (Blink/Calcite) generates Java bytecode for queries.
-   It unrolls loops and inlines virtual function calls to maximize CPU cache efficiency.

### Two-Phase Commit (2PC)
Used for Exactly-Once Sinks (Kafka, FileSystem).
1.  **Pre-Commit**: Write data to temp file / transaction. (On Checkpoint).
2.  **Commit**: Finalize transaction. (On Checkpoint Complete).
3.  **Abort**: Rollback.

### Advanced Reasoning
**The "Small File" Problem**
Streaming sinks often create millions of tiny files (1KB each).
-   **File Sink**: Supports "Compaction" (merging small files into large ones) as a post-process.
-   **Hive Sink**: Handles partition commit policies.

### Performance Implications
-   **PyFlink Overhead**: Even with Arrow, PyFlink is slower than Java. Use SQL for heavy lifting and Python only for complex logic that SQL can't express.
""",
    "Day5_Advanced_Topics_PyFlink_Interview.md": """# Day 5: Advanced Topics - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How does PyFlink work under the hood?**
    -   *A*: It uses a Gateway (Py4J) to talk to the JVM. Data is exchanged via memory mapped files or sockets, often using Arrow for performance.

2.  **Q: What is the trade-off of Async I/O?**
    -   *A*: It improves throughput for high-latency I/O, but increases checkpoint size (buffers in flight) and complexity.

3.  **Q: How do you handle "Small Files" in S3 sinks?**
    -   *A*: Use Flink's FileSink with rolling policies (size/time) and enable compaction.

### Production Challenges
-   **Challenge**: **Python Dependency Management**.
    -   *Scenario*: You need `numpy` on the cluster.
    -   *Fix*: Build a custom Docker image or use `add_python_archive` (VirtualEnv).

### Troubleshooting Scenarios
**Scenario**: PyFlink job is slow.
-   *Cause*: Too much serialization or using row-based UDFs instead of Vectorized.
-   *Fix*: Switch to Pandas UDFs.
"""
}

print("🚀 Populating Week 5 Advanced Flink Content...")

for filename, content in content_map.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {filename}")

print("✅ Week 5 Content Population Complete!")
