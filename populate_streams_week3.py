import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase2_Stream_Processing_Flink\Week3_Flink_Fundamentals"

content_map = {
    # --- Day 1: Intro ---
    "Day1_Intro_Stream_Processing_DeepDive.md": """# Day 1: Intro to Stream Processing - Deep Dive

## Deep Dive & Internals

### The Flink Runtime
-   **JobGraph**: The optimized logical plan (operators chained together).
-   **ExecutionGraph**: The physical plan (parallel tasks distributed across TaskManagers).
-   **Operator Chaining**: Flink fuses adjacent operators (e.g., Map -> Filter) into a single thread to reduce serialization/deserialization overhead and buffer exchange.

### Memory Management
Flink manages its own memory (off-heap) to avoid JVM GC pauses.
-   **Network Buffers**: For data exchange between TaskManagers.
-   **Managed Memory**: For internal data structures (hash tables, sort buffers) and State Backends (RocksDB).

### Advanced Reasoning
**Why "True Streaming"?**
Spark Streaming (legacy) used micro-batches. This meant latency was bounded by batch duration (seconds). Flink processes event-by-event, achieving sub-millisecond latency. This is critical for fraud detection or high-frequency trading.

### Performance Implications
-   **Backpressure**: Flink uses a credit-based flow control mechanism. If a downstream operator is slow, it stops granting credits to the upstream, naturally slowing down the source without data loss.
""",
    "Day1_Intro_Stream_Processing_Interview.md": """# Day 1: Intro to Stream Processing - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between Flink and Spark Streaming?**
    -   *A*: Flink is native streaming (event-at-a-time). Spark Structured Streaming is micro-batch (though it has a continuous processing mode now). Flink generally has lower latency and more advanced state management.

2.  **Q: What is a Task Slot?**
    -   *A*: A slice of resources in a TaskManager. It represents a fixed subset of memory. It does *not* enforce CPU isolation (threads share the CPU).

3.  **Q: Explain Operator Chaining.**
    -   *A*: Optimization where multiple operators are executed in the same thread to avoid thread switching and serialization overhead.

### Production Challenges
-   **Challenge**: **OOM (Out of Memory)**.
    -   *Scenario*: TaskManager crashes with Heap Space error.
    -   *Fix*: Check if you are buffering too much data in a `ListState` or if your window is too large. Tune `taskmanager.memory.process.size`.

### Troubleshooting Scenarios
**Scenario**: Job is stuck in "Created" state.
-   *Cause*: Not enough Task Slots available.
-   *Fix*: Scale up the cluster or reduce parallelism.
""",

    # --- Day 2: DataStream API ---
    "Day2_DataStream_API_Basics_Core.md": """# Day 2: DataStream API Basics

## Core Concepts & Theory

### Sources & Sinks
-   **Source**: Where data comes from (Kafka, File, Socket). `env.add_source(...)`.
-   **Sink**: Where data goes (Kafka, File, DB). `stream.add_sink(...)`.

### Transformations
-   **Map**: 1-to-1 transformation.
-   **FlatMap**: 1-to-N transformation (or 0). Good for splitting strings or filtering.
-   **Filter**: Keep only elements satisfying a condition.
-   **KeyBy**: Logically partitions the stream. All records with same key go to same parallel instance.

### Physical Partitioning
-   **Rescale**: Round-robin to a subset of downstream tasks.
-   **Rebalance**: Round-robin to ALL downstream tasks (handles skew).
-   **Broadcast**: Send element to ALL downstream tasks.

### Architectural Reasoning
**Why KeyBy is expensive?**
`keyBy` causes a **Network Shuffle**. Data must be serialized and sent over the network to the correct TaskManager. This is the most expensive operation in a distributed system. Minimize shuffles.

### Key Components
-   **StreamExecutionEnvironment**: The context for creating the job.
-   **DataStream**: The immutable collection of data.
""",
    "Day2_DataStream_API_Basics_DeepDive.md": """# Day 2: DataStream API - Deep Dive

## Deep Dive & Internals

### Serialization
Flink needs to serialize data to send it over the network.
-   **POJOs**: Flink analyzes your class. If it follows POJO rules (public fields, empty constructor), Flink uses a highly efficient custom serializer.
-   **Kryo**: Fallback for generic types. Slower.
-   **Tuples**: Extremely efficient.

### Type Erasure
Java erases generic types at runtime (`List<String>` becomes `List`). Flink uses **TypeInformation** to capture type info at compile time so it can serialize correctly at runtime.

### Advanced Reasoning
**Rich Functions**
Standard functions (`MapFunction`) are just lambdas. **RichFunctions** (`RichMapFunction`) give you access to the lifecycle:
-   `open()`: Called once during initialization (connect to DB here).
-   `close()`: Called at shutdown.
-   `getRuntimeContext()`: Access to State, Metrics, and Broadcast variables.

### Performance Implications
-   **Object Reuse**: Flink can be configured to reuse mutable objects (`enableObjectReuse()`) to reduce GC pressure. Dangerous if you hold references to objects.
""",
    "Day2_DataStream_API_Basics_Interview.md": """# Day 2: DataStream API - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between `map` and `flatMap`?**
    -   *A*: `map` produces exactly one output for every input. `flatMap` can produce 0, 1, or many.

2.  **Q: Why do we need `keyBy`?**
    -   *A*: To partition the state. Aggregations (sum, min, max) and Windows usually require a keyed stream so that the state is scoped to the key.

3.  **Q: What happens if you use a non-serializable object in a stream?**
    -   *A*: The job fails at submission time (or runtime) with `NotSerializableException`.

### Production Challenges
-   **Challenge**: **Serialization Bottleneck**.
    -   *Scenario*: CPU is high, but logic is simple.
    -   *Cause*: Using complex objects that fall back to Kryo serialization.
    -   *Fix*: Use POJOs or Tuples. Register custom serializers.

### Troubleshooting Scenarios
**Scenario**: `NullPointerException` in `open()`.
-   *Cause*: Trying to access runtime context before it's initialized (rare) or external resource failure.
""",

    # --- Day 3: Time Semantics ---
    "Day3_Time_Semantics_Watermarks_DeepDive.md": """# Day 3: Time Semantics - Deep Dive

## Deep Dive & Internals

### Watermark Propagation
Watermarks flow through the DAG.
-   **One-to-One**: Simple propagation.
-   **Many-to-One (Union/KeyBy)**: An operator receives watermarks from multiple upstream channels. It takes the **minimum** of all incoming watermarks.
    -   *Implication*: One slow upstream partition holds back the event time for the entire job.

### Idle Sources
If a Kafka partition has no data, it sends no watermarks. The downstream operator's watermark (min of all inputs) stalls.
-   **Fix**: `withIdleness(Duration)`. Marks a source as idle so it is ignored in the min calculation.

### Advanced Reasoning
**Late Data Handling**
What happens when an event arrives *after* the watermark has passed?
1.  **Default**: Dropped.
2.  **Allowed Lateness**: `allowedLateness(Time)`. Keep the window state around for a bit longer.
3.  **Side Output**: `sideOutputLateData(tag)`. Divert to a separate stream for manual handling.

### Performance Implications
-   **Watermark Interval**: `setAutoWatermarkInterval(200ms)`. Too frequent = CPU overhead. Too infrequent = jerky progress and higher latency for window results.
""",
    "Day3_Time_Semantics_Watermarks_Interview.md": """# Day 3: Time Semantics - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is a Watermark?**
    -   *A*: A timestamp that asserts "all events with timestamp < T have arrived". It triggers event-time timers (windows).

2.  **Q: How do you handle late data?**
    -   *A*: Allowed Lateness (update old result), Side Outputs (save for later), or just drop it.

3.  **Q: What is the difference between Ingestion Time and Event Time?**
    -   *A*: Ingestion is when Flink sees it. Event is when it happened. Event time allows deterministic replay.

### Production Challenges
-   **Challenge**: **Stalled Watermark**.
    -   *Scenario*: Windows are not closing.
    -   *Cause*: One Kafka partition is empty (idle).
    -   *Fix*: Use `withIdleness()`.

### Troubleshooting Scenarios
**Scenario**: Data is dropped unexpectedly.
-   *Cause*: Watermarks are too aggressive (assuming 1s lag when real lag is 5s).
-   *Fix*: Adjust `BoundedOutOfOrderness` strategy.
""",

    # --- Day 4: Windowing ---
    "Day4_Windowing_Strategies_Core.md": """# Day 4: Windowing Strategies

## Core Concepts & Theory

### Types of Windows
1.  **Tumbling**: Fixed size, non-overlapping. (e.g., "Every 5 minutes").
2.  **Sliding**: Fixed size, overlapping. (e.g., "Last 10 mins, updated every 1 min").
3.  **Session**: Dynamic size. Defined by a gap of inactivity. (e.g., "User session ends after 30 mins idle").
4.  **Global**: One giant window. Requires a custom trigger to fire.

### Window Lifecycle
-   **Creation**: When the first element for a key/window arrives.
-   **Accumulation**: Elements are added to the window state.
-   **Firing**: Trigger decides to compute the result.
-   **Purging**: Clearing the content.

### Architectural Reasoning
**Why Windows need State?**
To calculate "average price over 1 hour", Flink must store either all prices (ListState) or the running sum/count (ReducingState) until the hour is up. This state is managed by the State Backend.

### Key Components
-   **WindowAssigner**: Assigns element to window(s).
-   **Trigger**: When to evaluate.
-   **Evictor**: Which elements to keep (optional).
-   **WindowFunction**: The computation (Reduce, Aggregate, Process).
""",
    "Day4_Windowing_Strategies_DeepDive.md": """# Day 4: Windowing - Deep Dive

## Deep Dive & Internals

### Window State
-   **Incremental Aggregation** (`ReduceFunction`, `AggregateFunction`): Computes as data arrives. Stores only 1 value (e.g., sum). Efficient.
-   **Full Window Function** (`ProcessWindowFunction`): Stores ALL elements until trigger. Expensive (RAM/State), but allows access to metadata (start/end time) and iterating over all elements.
-   **Combined**: You can use `window.aggregate(Agg, Process)` to get efficiency + metadata.

### Session Window Implementation
Session windows are hard because they merge.
-   Initially, every element creates a new session window `[t, t+gap]`.
-   If two windows overlap, they are **merged** into a larger window.
-   State must be merged.

### Advanced Reasoning
**Sliding Window Optimization**
A sliding window (size 1hr, slide 1min) assigns every element to 60 windows! This explodes state.
-   **Optimization**: Panes (tumbling windows of GCD size) or slicing. Flink's default implementation actually duplicates the data into multiple buckets. Be careful with small slides.

### Performance Implications
-   **Large Windows**: A 24-hour window with full process function will blow up memory. Use incremental aggregation.
""",
    "Day4_Windowing_Strategies_Interview.md": """# Day 4: Windowing - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between Tumbling and Sliding windows?**
    -   *A*: Tumbling windows do not overlap. Sliding windows overlap.

2.  **Q: How does a Session Window work?**
    -   *A*: It groups elements by key and merges windows that are within a "gap" of each other. It has no fixed duration.

3.  **Q: When would you use `ProcessWindowFunction`?**
    -   *A*: When you need the window start/end timestamps or need to perform a calculation that requires all elements (like median).

### Production Challenges
-   **Challenge**: **State Explosion with Sliding Windows**.
    -   *Scenario*: Window(1hr, slide 1s).
    -   *Fix*: Don't do this. Use a larger slide or a different pattern.

### Troubleshooting Scenarios
**Scenario**: Window results are incorrect (too low).
-   *Cause*: Late data is being dropped.
-   *Fix*: Check `sideOutputLateData` to see if data is arriving after the watermark.
""",

    # --- Day 5: Triggers ---
    "Day5_Triggers_Evictors_Core.md": """# Day 5: Triggers & Evictors

## Core Concepts & Theory

### Triggers
A **Trigger** determines when a window is ready to be processed.
-   **EventTimeTrigger**: Fires when Watermark passes window end. (Default).
-   **ProcessingTimeTrigger**: Fires based on wall-clock time.
-   **CountTrigger**: Fires when N elements arrive.
-   **PurgingTrigger**: Fires and then clears the window.

### Evictors
An **Evictor** can remove elements from the window *before* or *after* the trigger fires.
-   **CountEvictor**: Keep only last N elements.
-   **DeltaEvictor**: Keep elements based on a delta threshold.

### Architectural Reasoning
**Custom Triggers**
You might want a window that fires "Every 1 minute OR when 1000 items arrive" (Early firing). This allows low latency updates for a long window.

### Key Components
-   `onElement()`: Called for every record.
-   `onEventTime()`: Called when watermark passes.
-   `onProcessingTime()`: Called when system timer fires.
""",
    "Day5_Triggers_Evictors_DeepDive.md": """# Day 5: Triggers - Deep Dive

## Deep Dive & Internals

### FIRE vs PURGE
-   **FIRE**: Call the window function, keep the state. (Good for early updates).
-   **PURGE**: Clear the state.
-   **FIRE_AND_PURGE**: Emit result and clear. (Standard for tumbling windows).

### ContinuousProcessingTimeTrigger
Fires periodically. Useful for "speculative" results.
-   e.g., "Show me the current top 10 trending items every 5 seconds, even though the 1-hour window isn't done."

### Advanced Reasoning
**The "Global Window" Trap**
A `GlobalWindow` never ends. The default trigger never fires. You *must* attach a custom trigger (like `CountTrigger`) or it will just accumulate state until OOM.

### Performance Implications
-   **Early Firing**: Increases downstream load (more updates).
-   **Evictors**: Force the use of `ProcessWindowFunction` (state must be kept to allow eviction). Prevents incremental aggregation optimization.
""",
    "Day5_Triggers_Evictors_Interview.md": """# Day 5: Triggers - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the default trigger for an Event Time window?**
    -   *A*: `EventTimeTrigger`. It fires once when the watermark passes the window end.

2.  **Q: Why would you use a custom trigger?**
    -   *A*: To get early results (speculative) before the window closes, or to handle late data specially.

3.  **Q: What is the difference between FIRE and PURGE?**
    -   *A*: FIRE emits a result but keeps the data. PURGE deletes the data.

### Production Challenges
-   **Challenge**: **Duplicate Results**.
    -   *Scenario*: Using a trigger that FIREs multiple times without PURGING.
    -   *Fix*: Ensure your downstream consumer can handle updates (idempotency or upserts).

### Troubleshooting Scenarios
**Scenario**: Global Window not producing output.
-   *Cause*: Forgot to set a Trigger.
-   *Fix*: `.trigger(CountTrigger.of(100))`.
"""
}

print("🚀 Populating Week 3 Flink Content...")

for filename, content in content_map.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {filename}")

print("✅ Week 3 Content Population Complete!")
