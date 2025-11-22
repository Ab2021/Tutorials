# Day 5: Unified Batch & Stream Processing

## Core Concepts & Theory

### The Lambda Architecture (Old)
-   **Speed Layer**: Streaming (Approximate, Fast). Storm/Flink.
-   **Batch Layer**: Hadoop/Spark (Correct, Slow).
-   **Serving Layer**: Merge results.
-   **Problem**: Maintain two codebases (Java vs SQL). "Logic Drift".

### The Kappa Architecture (New)
-   **Everything is a Stream**.
-   **Real-time**: Process latest data.
-   **Batch**: Replay old data (it's just a bounded stream).
-   **Engine**: Flink (or Spark Structured Streaming).
-   **Benefit**: One codebase.

### Flink's Unification
-   **DataStream API**: Works for both Bounded (Batch) and Unbounded (Stream) sources.
-   **Batch Mode**: Flink optimizes for batch (sort-merge shuffle instead of pipelined shuffle) when input is bounded.

### Architectural Reasoning
**Is Batch Dead?**
No.
-   **Ad-hoc Queries**: "Find all users who did X last year". This is Batch.
-   **Model Training**: Training ML models is usually Batch.
-   **But**: The *Engine* can be the same.

