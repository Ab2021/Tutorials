# Day 5: Advanced Topics - Deep Dive

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
