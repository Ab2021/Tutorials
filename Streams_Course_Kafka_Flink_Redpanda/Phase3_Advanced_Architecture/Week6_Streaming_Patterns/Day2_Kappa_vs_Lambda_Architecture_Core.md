# Day 2: Kappa vs Lambda Architecture

## Core Concepts & Theory

### Lambda Architecture
Hybrid approach (Big Data 1.0).
-   **Speed Layer**: Stream processing (Approximate, Low Latency).
-   **Batch Layer**: Hadoop/Spark (Accurate, High Latency).
-   **Serving Layer**: Merges results.
-   **Problem**: Maintaining two codebases (Batch + Stream) is painful.

### Kappa Architecture
Stream-only approach.
-   **Idea**: "Batch is just a stream with a bounded start and end."
-   **Single Codebase**: Use Flink for both real-time and historical reprocessing.
-   **Long Retention**: Kafka stores data for weeks/months/forever.

### Architectural Reasoning
**Why Kappa?**
Simplicity. You write the logic once (Flink SQL/DataStream). To recompute history (e.g., bug fix), you just start a new instance of the job reading from offset 0.

### Key Components
-   **Unified Engine**: Flink or Spark Structured Streaming.
-   **Tiered Storage**: Makes storing PBs of data in Kafka affordable, enabling Kappa.
