# Day 1: Intro to Stream Processing - Interview Prep

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
