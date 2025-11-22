# Day 5: Advanced Topics - Interview Prep

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
