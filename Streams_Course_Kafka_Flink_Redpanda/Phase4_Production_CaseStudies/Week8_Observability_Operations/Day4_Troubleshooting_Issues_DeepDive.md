# Day 4: Troubleshooting - Deep Dive

## Deep Dive & Internals

### Debugging Network Issues
Kafka/Flink are distributed. Network is often the culprit.
-   **TCP Retransmits**: High retransmits = Bad cable/switch.
-   **DNS Latency**: Java caches DNS. If IP changes, app might fail.
-   **Bandwidth Saturation**: Check `sar -n DEV`.

### Analyzing Thread Dumps
When a process is "stuck" (not processing, but CPU low):
1.  Take thread dump (`jstack <pid>`).
2.  Look for `BLOCKED` or `WAITING` threads.
3.  **Common Culprits**:
    -   Waiting on Lock (Deadlock).
    -   Waiting on I/O (Socket read).
    -   Waiting on external service (HTTP call without timeout).

### Analyzing Heap Dumps
When `OutOfMemoryError` occurs:
1.  Take heap dump (`jmap -dump:format=b,file=heap.bin <pid>`).
2.  Open in **Eclipse MAT** or **VisualVM**.
3.  Look at "Dominator Tree".
4.  **Common Culprits**:
    -   Huge `HashMap` (Caching without TTL).
    -   Large `byte[]` (Accumulating payloads).
    -   Flink State objects (RocksDB JNI objects).

### Advanced Reasoning
**Heisenbugs**
Bugs that disappear when you try to study them (e.g., enabling debug logging slows down the system enough to hide a race condition).
-   **Fix**: Distributed Tracing and high-resolution metrics are better than logs for race conditions.

### Performance Implications
-   **Logging Overhead**: `DEBUG` logging in a hot loop (processElement) can reduce throughput by 100x. Use `isDebugEnabled()` check.
