# Day 4: Testing & Monitoring

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
