# Day 1: Redpanda Architecture - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How does Redpanda differ from Kafka architecturally?**
    -   *A*: Redpanda is C++ (vs Java), uses Thread-Per-Core architecture (Seastar), has no Zookeeper dependency (built-in Raft), and is a single binary.

2.  **Q: What is the advantage of bypassing the Page Cache?**
    -   *A*: It gives the application full control over caching and eviction policies, preventing "neighbor noise" from other processes and avoiding double-buffering (storing data in both JVM heap and OS cache).

3.  **Q: Is Redpanda 100% compatible with Kafka?**
    -   *A*: It is compatible with the Kafka *API* (protocol). Existing Kafka clients work without modification. However, it does not support Kafka *plugins* (JARs) like custom Authorizers or Tiered Storage implementations (it has its own).

### Production Challenges
-   **Challenge**: **CPU Pinning in Containers**.
    -   *Scenario*: Running Redpanda in Docker/K8s without dedicated CPU limits.
    -   *Issue*: Seastar expects to own the core. If shared, performance degrades.
    -   *Fix*: Use `--cpuset-cpus` or K8s `cpu-manager-policy: static`.

### Troubleshooting Scenarios
**Scenario**: High latency on one specific node.
-   **Check**: Is the node overloaded? (Use `rpk cluster status`).
-   **Check**: Is there a "hot partition" on that node?
