# Day 3: Deployment - Interview Prep

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
