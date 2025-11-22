# Day 3: Deployment - Deep Dive

## Deep Dive & Internals

### Kubernetes Native Integration
Flink talks directly to the K8s API Server.
-   **Dynamic Resource Allocation**: If a job needs more slots, the JobManager asks K8s to spin up a new TaskManager Pod.
-   **Pod Templates**: Customize sidecars, volumes, and init containers.

### Reactive Mode
Allows Flink to scale automatically based on available resources.
-   If you add a TaskManager, Flink detects it and rescales the job (restarts with higher parallelism).
-   If a TaskManager dies, Flink rescales down instead of failing.
-   **Use Case**: Autoscaling on K8s (HPA).

### Advanced Reasoning
**Classloading**
-   **Parent-First**: Java default.
-   **Child-First**: Flink default. Allows user code to use different library versions than Flink core (avoids "Dependency Hell").

### Performance Implications
-   **Network Buffers**: In K8s, ensure `taskmanager.memory.network.fraction` is sufficient if pods are on different nodes.
