# Day 3: Deployment & High Availability

## Core Concepts & Theory

### Deployment Modes
1.  **Session Mode**: A long-running cluster. You submit multiple jobs to it. (Resources shared).
2.  **Application Mode**: The cluster is created *for* the job. The `main()` runs on the JobManager. (Better isolation).
3.  **Per-Job Mode** (Deprecated): Client runs `main()`, creates JobGraph, submits to cluster.

### High Availability (HA)
-   **Zookeeper / Kubernetes HA**:
    -   Stores metadata (JobGraph, Checkpoint pointers) in ZK/K8s ConfigMaps.
    -   If JobManager fails, a standby takes over and recovers from the metadata.

### Architectural Reasoning
**Why Application Mode on K8s?**
-   **Isolation**: If one job crashes the cluster, others are safe.
-   **GitOps**: The container image contains the JAR. `kubectl apply` deploys the job. No external client needed to submit.

### Key Components
-   **JobManager**: Coordinator.
-   **TaskManager**: Worker.
-   **BlobServer**: Distributes JARs.
