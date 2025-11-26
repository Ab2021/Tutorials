# Day 17: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the role of `etcd` in Kubernetes?
**Answer:**
*   **Role**: The "Source of Truth". It is a distributed, consistent Key-Value store.
*   **Content**: Stores the entire state of the cluster (Nodes, Pods, Configs, Secrets).
*   **Criticality**: If etcd is lost, the cluster is lost. Backups are essential.

### Q2: What is a "Sidecar Container"?
**Answer:**
*   **Pattern**: Running a helper container alongside the main application container *within the same Pod*.
*   **Use Cases**:
    *   **Logging**: Sidecar reads log files and ships them to Splunk.
    *   **Proxy**: Service Mesh (Istio) sidecar intercepts network traffic.
    *   **Config**: Sidecar watches a Git repo and updates config files.

### Q3: Explain the difference between `ReplicaSet` and `Deployment`.
**Answer:**
*   **ReplicaSet**: Ensures N copies of a pod are running.
*   **Deployment**: A higher-level abstraction. It manages ReplicaSets.
*   **Why Deployment?**: It handles **Rolling Updates**. When you update the image, Deployment creates a *new* ReplicaSet and slowly moves pods from the *old* ReplicaSet to the new one. You rarely use ReplicaSets directly.

---

## Scenario-Based Questions

### Q4: A Pod is stuck in `CrashLoopBackOff` status. How do you debug it?
**Answer:**
1.  **Logs**: `kubectl logs <pod_name>`. (Check for app errors).
2.  **Describe**: `kubectl describe pod <pod_name>`. (Check for OOMKilled, Liveness Probe failures).
3.  **Events**: `kubectl get events`.
4.  **Previous Logs**: `kubectl logs <pod_name> --previous` (If it crashed immediately).

### Q5: You updated a Deployment image, but the new pods are failing. How do you rollback?
**Answer:**
*   **Command**: `kubectl rollout undo deployment/my-dep`.
*   **Mechanism**: K8s keeps a history of ReplicaSets. It scales up the previous ReplicaSet and scales down the current (broken) one.

---

## Behavioral / Role-Specific Questions

### Q6: A developer asks "Why can't I just ssh into the node and restart the docker container?"
**Answer:**
*   **The K8s Way**: K8s manages the state. If you manually touch a container, K8s might think it's broken and kill it, or start a duplicate.
*   **Immutability**: We treat pods as "Cattle, not Pets". If it's broken, kill the pod (`kubectl delete pod`) and let K8s start a fresh one. Don't fix it in place.
