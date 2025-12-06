# Day 94: Fire and Forget: RayJob CRD
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 14: KubeRay & Production Ray

---

> **🎯 Focus Area:** Resources are expensive. Don't leave a GPU cluster running over the weekend. Use **RayJob** to spin up a cluster *only* for the duration of your training script, then shut it down.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the lifecycle of a `RayJob` (Pending -> Running -> Complete -> Teardown).
2.  **Write** a `RayJob` manifest that embeds a script (via ConfigMap).
3.  **Configure** `shutdownAfterJobFinishes` to save money.
4.  **Retrieve** job logs using `kubectl`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- K8s Cluster.

### Software Environment
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. Ray Job Submission (Gateway)
Ray has a Job Submission API.
1.  User sends code to Head Node (HTTP 8265).
2.  Head Node saves code.
3.  Head Node starts a Driver Process.
4.  Driver uses Cluster resources.

### 2. The RayJob Controller
The KubeRay Operator automates this:
1.  Creates `RayCluster` (Spec inside RayJob).
2.  Waits for `head-svc` to be ready.
3.  Executes `ray job submit` against the head.
4.  Monitors status.
5.  If complete, deletes `RayCluster` (if configured).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Script

First, put the code in a ConfigMap so the pod can read it.

#### 📁 `manifests/job-script.yaml`
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ray-job-code
  namespace: ray-system
data:
  train.py: |
    import ray
    import time
    ray.init()
    print("Cluster Resources:", ray.cluster_resources())
    time.sleep(10)
    print("Done!")
```

### 👨‍💻 Core Implementation: The RayJob

#### 📁 `manifests/ray-job.yaml`
```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: batch-training
  namespace: ray-system
spec:
  entrypoint: "python /home/ray/code/train.py"
  shutdownAfterJobFinishes: true
  ttlSecondsAfterFinished: 60 # Keep logs for 60s, then delete cluster
  
  rayClusterSpec:
    rayVersion: '2.9.0'
    headGroupSpec:
      template:
        spec:
          containers:
          - name: ray-head
            image: rayproject/ray:2.9.0
            volumeMounts:
            - name: code
              mountPath: /home/ray/code
          volumes:
          - name: code
            configMap:
              name: ray-job-code
    workerGroupSpecs:
    - replicas: 1
      groupName: small-group
      template:
        spec:
          containers:
          - name: ray-worker
            image: rayproject/ray:2.9.0
```

### 👨‍💻 Lab Operations

1.  Apply: `kubectl apply -f manifests/job-script.yaml -f manifests/ray-job.yaml`.
2.  Watch: `kubectl get rayjob -n ray-system -w`.
    *   Status: `Pending` -> `Running` -> `Succeeded`.
3.  Check Logs:
    ```bash
    kubectl logs -l ray.io/job-name=batch-training -n ray-system
    ```
4.  Wait 60s.
5.  Check Cluster: `kubectl get raycluster`. (Should be gone).

---

## 🔬 Lab Exercise: "Job Failure"

### Task
What happens if the code crashes?
1.  Modify ConfigMap to `raise ValueError("Boom")`.
2.  Submit Job.
3.  **Observation:** Job Status: `Failed`.
4.  **Behavior:** By default, cluster stays up for debugging? Or shuts down? Check `ttlSecondsAfterFinished`.
5.  **Fix:** Set `shutdownAfterJobFinishes: false` during development so you can `kubectl exec` in and debug.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Production Standard:** `RayJob` is the recommended way to run production training. `RayCluster` is mostly for development/playgrounds.
2.  **Entrypoint:** The entrypoint command runs on the **Head Node**. Ensure the Head Node has enough CPU/RAM to compile/distribute the work, logging overhead, etc.
3.  **Observability:** The `RayJob` CRD exposes the Job ID and submission status, making it easy for Argo Workflows or Airflow to track.

### API Summary
```bash
kubectl get rayjobs
```

---

**Day 94 Complete** ✅

*Next: Day 95 - RayService CRD - Hosting Models with Ray Serve.*
