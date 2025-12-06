# Day 44: Workload Management - Deployments, Jobs, and DaemonSets
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 7: Kubernetes Fundamentals

---

> **🎯 Focus Area:** Learn to manage the lifecycle of ML applications. Use **Deployments** for long-running services (Inference APIs) and **Jobs** for finite tasks (Training/Batch Processing).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Select** the correct Controller (Deployment vs Job vs StatefulSet) for a workload.
2.  **Create** a `Deployment` manifest for a scalable inference service.
3.  **Perform** a Rolling Update to change the model version without downtime.
4.  **Define** a `Job` for a training task that runs to completion.
5.  **Understand** why `DaemonSets` are crucial for GPU cluster administration.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Minikube or K8s Cluster.

### Software Environment
- `kubectl`.
- Docker image of a simple python app (we will use generic python images and inject code for demo).

---

## 📖 Theoretical Foundation

### 1. Stateless Services: `Deployment`
If you deploy Triton Inference Server, you want:
*   **Availability:** If a pod crashes, replace it.
*   **Scalability:** Run 10 copies behind a Load Balancer.
*   **Updates:** Upgrade from Triton 23.01 to 23.02 gradually.
*   **Solution:** The `Deployment` object manages a `ReplicaSet`, which manages `Pods`.

### 2. Batch Tasks: `Job`
If you run a Training script:
*   **Completion:** It must finish. It shouldn't restart forever if it succeeds.
*   **Parallelism:** You might run 5 hyperparameter search jobs.
*   **Solution:** The `Job` object creates Pods and watches for `exit code 0`.

### 3. Node Agents: `DaemonSet`
If you need to install NVIDIA Drivers or Monitoring Tools:
*   **Coverage:** You need exactly one pod per Node.
*   **Solution:** The `DaemonSet` ensures that as you add Nodes, they automatically get these pods.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Inference Deployment

We will simulate an Inference Service (stateless).

#### 📁 `manifests/inference-deploy.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference
  template:
    metadata:
      labels:
        app: inference
    spec:
      containers:
      - name: python-api
        image: python:3.9-slim
        command: ["python", "-c"]
        # Simple HTTP server simulation
        args:
          - |
            import http.server
            import socketserver
            import os
            
            PORT = 8080
            version = os.environ.get("MODEL_VERSION", "v1")
            
            class Handler(http.server.SimpleHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(f"Inference Response from {socket.gethostname()} | Model: {version}\n".encode())
            
            import socket
            with socketserver.TCPServer(("", PORT), Handler) as httpd:
                print("serving at port", PORT)
                httpd.serve_forever()
        env:
          - name: MODEL_VERSION
            value: "v1"
        ports:
        - containerPort: 8080
```

### 👨‍💻 Core Implementation: Training Job

A job that calculates Pi (simulation of intense compute) and exits.

#### 📁 `manifests/training-job.yaml`
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pi-calc-job
spec:
  # Retries before giving up
  backoffLimit: 4
  template:
    spec:
      containers:
      - name: pi
        image: perl
        command: ["perl",  "-Mbignum=bpi", "-wle", "print bpi(2000)"]
      restartPolicy: Never
```

### 👨‍💻 Lab Operations

```bash
# 1. Apply Deployment
kubectl apply -f manifests/inference-deploy.yaml

# 2. Check Replicas (Should allow 3)
kubectl get deployments
kubectl get pods -l app=inference

# 3. Scale Up
kubectl scale deployment inference-api --replicas=5

# 4. Rolling Update (Change version v1 -> v2)
kubectl set env deployment/inference-api MODEL_VERSION=v2

# 5. Watch the Rollout
kubectl rollout status deployment/inference-api

# 6. Run Job
kubectl apply -f manifests/training-job.yaml
kubectl get jobs
kubectl logs job/pi-calc-job
```

---

## 🔬 Lab Exercise: "Self-Healing Test"

### Task
Prove that Kubernetes maintains the desired state.
1.  Ensure you have 5 inference pods running (`kubectl get pods`).
2.  Delete one of them manually:
    ```bash
    kubectl delete pod inference-api-xxxxx-xxxxx
    ```
3.  Immediately run `kubectl get pods`.
4.  **Observation:** You will see the deleted pod Terminating, and a **NEW** pod "Pending" or "Running" immediately created. The Deployment controller noticed the count dropped to 4 and created 1 to get back to 5.

---

## 📝 Daily Summary

### Key Takeaways
1.  **ReplicaSets are hidden:** You rarely create ReplicaSets directly. Deployments create them for you (one RS for v1, new RS for v2 during updates).
2.  **Jobs for Glue Code:** Use K8s Jobs for things like "Data Prep script", "Database Migration", or "Batch Inference on S3 Bucket".
3.  **State is Hard:** Deployments assume statelessness. If you kill a pod and start a new one, the file saved in `/tmp` is gone. For persistent data, we need Volumes (Day 46) and StatefulSets.

### API Summary
```bash
kubectl scale deployment <name> --replicas=N
kubectl rollout history deployment <name>
kubectl rollout undo deployment <name>
```

---

**Day 44 Complete** ✅

*Next: Day 45 - K8s Networking - Services, Ingress, and how to actually reach your API.*
