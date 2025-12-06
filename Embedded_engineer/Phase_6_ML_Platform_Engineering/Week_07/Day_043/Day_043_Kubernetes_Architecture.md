# Day 43: Into the Cluster: Kubernetes Architecture for ML Engineers
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 7: Kubernetes Fundamentals

---

> **🎯 Focus Area:** Transition from running scripts on "My Machine" to running workloads on **Kubernetes (K8s)**, the operating system of the cloud. Understand the Control Plane, Worker Nodes, and the atomic unit of scheduling: the **Pod**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Map** ML concepts to K8s terms (Script -> Container, Machine -> Node, Process -> Pod).
2.  **Explain** the role of the API Server, Scheduler, and Kubelet.
3.  **Write** a declarative `pod.yaml` manifest to run a PyTorch container.
4.  **Deploy** a local cluster using `minikube` or `kind`.
5.  **Debug** a crashing pod using `kubectl logs` and `kubectl describe`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- A machine capable of running Docker/Minikube (4GB+ RAM).

### Software Environment
```bash
# Install kubectl (Client)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Install Minikube (Local Cluster)
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

### Prior Knowledge
- Docker Basics (Day 41).
- The "Cattle vs Pets" philosophy.

---

## 📖 Theoretical Foundation

### 1. The Distributed Operating System

Think of a Cluster like a single giant computer.
*   **Nodes** are CPU cores.
*   **RAM** is aggregated across nodes.
*   **Kubernetes** is the Kernel.

### 2. Architecture Components

**The Control Plane (Master Node):**
*   **API Server:** The only component you talk to (`kubectl`). It validates JSON/YAML.
*   **Etcd:** The Database. Stores the state of the cluster (Key-Value).
*   **Scheduler:** Decides *where* to put a new Pod (Node A or Node B?).
*   **Controller Manager:** The loop that says "You wanted 3 replicas, I see 2. Start one."

**The Worker Node:**
*   **Kubelet:** The agent. It talks to the API Server: "Do I have any new work?"
*   **Kube-Proxy:** Handles networking (IP Tables).
*   **Container Runtime:** Docker or Containerd (Actually runs the code).

### 3. The Pod

A **Pod** is the smallest deployable unit.
*   It is NOT just a container. It is a "logical host" that can consist of one or more containers sharing:
    *   Network Namespace (Same localhost, same IP).
    *   Storage Volumes.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Your First Manifest

We will define a Pod that runs a Jupyter Notebook server with PyTorch.

#### 📁 `manifests/jupyter-pod.yaml`
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-workspace
  labels:
    app: jupyter
    env: dev
spec:
  containers:
  - name: pytorch-notebook
    image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
    command: ["/bin/bash", "-c"]
    args:
      - pip install jupyterlab && jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token='secret'
    ports:
    - containerPort: 8888
    resources:
      requests:
        memory: "1Gi"
        cpu: "500m"
      limits:
        memory: "2Gi"
        cpu: "1000m"
```

### 👨‍💻 Lab: Deploying on Minikube

#### 📁 `src/k8s_lab.sh`
```bash
#!/bin/bash
# Day 43: K8s Lab
# Requirements: Minikube installed

# 1. Start Cluster
echo "Starting Minikube..."
minikube start --driver=docker

# 2. Check Nodes
echo "Checking Nodes..."
kubectl get nodes

# 3. Apply Manifest
echo "Creating Pod..."
kubectl apply -f manifests/jupyter-pod.yaml

# 4. Watch Creation
echo "Waiting for Pod to be Ready..."
kubectl wait --for=condition=Ready pod/ml-workspace --timeout=60s

# 5. Check Output
kubectl get pods -o wide

# 6. Port Forwarding (Access the pod from your laptop)
# K8s networks are private. We need to tunnel.
echo "Forwarding port 8888..."
# Run this in background or separate terminal:
# kubectl port-forward pod/ml-workspace 8888:8888
```

### 👨‍💻 Debugging Cheatsheet

When things break (and they will):

```bash
# 1. Is it running?
kubectl get pod ml-workspace

# 2. Why is it Pending/Error?
# "Events" section at the bottom is crucial
kubectl describe pod ml-workspace

# 3. Did the code crash?
kubectl logs ml-workspace

# 4. SSH into it (Shell access)
kubectl exec -it ml-workspace -- /bin/bash
```

---

## 🔬 Lab Exercise: "The CrashLoopBackOff"

### Task
Intentionally break a pod to learn debugging.
1.  Modify `jupyter-pod.yaml`: Change `command` to `["python", "non_existent_script.py"]`.
2.  Apply it.
3.  Observe status change: `ContainerCreating` -> `Error` -> `CrashLoopBackOff`.
4.  Run `kubectl logs ml-workspace` to see the Python error ("File not found").
5.  Run `kubectl describe pod ml-workspace` to see the Kubelet trying to restart it.

### Insight
K8s is self-healing. If a process dies, it restarts it. If it dies repeatedly, it backs off to prevent CPU thrashing.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Declarative > Imperative:** You don't say "Run this container". You say "I want a Pod with this Image", and K8s makes it happen.
2.  **Ephemerality:** Pods are mortal. If a Node dies, the Pod dies. We never rely on a specific Pod instance; we rely on **Controllers** (Deployments) to manage them (coming Day 44).
3.  **Resources:** Always set `requests` and `limits`. If you don't, one ML job can eat 100% of the Node's RAM and kill system services.

### API Summary
```bash
kubectl apply -f file.yaml
kubectl delete -f file.yaml
kubectl get pods
kubectl logs <pod>
kubectl exec -it <pod> -- bash
```

---

**Day 43 Complete** ✅

*Next: Day 44 - Workload Management - Deployments, Jobs, and how to scale stateless vs stateful ML apps.*
