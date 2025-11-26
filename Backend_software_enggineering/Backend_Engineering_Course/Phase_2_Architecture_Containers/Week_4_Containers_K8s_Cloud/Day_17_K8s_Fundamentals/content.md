# Day 17: Kubernetes Fundamentals

## 1. Why Kubernetes (K8s)?

Docker is great for *one* container. But what if you have 100 containers?
*   **Scheduling**: Which server has free RAM?
*   **Healing**: Container crashed? Restart it.
*   **Scaling**: Traffic spike? Add 10 copies.
*   **Updates**: Roll out v2.0 without downtime.

Kubernetes is the **Orchestrator** that handles this.

---

## 2. Architecture

### 2.1 The Cluster
*   **Control Plane (Master)**: The Brain.
    *   *API Server*: The entry point (REST API).
    *   *Scheduler*: Decides where to put pods.
    *   *Controller Manager*: Ensures desired state (e.g., "Keep 3 replicas").
    *   *Etcd*: The DB. Stores the cluster state (Key-Value).
*   **Worker Nodes**: The Muscle.
    *   *Kubelet*: Agent that talks to the Master.
    *   *Kube-Proxy*: Handles networking.
    *   *Container Runtime*: Docker/Containerd.

---

## 3. Core Objects

### 3.1 Pod
The atomic unit. A wrapper around one (or more) containers.
*   *Concept*: Treat a Pod like a "Logical Host". Containers in a pod share IP and Volume.
*   *Manifest*:
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: my-pod
    spec:
      containers:
      - name: nginx
        image: nginx:alpine
    ```

### 3.2 Deployment
Manages Pods. Handles Scaling and Updates.
*   *Desired State*: "I want 3 copies of Nginx".
*   *Self-Healing*: If a Pod dies, Deployment creates a new one to maintain count 3.
*   *Manifest*:
    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: nginx-dep
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: nginx
      template:
        metadata:
          labels:
            app: nginx
        spec:
          containers:
          - name: nginx
            image: nginx:alpine
    ```

### 3.3 Service
The Networking Abstraction.
*   *Problem*: Pods are ephemeral. They get new IPs when restarted.
*   *Solution*: A Service has a stable IP. It load balances traffic to matching Pods.
*   *Types*:
    1.  **ClusterIP** (Default): Internal only.
    2.  **NodePort**: Exposes port on each Node IP.
    3.  **LoadBalancer**: Asks Cloud Provider (AWS/GCP) for a real LB.

---

## 4. Kubectl
The CLI tool.
*   `kubectl get pods`
*   `kubectl apply -f deployment.yaml`
*   `kubectl logs my-pod`
*   `kubectl describe pod my-pod` (Debug magic).

---

## 5. Summary

Today we met the Captain.
*   **Cluster**: Master + Workers.
*   **Pod**: The atom.
*   **Deployment**: The manager.
*   **Service**: The phone number.

**Tomorrow (Day 18)**: We dive deeper into **ConfigMaps**, **Secrets**, and **Ingress** (Exposing apps to the world).
