# Day 29 Deep Dive: Kubernetes Architecture

## 1. Control Plane (Master)
*   **API Server:** The brain. REST API. Only component that talks to Etcd.
*   **Etcd:** The memory. Distributed KV store. Stores cluster state.
*   **Scheduler:** Decides where to run a Pod (based on CPU/RAM).
*   **Controller Manager:** Reconciles state (e.g., "I want 3 replicas, currently 2. Create 1").

## 2. Data Plane (Worker Node)
*   **Kubelet:** Agent. Talks to API Server. Manages containers (Docker/Containerd).
*   **Kube-Proxy:** Network proxy. Handles Service IP routing (IPTables).
*   **Pod:** Smallest unit. One or more containers sharing Network/Storage.

## 3. Key Concepts
*   **Deployment:** Manages Replicas. Handles Rolling Updates.
*   **Service:** Stable IP/DNS for a set of Pods.
*   **Ingress:** HTTP Load Balancer (L7) to expose services to outside.
*   **ConfigMap/Secret:** Inject configuration.

## 4. The Reconciliation Loop
*   **Declarative:** You say "I want X".
*   **Loop:**
    1.  Observe current state.
    2.  Compare with desired state.
    3.  Act to fix difference.
*   **Self-Healing:** If a node dies, Controller notices "Current < Desired" and creates Pods elsewhere.
