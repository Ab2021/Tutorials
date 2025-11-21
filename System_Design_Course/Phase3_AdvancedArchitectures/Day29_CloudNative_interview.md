# Day 29 Interview Prep: Cloud Native

## Q1: Docker vs VM?
**Answer:**
*   **VM:** Virtualizes Hardware. Has full OS (Kernel). Heavy (GBs). Slow boot.
*   **Docker:** Virtualizes OS (Kernel). Shares Host Kernel. Lightweight (MBs). Fast boot.

## Q2: What is a Sidecar?
**Answer:**
*   A container running in the same Pod as the main app.
*   Shares localhost and volumes.
*   **Uses:** Log shipping, Proxy (Envoy), Config watcher.

## Q3: Blue/Green vs Canary Deployment?
**Answer:**
*   **Blue/Green:** Run two full environments. Switch Router from Blue (Old) to Green (New). Instant rollback. Expensive (2x resources).
*   **Canary:** Rollout to 1% of users. Monitor metrics. Gradually increase to 10%, 50%, 100%. Low risk.

## Q4: How does Kubernetes Service Discovery work?
**Answer:**
*   **ClusterIP:** Stable internal IP.
*   **CoreDNS:** Maps `service-name` to `ClusterIP`.
*   **Kube-Proxy:** Intercepts traffic to ClusterIP and load balances to random Pod.
