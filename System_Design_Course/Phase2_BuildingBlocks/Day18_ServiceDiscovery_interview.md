# Day 18 Interview Prep: Service Discovery

## Q1: Why is DNS bad for Service Discovery?
**Answer:**
*   **TTL:** Clients cache DNS responses. If a server dies, the client keeps trying the dead IP until TTL expires.
*   **Load Balancing:** DNS usually returns a list of IPs (Round Robin), but doesn't know server load.

## Q2: Zookeeper vs Eureka?
**Answer:**
*   **Zookeeper:** CP (Consistent). If partition happens, writes fail. Good for Leader Election.
*   **Eureka:** AP (Available). If partition happens, it serves stale data. Better for Service Discovery (It's okay to try a dead IP occasionally, but it's not okay to stop discovery).

## Q3: Explain the Sidecar Pattern.
**Answer:**
*   Deploying a helper process (proxy) alongside the main application container.
*   The app talks to the Sidecar (localhost).
*   The Sidecar handles complexity (Discovery, Retries, TLS, Metrics).
*   **Example:** Envoy, Istio.

## Q4: How does Kubernetes handle discovery?
**Answer:**
*   **Internal DNS:** `my-service.default.svc.cluster.local`.
*   **Service IP (ClusterIP):** A virtual IP that load balances across Pods using IPTables/IPVS.
