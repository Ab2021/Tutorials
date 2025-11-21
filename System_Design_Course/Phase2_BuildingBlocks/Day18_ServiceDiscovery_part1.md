# Day 18 Deep Dive: Airbnb SmartStack

## 1. The Problem
*   Airbnb had a mix of Ruby, Java, Python services.
*   Client-side discovery (Ribbon) required writing libraries for all languages.
*   DNS was too slow (TTL issues).

## 2. SmartStack Architecture
*   **Sidecar Pattern:** Every container runs two extra processes: `Nerve` and `Synapse`.
*   **Nerve (Health Checker):**
    *   Runs on the *Service Node*.
    *   Checks if the service is healthy.
    *   Registers/Unregisters with Zookeeper.
*   **Synapse (Router):**
    *   Runs on the *Client Node*.
    *   Watches Zookeeper for changes.
    *   Configures a local HAProxy with the list of healthy IPs.
*   **Flow:**
    *   Client calls `localhost:port`.
    *   Local HAProxy forwards to `RemoteServiceIP`.

## 3. Benefits
*   **Language Agnostic:** Client just calls localhost.
*   **No Central LB:** Distributed routing.
*   **Fast:** Zookeeper updates propagate instantly.

## 4. Modern Equivalent: Kubernetes / Service Mesh
*   **K8s:** Uses Etcd + Kube-Proxy (IPTables).
*   **Istio:** Uses Envoy Proxy (Sidecar) to handle discovery, retries, and tracing.
