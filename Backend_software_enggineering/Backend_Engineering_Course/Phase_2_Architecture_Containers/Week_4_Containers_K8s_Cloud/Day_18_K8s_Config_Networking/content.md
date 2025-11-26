# Day 18: K8s Config & Networking

## 1. Configuration Management

Hardcoding DB URLs in your Docker image is a sin.
In K8s, we decouple config from code.

### 1.1 ConfigMap
Stores non-sensitive data (URLs, feature flags).
*   **Injection**:
    1.  **Env Var**: `DB_HOST=postgres-svc`
    2.  **Volume**: Mount a file `config.json` to `/app/config/`.

### 1.2 Secret
Stores sensitive data (Passwords, Keys).
*   **Encoding**: Stored as Base64.
*   **Encryption**: Etcd can encrypt secrets at rest (if configured).
*   **Usage**: Same as ConfigMap (Env Var or Volume).
    *   *Manifest*:
        ```yaml
        apiVersion: v1
        kind: Secret
        metadata:
          name: db-pass
        type: Opaque
        data:
          password: c3VwZXJzZWNyZXQ= # "supersecret" in base64
        ```

---

## 2. Networking: Ingress

We learned `NodePort` and `LoadBalancer` services. But they are Layer 4 (TCP).
What if we want Layer 7 (HTTP) routing?
*   `api.example.com` -> API Service
*   `example.com` -> Frontend Service

### 2.1 The Ingress Controller
An NGINX (or Traefik/Istio) pod that sits at the edge.
*   **Ingress Resource**: A rule definition.
    ```yaml
    apiVersion: networking.k8s.io/v1
    kind: Ingress
    metadata:
      name: my-ingress
    spec:
      rules:
      - host: api.example.com
        http:
          paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-svc
                port:
                  number: 80
    ```

---

## 3. Namespaces

Logical isolation within a cluster.
*   **Default**: `default`.
*   **Use Case**:
    *   `dev` namespace for developers.
    *   `prod` namespace for live traffic.
*   **DNS**: Services in the same namespace can talk via `http://mysvc`. Across namespaces: `http://mysvc.dev.svc.cluster.local`.

---

## 4. Summary

Today we made our apps configurable and accessible.
*   **ConfigMap/Secret**: Keep code generic, inject config at runtime.
*   **Ingress**: The smart router for HTTP traffic.
*   **Namespace**: Virtual clusters.

**Tomorrow (Day 19)**: We stop clicking buttons. We will define our entire infrastructure (K8s, VPC, DBs) as code using **Terraform**.
