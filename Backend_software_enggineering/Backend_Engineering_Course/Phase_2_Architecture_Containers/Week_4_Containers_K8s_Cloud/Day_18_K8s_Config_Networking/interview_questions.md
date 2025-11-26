# Day 18: Interview Questions & Answers

## Conceptual Questions

### Q1: Are Kubernetes Secrets actually secure?
**Answer:**
*   **By default, No.** They are just Base64 encoded strings in etcd. Anyone with API access can decode them.
*   **Hardening**:
    1.  **Encryption at Rest**: Configure etcd to encrypt secrets.
    2.  **RBAC**: Restrict who can `get secrets`.
    3.  **External Secrets**: Use Vault or AWS Secrets Manager and inject them into K8s pods at runtime (avoiding etcd storage of secrets).

### Q2: What is the difference between an Ingress and a LoadBalancer Service?
**Answer:**
*   **LoadBalancer**: Creates a physical Cloud LB (AWS ELB) for *each* service. Expensive ($$$). Layer 4.
*   **Ingress**: Creates *one* Cloud LB that points to the Ingress Controller. The Controller then routes traffic to 50 different services based on Host/Path. Cheaper. Layer 7.

### Q3: How do you restrict traffic between two namespaces (e.g., Dev cannot talk to Prod)?
**Answer:**
*   **Network Policies**.
*   By default, all pods can talk to all pods (Flat network).
*   A `NetworkPolicy` is like a firewall rule.
    ```yaml
    kind: NetworkPolicy
    spec:
      podSelector: {} # Select all in this namespace
      policyTypes: [Ingress]
      ingress: [] # Deny all incoming
    ```

---

## Scenario-Based Questions

### Q4: You updated a ConfigMap, but the Pod is still using the old config. Why?
**Answer:**
*   **Env Vars**: If config is injected as Env Vars, they are set at *start time*. You must restart the Pod to pick up changes.
*   **Volume Mount**: If mounted as a file, K8s updates the file eventually (can take minutes). The app must support "Hot Reloading" (watching the file for changes) to see it without restart.

### Q5: You have a microservice that needs to connect to a legacy database outside the K8s cluster. How do you model this?
**Answer:**
*   **Service without Selector**.
*   Create a Service `my-legacy-db`.
*   Manually create an `Endpoints` object that points to the external IP of the DB.
*   *Benefit**: The app just calls `my-legacy-db` and doesn't care if it's inside or outside K8s.

---

## Behavioral / Role-Specific Questions

### Q6: A junior dev committed a `secret.yaml` with a real password to Git. What do you do?
**Answer:**
1.  **Rotate**: Immediately change the password in the real system. The leaked one is dead.
2.  **Scrub**: Remove the file from Git history (BFG Repo-Cleaner) or squash commits, but assume it's compromised forever.
3.  **Prevent**: Add `pre-commit` hooks to scan for secrets (Talisman, Gitleaks). Use **Sealed Secrets** or **External Secrets Operator** so we never commit raw secrets.
