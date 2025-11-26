# Day 20: GitOps & Cloud Platform Services

## 1. GitOps: The Holy Grail

We have K8s (Manifests) and Terraform (HCL). Where do we store them? **Git**.
GitOps = Infrastructure as Code + Pull Requests + Convergence.

### 1.1 Push vs Pull
*   **Push (Traditional CI/CD)**: Jenkins runs `kubectl apply`.
    *   *Risk*: Jenkins needs Admin access to the cluster. If Jenkins is hacked, Cluster is doomed.
*   **Pull (GitOps)**: An agent (ArgoCD) *inside* the cluster watches Git. When Git changes, ArgoCD applies the change.
    *   *Security*: Cluster doesn't expose credentials. It pulls.
    *   *Drift*: If someone manually changes the cluster, ArgoCD detects it and reverts it to match Git.

### 1.2 ArgoCD
The standard GitOps tool for K8s.
*   **UI**: Visualizes the entire application tree.
*   **Sync**: Auto-syncs Git -> Cluster.

---

## 2. Cloud Platform Services

Don't reinvent the wheel. Use Managed Services.

### 2.1 Compute
*   **EC2 (VM)**: You manage OS, patching. (Good for legacy).
*   **EKS/GKE (K8s)**: They manage Control Plane, you manage Workers. (Good for microservices).
*   **Lambda (Serverless)**: You upload code. No servers. (Good for event-driven tasks).

### 2.2 Database
*   **RDS / Cloud SQL**: Managed Postgres/MySQL. Automated backups, patching, HA.
*   **DynamoDB / Firestore**: Serverless NoSQL. Infinite scale.

### 2.3 Storage
*   **S3 / GCS**: Object Storage. Infinite capacity. Cheap.
    *   *Use Case*: User uploads, Backups, Static Website Hosting.

---

## 3. The 12-Factor App (Cloud Native)

To survive in the cloud, apps must follow rules:
1.  **Codebase**: One repo, many deploys.
2.  **Dependencies**: Explicitly declared (Docker).
3.  **Config**: In Env Vars (ConfigMaps).
4.  **Backing Services**: Treat DBs as attached resources.
5.  **Build, Release, Run**: Strict separation.
6.  **Processes**: Stateless.
7.  **Port Binding**: Export services via port binding.
8.  **Concurrency**: Scale out via processes (replicas).
9.  **Disposability**: Fast startup, graceful shutdown.
10. **Dev/Prod Parity**: Keep them similar.
11. **Logs**: Treat logs as event streams (stdout).
12. **Admin Processes**: Run admin/management tasks as one-off processes.

---

## 4. Summary

Today we reached the Cloud Native summit.
*   **GitOps**: Git is the single source of truth.
*   **Managed Services**: Focus on code, not patching DBs.
*   **12-Factor**: The constitution of modern apps.

**Week 4 Wrap-Up**:
We have covered:
1.  Docker (Containerization).
2.  Kubernetes (Orchestration).
3.  K8s Config & Networking.
4.  Terraform (IaC).
5.  GitOps & Cloud Services.

**Next Week (Week 5)**: We return to code. We will master **Advanced API Design** (GraphQL, gRPC) and learn how to handle massive scale.
