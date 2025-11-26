# Day 20: Interview Questions & Answers

## Conceptual Questions

### Q1: Why is GitOps considered more secure than CI/CD Push?
**Answer:**
*   **Credential Exposure**: In Push, the CI server (Jenkins/GitHub Actions) needs `KUBECONFIG` with admin rights to the cluster. This is a high-value target.
*   **In Pull (GitOps)**: The cluster (ArgoCD) pulls from Git. No admin credentials leave the cluster.
*   **Audit**: Every change to the cluster is a Git Commit. You have a perfect audit trail.

### Q2: What is a "Cold Start" in Serverless (Lambda)?
**Answer:**
*   **Phenomenon**: When a Lambda function hasn't run for a while, AWS shuts down the container. The next request triggers a "Cold Start" (provision container + load runtime + start code), adding latency (100ms - 2s).
*   **Mitigation**:
    1.  **Provisioned Concurrency**: Pay to keep N instances warm.
    2.  **Lighter Runtimes**: Use Go/Rust instead of Java/Spring.

### Q3: Explain "Stateless Processes" (12-Factor App).
**Answer:**
*   **Rule**: The app should not store any state (user sessions, uploaded files) in local memory or disk.
*   **Why**: If the container crashes or scales down, that data is lost.
*   **Solution**: Store state in a backing service (Redis for sessions, S3 for files).

---

## Scenario-Based Questions

### Q4: You are using ArgoCD. A developer manually edits a Deployment in the cluster to increase replicas. What happens?
**Answer:**
*   **Drift Detected**: ArgoCD sees that the Cluster state (replicas=5) does not match Git state (replicas=3).
*   **Auto-Sync**: If enabled, ArgoCD immediately reverts the change back to 3.
*   **Lesson**: All changes *must* go through Git. No manual `kubectl edit`.

### Q5: When would you choose EC2 over Lambda?
**Answer:**
*   **Long-running tasks**: Lambda has a timeout (15 mins).
*   **Predictable Load**: If you have constant high traffic, EC2/EKS is cheaper than Lambda (which charges per request).
*   **Custom Hardware**: If you need GPUs or specific Kernel flags.

---

## Behavioral / Role-Specific Questions

### Q6: A manager asks "Why do we need Kubernetes? It seems complex. Can't we just use Heroku?"
**Answer:**
*   **Valid Point**: For simple apps, Heroku/Vercel is better (Lower Ops burden).
*   **The Tipping Point**:
    1.  **Cost**: Heroku gets expensive at scale.
    2.  **Control**: K8s allows custom networking, sidecars, and granular resource tuning.
    3.  **Vendor Lock-in**: K8s is standard. You can move from AWS to GCP easily.
*   **Strategy**: Start simple (PaaS), move to K8s only when the complexity is justified by scale or cost savings.
