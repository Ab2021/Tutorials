# GitOps with ArgoCD

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of GitOps, including:
- **Principles**: Declarative, Versioned, Immutable, and Continuously Reconciled.
- **ArgoCD**: Installing and configuring the leading GitOps tool for Kubernetes.
- **Workflows**: Implementing automated sync, rollback, and progressive delivery.
- **Advanced**: Managing multi-cluster deployments and ApplicationSets.
- **Comparison**: Understanding ArgoCD vs Flux.

---

## 📖 Theoretical Concepts

### 1. GitOps Principles

GitOps is a paradigm where Git is the single source of truth for infrastructure and application state.
- **Declarative**: Describe the desired state in YAML, not imperative scripts.
- **Versioned**: Every change is a Git commit. Full audit trail.
- **Immutable**: Don't patch running systems. Deploy new versions.
- **Continuously Reconciled**: A controller (ArgoCD) watches Git and ensures the cluster matches.

### 2. ArgoCD Architecture

- **Application**: A CRD that points to a Git repo and a K8s cluster.
- **Sync**: The process of making the cluster match Git.
- **Sync Policy**: Manual (you click "Sync") or Automated (ArgoCD syncs on every commit).
- **Health**: ArgoCD checks if resources are healthy (Deployment has desired replicas).

### 3. Progressive Delivery (Argo Rollouts)

Traditional K8s Deployments only support Rolling Updates.
**Argo Rollouts** adds:
- **Blue/Green**: Deploy v2 alongside v1. Switch traffic 100% when ready.
- **Canary**: Gradually shift traffic (5% -> 25% -> 50% -> 100%).
- **Analysis**: Pause the rollout if error rate spikes (integrates with Prometheus).

### 4. Multi-Cluster Management

One ArgoCD instance can manage 100 clusters.
- **Cluster Secret**: Credentials to talk to remote clusters.
- **ApplicationSet**: Generate N applications from a template (e.g., Deploy to Dev, Staging, Prod).

---

## 🔧 Practical Examples

### ArgoCD Application (`app.yaml`)

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: guestbook
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/argoproj/argocd-example-apps
    targetRevision: HEAD
    path: guestbook
  destination:
    server: https://kubernetes.default.svc
    namespace: default
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

### Argo Rollouts (Canary)

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: my-app
spec:
  replicas: 5
  strategy:
    canary:
      steps:
      - setWeight: 20
      - pause: {duration: 1m}
      - setWeight: 50
      - pause: {duration: 1m}
```

### CLI Commands

```bash
# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Get admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# Create application
argocd app create guestbook --repo https://github.com/argoproj/argocd-example-apps --path guestbook --dest-server https://kubernetes.default.svc --dest-namespace default
```

---

## 🎯 Hands-on Labs

- [Lab 22.1: Gitops Principles](./labs/lab-22.1-gitops-principles.md)
- [Lab 22.2: Argocd Installation](./labs/lab-22.2-argocd-installation.md)
- [Lab 22.3: Application Deployment](./labs/lab-22.3-application-deployment.md)
- [Lab 22.4: Sync Strategies](./labs/lab-22.4-sync-strategies.md)
- [Lab 22.5: Multi Cluster](./labs/lab-22.5-multi-cluster.md)
- [Lab 22.6: Argocd Projects](./labs/lab-22.6-argocd-projects.md)
- [Lab 22.7: Automated Rollbacks](./labs/lab-22.7-automated-rollbacks.md)
- [Lab 22.8: Progressive Delivery](./labs/lab-22.8-progressive-delivery.md)
- [Lab 22.9: Flux Comparison](./labs/lab-22.9-flux-comparison.md)
- [Lab 22.10: Gitops Best Practices](./labs/lab-22.10-gitops-best-practices.md)

---

## 📚 Additional Resources

### Official Documentation
- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [Argo Rollouts](https://argoproj.github.io/argo-rollouts/)

### Community
- [CNCF ArgoCD Project](https://www.cncf.io/projects/argo/)

---

## 🔑 Key Takeaways

1.  **Git is the Source of Truth**: Never `kubectl apply` manually in production. Commit to Git.
2.  **Observability**: ArgoCD UI shows you exactly what's deployed and if it matches Git.
3.  **Rollback = Git Revert**: To undo a deployment, revert the Git commit.
4.  **Separation of Concerns**: Developers push code. CI builds images. ArgoCD deploys manifests.

---

## ⏭️ Next Steps

1.  Complete the labs to automate your K8s deployments.
2.  Proceed to **[Module 23: Serverless & Functions](../module-23-serverless-functions/README.md)** to explore event-driven architectures.
