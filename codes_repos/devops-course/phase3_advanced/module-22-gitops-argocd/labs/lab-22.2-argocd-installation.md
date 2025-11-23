# Lab 22.2: Argo CD Installation

## Objective
Install and configure Argo CD for GitOps.

## Learning Objectives
- Install Argo CD
- Access Argo CD UI
- Configure Git repositories
- Deploy first application

---

## Install Argo CD

```bash
# Create namespace
kubectl create namespace argocd

# Install Argo CD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for pods
kubectl wait --for=condition=Ready pods --all -n argocd --timeout=300s
```

## Access UI

```bash
# Port forward
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Get admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# Login
argocd login localhost:8080 --username admin --password <password>
```

## Add Repository

```bash
argocd repo add https://github.com/myorg/gitops-repo \
  --username myuser \
  --password mytoken
```

## Deploy Application

```yaml
# application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/gitops-repo
    targetRevision: HEAD
    path: apps/myapp
  destination:
    server: https://kubernetes.default.svc
    namespace: default
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

```bash
kubectl apply -f application.yaml
argocd app sync myapp
```

## Success Criteria
✅ Argo CD installed  
✅ UI accessible  
✅ Repository added  
✅ Application deployed  

**Time:** 40 min
