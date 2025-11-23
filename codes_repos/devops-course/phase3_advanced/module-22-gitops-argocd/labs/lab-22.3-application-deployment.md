# Lab 22.3: Application Deployment

## Objective
Deploy and manage applications with Argo CD.

## Learning Objectives
- Create Argo CD applications
- Configure sync policies
- Monitor deployments
- Handle rollbacks

---

## Create Application

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: frontend
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/app-manifests
    targetRevision: main
    path: frontend
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

## Manual Sync

```bash
# Sync application
argocd app sync frontend

# Get status
argocd app get frontend

# View diff
argocd app diff frontend
```

## Rollback

```bash
# List history
argocd app history frontend

# Rollback to revision
argocd app rollback frontend 5
```

## Health Checks

```yaml
spec:
  source:
    path: frontend
  ignoreDifferences:
  - group: apps
    kind: Deployment
    jsonPointers:
    - /spec/replicas
```

## Success Criteria
✅ Application deployed via Argo CD  
✅ Auto-sync working  
✅ Rollback tested  
✅ Health monitoring active  

**Time:** 40 min
