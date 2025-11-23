# Lab 22.4: Argo CD Sync Strategies

## Objective
Configure advanced sync strategies in Argo CD.

## Learning Objectives
- Use sync waves
- Implement sync hooks
- Configure sync options
- Handle sync failures

---

## Sync Waves

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: database-config
  annotations:
    argocd.argoproj.io/sync-wave: "0"  # Deploy first
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  annotations:
    argocd.argoproj.io/sync-wave: "1"  # Deploy second
```

## Sync Hooks

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration
  annotations:
    argocd.argoproj.io/hook: PreSync
    argocd.argoproj.io/hook-delete-policy: HookSucceeded
spec:
  template:
    spec:
      containers:
      - name: migrate
        image: myapp:latest
        command: ["python", "migrate.py"]
      restartPolicy: Never
```

## Sync Options

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
spec:
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
      - PruneLast=true
      - RespectIgnoreDifferences=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

## Success Criteria
✅ Sync waves working  
✅ Hooks executing  
✅ Sync options configured  
✅ Retries functional  

**Time:** 40 min
