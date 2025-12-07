# Days 141-147: Week 21 - ArgoCD Advanced Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## ArgoCD Advanced Quick Reference

### App of Apps Pattern
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: root
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/org/platform
    path: apps
  destination:
    server: https://kubernetes.default.svc
    namespace: argocd
```

### ApplicationSet
```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: ml-apps
spec:
  generators:
  - list:
      elements:
      - env: dev
        cluster: dev-cluster
      - env: prod
        cluster: prod-cluster
  template:
    metadata:
      name: 'ml-api-{{env}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/org/ml-api
        path: 'envs/{{env}}'
      destination:
        server: '{{cluster}}'
```

### Image Updater
```yaml
metadata:
  annotations:
    argocd-image-updater.argoproj.io/image-list: app=myorg/myapp
    argocd-image-updater.argoproj.io/app.update-strategy: semver
    argocd-image-updater.argoproj.io/write-back-method: git
```

### Sync Waves
```yaml
metadata:
  annotations:
    argocd.argoproj.io/sync-wave: "1"  # Lower = earlier
```

---

## 📝 Week 21 Summary
| Day | Topic |
|-----|-------|
| 141 | App of Apps |
| 142 | ApplicationSet |
| 143 | Image Updater |
| 144 | Sync Waves |
| 145 | Hooks |
| 146 | Notifications |
| 147 | Project |
