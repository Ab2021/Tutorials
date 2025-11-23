# Lab 22.5: Multi-Cluster Management

## Objective
Manage multiple Kubernetes clusters with Argo CD.

## Learning Objectives
- Add multiple clusters
- Deploy to different clusters
- Use ApplicationSets
- Implement cluster generators

---

## Add Clusters

```bash
# Add cluster
argocd cluster add my-cluster-context

# List clusters
argocd cluster list

# Remove cluster
argocd cluster rm https://my-cluster
```

## Deploy to Multiple Clusters

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp-prod
spec:
  destination:
    server: https://prod-cluster
    namespace: default
---
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp-staging
spec:
  destination:
    server: https://staging-cluster
    namespace: default
```

## ApplicationSet

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: cluster-apps
spec:
  generators:
  - list:
      elements:
      - cluster: prod
        url: https://prod-cluster
      - cluster: staging
        url: https://staging-cluster
  template:
    metadata:
      name: '{{cluster}}-myapp'
    spec:
      source:
        repoURL: https://github.com/myorg/apps
        path: apps/myapp
      destination:
        server: '{{url}}'
        namespace: default
```

## Success Criteria
✅ Multiple clusters added  
✅ Apps deployed to each cluster  
✅ ApplicationSets working  

**Time:** 45 min
