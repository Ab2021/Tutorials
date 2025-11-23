# Lab 22.1: GitOps Principles

## Objective
Understand and implement GitOps principles.

## Learning Objectives
- Understand GitOps workflow
- Implement declarative infrastructure
- Use Git as source of truth
- Automate deployments

---

## GitOps Workflow

```
1. Developer commits to Git
2. CI builds and tests
3. CI updates manifest repo
4. GitOps operator detects change
5. Operator applies to cluster
6. Cluster state matches Git
```

## Repository Structure

```
gitops-repo/
├── apps/
│   ├── frontend/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── backend/
│       ├── deployment.yaml
│       └── service.yaml
├── infrastructure/
│   ├── namespaces.yaml
│   └── rbac.yaml
└── kustomization.yaml
```

## Kustomize

```yaml
# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - apps/frontend/deployment.yaml
  - apps/frontend/service.yaml
  - apps/backend/deployment.yaml

images:
  - name: myapp
    newTag: v1.2.3
```

## Success Criteria
✅ Git repo structured  
✅ Declarative manifests  
✅ Kustomize configured  
✅ GitOps workflow understood  

**Time:** 35 min
