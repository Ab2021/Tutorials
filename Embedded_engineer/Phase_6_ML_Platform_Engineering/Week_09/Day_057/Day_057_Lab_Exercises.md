# Days 57-63: Week 9 - Helm, Operators & GitOps Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 57: Helm Basics

```bash
# Create chart
helm create ml-platform

# Install
helm install ml-platform ./ml-platform

# Upgrade
helm upgrade ml-platform ./ml-platform --set replicas=5
```

---

## Day 58: Helm Advanced

```yaml
# values.yaml
replicaCount: 3
image:
  repository: myorg/ml-api
  tag: v1.0.0
resources:
  limits:
    nvidia.com/gpu: 1
```

---

## Day 59: Operators

```bash
# View installed operators
kubectl get csv -A

# Example: Install KubeRay operator
helm install kuberay-operator kuberay/kuberay-operator
```

---

## Day 60: KubeRay

```yaml
# RayCluster
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ml-cluster
spec:
  headGroupSpec:
    rayStartParams:
      dashboard-host: '0.0.0.0'
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.9.0-py310-gpu
  workerGroupSpecs:
  - replicas: 4
    groupName: gpu-workers
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.9.0-py310-gpu
          resources:
            limits:
              nvidia.com/gpu: 1
```

---

## Day 61: GitOps Principles

```bash
# Git repo structure
├── apps/
│   ├── ml-api/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── ray-cluster/
│       └── raycluster.yaml
└── kustomization.yaml
```

---

## Day 62: ArgoCD

```bash
# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Create application
argocd app create ml-platform \
  --repo https://github.com/org/ml-platform \
  --path apps \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace ml
```

---

## Day 63: Week 9 Project

```yaml
# ApplicationSet for multi-env
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: ml-platform
spec:
  generators:
  - list:
      elements:
      - env: dev
      - env: staging
      - env: prod
  template:
    spec:
      project: default
      source:
        repoURL: https://github.com/org/ml-platform
        path: 'envs/{{env}}'
      destination:
        server: https://kubernetes.default.svc
        namespace: 'ml-{{env}}'
```

---

## 📝 Week 9 Summary
| Day | Topic | Tool |
|-----|-------|------|
| 57 | Helm Basics | helm create |
| 58 | Helm Advanced | Values, hooks |
| 59 | Operators | CRDs, Controllers |
| 60 | KubeRay | RayCluster CRD |
| 61 | GitOps | Git as source |
| 62 | ArgoCD | App sync |
| 63 | Project | Multi-env |
