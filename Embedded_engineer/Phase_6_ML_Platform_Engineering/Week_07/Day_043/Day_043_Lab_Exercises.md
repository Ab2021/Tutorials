# Days 43-49: Week 7 - Kubernetes Fundamentals Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 43: K8s Architecture

```bash
# Deploy local cluster
kind create cluster --name ml-platform

# Verify components
kubectl get nodes
kubectl get pods -n kube-system
```

---

## Day 44: Workload Management

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: api
        image: python:3.10
        command: ["python", "-m", "http.server", "8080"]
        ports:
        - containerPort: 8080
```

---

## Day 45: Networking

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api
spec:
  type: ClusterIP
  selector:
    app: ml-api
  ports:
  - port: 80
    targetPort: 8080
---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api
spec:
  rules:
  - host: ml.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-api
            port:
              number: 80
```

---

## Day 46: Storage

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

---

## Day 47: Configuration

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
data:
  MODEL_PATH: "/models/bert"
  BATCH_SIZE: "32"
---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-credentials
type: Opaque
stringData:
  API_KEY: "super-secret-key"
```

---

## Day 48: RBAC

```yaml
# role.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ml-developer
  namespace: ml-team
rules:
- apiGroups: ["", "apps"]
  resources: ["pods", "deployments"]
  verbs: ["get", "list", "create", "delete"]
```

---

## Day 49: Week 7 Project

```bash
# Deploy microservices app
kubectl apply -f manifests/
kubectl get pods -w
kubectl port-forward svc/ml-api 8080:80
curl localhost:8080
```

---

## 📝 Week 7 Summary
| Day | Topic | Key Resource |
|-----|-------|--------------|
| 43 | Architecture | Nodes, Pods |
| 44 | Workloads | Deployment, Job |
| 45 | Networking | Service, Ingress |
| 46 | Storage | PVC, StorageClass |
| 47 | Config | ConfigMap, Secret |
| 48 | RBAC | Role, RoleBinding |
| 49 | Project | Full deployment |
