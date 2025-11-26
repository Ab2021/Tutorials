# Day 18: Kubernetes Configuration & Networking

## Table of Contents
1. [Namespaces](#1-namespaces)
2. [Ingress](#2-ingress)
3. [Network Policies](#3-network-policies)
4. [Persistent Volumes](#4-persistent-volumes)
5. [StatefulSets](#5-statefulsets)
6. [DaemonSets](#6-daemonsets)
7. [Jobs & CronJobs](#7-jobs--cronjobs)
8. [RBAC](#8-rbac)
9. [Helm](#9-helm)
10. [Summary](#10-summary)

---

## 1. Namespaces

### 1.1 What are Namespaces?

**Namespace**: Virtual clusters within physical cluster.

**Use cases**:
- Multi-tenancy (team-a, team-b)
- Environments (dev, staging, prod)
- Resource isolation

### 1.2 Default Namespaces

```bash
kubectl get namespaces

NAME              STATUS   AGE
default           Active   10d
kube-system       Active   10d  # K8s internal components
kube-public       Active   10d  # Public resources
kube-node-lease   Active   10d  # Node heartbeats
```

### 1.3 Creating Namespaces

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: production
```

```bash
kubectl create namespace production

# Use namespace
kubectl apply -f deployment.yaml -n production
kubectl get pods -n production
```

### 1.4 Resource Quotas

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: compute-quota
  namespace: production
spec:
  hard:
    requests.cpu: "100"
    requests.memory: "200Gi"
    pods: "50"
```

**Prevents one team from using all cluster resources.**

---

## 2. Ingress

### 2.1 Why Ingress?

**Without Ingress** (multiple LoadBalancers):
```
Service A → LoadBalancer ($$$)
Service B → LoadBalancer ($$$)
Service C → LoadBalancer ($$$)
```

**With Ingress** (single entry point):
```
           Ingress (1 LoadBalancer)
                    ↓
  /api → Service A
 /web → Service B
/admin → Service C
```

### 2.2 Ingress Controller

**Install NGINX Ingress Controller**:
```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml
```

### 2.3 Basic Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
spec:
  rules:
  - host: myapp.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 80
      - path: /web
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80
```

**Result**:
```
http://myapp.com/api → api-service
http://myapp.com/web → web-service
```

### 2.4 TLS/SSL

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
spec:
  tls:
  - hosts:
    - myapp.com
    secretName: tls-secret
  rules:
  - host: myapp.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80
```

**Create TLS secret**:
```bash
kubectl create secret tls tls-secret \
  --cert=tls.crt \
  --key=tls.key
```

---

## 3. Network Policies

### 3.1 Default Behavior

**By default, all pods can talk to all pods** (no firewall).

### 3.2 Deny All Traffic

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
  namespace: production
spec:
  podSelector: {}  # Apply to all pods
  policyTypes:
  - Ingress
  - Egress
```

### 3.3 Allow Specific Traffic

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8000
```

**Result**: Only frontend pods can connect to backend on port 8000.

---

## 4. Persistent Volumes

### 4.1 The Problem

**Pods are ephemeral** → data lost when pod dies.

### 4.2 PersistentVolumeClaim (PVC)

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: db-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
```

**Use in Pod**:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: db-pod
spec:
  containers:
  - name: postgres
    image: postgres:15
    volumeMounts:
    - mountPath: /var/lib/postgresql/data
      name: db-storage
  volumes:
  - name: db-storage
    persistentVolumeClaim:
      claimName: db-pvc
```

**Result**: Data persists across pod restarts.

---

## 5. StatefulSets

### 5.1 Deployments vs StatefulSets

**Deployment**:
```
Pods: nginx-abc123-xyz, nginx-def456-uvw
Names random, order doesn't matter
```

**StatefulSet**:
```
Pods: db-0, db-1, db-2
Stable names, ordered creation
```

### 5.2 StatefulSet Example

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 10Gi
```

**Result**:
```
postgres-0: PVC data-postgres-0
postgres-1: PVC data-postgres-1
postgres-2: PVC data-postgres-2
```

**Each pod gets its own persistent volume.**

---

## 6. DaemonSets

### 6.1 What is a DaemonSet?

**DaemonSet**: Runs one pod per node.

**Use cases**:
- Log collectors (Fluentd on every node)
- Node monitoring (Prometheus Node Exporter)

### 6.2 Example

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluent-bit
spec:
  selector:
    matchLabels:
      app: fluent-bit
  template:
    metadata:
      labels:
        app: fluent-bit
    spec:
      containers:
      - name: fluent-bit
        image: fluent/fluent-bit:latest
        volumeMounts:
        - name: varlog
          mountPath: /var/log
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
```

**Result**: Fluent-bit pod on EVERY node collecting logs.

---

## 7. Jobs & CronJobs

### 7.1 Job (Run Once)

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration
spec:
  template:
    spec:
      containers:
      - name: migrate
        image: myapp:latest
        command: ["python", "manage.py", "migrate"]
      restartPolicy: Never
  backoffLimit: 3
```

### 7.2 CronJob (Scheduled)

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: backup
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: backup-tool:latest
            command: ["/backup.sh"]
          restartPolicy: OnFailure
```

---

## 8. RBAC

### 8.1 ServiceAccount

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pod-reader
  namespace: default
```

### 8.2 Role

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
```

### 8.3 RoleBinding

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
subjects:
- kind: ServiceAccount
  name: pod-reader
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

---

## 9. Helm

### 9.1 What is Helm?

**Helm**: Package manager for Kubernetes (like npm for Node.js).

### 9.2 Installing Chart

```bash
# Add repo
helm repo add bitnami https://charts.bitnami.com/bitnami

# Install
helm install my-release bitnami/postgresql

# List
helm list

# Uninstall
helm uninstall my-release
```

### 9.3 Custom Values

```yaml
# values.yaml
replicaCount: 3
image:
  repository: myapp
  tag: v1.0.0
service:
  type: LoadBalancer
  port: 80
```

```bash
helm install myapp ./mychart -f values.yaml
```

---

## 10. Summary

### 10.1 Key Takeaways

1. ✅ **Namespaces** - Virtual clusters, resource isolation
2. ✅ **Ingress** - HTTP routing, single entry point
3. ✅ **Network Policies** - Firewall rules for pods
4. ✅ **Persistent Volumes** - Stateful data
5. ✅ **StatefulSets** - Databases, ordered pods
6. ✅ **DaemonSets** - One pod per node
7. ✅ **Jobs/CronJobs** - Batch processing
8. ✅ **RBAC** - Access control
9. ✅ **Helm** - Package manager

### 10.2 Tomorrow (Day 19): Terraform & Infrastructure as Code

- **Terraform basics**: Providers, resources, state
- **AWS infrastructure**: VPC, EC2, RDS
- **Modules**: Reusable infrastructure
- **Remote state**: S3 backend
- **Best practices**: Workspaces, locking

See you tomorrow! 🚀

---

**File Statistics**: ~900 lines | K8s Config & Networking mastered ✅
