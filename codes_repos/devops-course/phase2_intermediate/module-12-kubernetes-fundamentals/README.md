# Kubernetes Fundamentals

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of Kubernetes, including:
- **Architecture**: How the Control Plane and Worker Nodes collaborate.
- **Workloads**: Managing Pods, Deployments, and StatefulSets.
- **Networking**: Exposing applications using Services and Ingress.
- **Configuration**: Decoupling config from code using ConfigMaps and Secrets.
- **Storage**: Persisting data with PVs and PVCs.

---

## 📖 Theoretical Concepts

### 1. Kubernetes Architecture

Kubernetes (K8s) is a container orchestration platform.
- **Control Plane** (The Brain):
  - **API Server**: The entry point (REST API).
  - **Etcd**: The database (Key-Value store) for all cluster data.
  - **Scheduler**: Decides where to run pods based on resources.
  - **Controller Manager**: Ensures the desired state (e.g., "I want 3 replicas").
- **Worker Nodes** (The Muscle):
  - **Kubelet**: The agent that talks to the API Server and runs containers.
  - **Kube-proxy**: Maintains network rules.
  - **Container Runtime**: Docker/Containerd.

### 2. Core Objects

- **Pod**: The smallest deployable unit. Usually one container (or sidecars).
- **ReplicaSet**: Ensures a specified number of pod replicas are running.
- **Deployment**: Manages ReplicaSets. Supports rolling updates and rollbacks. Best for stateless apps.
- **StatefulSet**: For stateful apps (Databases). Stable network IDs (`web-0`, `web-1`).
- **DaemonSet**: Runs one pod on *every* node (e.g., Logging agents).

### 3. Networking

- **Service**: A stable IP address for a set of pods.
  - **ClusterIP**: Internal only.
  - **NodePort**: Exposes port on every node IP.
  - **LoadBalancer**: Provisions cloud LB (AWS ELB).
- **Ingress**: HTTP/HTTPS routing (Layer 7). Hostname based (`app.example.com`).

### 4. Configuration & Storage

- **ConfigMap**: Non-sensitive data (Env vars, config files).
- **Secret**: Sensitive data (Passwords, Keys). Base64 encoded.
- **PersistentVolume (PV)**: Storage resource (e.g., EBS volume).
- **PersistentVolumeClaim (PVC)**: A request for storage by a user.

---

## 🔧 Practical Examples

### Basic Deployment (`deployment.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
```

### Service (`service.yaml`)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
  type: ClusterIP
```

### Common Commands

```bash
# Apply config
kubectl apply -f deployment.yaml

# Get pods
kubectl get pods

# Get logs
kubectl logs -f nginx-deployment-xxxx

# Exec into pod
kubectl exec -it nginx-deployment-xxxx -- /bin/bash
```

---

## 🎯 Hands-on Labs

- [Lab 12.1: Introduction to Kubernetes (K8s)](./labs/lab-12.1-intro-k8s.md)
- [Lab 12.2: Pods & Deployments](./labs/lab-12.2-pods-deployments.md)
- [Lab 12.3: Services & Networking](./labs/lab-12.3-services-networking.md)
- [Lab 12.4: Kubernetes Capstone Project](./labs/lab-12.4-k8s-project.md)
- [Lab 12.5: Configmaps Secrets](./labs/lab-12.5-configmaps-secrets.md)
- [Lab 12.6: Persistent Volumes](./labs/lab-12.6-persistent-volumes.md)
- [Lab 12.7: Namespaces](./labs/lab-12.7-namespaces.md)
- [Lab 12.8: Ingress Controllers](./labs/lab-12.8-ingress-controllers.md)
- [Lab 12.9: Rolling Updates](./labs/lab-12.9-rolling-updates.md)
- [Lab 12.10: K8S Troubleshooting](./labs/lab-12.10-k8s-troubleshooting.md)

---

## 📚 Additional Resources

### Official Documentation
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)

### Interactive Tutorials
- [Killercoda](https://killercoda.com/kubernetes)
- [Kubernetes by Example](https://kubernetesbyexample.com/)

---

## 🔑 Key Takeaways

1.  **Declarative API**: You tell K8s *what* you want, not *how* to do it.
2.  **Self-Healing**: If a pod crashes, K8s restarts it. If a node dies, K8s moves work elsewhere.
3.  **Labels & Selectors**: The glue that connects objects (e.g., Service -> Pods).
4.  **Immutability**: Don't patch running pods. Update the Deployment image tag.

---

## ⏭️ Next Steps

1.  Complete the labs to deploy your first cluster.
2.  Proceed to **[Module 13: Advanced CI/CD](../module-13-advanced-cicd/README.md)** to automate deployments to Kubernetes.
