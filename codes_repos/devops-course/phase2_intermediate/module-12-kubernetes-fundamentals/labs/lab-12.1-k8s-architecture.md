# Lab 12.1: Kubernetes Architecture and Setup

## Objective
Understand Kubernetes architecture and set up a local cluster using Minikube or kind.

## Prerequisites
- Docker installed
- Basic container knowledge
- 4GB+ RAM available

## Learning Objectives
- Understand K8s control plane and worker nodes
- Set up local Kubernetes cluster
- Use kubectl to interact with cluster
- Deploy first application

---

## Part 1: Kubernetes Architecture

### Control Plane Components

**API Server:** Front-end for K8s control plane  
**etcd:** Key-value store for cluster data  
**Scheduler:** Assigns pods to nodes  
**Controller Manager:** Runs controller processes  

### Worker Node Components

**Kubelet:** Agent that runs on each node  
**Kube-proxy:** Network proxy  
**Container Runtime:** Docker, containerd, CRI-O  

---

## Part 2: Install Minikube

### macOS/Linux

```bash
# Install Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/kubectl

# Start cluster
minikube start --driver=docker

# Verify
kubectl cluster-info
kubectl get nodes
```

---

## Part 3: First Deployment

### Create Nginx Deployment

```bash
kubectl create deployment nginx --image=nginx:alpine

# Check deployment
kubectl get deployments
kubectl get pods

# Expose as service
kubectl expose deployment nginx --port=80 --type=NodePort

# Get service URL
minikube service nginx --url

# Test
curl $(minikube service nginx --url)
```

### Using YAML

```yaml
# nginx-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
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
        image: nginx:alpine
        ports:
        - containerPort: 80
```

```bash
kubectl apply -f nginx-deployment.yaml
kubectl get pods -l app=nginx
```

---

## Part 4: Basic kubectl Commands

```bash
# Get resources
kubectl get pods
kubectl get deployments
kubectl get services
kubectl get all

# Describe resource
kubectl describe pod <pod-name>

# Logs
kubectl logs <pod-name>
kubectl logs -f <pod-name>  # Follow

# Execute command
kubectl exec -it <pod-name> -- sh

# Delete
kubectl delete deployment nginx
kubectl delete -f nginx-deployment.yaml
```

---

## Part 5: Namespaces

```bash
# List namespaces
kubectl get namespaces

# Create namespace
kubectl create namespace dev

# Deploy to namespace
kubectl create deployment nginx --image=nginx -n dev

# Set default namespace
kubectl config set-context --current --namespace=dev
```

---

## Success Criteria

✅ Minikube cluster running  
✅ kubectl configured and working  
✅ Deployed application successfully  
✅ Accessed application via service  
✅ Understand basic kubectl commands  

---

## Key Learnings

- **K8s is declarative** - Describe desired state in YAML
- **Pods are ephemeral** - Can be deleted/recreated anytime
- **Services provide stable endpoints** - Pods come and go
- **Namespaces isolate resources** - Multi-tenancy

**Estimated Time:** 45 minutes  
**Difficulty:** Intermediate
