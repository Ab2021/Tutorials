# Lab 12.3: Services and Networking

## Objective
Expose applications using Kubernetes Services and understand networking concepts.

## Prerequisites
- Minikube running
- Deployment from Lab 12.2

## Learning Objectives
- Create ClusterIP, NodePort, and LoadBalancer services
- Understand service discovery
- Use DNS for service communication
- Implement Ingress for HTTP routing

---

## Part 1: Service Types

### ClusterIP (Default)

```yaml
# service-clusterip.yaml
apiVersion: v1
kind: Service
metadata:
  name: web-service
spec:
  type: ClusterIP
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 80
```

```bash
kubectl apply -f service-clusterip.yaml
kubectl get svc

# Test from within cluster
kubectl run test --image=busybox -it --rm -- wget -O- web-service
```

### NodePort

```yaml
# service-nodeport.yaml
apiVersion: v1
kind: Service
metadata:
  name: web-nodeport
spec:
  type: NodePort
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 80
    nodePort: 30080  # Optional: 30000-32767
```

```bash
kubectl apply -f service-nodeport.yaml

# Access via Minikube
minikube service web-nodeport --url
curl $(minikube service web-nodeport --url)
```

### LoadBalancer

```yaml
# service-lb.yaml
apiVersion: v1
kind: Service
metadata:
  name: web-lb
spec:
  type: LoadBalancer
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 80
```

```bash
kubectl apply -f service-lb.yaml

# In Minikube, use tunnel
minikube tunnel  # Run in separate terminal

# Get external IP
kubectl get svc web-lb
```

---

## Part 2: Service Discovery

### DNS-Based Discovery

```yaml
# backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: hashicorp/http-echo
        args:
        - "-text=Backend Response"
        ports:
        - containerPort: 5678
---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: backend
  ports:
  - port: 80
    targetPort: 5678
```

### Frontend Calling Backend

```yaml
# frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: busybox
        command: ['sh', '-c', 'while true; do wget -O- backend-service; sleep 5; done']
```

```bash
kubectl apply -f backend-deployment.yaml
kubectl apply -f frontend-deployment.yaml

# Check logs
kubectl logs -f deployment/frontend
# Output: Backend Response
```

---

## Part 3: Ingress

### Enable Ingress Addon

```bash
minikube addons enable ingress
kubectl get pods -n ingress-nginx
```

### Create Ingress Resource

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: myapp.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: backend-service
            port:
              number: 80
```

```bash
kubectl apply -f ingress.yaml

# Get Ingress IP
kubectl get ingress

# Add to /etc/hosts
echo "$(minikube ip) myapp.local" | sudo tee -a /etc/hosts

# Test
curl http://myapp.local
curl http://myapp.local/api
```

---

## Part 4: Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-policy
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
      port: 5678
```

```bash
kubectl apply -f network-policy.yaml

# Test: frontend can access backend
kubectl exec -it deployment/frontend -- wget -O- backend-service

# Test: other pods cannot access backend
kubectl run test --image=busybox -it --rm -- wget -O- backend-service
# Should timeout
```

---

## Challenges

### Challenge 1: Multi-Path Ingress

Create an Ingress with multiple paths routing to different services.

### Challenge 2: Headless Service

Create a headless service (clusterIP: None) for StatefulSet.

---

## Success Criteria

✅ Created all service types  
✅ Tested service discovery via DNS  
✅ Configured Ingress for HTTP routing  
✅ Implemented Network Policy  

---

## Key Learnings

- **ClusterIP for internal communication** - Default and most common
- **NodePort for external access** - Development/testing
- **LoadBalancer for production** - Cloud providers only
- **Ingress for HTTP/HTTPS routing** - Layer 7 load balancing
- **DNS format:** `<service-name>.<namespace>.svc.cluster.local`

**Estimated Time:** 50 minutes  
**Difficulty:** Intermediate
