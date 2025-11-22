# Lab 12.2: Pods and Deployments

## Objective
Create and manage Pods and Deployments in Kubernetes.

## Prerequisites
- Minikube running (Lab 12.1)
- kubectl configured

## Learning Objectives
- Create Pods directly and via Deployments
- Understand Pod lifecycle
- Scale deployments
- Perform rolling updates
- Rollback deployments

---

## Part 1: Creating Pods

### Imperative Approach

```bash
kubectl run nginx --image=nginx:alpine
kubectl get pods
kubectl describe pod nginx
kubectl logs nginx
kubectl delete pod nginx
```

### Declarative Approach

```yaml
# pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
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
kubectl apply -f pod.yaml
kubectl get pods
```

---

## Part 2: Deployments

### Create Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
```

```bash
kubectl apply -f deployment.yaml
kubectl get deployments
kubectl get pods
kubectl get rs  # ReplicaSets
```

---

## Part 3: Scaling

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment web-app --replicas=5

# Verify
kubectl get pods

# Scale down
kubectl scale deployment web-app --replicas=2
```

### Auto-Scaling

```bash
kubectl autoscale deployment web-app --min=2 --max=10 --cpu-percent=80
kubectl get hpa  # Horizontal Pod Autoscaler
```

---

## Part 4: Rolling Updates

### Update Image

```bash
# Update to new version
kubectl set image deployment/web-app nginx=nginx:1.22

# Watch rollout
kubectl rollout status deployment/web-app

# Check history
kubectl rollout history deployment/web-app
```

### Update via YAML

```yaml
# Update deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: nginx
        image: nginx:1.22  # Changed from 1.21
```

```bash
kubectl apply -f deployment.yaml
kubectl rollout status deployment/web-app
```

---

## Part 5: Rollback

### Rollback to Previous Version

```bash
# Rollback
kubectl rollout undo deployment/web-app

# Rollback to specific revision
kubectl rollout undo deployment/web-app --to-revision=1

# Check status
kubectl rollout status deployment/web-app
```

---

## Part 6: Pod Lifecycle

### Lifecycle Hooks

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: lifecycle-demo
spec:
  containers:
  - name: nginx
    image: nginx
    lifecycle:
      postStart:
        exec:
          command: ["/bin/sh", "-c", "echo Hello from postStart > /usr/share/message"]
      preStop:
        exec:
          command: ["/bin/sh", "-c", "nginx -s quit; while killall -0 nginx; do sleep 1; done"]
```

---

## Challenges

### Challenge 1: Blue-Green Deployment

Implement blue-green deployment using labels and services.

### Challenge 2: Init Containers

Add an init container that waits for a database to be ready.

---

## Success Criteria

✅ Created Pods and Deployments  
✅ Scaled deployments up and down  
✅ Performed rolling update  
✅ Successfully rolled back  
✅ Understand Pod lifecycle  

---

## Key Learnings

- **Never create Pods directly in production** - Use Deployments
- **Deployments manage ReplicaSets** - ReplicaSets manage Pods
- **Rolling updates are zero-downtime** - Old pods replaced gradually
- **Resource limits prevent resource starvation** - Always set them

**Estimated Time:** 50 minutes  
**Difficulty:** Intermediate
