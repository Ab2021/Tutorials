# Lab: Day 17 - Hello Kubernetes

## Goal
Deploy your first application to a local Kubernetes cluster. You will create a Deployment and a Service.

## Prerequisites
- `kubectl` installed.
- `minikube` OR `kind` (Kubernetes in Docker) installed.

## Directory Structure
```
day17/
├── deployment.yaml
├── service.yaml
└── README.md
```

## Step 1: Start Cluster
```bash
# Option A: Minikube
minikube start

# Option B: Kind
kind create cluster
```

## Step 2: The Deployment (`deployment.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hello-k8s
spec:
  replicas: 2
  selector:
    matchLabels:
      app: hello
  template:
    metadata:
      labels:
        app: hello
    spec:
      containers:
      - name: nginx
        image: nginx:alpine
        ports:
        - containerPort: 80
```

## Step 3: The Service (`service.yaml`)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: hello-service
spec:
  type: NodePort # Expose on a port on the node
  selector:
    app: hello
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
      nodePort: 30007
```

## Step 4: Apply & Verify

1.  **Apply**:
    ```bash
    kubectl apply -f deployment.yaml
    kubectl apply -f service.yaml
    ```

2.  **Verify**:
    ```bash
    kubectl get pods
    kubectl get svc
    ```

3.  **Access**:
    *   **Minikube**: `minikube service hello-service` (Opens browser).
    *   **Kind/Docker**: `curl http://localhost:30007` (Might require port forwarding: `kubectl port-forward svc/hello-service 8080:80`).

4.  **Scale**:
    ```bash
    kubectl scale deployment hello-k8s --replicas=5
    kubectl get pods -w
    ```
    Watch 3 new pods appear.

## Challenge
Update the image in `deployment.yaml` to `nginx:latest` and apply. Watch the rolling update:
`kubectl rollout status deployment/hello-k8s`
