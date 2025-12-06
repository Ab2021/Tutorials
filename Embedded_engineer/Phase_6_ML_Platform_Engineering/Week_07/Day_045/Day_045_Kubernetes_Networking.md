# Day 45: Exposing AI Models: K8s Services & Ingress
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 7: Kubernetes Fundamentals

---

> **🎯 Focus Area:** Your model is running, but no one can reach it. Master **Services** to provide stable internal IPs and **Ingress** to route external HTTP traffic to your Inference APIs.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** why Pod IPs are unreliable and how Services provide a stable interface.
2.  **Differentiate** between `ClusterIP`, `NodePort`, and `LoadBalancer`.
3.  **Create** a Service to load balance traffic across 3 inference replicas.
4.  **Use** `kube-dns` to discover services by name.
5.  **Visualize** how Ingress acts as an L7 Reverse Proxy (Nginx) for the cluster.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Minikube/K8s Cluster.
- *Note:* Ingress requires an Ingress Controller enabled (`minikube addons enable ingress`).

### Software Environment
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. The Dynamic IP Problem
Pods are ephemeral. If you upgrade `inference-api`, old pods die (IP 10.1.x.x), new pods rise (IP 10.1.y.y).
If your Frontend App hardcodes the Pod IP, it breaks instantly.

### 2. The Service Abstraction
A **Service** is a persistent Virtual IP (VIP).
*   **Selector:** It looks for pods with specific labels (e.g., `app: inference`).
*   **Load Balancing:** Traffic sent to the Service VIP is round-robined to the matching Pods.
*   **DNS:** K8s automatically creates a DNS entry: `my-service.my-namespace.svc.cluster.local`.

### 3. Service Types
*   **ClusterIP (Default):** Only accessible *inside* the cluster. Good for Database <-> Backend.
*   **NodePort:** Opens a static port (e.g., 30007) on *every Node IP*. Accessible externally via `NodeIP:30007`.
*   **LoadBalancer:** Provision a Cloud Load Balancer (AWS ALB, Google CLB) that points to the NodePorts.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Service

We expose the `inference-api` deployment from Day 44.

#### 📁 `manifests/inference-service.yaml`
```yaml
apiVersion: v1
kind: Service
metadata:
  name: inference-svc
spec:
  # Expose it within cluster only
  type: ClusterIP
  selector:
    # Traffic goes to pods with this label
    app: inference
  ports:
    - protocol: TCP
      port: 80        # The port the Service listens on
      targetPort: 8080 # The port the Container listens on
```

### 👨‍💻 Lab: Service Discovery

1.  **Apply** the service:
    ```bash
    kubectl apply -f manifests/inference-service.yaml
    ```
2.  **Verify**:
    ```bash
    kubectl get svc
    # Output: inference-svc   ClusterIP   10.96.x.x   80/TCP
    ```
3.  **Test Connectivity (Internal):**
    We need to be *inside* the cluster to hit a ClusterIP. Let's spawn a temporary `curl` pod.
    ```bash
    kubectl run test-curl --image=curlimages/curl -it --restart=Never -- /bin/sh
    ```
    Inside the pod:
    ```bash
    # Test by DNS Name
    curl http://inference-svc
    
    # You should see: "Inference Response from inference-api-xyz | Model: v1"
    # Run it multiple times. You might see different hostnames (Load Balancing).
    ```

### 👨‍💻 Ingress (External Access)

If we want `http://ai.example.com` to reach our model.

#### 📁 `manifests/inference-ingress.yaml`
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: inference-ingress
  annotations:
    # Use Nginx Controller
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: ai.local
    http:
      paths:
      - path: /predict
        pathType: Prefix
        backend:
          service:
            name: inference-svc
            port:
              number: 80
```

*Note: On Minikube, you need `minikube tunnel` or configure `/etc/hosts` to map `ai.local` to the Minikube IP.*

---

## 🔬 Lab Exercise: "Zero Downtime Deploy"

### Task
Observe how Services handle Pod turnover.
1.  Run a loop in your terminal (or `test-curl` pod) hitting the service:
    ```bash
    while true; do curl -s http://inference-svc; echo; sleep 0.5; done
    ```
2.  In another terminal, update the deployment image:
    ```bash
    # Setting an env var triggers a rollout
    kubectl set env deployment/inference-api MODEL_VERSION=v3
    ```
3.  **Observation:**
    *   The loop continues printing responses.
    *   You see `Model: v1` mixed with `Model: v3`.
    *   Once rollout finishes, only `v3`.
    *   **No Connection Refused errors.** The Service removes terminating pods from the load balancer automatically.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Labels are Key:** Services rely heavily on `selector: app=inference`. If you mistype the label in the Deployment, the Service will point to 0 endpoints (Blackhole).
2.  **Cluster DNS:** You never memorize IPs. Services communicate via `http://service-name`.
3.  **Ingress vs LoadBalancer:** Use `LoadBalancer` for L4 (TCP/UDP). Use `Ingress` for L7 (HTTP/HTTPS) when you want path-based routing (`/v1`, `/v2`) and SSL termination.

### API Summary
```bash
kubectl get svc
kubectl describe svc <name> # Check "Endpoints" list to see if it found Pods
```

---

**Day 45 Complete** ✅

*Next: Day 46 - Storage in Kubernetes - Persistent Volumes (PV/PVC) and deploying a Database for our ML Metadata.*
