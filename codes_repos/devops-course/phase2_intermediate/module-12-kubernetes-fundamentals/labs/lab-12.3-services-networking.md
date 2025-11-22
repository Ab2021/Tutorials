# Lab 12.3: Services & Networking

## 🎯 Objective

Expose your application to the world. Pods have dynamic IPs (they change when they restart). **Services** provide a stable IP and DNS name to access a set of Pods.

## 📋 Prerequisites

-   Completed Lab 12.2.
-   Minikube running.

## 📚 Background

### Service Types
1.  **ClusterIP** (Default): Internal only. Accessible within the cluster.
2.  **NodePort**: Opens a port (30000-32767) on every Node. Good for development.
3.  **LoadBalancer**: Asks the Cloud Provider (AWS/GCP) for a real Load Balancer.
4.  **ExternalName**: DNS alias.

---

## 🔨 Hands-On Implementation

### Part 1: The Setup (Deployment) 📦

1.  **Reuse the Deployment from Lab 12.2:**
    Ensure `nginx-deployment` is running with 3 replicas.
    ```bash
    kubectl get pods -l app=nginx
    ```

### Part 2: ClusterIP (Internal) 🔒

1.  **Create `nginx-service.yaml`:**
    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: nginx-internal
    spec:
      selector:
        app: nginx
      ports:
        - protocol: TCP
          port: 80        # Service Port
          targetPort: 80  # Container Port
      type: ClusterIP
    ```

2.  **Apply:**
    ```bash
    kubectl apply -f nginx-service.yaml
    ```

3.  **Test Internal Access:**
    You can't curl it from your laptop. You must be *inside* the cluster.
    ```bash
    # Run a temporary pod to curl
    kubectl run curlpod --image=curlimages/curl -it --rm -- sh
    # Inside the pod:
    curl http://nginx-internal
    exit
    ```
    *Result:* You see the Nginx HTML. DNS resolution works!

### Part 3: NodePort (External-ish) 🔓

1.  **Create `nginx-nodeport.yaml`:**
    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: nginx-nodeport
    spec:
      selector:
        app: nginx
      ports:
        - port: 80
          targetPort: 80
          nodePort: 30007 # Fixed port (optional)
      type: NodePort
    ```

2.  **Apply:**
    ```bash
    kubectl apply -f nginx-nodeport.yaml
    ```

3.  **Access:**
    -   **Minikube**: `minikube service nginx-nodeport --url`.
    -   **Real Cluster**: `http://<NODE-IP>:30007`.

### Part 4: LoadBalancer (Production) ⚖️

1.  **Create `nginx-lb.yaml`:**
    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: nginx-lb
    spec:
      selector:
        app: nginx
      ports:
        - port: 80
          targetPort: 80
      type: LoadBalancer
    ```

2.  **Apply:**
    ```bash
    kubectl apply -f nginx-lb.yaml
    ```

3.  **Access (Minikube Tunnel):**
    In Minikube, LoadBalancers stay "Pending" unless you run `minikube tunnel`.
    Open a new terminal:
    ```bash
    minikube tunnel
    ```
    Now check `kubectl get svc`. You should see an `EXTERNAL-IP`.

---

## 🎯 Challenges

### Challenge 1: Service Discovery (Difficulty: ⭐⭐)

**Task:**
Inside the `curlpod`, run `nslookup nginx-internal`.
*Observation:* It resolves to `nginx-internal.default.svc.cluster.local`.
This is how microservices talk to each other (e.g., Backend -> Database).

### Challenge 2: Multi-Port Service (Difficulty: ⭐⭐⭐)

**Task:**
Create a service that exposes both port 80 (HTTP) and 443 (HTTPS).
*Hint:* The `ports` array in YAML can have multiple entries.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
```yaml
ports:
  - name: http
    port: 80
    targetPort: 80
  - name: https
    port: 443
    targetPort: 443
```
</details>

---

## 🔑 Key Takeaways

1.  **Labels match Selectors**: The Service finds pods where `metadata.labels` matches `spec.selector`.
2.  **Stable IP**: Pods die, Services survive. Always talk to the Service IP/DNS.
3.  **Ingress**: In real life, we rarely use LoadBalancer for *every* service (expensive). We use **Ingress** (Module 13).

---

## ⏭️ Next Steps

We have the components. Let's build a full app.

Proceed to **Lab 12.4: Kubernetes Capstone Project**.
