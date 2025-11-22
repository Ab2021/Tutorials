# Lab 21.3: Service Mesh with Istio

## 🎯 Objective

Control the traffic. Kubernetes controls *deployment*, but Istio controls *networking*. You will install Istio, inject sidecars, and perform a **Canary Deployment** (Traffic Splitting).

## 📋 Prerequisites

-   Minikube running (`minikube start --cpus 4 --memory 8192`).
-   Istio CLI installed (`istioctl`).

## 📚 Background

### The Sidecar Pattern
Istio injects a proxy (Envoy) into every Pod.
App Container <-> Envoy Proxy <-> Network.
This allows Istio to intercept all traffic, encrypt it (mTLS), and route it (Canary).

---

## 🔨 Hands-On Implementation

### Part 1: Install Istio ⛵

1.  **Install:**
    ```bash
    istioctl install --set profile=demo -y
    ```

2.  **Enable Injection:**
    Label the default namespace so Istio knows to inject sidecars.
    ```bash
    kubectl label namespace default istio-injection=enabled
    ```

### Part 2: Deploy Sample App (Bookinfo) 📚

1.  **Deploy:**
    ```bash
    kubectl apply -f https://raw.githubusercontent.com/istio/istio/master/samples/bookinfo/platform/kube/bookinfo.yaml
    ```

2.  **Verify Sidecars:**
    ```bash
    kubectl get pods
    ```
    *Result:* `2/2` in the Ready column. (1 App + 1 Sidecar).

### Part 3: The Gateway 🚪

1.  **Create Gateway:**
    ```bash
    kubectl apply -f https://raw.githubusercontent.com/istio/istio/master/samples/bookinfo/networking/bookinfo-gateway.yaml
    ```

2.  **Access:**
    `minikube tunnel` (in separate terminal).
    Go to `http://localhost/productpage`.

### Part 4: Traffic Splitting (Canary) 🐤

We have 3 versions of `reviews` (v1: no stars, v2: black stars, v3: red stars).
By default, K8s round-robins (33% each). Let's force 100% to v1, then shift 10% to v2.

1.  **Destination Rules (Define Subsets):**
    ```bash
    kubectl apply -f https://raw.githubusercontent.com/istio/istio/master/samples/bookinfo/networking/destination-rule-all.yaml
    ```

2.  **Virtual Service (Route 100% to v1):**
    Create `virtual-service-all-v1.yaml`:
    ```yaml
    apiVersion: networking.istio.io/v1alpha3
    kind: VirtualService
    metadata:
      name: reviews
    spec:
      hosts:
      - reviews
      http:
      - route:
        - destination:
            host: reviews
            subset: v1
    ```
    Apply it. Refresh browser. You ONLY see v1 (no stars).

3.  **Shift 10% to v2:**
    Update YAML:
    ```yaml
    ...
      http:
      - route:
        - destination:
            host: reviews
            subset: v1
          weight: 90
        - destination:
            host: reviews
            subset: v2
          weight: 10
    ```
    Apply. Refresh 10 times. You should see black stars once.

---

## 🎯 Challenges

### Challenge 1: Fault Injection (Difficulty: ⭐⭐⭐)

**Task:**
Inject a 5-second delay for user `jason`.
*Hint:* `fault: delay: percentage: 100 fixedDelay: 5s`.
*Goal:* Test if the frontend handles backend latency gracefully.

### Challenge 2: mTLS Strict Mode (Difficulty: ⭐⭐)

**Task:**
Verify that mTLS is enabled.
Try to curl the `reviews` pod from a non-injected pod (e.g., a plain busybox).
It should fail (connection reset) if Strict mTLS is on.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```yaml
- match:
    - headers:
        end-user:
          exact: jason
  fault:
    delay:
      percentage:
        value: 100
      fixedDelay: 5s
```
</details>

---

## 🔑 Key Takeaways

1.  **Observability**: Istio (via Kiali) generates a map of your microservices automatically.
2.  **Security**: mTLS encrypts traffic between pods without changing application code.
3.  **Resilience**: You can configure Retries, Timeouts, and Circuit Breakers in YAML.

---

## ⏭️ Next Steps

We have a Service Mesh. Now let's automate everything with GitOps.

Proceed to **Module 22: GitOps & ArgoCD**.
