# Lab 23.2: Knative Serving (Kubernetes Serverless)

## 🎯 Objective

Scale to Zero. Run serverless workloads on your own Kubernetes cluster. **Knative** provides the "Serverless" experience (auto-scaling 0->N, traffic splitting) on top of K8s.

## 📋 Prerequisites

-   Minikube running.
-   Knative Serving installed (This is heavy. Ensure 4CPUs/8GB RAM).
    *Alternative:* Use Google Cloud Run (which is managed Knative) if local resources are tight.

## 📚 Background

### Components
-   **Serving**: Manages deployments, revisions, and scaling.
-   **Eventing**: Manages event sources and subscriptions (Broker/Trigger).

---

## 🔨 Hands-On Implementation

### Part 1: Install Knative (Quickstart) 🏎️

1.  **Install Plugin:**
    `brew install knative/client/kn` (or download binary).
    `brew install knative-sandbox/kn-plugins/quickstart`.

2.  **Run Quickstart:**
    ```bash
    kn quickstart minikube
    ```
    *Note:* This sets up a Minikube cluster with Knative pre-installed. It might take 5-10 mins.

### Part 2: Deploy a Service 🚀

1.  **Create `hello.yaml`:**
    ```yaml
    apiVersion: serving.knative.dev/v1
    kind: Service
    metadata:
      name: hello
    spec:
      template:
        spec:
          containers:
          - image: gcr.io/knative-samples/helloworld-go
            env:
            - name: TARGET
              value: "Knative"
    ```

2.  **Apply:**
    ```bash
    kubectl apply -f hello.yaml
    ```

3.  **Access:**
    Get the URL:
    ```bash
    kubectl get ksvc hello
    ```
    Curl it (might need `minikube tunnel` or magic DNS).

### Part 3: Scale to Zero 📉

1.  **Wait:**
    Don't send any traffic for 60-90 seconds.

2.  **Observe:**
    ```bash
    kubectl get pods
    ```
    *Result:* No pods running. Terminating.

3.  **Wake Up:**
    Send a curl request.
    *Result:* It hangs for 2-3 seconds (Cold Start), then responds.
    `kubectl get pods` -> Running.

### Part 4: Traffic Splitting 🚦

1.  **Update Revision:**
    Change `TARGET` to "Knative v2" in YAML.
    Apply.
    This creates a new **Revision** (`hello-00002`).

2.  **Split Traffic:**
    ```bash
    kn service update hello --traffic hello-00001=50,hello-00002=50
    ```

3.  **Verify:**
    Curl 10 times. You see 50/50 split.

---

## 🎯 Challenges

### Challenge 1: Concurrency Limit (Difficulty: ⭐⭐⭐)

**Task:**
Set `containerConcurrency: 1`.
Send 10 concurrent requests (using `hey` or `ab`).
*Result:* Knative should scale up to 10 pods (1 pod per request).

### Challenge 2: Cloud Run (Difficulty: ⭐)

**Task:**
If you have a GCP account, deploy the same container to Cloud Run.
`gcloud run deploy ...`
*Observation:* The experience is identical. Cloud Run IS Knative.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```yaml
spec:
  template:
    spec:
      containerConcurrency: 1
```
</details>

---

## 🔑 Key Takeaways

1.  **Portability**: Unlike AWS Lambda, Knative runs anywhere Kubernetes runs. No vendor lock-in.
2.  **Complexity**: Installing and managing Knative is hard. Using it is easy.
3.  **Developer Experience**: `kn` CLI is very friendly compared to raw `kubectl`.

---

## ⏭️ Next Steps

We have Serverless. Now let's monitor it all.

Proceed to **Module 24: Advanced Monitoring**.
