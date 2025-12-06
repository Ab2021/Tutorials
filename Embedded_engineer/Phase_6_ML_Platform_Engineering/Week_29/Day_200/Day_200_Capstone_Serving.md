# Day 200: Titan Phase 3: The Serving Pipeline with KServe
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 29: Capstone Project Part 1

---

> **🎯 Focus Area:** We have a trained model artifact in S3. Now we must serve it to millions of users with < 100ms latency. **KServe** (built on top of Knative & Istio) gives us Scale-to-Zero, Canary Rollouts, and standardized Inference Protocols.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** the KServe Stack (Control Plane) on the `titan-inf` cluster.
2.  **Define** an `InferenceService` to serve a Scikit-Learn model and a PyTorch model.
3.  **Implement** a Canary Rollout (shift 10% traffic to v2) using KServe declarative config.
4.  **Create** an Inference Graph to chain Pre-processing -> Model -> Post-processing.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.
- Access to `titan-inf` Cluster.

### Software Environment
- `helm install kserve`.

---

## 📖 Theoretical Foundation

### 1. KServe Architecture
*   **Knative Serving:** Handles Autoscaling (Request-based), Scale-to-Zero, and Revision management.
*   **Istio:** Handles Traffic Splitting (Canary) and Ingress.
*   **ModelMesh:** Multi-Model Serving (Pack 100 models into 1 Pod) for high density.

### 2. Predictor, Transformer, Explainer
Standard InferenceService components:
*   **Predictor:** Runs the model (TensorFlow, PyTorch, XGBoost).
*   **Transformer:** Pre/Post-processing (image decoding, JSON formatting).
*   **Explainer:** Why did it predict this? (Alibi Explain / SHAP).

---

## 💻 Implementation

### 👨‍💻 Infrastructure: KServe Installation

Needs Istio and Knative first.

```bash
# 1. Install Istio (Day 57)
# 2. Install Knative Serving
kubectl apply -f https://github.com/knative/serving/releases/download/v1.10.0/serving-core.yaml

# 3. Install KServe
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve-runtimes.yaml
```

### 👨‍💻 Core Implementation: The Inference Service (Simple)

Serve a Sklearn model from S3.

#### 📁 `manifests/serving/sklearn-v1.yaml`
```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "sklearn-iris"
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: "s3://my-bucket/models/iris/v1"
```

### 👨‍💻 Core Implementation: Canary Rollout (Traffic Splitting)

Deploy v2 and send 10% traffic.

#### 📁 `manifests/serving/sklearn-v2-canary.yaml`
```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "sklearn-iris"
spec:
  predictor:
    canaryTrafficPercent: 10 # Magic Number
    model:
      modelFormat:
        name: sklearn
      storageUri: "s3://my-bucket/models/iris/v2"
```
**Mechanism:** KServe creates a new Knative Revision for v2. Istio VirtualService is updated to route 90% to Revision-1 and 10% to Revision-2.

### 👨‍💻 Core Implementation: Inference Graph (Chaining)

Dog Breed Classifier: Image -> Preprocess -> ResNet -> Postprocess -> Breed Name.

#### 📁 `manifests/serving/dog-breed-pipeline.yaml`
```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: InferenceGraph
metadata:
  name: dog-breed-pipeline
spec:
  nodes:
    root:
      routerType: Sequence
      steps:
      - serviceName: image-transformer # Decodes Base64 -> Tensor
      - serviceName: resnet-predictor  # Predicts Tensor -> Logits
      - serviceName: breed-transformer # Maps Logits -> "Golden Retriever"
```

---

## 🔬 Lab Exercise: "The Cold Start"

### Task
Measure Latency.
1.  **State:** No traffic for 5 mins. Pods = 0.
2.  **Request:** Send 1 request.
3.  **Observation:**
    *   Knative Activator buffers request.
    *   Autoscaler launches Pod.
    *   Pod pulls image + model (Time: ~5s).
    *   Response returned. Total Latency: 5s. (Too slow for User).
4.  **Fix:** `minReplicas: 1` (Keep warm) OR optimize startup time (smaller image).

---

## 📖 Advanced Theory: V2 Inference Protocol
KServe supports the industry standard **KServe V2 Protocol** (GRPC/HTTP).
*   **Endpoints:** `/v2/models/{name}/infer`, `/v2/models/{name}/ready`.
*   **Payload:** Standardized JSON with `inputs: [{name: "input0", shape: [1, 3], datatype: "FP32", data: [...]}]`.
*   **Benefit:** Switch backends (Triton vs TorchServe vs ONNX) without changing client code.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Serverless for ML:** KServe provides the "Serverless" experience for ML. Data Scientists don't manage Pods; they manage "Models".
2.  **Canary is Critical:** Never replace v1 with v2 instantly. Use traffic splitting to catch regression bugs (e.g., v2 is 50% slower) before they affect all users.
3.  **Transformer Pattern:** Decouple "Math" (GPU) from "Logic" (CPU). Run Transformers on CPUs, run Predictors on GPUs.

### API Summary
```bash
kubectl get isvc
kubectl get revisions
```

---

**Day 200 Complete** ✅

*Next: Day 201 - Capstone Part 5 - The Observability Pipeline.*
