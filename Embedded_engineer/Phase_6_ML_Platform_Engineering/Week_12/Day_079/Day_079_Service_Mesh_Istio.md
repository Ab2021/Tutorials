# Day 79: The Traffic Cop: Service Mesh (Istio) for ML
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 12: Networking for Distributed ML

---

> **🎯 Focus Area:** K8s Services are dumb (Round Robin). **Istio** is smart. Learn to use Envoy Sidecars to perform **Canary Rollouts**, **Traffic Mirroring**, and **mTLS** for your ML models.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the limitations of `kube-proxy` for advanced traffic shaping.
2.  **Deploy** Istio to a cluster and enable sidecar injection.
3.  **Configure** a Canary Release (90% Traffic to Model V1, 10% to Model V2).
4.  **Implement** Shadow Mode (Traffic Mirroring) to test Model V2 with real data safely.
5.  **Visualize** the Mesh using Kiali.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- K8s Cluster (Istio is heavy, requires ~2GB RAM overhead).

### Software Environment
- `istioctl`.

---

## 📖 Theoretical Foundation

### 1. The Sidecar Pattern
Istio injects a small container (`envoy-proxy`) into every Pod.
*   Application sends request to `locahost`.
*   Envoy intercepts it.
*   Envoy checks Rules (VirtualService).
*   Envoy routes to destination Envoy.
*   **Result:** Intelligent routing without changing app code.

### 2. Traffic Splitting
K8s Service: `random(pod_list)`.
Istio VirtualService:
```yaml
route:
- destination: model-v1
  weight: 90
- destination: model-v2
  weight: 10
```
This is essential for **A/B Testing** models in production.

### 3. Traffic Mirroring (Shadowing)
Send 100% of live traffic to V1 (User gets V1 reponse).
Send a *copy* of traffic to V2 (User ignores reponse).
**Use Case:** Verify V2 doesn't crash or have high latency on real data, with zero risk to user.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Install Istio

```bash
istioctl install --set profile=demo -y
kubectl label namespace default istio-injection=enabled
# Restart pods to get sidecars
kubectl rollout restart deploy
```

### 👨‍💻 Core Implementation: Canary Rollout

Assume we have two deployments: `resnet-v1` and `resnet-v2`.

#### 1. DestinationRule (Define Subsets)
```yaml
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: resnet
spec:
  host: resnet-service
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
```

#### 2. VirtualService (Define Weights)
```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: resnet-route
spec:
  hosts:
  - resnet-service
  http:
  - route:
    - destination:
        host: resnet-service
        subset: v1
      weight: 90
    - destination:
        host: resnet-service
        subset: v2
      weight: 10
```

### 👨‍💻 Core Implementation: Shadowing

```yaml
  http:
  - route:
    - destination:
        host: resnet-service
        subset: v1
      weight: 100
    mirror:
      host: resnet-service
      subset: v2
    mirrorPercentage:
      value: 100.0
```

---

## 🔬 Lab Exercise: "The Latency Spike"

### Task
Simulate a slow V2 model.
1.  Deploy V2 with a `time.sleep(2)` in the code.
2.  Route 50/50 traffic.
3.  Open Kiali Dashboard (`istioctl dashboard kiali`).
4.  **Observation:** You see the traffic graph. The edge to V2 turns Red (High Latency).
5.  **Circuit Breaking:** Configure Istio to automatically cut off V2 if response time > 1s (Pool Ejection).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Decoupling:** Traffic Control is separated from Infrastructure. You don't scale pods to change weights; you change YAML.
2.  **Safety:** Traffic Mirroring is the safest way to deploy a new ML model. You can compare `V1_Prediction` vs `V2_Prediction` offline to ensure accuracy matches.
3.  **Complexity:** Istio adds complexity. For simple apps, K8s Ingress is enough. For ML Platforms, Istio is often worth it.

### API Summary
```bash
istioctl analyze
kubectl get virtualservice
```

---

**Day 79 Complete** ✅

*Next: Day 80 - High Performance Networking - SR-IOV, RDMA, and bypassing the Kernel.*
