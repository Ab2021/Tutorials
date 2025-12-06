# Day 177: The Global Controller: Advanced Karmada Scheduling
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 26: Multi-Cluster & Federation

---

> **🎯 Focus Area:** You need to deploy the same app to `China` and `US`. But in China, you must use `registry.cn/image` and enable `RedactedMode`. In the US, you use `docker.io/image`. **OverridePolicies** allow context-aware customization.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Apply** ClusterOverridePolicies to modify manifests based on the target cluster's region.
2.  **Configure** FederatedHPA to scale the *global* replica count based on aggregate metrics.
3.  **Implement** Multi-Cluster Failover logic with Taint/Tolerations.
4.  **Debug** propagation failures using `karmadactl describe`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine (Kind clusters from Day 176).

### Software Environment
- `karmadactl`.

---

## 📖 Theoretical Foundation

### 1. The Context Problem
Kubernetes manifests are usually static.
But environments differ:
*   **Images:** Different registries for speed/compliance.
*   **Resources:** Cloud nodes have 64GB RAM, Edge nodes have 4GB.
*   **Config:** `ENABLE_GDPR=true` in EU, `false` elsewhere.

### 2. OverridePolicy
Intercepts the manifest *before* it is pushed to the Member Cluster.
Apply patches (JSON Patch) based on clusters.

### 3. FederatedHPA
Native HPA scales Deployment -> ReplicaSet -> Pods.
FederatedHPA scales PropagationPolicy -> Member Clusters -> Deployments.
It calculates `Total Replicas Needed` and splits them.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Override Policy

Scenario: Use distinct image repositories for US vs CN.

#### 📁 `manifests/override-image.yaml`
```yaml
apiVersion: policy.karmada.io/v1alpha1
kind: ClusterOverridePolicy
metadata:
  name: image-localization
spec:
  resourceSelectors:
    - apiVersion: apps/v1
      kind: Deployment
      name: inference-server
  overrideRules:
    - targetCluster:
        clusterNames: ["member-cn"] # Target China Cluster
      overriders:
        imageOverrider:
          - component: "Container"
            operator: replace
            value: "registry.cn/inference:v1" # Replace default image
    - targetCluster:
        clusterNames: ["member-us"]
      overriders:
        plaintext:
          - path: "/spec/template/spec/containers/0/env"
            operator: add
            value:
              - name: REGION
                value: "US"
```

### 👨‍💻 Infrastructure: Federated HPA

Scaling based on Global Traffic.

#### 📁 `manifests/fhpa.yaml`
```yaml
apiVersion: autoscaling.karmada.io/v1alpha1
kind: FederatedHPA
metadata:
  name: global-scaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-server
  minReplicas: 10
  maxReplicas: 100
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 80
```
**Logic:**
1.  Karmada collects CPU usage from `member-us` and `member-cn`.
2.  Aggregates (Average).
3.  If Avg > 80%, calculates new Desired Replicas (e.g., 120).
4.  Updates PropagationPolicy to split 120 replicas (e.g., 60 US, 60 CN).

### 👨‍💻 Core Implementation: Debugging Propagation

When things don't show up.

#### 📁 `debug_guide.sh`
```bash
# 1. Check Policy Status
kubectl get propagationpolicy

# 2. Check Binding (The link between Policy and Cluster)
kubectl get resourcebinding
# NAME              AGE   SCHEDULED   APPLIED
# nginx-binding     5m    True        False   <-- Issue: Applied=False means Push failed.

# 3. Describe Binding
kubectl describe resourcebinding nginx-binding
# Events:
# Warning  ApplyFailed  Server "member1" not reachable

# 4. Check Work (The actual object in the member namespace)
kubectl get work -n karmada-es-member1
```

---

## 🔬 Lab Exercise: "The Edge Constraint"

### Task
Deploy a heavy model to Cloud, light model to Edge.
1.  **Setup:**
    *   Cluster `cloud`: Label `type=cloud`.
    *   Cluster `edge`: Label `type=edge`.
2.  **Deployment:** `name: model-server`.
3.  **OverridePolicy:**
    *   If `type=edge`: Replace args `--model=llama-70b` with `--model=llama-7b`.
    *   If `type=edge`: Set `resources.requests.memory` to `8Gi` (Limit).
4.  **Verify:**
    *   `kubectl --context cloud get deploy model-server -o yaml` -> Shows 70B.
    *   `kubectl --context edge get deploy model-server -o yaml` -> Shows 7B.

---

## 📖 Advanced Theory: Multi-Cluster Service (MCS)
How does a pod in US call a service in EU?
Standard K8s DNS (`svc.cluster.local`) is cluster-local.
**MCS (API Spec):**
*   Export Service in Cluster A.
*   Import Service in Cluster B.
*   DNS: `my-svc.my-ns.svc.clusterset.local`.
*   Requires a flattened network (Submariner/Istio) or Gateway approach.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Don't Fork Manifests:** Do not maintain `deployment-us.yaml` and `deployment-cn.yaml`. Maintain ONE `deployment.yaml` and use OverridePolicies. This is "DRY" (Don't Repeat Yourself) for Ops.
2.  **Status Sync:** Karmada aggregates status. `kubectl get deploy` on Host shows `20/20 Ready`, meaning 10 from US and 10 from CN are all ready.
3.  **Latency:** Federated HPA is slower than Local HPA. Network RTT to Host adds delay. Use Local HPA for critical spikes, Federated HPA for capacity planning.

### API Summary
```yaml
kind: ClusterOverridePolicy
overriders:
  imageOverrider: ...
```

---

**Day 177 Complete** ✅

*Next: Day 178 - Istio Mesh Federation - Networking across Oceans.*
