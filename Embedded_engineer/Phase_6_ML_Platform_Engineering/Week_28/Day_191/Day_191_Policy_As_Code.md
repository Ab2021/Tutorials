# Day 191: The Guardrails: Policy as Code with Kyverno
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 28: Platform Engineering Practices

---

> **🎯 Focus Area:** You gave developers self-service access via Backstage (Day 190). Now they are deploying "bitcoin-miners" running as "root". You need **Policy as Code** to block bad resources *before* they enter the cluster.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Compare** OPA (Rego) vs Kyverno (YAML) for Kubernetes Admission Control.
2.  **Write** a Kyverno Policy to Validating resources (e.g., "Must have `cost-center` label").
3.  **Write** a Kyverno Policy to Mutate resources (e.g., "Inject sidecar automatically").
4.  **Audit** existing resources to find violations without breaking production.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `helm install kyverno`.
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. Admission Controllers
Kubernetes API Flow:
Request -> AuthN -> AuthZ -> **Mutating Admission** -> Schema Validation -> **Validating Admission** -> Etcd.
*   **Mutation:** Change the object (e.g., set default CPU limit).
*   **Validation:** Accept or Reject (e.g., Reject checks running as root).

### 2. OPA vs Kyverno
*   **OPA (Gatekeeper):** Uses **Rego** language. Powerful, complex, general-purpose (works with Terraform, Envoy).
*   **Kyverno:** Uses **YAML**. Designed specifically for Kubernetes. Easier learning curve. We will use Kyverno.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Use Case 1 - Enforce Labels

Ensure every deployment has an owner for billing.

#### 📁 `manifests/require-labels.yaml`
```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-labels
spec:
  validationFailureAction: enforce # Block it. (Use 'audit' for warning)
  background: true # Scan existing resources
  rules:
  - name: check-owner-label
    match:
      any:
      - resources:
          kinds:
          - Pod
    validate:
      message: "The label 'owner' is required."
      pattern:
        metadata:
          labels:
            owner: "?*" # Wildcard: Must exist and not be empty
```

### 👨‍💻 Infrastructure: Use Case 2 - Secure Supply Chain

Block images not from our Trusted Registry.

#### 📁 `manifests/restrict-image-registry.yaml`
```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: restrict-registry
spec:
  validationFailureAction: enforce
  rules:
  - name: validate-registry
    match:
      any:
      - resources:
          kinds:
          - Pod
    validate:
      message: "Images must pull from harbor.mycorp.com"
      pattern:
        spec:
          containers:
          - image: "harbor.mycorp.com/*"
```

### 👨‍💻 Infrastructure: Use Case 3 - Mutation (Auto-Injection)

Automatically add a "cost-agent" sidecar to every ML Job.

#### 📁 `manifests/inject-sidecar.yaml`
```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: inject-sidecar
spec:
  rules:
  - name: inject-agent
    match:
      any:
      - resources:
          kinds:
          - Job
          selector:
            matchLabels:
              type: ml-training
    mutate:
      patchStrategicMerge:
        spec:
          template:
            spec:
              containers:
              - name: cost-agent
                image: cost-agent:v1
```

---

## 🔬 Lab Exercise: " The Rejection"

### Task
Test the Guardrails.
1.  **Apply** the `require-labels` policy.
2.  **Create** `bad-pod.yaml` (No labels).
3.  **Run:** `kubectl apply -f bad-pod.yaml`.
4.  **Result:**
    ```text
    Error from server: error when creating "bad-pod.yaml": admission webhook "validate.kyverno.svc-fail" denied the request: 
    resource Pod/default/bad-pod was blocked due to the following policies:
    require-labels:
      check-owner-label: "The label 'owner' is required."
    ```
5.  **Fix:** Add `labels: owner: me`. Apply succeeds.

---

## 📖 Advanced Theory: Generation
Kyverno can also **Generate** resources.
*   **Use Case:** Multi-Tenancy.
*   **Trigger:** When a new `Namespace` is created.
*   **Action:** Generate a `NetworkPolicy` (Deny-All), a `ResourceQuota`, and a `RoleBinding` inside that new namespace automatically.
*   **Result:** Instant, secure environment provisioning.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Audit Mode First:** Never deploy a new policy as `enforce` immediately. You will break Prod. Deploy as `audit`, check the Policy Reports, fix existing violations, THEN switch to `enforce`.
2.  **Shift Left:** Run these checks in CI (using `kyverno-cli` or `conftest`) so developers get feedback *before* they push to the cluster.
3.  **Governance:** Compliance auditors (SOC2) love "Policy as Code". You can prove exactly what controls are in place by showing the Git repo.

### API Summary
```yaml
validationFailureAction: enforce
validate:
  pattern: ...
```

---

**Day 191 Complete** ✅

*Next: Day 192 - Infrastructure as Code - Drift Detection.*
