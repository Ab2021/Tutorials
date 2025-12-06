# Day 47: Configuration & Secrets - The Twelve-Factor App
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 7: Kubernetes Fundamentals

---

> **🎯 Focus Area:** separate Code from Config. Use **ConfigMaps** for hyperparameters and **Secrets** for credentials to make your ML Applications portable and secure.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Create** ConfigMaps from literals and files.
2.  **Create** Secrets and understand Base64 encoding behavior.
3.  **Inject** configuration into Pods as Environment Variables (`valueFrom`).
4.  **Mount** configuration as Files (Volume Mounts) for complex configs (e.g., `config.json`).
5.  **Refactor** the Postgres Deployment to remove hardcoded passwords.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Minikube/K8s Cluster.

### Software Environment
- `kubectl`.
- `base64` terminal tool.

---

## 📖 Theoretical Foundation

### 1. The Separation of Concerns
If you bake `BATCH_SIZE=32` into your Docker Image, you must rebuild the image to change it to 64.
K8s allows "Late Binding": Image is generic; Config is injected at runtime.

### 2. ConfigMap vs Secret
*   **ConfigMap:** Plain text. Stored in Etcd. Visible in `kubectl describe`. Use for URLs, tuning params.
*   **Secret:** Base64 encoded. Stored in Etcd (optionally encrypted at rest). Obscured in `kubectl describe` (shows bytes). Use for passwords, API keys.

### 3. Usage Patterns
1.  **Env Injection:** Good for simple strings (`$DB_HOST`).
2.  **Volume Mount:** Good for config files (`/etc/training/hyperparams.json`). If the ConfigMap updates, the file inside the pod updates automatically (eventually).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Creating Resources

#### 📁 `manifests/config-layer.yaml`
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-hyperparams
data:
  # Key-Value pairs
  LEARNING_RATE: "0.001"
  BATCH_SIZE: "64"
  OPTIMIZER: "Adam"
---
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
stringData:
  # 'stringData' allows us to write plain text, K8s will base64 encode it for us.
  # If using 'data', you must provide base64 strings.
  password: "SuperSecurePassword123!"
```

### 👨‍💻 Core Implementation: Consuming in Pod

#### 📁 `manifests/secure-pod.yaml`
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-trainer
spec:
  containers:
  - name: trainer
    image: python:3.9
    command: ["/bin/sh", "-c", "env && sleep 3600"]
    env:
      # 1. Load from ConfigMap
      - name: LR
        valueFrom:
          configMapKeyRef:
            name: ml-hyperparams
            key: LEARNING_RATE
      
      # 2. Load from Secret
      - name: DB_PASS
        valueFrom:
          secretKeyRef:
            name: db-credentials
            key: password
```

### 👨‍💻 Lab Operations

```bash
# 1. Apply Configs
kubectl apply -f manifests/config-layer.yaml

# 2. Verify Secret encoding
kubectl get secret db-credentials -o yaml
# You will see 'password: U3VwZXRTZWN...'. 
# Decode it:
echo "U3VwZXRTZWN1cmVQYXNzd29yZDEyMyE=" | base64 --decode

# 3. Apply Pod
kubectl apply -f manifests/secure-pod.yaml

# 4. Check Env Vars inside Pod
kubectl logs secure-trainer
# You should see LR=0.001 and DB_PASS=SuperSecurePassword123!
```

### 👨‍💻 Advanced: Mounting a Config File

Sometimes your app expects a `config.json`.

```yaml
# In ConfigMap
data:
  config.json: |
    {
      "model": "resnet50",
      "layers": 50
    }

# In Pod Spec
volumes:
  - name: config-vol
    configMap:
      name: my-config
containers:
  - volumeMounts:
    - name: config-vol
      mountPath: /app/config
```
This results in a file `/app/config/config.json` inside the container.

---

## 🔬 Lab Exercise: "Immutable Config"

### Task
Update a ConfigMap and see if the Pod notices.
1.  Edit `ml-hyperparams` ConfigMap: change `LEARNING_RATE` to `0.01`.
2.  Check the running Env Var pod.
    *   **Env Vars:** DO NOT UPDATE. They are set at start time. You must restart the pod.
    *   **Volume Mounts:** DO UPDATE (Takes ~60s cache sync).

### Insight
For simple Env Vars, this reinforces the "Immutable Infrastructure" pattern. Don't change config of running pods; perform a Rollout (Restart) to apply new config safely.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Don't check in Secrets:** Never commit `secret.yaml` with real passwords to Git. Use tools like `SealedSecrets` or `ExternalSecrets` (Vault integration) in production.
2.  **StringData:** Use `stringData` in manifests for convenience, but remember `kubectl get -o yaml` reveals the base64 string, which is trivially reversible. Secrets are about *obfuscation* and intent, not encryption (unless Etcd encryption is on).
3.  **Portability:** ConfigMaps allow you to deploy the *exact same* Docker Image to Dev (DEBUG=True) and Prod (DEBUG=False).

### API Summary
```bash
kubectl create configmap my-config --from-literal=key=val
kubectl create secret generic my-secret --from-literal=pass=123
```

---

**Day 47 Complete** ✅

*Next: Day 48 - RBAC (Role Based Access Control) - "Who can delete my production database?"*
