# Day 157: Keeping Secrets: Confidential Computing
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 23: Security & Governance

---

> **🎯 Focus Area:** You committed `.env` to GitHub. Now your AWS bill is $50,000. **Secrets Management** ensures credentials are injected only at runtime, never stored on disk, and rotated automatically.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** the External Secrets Operator (ESO) to sync AWS Secrets Manager with Kubernetes.
2.  **Mount** secrets as a Ramdisk Volume (Tmpfs) so they never touch the disk.
3.  **Implement** Automatic Key Rotation logic for Database credentials.
4.  **Audit** secret usage to see which Pod accessed which key.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine with `kubectl`.

### Software Environment
- `pip install hvac` (HashiCorp Vault Client) or `boto3`.

---

## 📖 Theoretical Foundation

### 1. The Maturity Model
1.  **Code:** `PASSWORD = "hunter2"` (Terrible).
2.  **Env Var:** `docker run -e PASS=...`. (Better, but visible in `docker inspect`).
3.  **K8s Secret:** Encoded (Base64), not Encrypted (unless Etcd encryption on).
4.  **External Secret:** Secrets live in Vault/AWS. Synced to K8s only when needed.
5.  **Dynamic Secret:** DB creates a username `app-123` valid for 5 minutes, then deletes it.

### 2. Sidecar Injection
Instead of Env Vars, a Sidecar (Vault Agent) writes the secret to `/vault/secrets/config`. The App watches this file. If secret rotates, file updates, app reloads.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: External Secrets Operator

Do not create Secrets manually. Let ESO fetch them from AWS.

```bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets -n kube-system
```

#### 📁 `manifests/secret-store.yaml`
```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-backend
  namespace: prod
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt: # IRSA Auth
          serviceAccountRef:
            name: secrets-sa
```

#### 📁 `manifests/db-creds.yaml`
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-creds
  namespace: prod
spec:
  refreshInterval: 1h # Check for rotation hourly
  secretStoreRef:
    name: aws-backend
    kind: SecretStore
  target:
    name: db-secret-k8s # The resulting K8s Secret
  data:
  - secretKey: username
    remoteRef:
      key: prod/db/creds
      property: username
  - secretKey: password
    remoteRef:
      key: prod/db/creds
      property: password
```

### 👨‍💻 Core Implementation: File Mount (Better than Env)

Env vars are leaked in Crash Dumps. Files on Ramdisk are safer.

#### 📁 `manifests/deployment-mount.yaml`
```yaml
spec:
  containers:
  - name: app
    volumeMounts:
    - name: secret-vol
      mountPath: "/etc/secrets"
      readOnly: true
  volumes:
  - name: secret-vol
    secret:
      secretName: db-secret-k8s
```

#### 📁 `src/db_connector.py`
```python
import os
import time

class SecretWatcher:
    def __init__(self, path="/etc/secrets/password"):
        self.path = path
        self.last_load = 0
        self.secret = None

    def get_password(self):
        # Check if file changed (Rotation)
        current_mtime = os.path.getmtime(self.path)
        if current_mtime > self.last_load:
            print("Reloading Secret...")
            with open(self.path, "r") as f:
                self.secret = f.read().strip()
            self.last_load = current_mtime
        return self.secret

# Usage
watcher = SecretWatcher()
# db.connect(password=watcher.get_password())
```

---

## 🔬 Lab Exercise: "Rotation Day"

### Task
Handle invalid credentials without crashing.
1.  **Scenario:** AWS rotates the DB password at 12:00.
2.  **State:** The K8s Secret updates at 12:01 (RefreshInterval).
3.  **Gap:** For 1 minute, the App has the old password.
4.  **Handling:**
    ```python
    try:
        connect_db()
    except AuthError:
        # Force reload secret
        watcher.force_reload()
        connect_db() # Retry
    ```
5.  **Test:** Manually change secret in AWS. Watch app logs. It should fail once, then recover.

---

## 📖 Advanced Theory: Sealed Secrets
If you *must* commit secrets to Git (GitOps), use **Sealed Secrets**.
1.  `kubeseal < secret.yaml > sealed.json`.
2.  This encrypts the secret using the Cluster's Public Key.
3.  Commit `sealed.json`. Safe to public repo.
4.  Only the Controller (with Private Key) can decrypt it.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Mounts > Env:** Using `volumeMount` is safer than `envFrom`. It allows updates without restarting the Pod.
2.  **Least Privilege:** The EC2 instance running the build agent should NOT have access to Prod DB secrets. Only the Prod Pods should.
3.  **Encryption at Rest:** Ensure Etcd (K8s DB) is encrypted. Otherwise, anyone with `etcdctl` access can read all secrets in plain text.

### API Summary
```bash
kubectl create secret generic my-secret --from-literal=key=val
```

---

**Day 157 Complete** ✅

*Next: Day 158 - Privacy Preserving ML - Federated Learning.*
