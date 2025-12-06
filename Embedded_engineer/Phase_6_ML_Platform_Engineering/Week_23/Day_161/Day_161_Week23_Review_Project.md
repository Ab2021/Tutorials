# Day 161: Week 23 Review & Project - The Fortress (DevSecOps)
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 23: Security & Governance

---

> **🎯 Focus Area:** Security is not a feature you add at the end. It is the pipeline itself. We will build a pipeline that rejects unsecured code, signs valid artifacts, and deploys them into a hardened bunker.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Integrate** `trivy` (Scanning) and `cosign` (Signing) into GitHub Actions.
2.  **Enforce** Policy-as-Code using OPA Gatekeeper or Kyverno.
3.  **Config** a "Break Glass" procedure for emergency access to production.
4.  **Demonstrate** end-to-end provenance: Commit -> Build -> Sign -> Deploy.

---

## 📚 Week 23 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 155 | Threat Modeling | "I can hack a model by adding noise to an image." |
| 156 | RBAC | "The Inference Pod doesn't need Write access to anything." |
| 157 | Secrets | "Environment Variables leak in crash dumps. Use Mounts." |
| 158 | Federated Learning | "Training on the phone keeps data private." |
| 159 | Compliance | "Unlearning a user is harder than deleting a row." |
| 160 | Secure Deploy | "If the image isn't signed, the cluster rejects it." |

---

## 🏗️ Final Project: "SecureFinTech Pipeline"

### Scenario
We are deploying a Credit Approval Model.
**Policy:**
1.  No Critical CVEs in dependencies.
2.  Image must be signed by "Build Bot".
3.  Pod must look up DB Password from AWS Secrets Manager (via ESO).
4.  No Internet Egress allowed.

### Step 1: The Code (Checking Dependencies)

#### 📁 `project/requirements.txt`
```text
# Bad: numpy (no version)
# Good: Pinned Version
numpy==1.22.4
torch==1.13.1
flask==2.2.2
```

### Step 2: The Pipeline (GitHub Actions)

#### 📁 `.github/workflows/secure-pipeline.yaml`
```yaml
name: Secure Build
on: [push]

jobs:
  build-sign:
    runs-on: ubuntu-latest
    permissions:
      id-token: write # For Sigstore signing
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v3

      # 1. Dependency Scan
      - name: Safety Check
        run: |
          pip install safety
          safety check -r project/requirements.txt

      # 2. Build Image
      - name: Build
        run: docker build -t ghcr.io/myorg/credit-model:${{ github.sha }} .

      # 3. Vulnerability Scan (Trivy)
      - name: Trivy Scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'ghcr.io/myorg/credit-model:${{ github.sha }}'
          format: 'table'
          exit-code: '1' # Fail pipeline on CRITICAL
          ignore-unfixed: true
          vuln-type: 'os,library'
          severity: 'CRITICAL,HIGH'

      # 4. Push & Sign
      - name: Push & Sign
        uses: sigstore/cosign-installer@main
        run: |
          docker push ghcr.io/myorg/credit-model:${{ github.sha }}
          cosign sign --yes ghcr.io/myorg/credit-model:${{ github.sha }}
```

### Step 3: The Cluster Policy (Kyverno)

#### 📁 `infra/policy.yaml`
```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: enforce-security
spec:
  validationFailureAction: enforce
  rules:
    # Rule 1: Must be Signed
    - name: check-image
      match:
        resources:
          kinds: [Pod]
      verifyImages:
      - image: "ghcr.io/myorg/*"
        keyless: # Uses OIDC Sigstore
          issuer: "https://token.actions.githubusercontent.com"
          subject: "https://github.com/myorg/repo/.github/workflows/secure-pipeline.yaml@refs/heads/main"

    # Rule 2: Cannot Run as Root
    - name: validate-runAsNonRoot
      match:
        resources:
          kinds: [Pod]
      validate:
        message: "Running as root is forbidden"
        pattern:
          spec:
            containers:
            - securityContext:
                runAsNonRoot: true
```

### Step 4: The Deployment Manifest

#### 📁 `manifests/deployment.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: credit-model
  namespace: secure-fintech
spec:
  template:
    spec:
      serviceAccountName: credit-sa # IRSA for Secrets
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: model
        image: ghcr.io/myorg/credit-model:sha-123
        volumeMounts:
        - name: secrets
          mountPath: /etc/secrets
          readOnly: true
      volumes:
      - name: secrets
        csi:
          driver: secrets-store.csi.k8s.io
          readOnly: true
          volumeAttributes:
            secretProviderClass: "aws-secrets"
```

---

## 🔬 Lab Exercise: "The Pentest"

### Task
Try to break in.
1.  **Attack:** Modify `requirements.txt` to include `pickle-mixin` (imaginary vulnerable lib).
    *   **Defense:** `safety check` fails pipeline.
2.  **Attack:** Build image locally and push to registry, skipping scan. (Unsigned).
    *   **Defense:** Kyverno `check-image` blocks deployment: `signature verification failed`.
3.  **Attack:** Change `Dockerfile` to `USER root`.
    *   **Defense:** Kyverno `validate-runAsNonRoot` blocks deployment.
4.  **Attack:** Exec into pod and try `apt-get install nmap`.
    *   **Defense:** Pod is running as User 1000 (Non-root). Cannot install packages. Filesystem is ReadOnly.

---

## 📝 Success Criteria
1.  **Automation:** Security checks happen on every Commit.
2.  **Immutability:** Once an image is signed, it cannot be changed without invalidating the signature.
3.  **Auditability:** We know exactly *who* signed the image (GitHub Actions via OIDC) and *what* code produced it.

---

**Week 23 Complete** ✅
**Phase 6D In Progress**

*Next Phase: Cost Optimization - Saving Money.*
