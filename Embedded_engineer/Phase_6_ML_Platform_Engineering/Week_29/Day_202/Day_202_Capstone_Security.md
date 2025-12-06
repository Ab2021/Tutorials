# Day 202: Titan Phase 5: Zero Trust Security & RBAC
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 29: Capstone Project Part 1

---

> **🎯 Focus Area:** "Titan" is now live. But anyone can access the endpoint. We need **Zero Trust**. No one is trusted by default. Every request must carry a verifiable Identity (OIDC/JWT) and must be explicitly authorized (RBAC).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** OIDC Integration (Keycloak) for JupyterHub and KServe using **Dex**.
2.  **Enforce** Kubernetes RBAC (isolate `team-a` from `team-b`).
3.  **Implement** Istio AuthorizationPolicy to block unauthorized service-to-service calls.
4.  **Audit** Network Access using Cilium Hubble or Calico logs.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.
- Access to `titan-control`.

### Software Environment
- `helm install dex`.
- Keycloak (or Google/GitHub OIDC provider).

---

## 📖 Theoretical Foundation

### 1. Identity vs Access
*   **Authentication (AuthN):** "Who are you?" (OIDC/Dex). Login via GitHub. Result: JWT Token.
*   **Authorization (AuthZ):** "Can you do this?" (RBAC/Istio). "Can user `alice` GET `/models/gpt`?".

### 2. Zero Trust Network
*   **Perimeter Security (Old):** VPN into network, then you can access everything.
*   **Zero Trust (Titan):** Being "inside" the VPC means nothing. You need a valid mTLS certificate + valid JWT for *every single request*.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Dex (OIDC Connector)

Unifies GitHub, Google, and LDAP into a single OIDC provider.

#### 📁 `manifests/dex-config.yaml`
```yaml
config:
  issuer: https://auth.titan.ai
  storage:
    type: kubernetes
  connectors:
  - type: github
    id: github
    name: GitHub
    config:
      clientID: $GITHUB_CLIENT_ID
      clientSecret: $GITHUB_CLIENT_SECRET
      redirectURI: https://auth.titan.ai/callback
  staticClients:
  - id: kubeflow-client
    redirectURIs:
    - 'https://kubeflow.titan.ai/oauth2/callback'
    name: 'Kubeflow'
    secret: p@ssword
```

### 👨‍💻 Infrastructure: Kubernetes RBAC (Team Isolation)

Team A cannot see Team B's Pods.

#### 📁 `manifests/rbac/team-isolation.yaml`
```yaml
# 1. Namespace
apiVersion: v1
kind: Namespace
metadata:
  name: team-a
---
# 2. Role (What can they do?)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: team-a
  name: ml-developer
rules:
- apiGroups: ["", "apps", "batch", "ray.io", "serving.kserve.io"]
  resources: ["pods", "deployments", "jobs", "rayjobs", "inferenceservices"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
# 3. Binding (Who gets the role?)
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  namespace: team-a
  name: alice-bind
subjects:
- kind: User
  name: alice@gmail.com # Mapped from OIDC Email
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: ml-developer
  apiGroup: rbac.authorization.k8s.io
```

### 👨‍💻 Core Implementation: Istio AuthorizationPolicy (Service Mesh)

Block calls to the Model if the user is not in "data-scientists" group.

#### 📁 `manifests/istio-authz.yaml`
```yaml
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: restrict-model-access
  namespace: team-a
spec:
  selector:
    matchLabels:
      app: fraud-detector
  action: ALLOW
  rules:
  # Rule 1: Allow Requests with Valid JWT issued by Dex
  - from:
    - source:
        requestPrincipals: ["https://auth.titan.ai/*"]
    when:
    - key: request.auth.claims[groups]
      values: ["data-scientists"]
```

### 👨‍💻 Infrastructure: NetworkPolicy (Cilium/Calico)

Layer 3/4 Lockdown.

#### 📁 `manifests/netpol-default-deny.yaml`
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny
  namespace: team-a
spec:
  podSelector: {} # Select ALL pods
  policyTypes:
  - Ingress
  - Egress
  # Empty Ingress/Egress means DENY ALL
```

#### 📁 `manifests/netpol-allow-dns.yaml`
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
  namespace: team-a
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system # Allow CoreDNS
    ports:
    - protocol: UDP
      port: 53
```

---

## 🔬 Lab Exercise: "The Intruder"

### Task
Simulate unauthorized access.
1.  **Context:** User `bob` (Marketing) tries to access `fraud-detector` (Finance).
2.  **Action:** `curl -H "Authorization: Bearer $BOB_TOKEN" https://fraud.titan.ai/predict`.
3.  **Layer 1 (Ingress):** Allowed (Public Endpoint).
4.  **Layer 2 (Istio AuthZ):** Istio Sidecar checks JWT `groups`. Bob has `marketing`. Policy requires `data-scientists`.
5.  **Result:** `403 Forbidden` (RBAC: Access denied).
6.  **Simulate Breach:** Bob manages to Exec into a pod in `marketing` namespace. Tries `curl 10.1.1.5` (Fraud Pod IP).
7.  **Layer 3 (NetworkPolicy):** `default-deny` drops the packet. `Connection Timed Out`.

---

## 📖 Advanced Theory: Workload Identity
How does the `fraud-detector` Pod authenticate to AWS S3?
**IRSA (IAM Roles for Service Accounts):**
1.  K8s ServiceAccount `fraud-sa` is annotated with AWS Role ARN.
2.  EKS OIDC Provider issues a JWT to the Pod.
3.  Pod sends JWT to AWS STS (`AssumeRoleWithWebIdentity`).
4.  AWS returns temp Access Key.
5.  **Zero Trust:** No long-lived keys. Access revoked if Pod dies.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Defense in Depth:** Don't rely on one layer. If NetworkPolicy fails, Istio catches it. If Istio fails, App logic catches it.
2.  **Least Privilege:** Give teams `Role` (Namespace scoped), not `ClusterRole`. Give Pods ReadOnly S3 access, not Admin.
3.  **Auditing:** `kubectl get events` isn't enough. Enable Kubernetes Audit Logs and ship them to S3/Loki. You need to know *who* ran `kubectl exec`.

### API Summary
```bash
istioctl x authz check # Verify policies
```

---

**Day 202 Complete** ✅

*Next: Day 203 - Week 29 Review & Project - Titan Part 1 Delivery.*
