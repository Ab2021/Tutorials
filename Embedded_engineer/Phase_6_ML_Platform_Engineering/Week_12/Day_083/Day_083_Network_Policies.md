# Day 83: Zero Trust: Network Policies
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 12: Networking for Distributed ML

---

> **🎯 Focus Area:** By default, a hacked Jupyter Notebook in `dev` can connect to the Database in `prod`. Use **Network Policies** to enforce namespace isolation and lock down egress.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Understand** the "Flat Network" model of Kubernetes and its security risks.
2.  **Write** a "Default Deny" policy to whitelist traffic.
3.  **Allow** specific Ingress (e.g., only Frontend can call Backend).
4.  **Block** Egress (Prevent model weights from being uploaded to unauthorized S3 buckets).
5.  **Verify** policies using `netshoot`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Cluster with CNI that supports NetPol (Calico, Cilium, EKS VPC CNI).
- *Warning:* Minikube with default `kubenet` does NOT support policies. You must start minikube with `--cni calico`.

### Software Environment
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. The Firewall Rule
NetworkPolicies are highly specific firewalls applied to Pods via **Label Selectors**.
*   **Ingress:** Incoming traffic.
*   **Egress:** Outgoing traffic.
*   **Stateful:** Like AWS Security Groups. Return traffic is automatically allowed.

### 2. The Default Deny Pattern
Best Practice: Start by blocking EVERYTHING. Then punch holes for what you need.
If you apply *any* Policy to a pod, it switches from "Allow All" to "Deny All except...".

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Default Deny

Apply this to *every* namespace.

#### 📁 `manifests/default-deny.yaml`
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: sensitive-training
spec:
  podSelector: {} # Selects ALL pods
  policyTypes:
  - Ingress
  - Egress
```
*Effect:* All pods are silenced. Even DNS fails.

### 👨‍💻 Core Implementation: Training Job Rules

We need to allow:
1.  **Egress:** To DNS (UDP 53) and S3 (TCP 443).
2.  **Ingress:** From Prometheus (TCP 9090).
3.  **Ingress:** From other Training Ranks (All Ports for DDP).

#### 📁 `manifests/training-policy.yaml`
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-training
  namespace: sensitive-training
spec:
  podSelector:
    matchLabels:
      app: pytorch-ddp
  policyTypes:
  - Ingress
  - Egress
  
  ingress:
  # Allow other ranks to talk to me
  - from:
    - podSelector:
        matchLabels:
          app: pytorch-ddp
  # Allow monitoring
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
      podSelector:
          matchLabels:
            app: prometheus
    ports:
    - port: 9090

  egress:
  # Allow DNS
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - port: 53
      protocol: UDP
  # Allow S3 (CIDR Block - hard in cloud, easier with DNS policies in Cilium)
  - to:
    - ipBlock: # Allow All Public Internet (Simplified)
        cidr: 0.0.0.0/0
        except:
        - 10.0.0.0/8 # Block Private Network (except Explicit Allows)
```

---

## 🔬 Lab Exercise: "The Break-in"

### Task
Verify Isolation.
1.  Create `namespace: dev` and `namespace: prod`.
2.  Deploy `nginx` in `prod` with a Policy "Allow from prod only".
3.  Deploy `hacker` in `dev`.
4.  Try `curl nginx.prod`.
    *   **Result:** Connection Timed Out.
5.  Remove the Policy.
    *   **Result:** 200 OK.

---

## 📝 Daily Summary

### Key Takeaways
1.  **UDP DNS:** The most common mistake is forgetting to allow UDP port 53 in Egress. If you forget, your pod hangs on startup.
2.  **Cilium:** Standard K8s NetPol is L3/L4 (IP/Port). Cilium extends this to L7 (HTTP Methods, DNS Names). You can say "Allow egress to `*.aws.amazon.com` only".
3.  **Scope:** Policies are Namespaced. You generally cannot write a single policy that covers the whole cluster (unless using Calico GlobalNetworkPolicy).

### API Summary
```bash
kubectl get netpol
```

---

**Day 83 Complete** ✅

*Next: Day 84 - Week 12 Review & Project - Designing a High-Performance Distributed Training Network.*
