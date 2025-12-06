# Day 76: The Warehouse: Registry Management with Harbor
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 11: Container Optimization

---

> **🎯 Focus Area:** Your images are 5GB. Pulling them from Docker Hub every time is slow and hits Rate Limits. Deploy **Harbor**, an enterprise-grade Private Registry with built-in Scanning and Caching.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** Harbor using Helm.
2.  **Configure** a "Proxy Cache" project to mirror Docker Hub (Bye-bye `toomanyrequests`).
3.  **Enforce** CVE Policies (e.g., "Prevent pulling images with Critical vulnerabilities").
4.  **Replicate** images between two Harbor instances (Geo-Redundancy).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- K8s Cluster (Harbor is somewhat heavy: needs Postgres + Redis).

### Software Environment
- `helm`.

---

## 📖 Theoretical Foundation

### 1. The Registry Protocol (OCI)
A Registry is just a web server handling `PUT /v2/<name>/blobs`.
*   **Docker Hub:** The public SaaS.
*   **ECR/GCR:** Cloud managed.
*   **Harbor:** Self-hosted. Features: UI, RBAC (LDAP/OIDC), Signing (Notary), Scanning (Trivy).

### 2. Proxy Cache
K8s pulls `nginx:latest`.
1.  Kubelet asks Harbor.
2.  Harbor checks local disk. Miss.
3.  Harbor pulls from Docker Hub.
4.  Harbor saves to disk.
5.  Harbor serves Kubelet.
6.  *Next time:* Hit. Fast. No rate limit usage.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Installing Harbor

```bash
helm repo add harbor https://helm.goharbor.io
helm repo update

# Install (minimal)
helm install harbor harbor/harbor \
  --namespace harbor-system \
  --create-namespace \
  --set expose.type=nodePort \
  --set externalURL=http://localhost:30002
```

### 👨‍💻 Core Implementation: Configuring Proxy

1.  Login to UI (`admin` / `Harbor12345`).
2.  **Registries:** Add Endpoint -> Provider: Docker Hub -> Name: `docker-hub`.
3.  **Projects:** New Project -> Name: `hub-proxy`.
    *   **Proxy Cache:** Toggle ON.
    *   **Registry:** Select `docker-hub`.
4.  **Usage:**
    Instead of `docker pull redis`, use:
    `docker pull core.harbor.domain/hub-proxy/library/redis`.

### 👨‍💻 Core Implementation: Vulnerability Gate

1.  Go to Project -> Configuration.
2.  **Prevent Vulnerable Images from running:** Tick "Critical".
3.  Now, if `trivy` finds a Critical CVE, `docker pull` will fail with "Forbidden: Vulnerability policy violation".

---

## 🔬 Lab Exercise: "The Air-Gapped Simulation"

### Task
Simulate a disconnected environment.
1.  Pull `python:3.9` into Harbor.
2.  Disconnect your internet (or block docker.io).
3.  Pull from Harbor.
4.  **Observation:** It works. This is mandatory for Defense/Finance environments.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Cache everything:** Configure proxies for `gcr.io`, `quay.io`, and `docker.io`. Save bandwidth.
2.  **Garbage Collection:** Harbor allows setting retention policies (e.g., "Keep last 5 tags"). 5GB images fill disk fast.
3.  **Robot Accounts:** Don't use your username for CI. Create Robot Accounts in Harbor with limited "Push" scope.

### API Summary
```bash
helm install harbor harbor/harbor
```

---

**Day 76 Complete** ✅

*Next: Day 77 - Week 11 Review & Project - Zero-Vulnerability Base Images.*
