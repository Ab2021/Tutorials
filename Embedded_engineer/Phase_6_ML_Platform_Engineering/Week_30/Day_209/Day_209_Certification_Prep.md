# Day 209: The Validation: Certification Prep (CKA/CKAD/CKS)
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 30: Capstone Project Part 2

---

> **🎯 Focus Area:** You have the skills. Now get the badge. **CKA (Administrator)** and **CKS (Security)** are the gold standards for Platform Engineers. Today we run a rapid-fire drill to prepare you for the exam.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Solve** 5 common CKA Troubleshooting scenarios (Broken Kubelet, TLS expiry).
2.  **Solve** 5 CKAD scenarios (Multi-Container Pods, NetworkPolicies).
3.  **Solve** 5 CKS scenarios (Runtime Security, Image Scanning).
4.  **Navigate** the Kubernetes Documentation quickly (Allowed during exam).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine (Kind Cluster).

---

## 📖 Theoretical Foundation

### 1. The Exam Format
*   **CKA:** 2 hours. 17 questions. Hands-on (Terminal).
*   **Strategy:** Speed is key. Do not write YAML from scratch. Use `kubectl run --dry-run=client -o yaml`.
*   **Alias:** `alias k=kubectl`. `export do="--dry-run=client -o yaml"`.

---

## 💻 Implementation

### 👨‍💻 Drill 1: The Broken Worker Node (CKA)

**Scenario:** Node `worker-1` is `NotReady`. fix it.
**Investigation:**
1.  `ssh worker-1`.
2.  `sudo systemctl status kubelet`. (Status: Active? Inactive?)
3.  `journalctl -u kubelet -f`.
    *   *Error:* `Client certificate not signed`.
    *   *Fix:* Check `/etc/kubernetes/kubelet.conf`. Does it point to the right CA?
    *   *Error:* `Swap is enabled`.
    *   *Fix:* `sudo swapoff -a`.

### 👨‍💻 Drill 2: Network Policy Isolation (CKAD)

**Scenario:** Create a Policy `deny-all` in namespace `foo`, but allow traffic from namespace `bar`.
**Solution:**
```bash
# 1. Deny All
k create -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny
  namespace: foo
spec:
  podSelector: {}
  policyTypes: [Ingress]
EOF

# 2. Allow Bar
k create -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-bar
  namespace: foo
spec:
  podSelector: {}
  policyTypes: [Ingress]
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: bar
EOF
```

### 👨‍💻 Drill 3: Secret Decoding (CKS)

**Scenario:** There is a secret `db-pass` in namespace `secure`. Decode it.
**Solution:**
```bash
k get secret db-pass -n secure -o jsonpath='{.data.password}' | base64 --decode
```

### 👨‍💻 Drill 4: Sidecar Logging (CKAD)

**Scenario:** Pod `app` writes logs to `/var/log/app.log`. It does NOT log to stdout. Add a sidecar to stream it to stdout.
**Solution:**
```yaml
containers:
- name: app
  image: busybox
  command: ["sh", "-c", "while true; do echo $(date) >> /var/log/app.log; sleep 1; done"]
  volumeMounts:
  - name: varlog
    mountPath: /var/log
- name: stream-sidecar
  image: busybox
  command: ["sh", "-c", "tail -f /var/log/app.log"] # The Fix
  volumeMounts:
  - name: varlog
    mountPath: /var/log
volumes:
- name: varlog
  emptyDir: {}
```

### 👨‍💻 Drill 5: ETCD Backup (CKA)

**Scenario:** Backup ETCD.
**Solution:**
```bash
ETCDCTL_API=3 etcdctl --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  snapshot save /tmp/snapshot.db
```

---

## 🔬 Lab Exercise: "The Time Attack"

### Task
Complete in 10 minutes.
1.  Create namespace `x`.
2.  Create deployment `web` in `x` with image `nginx`, 3 replicas.
3.  Expose `web` as Service `web-svc` on port 80.
4.  Create a CronJob `checker` that curls `web-svc` every minute.

**Solution (Fastest Way):**
```bash
k create ns x
k create deploy web --image=nginx --replicas=3 -n x
k expose deploy web --name=web-svc --port=80 -n x
k create cronjob checker --image=curlimages/curl --schedule="*/1 * * * *" -n x -- curl http://web-svc
```

---

## 📖 Advanced Theory: CKS Security Context
**Scenario:** Ensure key files cannot be modified.
**Context:**
```yaml
securityContext:
  readOnlyRootFilesystem: true
  runAsUser: 1000
  runAsGroup: 3000
```
This forces the container to write *only* to mounted volumes (`emptyDir` or PVCs). Any write to `/etc` or `/bin` fails.

---

## 📝 Daily Summary

### Key Takeaways
1.  **RTFM:** You can use `kubernetes.io/docs` during the exam. Search "PersistentVolume" -> Copy YAML -> Edit. Don't memorize YAML.
2.  **Context:** Always check which cluster context you are in. `kubectl config use-context k8s`. Getting 100% right on the wrong cluster = 0 points.
3.  **Tmux:** Learn basic tmux/screen. It helps to have one pane for `kubectl explain` and another for editing `vi`.

### API Summary
```bash
kubectl run pod1 --image=nginx --restart=Never # Pod
kubectl create deploy dep1 --image=nginx --replicas=2 # Deployment
```

---

**Day 209 Complete** ✅

*Next: Day 210 - The End & The Beginning.*
