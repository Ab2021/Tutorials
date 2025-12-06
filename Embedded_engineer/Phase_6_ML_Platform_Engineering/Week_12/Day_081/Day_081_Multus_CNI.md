# Day 81: The Two-Headed Beast: Multus CNI
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 12: Networking for Distributed ML

---

> **🎯 Focus Area:** High-Performance Computing (HPC) requires network separation. Separation of **Control Plane** (K8s API) and **Data Plane** (Gradients). Use **Multus** to attach multiple interfaces to a single Pod.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the "Meta-Plugin" architecture of Multus.
2.  **Define** a `NetworkAttachmentDefinition` (NAD) CRD.
3.  **Deploy** a Pod with `eth0` (Calico) and `net1` (Macvlan/SR-IOV).
4.  **Configure** Routing tables so Traffic A goes out `eth0` and Traffic B goes out `net1`.
5.  **Troubleshoot** IPAM conflicts in secondary interfaces.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Cluster with Multus installed.

### Software Environment
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. The Single Interface Myth
Default K8s: Pod gets `eth0`.
*   All traffic (Prometheus metrics, Logs, S3 downloads, GPU Gradients) fights for `eth0`.
*   **Problem:** S3 download saturates link; GPU training stalls.

### 2. Enter Multus
Multus acts as the "Master CNI".
*   It calls the **Default Delegate** (e.g., Calico) to create `eth0` (for K8s Services/DNS).
*   It looks at Pod Annotations.
*   It calls **Additional Delegates** (e.g., Macvlan, SR-IOV) to create `net1`, `net2`...

### 3. IPAM (IP Address Management)
*   **Cluster-wide:** Calico IPAM handles `eth0`.
*   **Secondary:** `host-local` or `dhcp` handles `net1`. You need to define CIDR ranges for the secondary network carefully.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Install Multus

Download the thin-client daemonset.

```bash
kubectl apply -f https://raw.githubusercontent.com/k8snetworkplumbingwg/multus-cni/master/deployments/multus-daemonset.yml
```

### 👨‍💻 Core Implementation: Define Secondary Network

We create a `NetworkAttachmentDefinition` (NAD). This tells Multus *how* to create the extra interface.

#### 📁 `manifests/macvlan-conf.yaml`
```yaml
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
  name: macvlan-conf
spec:
  config: '{
      "cniVersion": "0.3.0",
      "type": "macvlan",
      "master": "eth0", # Attach to host eth0
      "mode": "bridge",
      "ipam": {
        "type": "host-local",
        "subnet": "192.168.1.0/24",
        "rangeStart": "192.168.1.200",
        "rangeEnd": "192.168.1.216",
        "gateway": "192.168.1.1"
      }
    }'
```

### 👨‍💻 Core Implementation: The Multi-Homed Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: multi-pod
  annotations:
    # Magic Annotation
    k8s.v1.cni.cncf.io/networks: macvlan-conf
spec:
  containers:
  - name: app
    image: nicolaka/netshoot
    command: ["tail", "-f", "/dev/null"]
```

---

## 🔬 Lab Exercise: "Interface Verification"

### Task
1.  Apply the manifests.
2.  Exec into the pod:
    ```bash
    kubectl exec -it multi-pod -- ip addr
    ```
3.  **Observation:**
    *   `eth0`: 10.244.x.x (Calico Overlay).
    *   `net1`: 192.168.1.200 (Macvlan Underlay).
4.  **Routing Test:**
    *   `ping google.com` -> Goes via `eth0` gateway.
    *   `ping 192.168.1.201` (Another pod) -> Goes via `net1` (Direct L2).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Isolation:** Use `eth0` for Control Plane (Health Checks, Kube-API). Use `net1` for Data Plane (NCCL, RDMA).
2.  **Performance:** Macvlan/SR-IOV secondary interfaces often bypass iptables/conntrack, providing significant CPU savings.
3.  **Whereabouts:** A popular IPAM plugin often used with Multus to manage cluster-wide IP pools for secondary interfaces (better than `host-local`).

### API Summary
```bash
kubectl get net-attach-def
```

---

**Day 81 Complete** ✅

*Next: Day 82 - Service Discovery & DNS - When things go wrong, it's always DNS.*
