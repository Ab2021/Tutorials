# Lab 24.2: Thanos (Long-Term Storage)

## 🎯 Objective

Scale Prometheus. Prometheus is not designed for long-term storage (months/years). **Thanos** extends Prometheus by uploading old blocks to Object Storage (S3) and providing a global query view across multiple clusters.

## 📋 Prerequisites

-   Minikube running.
-   S3 Bucket (or MinIO local).

## 📚 Background

### Architecture
-   **Sidecar**: Runs next to Prometheus. Uploads data to S3.
-   **Store Gateway**: Reads data from S3.
-   **Querier**: Aggregates data from Sidecars (Real-time) and Store Gateways (Historical).
-   **Compactor**: Downsamples data (Raw -> 5m -> 1h) to save space.

---

## 🔨 Hands-On Implementation

### Part 1: Install MinIO (S3 Emulation) 🪣

1.  **Install:**
    ```bash
    helm repo add minio https://charts.min.io/
    helm install minio minio/minio --set rootUser=admin,rootPassword=password123,mode=standalone
    ```

2.  **Create Bucket:**
    Port-forward MinIO console (9001). Login. Create bucket `thanos`.

### Part 2: Configure Prometheus with Sidecar 🏎️

We need to tell the Prometheus Operator to inject the Thanos Sidecar.

1.  **Create `thanos-config.yaml`:**
    ```yaml
    type: s3
    config:
      bucket: thanos
      endpoint: minio.default.svc:9000
      access_key: admin
      secret_key: password123
      insecure: true
    ```

2.  **Create Secret:**
    ```bash
    kubectl create secret generic thanos-objstore-config --from-file=thanos.yaml=thanos-config.yaml -n monitoring
    ```

3.  **Upgrade Helm Chart:**
    Update `values.yaml` for `kube-prometheus-stack`:
    ```yaml
    prometheus:
      prometheusSpec:
        thanos:
          objectStorageConfig:
            key: thanos.yaml
            name: thanos-objstore-config
    ```
    `helm upgrade monitoring ... -f values.yaml`.

### Part 3: Install Thanos Components 🏛️

1.  **Install Bitnami Chart:**
    ```bash
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm install thanos bitnami/thanos --set objstoreConfig=...
    ```
    *(Note: Full setup is complex. For this lab, focus on the Sidecar uploading data).*

2.  **Verify Upload:**
    Check MinIO bucket. You should see folders like `01G...` (ULIDs). These are Prometheus TSDB blocks.

### Part 4: The Global View 🌍

1.  **Thanos Querier:**
    The Querier connects to the Sidecar (gRPC).
    Open Thanos UI.
    Query `up`.
    It looks like Prometheus, but it can see data from S3!

---

## 🎯 Challenges

### Challenge 1: Downsampling (Difficulty: ⭐⭐⭐⭐)

**Task:**
Enable the **Compactor**.
Wait 2 hours (or force compaction).
Check S3. You should see `5m` and `1h` folders.
*Benefit:* Querying a year of data is fast because you scan 1h chunks, not raw data.

### Challenge 2: Multi-Cluster (Difficulty: ⭐⭐⭐⭐⭐)

**Task:**
Conceptual.
Cluster A (US) + Cluster B (EU).
Both upload to S3.
One central Thanos Querier in US reads from S3.
*Result:* Single pane of glass for global infrastructure.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Run `thanos compact --data-dir /data --objstore.config-file bucket_config.yaml`.
</details>

---

## 🔑 Key Takeaways

1.  **Unlimited Retention**: S3 is cheap. Store metrics forever.
2.  **Global View**: Query across all clusters without federation.
3.  **High Availability**: If Prometheus crashes, data is safe in S3.

---

## ⏭️ Next Steps

We have monitoring. Now let's break things on purpose.

Proceed to **Module 25: Chaos Engineering**.
