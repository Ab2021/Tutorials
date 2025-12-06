# Day 180: The Infinity Gauntlet: Multi-Cluster Observability with Thanos
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 26: Multi-Cluster & Federation

---

> **🎯 Focus Area:** You have 10 clusters. Each has a Prometheus server. To see global CPU usage, you need to open 10 tabs. **Thanos** aggregates these metrics into a Single Pane of Glass and provides unlimited long-term storage in S3.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** Thanos Sidecar alongside Prometheus to ship blocks to Object Storage.
2.  **Configure** Thanos Store Gateway to query historical data from S3.
3.  **Deploy** Thanos Querier to aggregate metrics from multiple clusters (Sidecars + Store).
4.  **Visualize** a Global Dashboard in Grafana showing "All Clusters" health.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine (3 Kind Clusters).

### Software Environment
- `helm install prometheus-community/kube-prometheus-stack`.

---

## 📖 Theoretical Foundation

### 1. The Prometheus Bottleneck
Prometheus is designed for reliability, not scalability.
*   **Storage:** Local disk only. Hard to keep > 14 days of data.
*   **View:** Cluster-local only.
*   **HA:** Running two Prometheus instances creates duplicate data, hard to de-dupe.

### 2. Thanos Architecture
*   **Sidecar:** Runs in the Prometheus Pod. Uploads TSDB blocks (2h chunks) to S3. Serves real-time queries via gRPC.
*   **Store:** Gateway that reads historical blocks from S3.
*   **Querier:** The "Brain". Connects to Sidecars (Real-time) + Stores (History). Performs De-duplication.
*   **Compactor:** Downsamples old data (Raw -> 5m resolution -> 1h resolution).

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Object Storage Config (S3)

Secret for Thanos to talk to S3.

#### 📁 `manifests/objstore.yaml`
```yaml
type: S3
config:
  bucket: "thanos-metrics"
  endpoint: "s3.us-east-1.amazonaws.com"
  access_key: "..."
  secret_key: "..."
```

### 👨‍💻 Infrastructure: Prometheus with Sidecar (Cluster 1, 2, 3)

Deploy this on EVERY cluster.

#### 📁 `manifests/prometheus-values.yaml`
```yaml
prometheus:
  prometheusSpec:
    # Enable Thanos Sidecar
    thanos:
      baseImage: quay.io/thanos/thanos
      version: v0.31.0
      objectStorageConfig:
        key: objstore.yaml
        name: thanos-objstore
    
    # Expose Sidecar via Service
    service:
      ports:
        - name: grpc
          port: 10901
          targetPort: 10901
      type: LoadBalancer # Or use Ingress/East-West Gateway in Mesh
```

### 👨‍💻 Infrastructure: Global Querier (Observer Cluster)

The central point.

#### 📁 `manifests/thanos-query.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: thanos-query
spec:
  template:
    spec:
      containers:
      - name: thanos-query
        image: quay.io/thanos/thanos:v0.31.0
        args:
        - query
        - --store=dns+prometheus-sidecar.cluster1:10901
        - --store=dns+prometheus-sidecar.cluster2:10901
        - --store=dns+thanos-store-gateway:10901 # History from S3
```

---

## 🔬 Lab Exercise: "Time Travel"

### Task
Query History.
1.  **Setup:** Run Prometheus without Thanos (Retention 2h).
2.  **Wait:** e.g., 3 hours.
3.  **Query:** `rate(http_requests_total[1h] offset 4h)`.
4.  **Result:** Empty (Data lost).
5.  **Setup:** Enable Thanos Sidecar + Store.
6.  **Query:** Same query.
7.  **Result:** **Success**. The Store component pulls the data from S3, even though it was deleted from Prometheus disk.

---

## 📖 Advanced Theory: Downsampling
Storing 1 year of raw metrics (15s scrape interval) is expensive and slow to query.
Thanos Compactor creates:
*   **Raw:** 15s interval. (Retain 14 days).
*   **5m:** Downsampled. (Retain 3 months).
*   **1h:** Downsampled. (Retain 5 years).
When you query a 1-year graph, Thanos automatically picks the 1h chunks. Speedup: 240x.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Global View:** You can write PromQL referencing multiple clusters: `sum by (cluster) (machine_cpu_cores)`.
2.  **Cost:** S3 API "PUT" costs add up. The Sidecar writes every 2 hours. The Compactor reads/writes heavily. Monitor your S3 bill.
3.  **Pull vs Push:** Cortex (Mimir) uses a Push model (Remote Write). Thanos uses a Pull model (Sidecar). Thanos is generally cheaper (less ingest bandwidth) but requires incoming connectivity to clusters.

### API Summary
```bash
./thanos query --store=...
```

---

**Day 180 Complete** ✅

*Next: Day 181 - Week 26 Review & Project - The Global Platform.*
