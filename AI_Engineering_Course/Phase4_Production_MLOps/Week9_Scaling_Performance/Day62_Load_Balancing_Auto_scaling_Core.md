# Day 62: Load Balancing & Auto-scaling
## Core Concepts & Theory

### The Scaling Challenge

**Problem:** LLM traffic is bursty and compute-intensive.
- **Bursty:** Traffic spikes during day, drops at night.
- **Long-Running:** Requests take seconds (not milliseconds like HTTP).
- **Stateful:** KV cache makes requests stateful (for multi-turn).

**Goal:** Maintain low latency (TTFT/TPOT) while minimizing cost (GPU hours).

### 1. Load Balancing Strategies

**Round Robin:**
- Rotate requests sequentially.
- **Flaw:** LLM requests vary wildly in duration (10 tokens vs 1000 tokens). One replica might get stuck with long requests while others are idle.

**Least Connections (Outstanding Requests):**
- Send to replica with fewest active requests.
- **Better:** Accounts for current load.

**Peak EWMA (Exponential Weighted Moving Average):**
- Track latency of each replica.
- Send to replica with lowest predicted latency.
- **Best:** Accounts for "slow" replicas (e.g., thermal throttling).

**Capacity-Aware (Token Budget):**
- Track estimated token load on each replica.
- Don't send if `current_load + estimated_new_load > max_capacity`.

### 2. Auto-scaling Metrics

**CPU/Memory Usage:**
- **Bad for LLMs:** GPU memory is always high (weights loaded). GPU utilization fluctuates.

**Concurrency (Active Requests):**
- **Good:** Scale up if `avg_concurrent_requests > target`.
- **Target:** e.g., 10 requests per replica.

**Queue Depth:**
- **Better:** Scale up if requests are waiting in the queue.
- **Metric:** `pending_requests / active_replicas`.

**Token Throughput:**
- **Advanced:** Scale based on tokens/sec.

### 3. Scaling Patterns

**Scale Out (Horizontal):**
- Add more replicas (Pods).
- **Pros:** Linear throughput increase.
- **Cons:** Cold start (loading weights takes 30s-1min).

**Scale Up (Vertical):**
- Move to larger GPU (A10G -> A100).
- **Cons:** Requires restart, limited by hardware availability.

**Zero-to-One (Serverless):**
- Scale to 0 when idle.
- **Challenge:** The "Cold Start" penalty is massive for LLMs.

### 4. Cold Start Mitigation

**Model Baking:**
- Bake model weights into the Docker image.
- **Pros:** No download time.
- **Cons:** Huge images (20GB+), slow pull time.

**Volume Mounting (NFS/EFS):**
- Mount shared volume with weights.
- **Pros:** Fast startup if network is fast.
- **Cons:** Network bandwidth bottleneck during storm.

**Lazy Loading:**
- Stream weights from S3 directly to GPU memory.
- **Pros:** Parallelize download and load.

### 5. Multi-Model Serving (MMS)

**Concept:** Serve multiple LoRA adapters on one base model.
- **Base Model:** Llama-3-70B (Loaded once).
- **Adapters:** Finance, Medical, Coding (Swapped on demand).

**Mechanism:**
- Request comes with `adapter_id`.
- Server loads adapter weights (small, ~100MB) into GPU.
- Performs inference.
- **Benefit:** Serve 100 specialized models with cost of 1 base model.

### 6. Request Queue Management

**Priority Queues:**
- **Paid Users:** High priority.
- **Free Users:** Low priority.

**Deadline-Aware Queue:**
- Drop requests that have waited longer than `timeout`.
- "Fail fast" is better than "Fail slow".

### 7. Distributed Tracing

**Problem:** Request hits Gateway -> Router -> Model Replica -> Tokenizer. Where is the latency?
- **Tools:** OpenTelemetry, Jaeger.
- **Trace:**
  - `gateway_latency`: 5ms
  - `queue_latency`: 200ms (Bottleneck!)
  - `inference_latency`: 5000ms

### 8. Rate Limiting

**Token Bucket:**
- Allow bursts but enforce average rate.
- **Metric:** Requests/min or Tokens/min.

**Global vs Local:**
- **Global:** Redis-backed (accurate but slower).
- **Local:** In-memory (fast but approximate).

### 9. Deployment Architectures

**Synchronous (Real-time):**
- User waits for response.
- **Focus:** Latency.

**Asynchronous (Batch):**
- User submits job, polls for result.
- **Focus:** Throughput, Cost.
- **Strategy:** Pack batches to 100% GPU utilization.

### 10. Cost Optimization (FinOps)

**Spot Instances:**
- Use interruptible GPUs (60-90% cheaper).
- **Strategy:** Checkpointing, fast recovery.

**Mixed Instances:**
- Use A100 for prompt processing (fast).
- Use A10 for decoding (cheaper).

### Summary

**Scaling Strategy:**
1.  **Load Balancer:** Use **Least Outstanding Requests**.
2.  **Auto-scaler:** Scale on **Queue Depth** or **Concurrency**.
3.  **Cold Start:** Use **Model Baking** or **Fast Network Storage**.
4.  **Multi-Model:** Use **LoRA adapters** to consolidate traffic.
5.  **Queue:** Implement **Priority Queues** and **Timeouts**.

### Next Steps
In the Deep Dive, we will implement a Load Balancer simulation with different strategies and an Auto-scaler logic.
