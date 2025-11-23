# Day 62: Load Balancing & Auto-scaling
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why is "CPU Utilization" a bad metric for auto-scaling LLMs?

**Answer:**
- **GPU Dominance:** LLM inference runs on GPUs. CPU is mostly idle (dispatching kernels).
- **Memory Bound:** GPU memory is always near 100% (weights loaded).
- **Compute Spikes:** GPU compute utilization fluctuates wildly between prefill (100%) and decode (low).
- **Better Metrics:** **Queue Depth** (how many requests are waiting), **Concurrency** (active requests), or **Token Latency** (TPOT).

#### Q2: Explain the "Cold Start" problem in LLM serving and how to mitigate it.

**Answer:**
- **Problem:** When a new replica starts, it must download weights (e.g., 140GB for 70B model) and load them into GPU memory. This takes 30s to 5 minutes.
- **Mitigation:**
  - **Model Baking:** Include weights in the Docker image (fastest startup, but large image).
  - **Fast Storage:** Use high-throughput shared storage (Amazon FSx, localized SSDs).
  - **Over-provisioning:** Keep a "warm pool" of replicas ready.
  - **Lazy Loading:** Stream weights and start compiling kernels immediately.

#### Q3: What is Multi-Model Serving (MMS) and why is it cost-effective?

**Answer:**
- **Concept:** Serving multiple fine-tuned variants (adapters) on a single shared base model.
- **Mechanism:** The base model (heavy) stays in GPU memory. When a request comes for "Finance-LoRA", only the small LoRA weights are loaded/swapped.
- **Benefit:** Instead of deploying 10 replicas for 10 tasks (expensive), deploy 1 replica that handles all 10 tasks dynamically. Reduces GPU cost by 10x.

#### Q4: How does Least Outstanding Requests load balancing work?

**Answer:**
- **Algorithm:** The load balancer tracks the number of active requests sent to each replica. New requests go to the replica with the minimum count.
- **Why for LLMs:** LLM requests have high variance (generating 10 tokens vs 1000 tokens). Round Robin might send a request to a replica that is still busy with a long generation. Least Outstanding ensures we utilize idle/less-busy replicas.

#### Q5: What is the difference between Scale Up and Scale Out?

**Answer:**
- **Scale Up (Vertical):** Moving to a more powerful instance (e.g., A10 -> A100). Good for reducing latency of single requests (larger model fits in memory).
- **Scale Out (Horizontal):** Adding more instances (replicas). Good for increasing total throughput (handling more users).

---

### Production Challenges

#### Challenge 1: Auto-scaler Oscillation (Flapping)

**Scenario:** System scales up to 10 replicas, then immediately down to 2, then up to 10.
**Root Cause:**
- Scaling metric is too volatile (spikes).
- Cooldown period is too short.
- Scale-down threshold is too close to scale-up threshold.
**Solution:**
- **Smoothing:** Use moving average (EWMA) for the metric.
- **Hysteresis:** Add a gap between scale-up and scale-down thresholds.
- **Cooldown:** Enforce a 5-10 min cooldown after scaling actions.

#### Challenge 2: "Thundering Herd" on Recovery

**Scenario:** One replica crashes. Load balancer retries all its requests on the remaining replicas, causing them to crash (OOM/Timeout). Cascade failure.
**Root Cause:** Retry storms and lack of circuit breaking.
**Solution:**
- **Circuit Breaker:** Stop sending requests to a failing replica immediately.
- **Exponential Backoff:** Clients should retry with jitter.
- **Shed Load:** Drop low-priority requests during high load.

#### Challenge 3: Uneven Load Distribution (Hotspotting)

**Scenario:** One replica is at 100% load, others are at 10%.
**Root Cause:**
- Sticky sessions (routing same user to same replica).
- Very long requests (Elephant flows) hogging one replica.
**Solution:**
- **Least Connections:** Switch to dynamic load balancing.
- **Request Splitting:** (Advanced) Split long generation into chunks? (Hard for autoregressive).
- **Timeout:** Kill requests that take too long.

#### Challenge 4: GPU Memory Fragmentation over Time

**Scenario:** Replica runs fine for 2 days, then OOMs.
**Root Cause:** PyTorch memory fragmentation or memory leak in the serving engine.
**Solution:**
- **Periodic Restart:** Scheduled rolling restart of replicas every 24h.
- **PagedAttention:** Use vLLM to minimize fragmentation.

#### Challenge 5: Multi-Region Latency

**Scenario:** Users in Europe experience high latency accessing US-East model.
**Root Cause:** Speed of light (network latency).
**Solution:**
- **Edge Deployment:** Deploy replicas in multiple regions.
- **Geo-DNS:** Route user to nearest region.
- **Global Load Balancer:** Manage traffic across regions.

### System Design Scenario: Global LLM API

**Requirement:** Serve 1M users globally with <100ms latency.
**Design:**
1.  **Regions:** Deploy clusters in US, EU, Asia.
2.  **Routing:** Geo-DNS routing.
3.  **Scaling:** KEDA (Kubernetes Event-driven Autoscaling) based on Prometheus metrics (Queue Depth).
4.  **Queue:** Kafka/RabbitMQ for async jobs, HTTP for sync chat.
5.  **Caching:** Global Redis cache (replicated) for common queries.
6.  **Cost:** Use Spot instances for the async queue processing.

### Summary Checklist for Production
- [ ] **Metric:** Scale on **Concurrency** or **Queue Depth**.
- [ ] **Balancing:** Use **Least Outstanding Requests**.
- [ ] **Startup:** Optimize **Cold Start** (Baking/Lazy Loading).
- [ ] **Safety:** Implement **Circuit Breakers** and **Rate Limits**.
- [ ] **Observability:** Use **Distributed Tracing** to find bottlenecks.
