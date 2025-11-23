# Day 43: Production LLM Deployment
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between API-based and self-hosted LLM deployment?

**Answer:**
- **API-based:** Use OpenAI, Anthropic, Cohere APIs. No infrastructure management, pay per token, instant scaling. Good for prototypes and variable load.
- **Self-hosted:** Deploy open-source models (LLaMA, Mistral) on your infrastructure. Full control, no per-token cost, data privacy. Good for high volume and sensitive data.
- **Trade-off:** API is easier but expensive at scale. Self-hosted is complex but cheaper for high volume.

#### Q2: What is PagedAttention and why is it important?

**Answer:**
- **Problem:** KV cache memory is fragmented. Traditional approach pre-allocates max sequence length, wasting memory.
- **PagedAttention (vLLM):** Allocates KV cache in pages (blocks). Only allocates blocks as needed. Can share blocks across sequences.
- **Benefits:** 2-4x memory reduction, 24x higher throughput, enables prefix caching.
- **Impact:** Can serve more requests with same GPU memory.

#### Q3: Explain continuous batching.

**Answer:**
- **Static Batching:** Wait for batch to fill, process entire batch, wait for all to complete.
- **Continuous Batching:** Process requests as they arrive. Add new requests and remove completed ones dynamically.
- **Benefits:** Lower latency (no waiting for batch), higher throughput (GPU always busy).
- **Used by:** vLLM, TGI.

#### Q4: How do you optimize LLM deployment costs?

**Answer:**
- **Model Selection:** Use smallest model that meets quality bar (GPT-3.5 vs GPT-4).
- **Quantization:** INT8 (2x reduction), INT4 (4x reduction).
- **Caching:** Cache responses for common queries.
- **Batching:** Increase throughput per GPU.
- **Spot Instances:** Use spot/preemptible instances (70% cheaper).
- **Routing:** Route simple queries to smaller models.

#### Q5: What metrics should you monitor for LLM services?

**Answer:**
- **Latency:** p50, p95, p99 (target <1s for interactive).
- **Throughput:** Requests per second, tokens per second.
- **Error Rate:** % of failed requests.
- **Cost:** $ per 1K requests or tokens.
- **Quality:** User satisfaction, hallucination rate.
- **Resources:** GPU utilization, memory usage.

---

### Production Challenges

#### Challenge 1: High Latency

**Scenario:** p95 latency is 5 seconds. Users complain about slowness.
**Diagnosis:**
- Check GPU utilization (should be >80%).
- Check batch size (too small = underutilized GPU).
- Check model size (70B model on single GPU = slow).
**Solution:**
- **Quantization:** Use INT8 or INT4 to fit larger batches.
- **Batching:** Increase batch size (dynamic batching).
- **Smaller Model:** Use 7B instead of 70B if quality allows.
- **Streaming:** Stream tokens as generated (perceived latency reduction).

#### Challenge 2: OOM (Out of Memory)

**Scenario:** GPU runs out of memory during inference.
**Root Cause:** KV cache grows with sequence length. Long sequences exhaust memory.
**Solution:**
- **PagedAttention:** Use vLLM for efficient memory management.
- **Quantization:** Reduce model size (INT8, INT4).
- **Shorter Context:** Limit max sequence length.
- **Larger GPU:** Upgrade to A100 (80 GB) or H100 (80 GB).

#### Challenge 3: Cost Explosion

**Scenario:** Monthly bill is $50k. Budget is $10k.
**Analysis:**
- Check token usage (are users sending very long prompts?).
- Check model usage (are you using GPT-4 for everything?).
**Solution:**
- **Caching:** Cache responses for common queries (50% hit rate = 50% cost reduction).
- **Model Routing:** Use GPT-3.5 for simple queries, GPT-4 for complex.
- **Self-Hosting:** If volume is high (>10M tokens/day), self-host is cheaper.
- **Rate Limiting:** Limit tokens per user per day.

#### Challenge 4: Autoscaling Lag

**Scenario:** Traffic spikes. Autoscaling takes 5 minutes to spin up new pods. Users see errors.
**Root Cause:** Cold start time (loading model into GPU memory takes time).
**Solution:**
- **Pre-warming:** Keep minimum replicas running (min=2 instead of min=0).
- **Faster Scaling:** Use smaller models (load faster).
- **Queue:** Queue requests during scale-up instead of rejecting.
- **Predictive Scaling:** Scale up before traffic spike (based on historical patterns).

#### Challenge 5: Monitoring Blind Spots

**Scenario:** Users report poor quality, but metrics look good (latency OK, error rate low).
**Root Cause:** Not monitoring quality metrics.
**Solution:**
- **User Feedback:** Track thumbs up/down.
- **Hallucination Detection:** Sample outputs and check for hallucinations.
- **A/B Testing:** Compare new model version against baseline.
- **Logging:** Log all requests/responses for manual review.

### Summary Checklist for Production
- [ ] **Serving:** Use **vLLM** or **TGI** for efficient serving.
- [ ] **Quantization:** Use **INT8** for 2x memory reduction.
- [ ] **Batching:** Enable **continuous batching**.
- [ ] **Monitoring:** Track **latency, throughput, cost, quality**.
- [ ] **Autoscaling:** Set **min replicas ≥ 2** to avoid cold starts.
- [ ] **Caching:** Cache **common queries** (50%+ hit rate).
- [ ] **Alerts:** Set alerts for **p95 latency > 2s**, **error rate > 1%**.
