# Day 70: Cost Management & FinOps
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How do you calculate the cost of an LLM request?

**Answer:**
- **API (OpenAI):** `Cost = (Input_Tokens * Price_In) + (Output_Tokens * Price_Out)`. Note that Output tokens are usually 3x more expensive.
- **Self-Hosted:** `Cost = (GPU_Hourly_Rate / Requests_Per_Hour)`. Utilization is key here. If GPU is idle, cost per request skyrockets.

#### Q2: What is Model Cascading and when should you use it?

**Answer:**
- **Concept:** Using a hierarchy of models (Cheap -> Expensive).
- **Strategy:** Send query to 7B model first. If it refuses or has low confidence, send to GPT-4.
- **Use Case:** High-volume chatbots where 80% of queries are simple ("Reset password"), but 20% require reasoning.
- **Benefit:** Massive cost reduction (often 10x) with minimal quality loss.

#### Q3: Explain the economics of Fine-tuning vs RAG vs Long Context.

**Answer:**
- **RAG:** Cheap storage, expensive retrieval/generation. Good for knowledge.
- **Long Context:** Very expensive (quadratic attention cost, though linear with FlashAttention, still high token count). Good for "needle in haystack".
- **Fine-tuning:** High upfront cost (Training), cheap inference (smaller model, less prompting). Good for style/format/specific tasks.
- **Winner:** Usually RAG + Small Fine-tuned Model is the most cost-effective.

#### Q4: Why are Spot Instances risky for LLM inference?

**Answer:**
- **Preemption:** Cloud provider can take the instance back with 2 minutes notice.
- **Impact:** If running a stateful service (training), you lose progress. If inference, requests drop.
- **Mitigation:** Use for stateless inference behind a load balancer. If node dies, LB reroutes. Use for checkpointed training.

#### Q5: What is "Token Padding" and how does it waste money?

**Answer:**
- In batching, sequences are padded to the length of the longest sequence in the batch.
- **Waste:** You pay compute for processing padding tokens (zeros), which contribute nothing.
- **Solution:** Continuous Batching (vLLM) removes padding waste.

---

### Production Challenges

#### Challenge 1: The "Infinite Loop" Bill

**Scenario:** A bug in the agent loop causes it to call GPT-4 10,000 times in an hour. Bill: $500.
**Root Cause:** No budget caps.
**Solution:**
- **Hard Limits:** Set monthly budget caps in OpenAI/AWS console.
- **Circuit Breaker:** Stop agent after N steps.
- **Rate Limiting:** Limit requests per user.

#### Challenge 2: Low Utilization of Self-Hosted GPUs

**Scenario:** You rented an A100 ($4/hr). Traffic is low at night. Cost per request is $10.
**Root Cause:** Paying for idle compute.
**Solution:**
- **Auto-scaling:** Scale to zero (Serverless) or down to 1 node.
- **Batching:** Queue low-priority requests and run them in a batch when traffic picks up.
- **Spot:** Use Spot instances for base capacity? (Risky). Better: Use Spot for burst.

#### Challenge 3: Token Bloat in RAG

**Scenario:** Retrieving 10 documents (2k tokens each) for every query. Input cost is huge.
**Root Cause:** Poor retrieval relevance.
**Solution:**
- **Re-ranking:** Retrieve 50 docs, re-rank with cheap model, send top 3 to LLM.
- **Summarization:** Summarize docs before sending to LLM.

#### Challenge 4: Unexpected API Price Hikes

**Scenario:** Provider changes pricing or you accidentally switch to a newer, more expensive model version.
**Root Cause:** Lack of monitoring.
**Solution:**
- **Cost Anomaly Detection:** Alert if hourly spend spikes > 50%.
- **Model Pinning:** Pin specific versions (`gpt-4-0613`) instead of `gpt-4` (which might point to a new expensive one).

#### Challenge 5: Development vs Production Cost

**Scenario:** Devs use GPT-4 for testing. Prod uses Llama-3. Devs assume it works, but Prod fails. Or Devs spend $1k testing.
**Root Cause:** Environment mismatch.
**Solution:**
- **Dev Budget:** Cap Dev environment spend.
- **Local LLMs:** Force Devs to use Ollama/Local models for basic testing.

### System Design Scenario: Cost-Optimized Summarization Service

**Requirement:** Summarize 1M articles/day. Budget: $100/day.
**Design:**
1.  **Model:** Fine-tuned Mistral-7B (Cheap, good at summarization).
2.  **Compute:** Spot Instances (A10G).
3.  **Batching:** Run as a batch job, not real-time API. Pack batches tightly.
4.  **Compression:** Truncate articles to 4k tokens.
5.  **Fallback:** None (Accept 1% failure rate from Spot interruptions).

### Summary Checklist for Production
- [ ] **Budget:** Set **Hard Caps** on API spend.
- [ ] **Monitoring:** Dashboard for **Cost per User**.
- [ ] **Optimization:** Use **Caching** and **Cascading**.
- [ ] **Compute:** Use **Spot Instances** for batch workloads.
- [ ] **Model:** **Fine-tune small models** to replace large ones.
