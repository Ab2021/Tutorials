# Day 84: Capstone Project Phase 3 - Refinement & Production
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How do you measure the success of an AI Agent?

**Answer:**
- **Technical Metrics:** Latency (P99), Error Rate, RAGAS Scores (Faithfulness).
- **Business Metrics:** Deflection Rate (Support tickets avoided), User CSAT (Thumbs up), Time Saved.
- **Cost Metrics:** Cost per Query.

#### Q2: What is "Time to First Token" (TTFT) and why does it matter?

**Answer:**
- The time from user hitting "Send" to seeing the first word appear.
- **Psychology:** Users perceive the system as "fast" if TTFT < 1s, even if the full answer takes 10s.
- **Optimization:** Use Streaming. Don't wait for the full generation.

#### Q3: Explain "Blue/Green Deployment" for AI Models.

**Answer:**
- **Blue:** Current Prod (e.g., GPT-3.5 based).
- **Green:** New Version (e.g., GPT-4 based).
- **Traffic:** Route 1% to Green. Monitor metrics. If good, scale to 100%.
- **Rollback:** Instant switch back to Blue if Green fails.

#### Q4: How do you handle "Model Drift"?

**Answer:**
- **Definition:** The model's behavior changes over time (e.g., OpenAI updates GPT-4).
- **Detection:** Continuous Eval. Run the Golden Dataset every night. If scores drop, alert.
- **Mitigation:** Pin model versions (`gpt-4-0613`).

#### Q5: What is the difference between "Faithfulness" and "Relevance"?

**Answer:**
- **Faithfulness:** Is the answer true to the *context*? (Did it hallucinate?).
- **Relevance:** Is the answer useful to the *user*? (Did it answer the question?).
- You can have a faithful answer ("The sky is blue") that is irrelevant ("What time is it?").

---

### Production Challenges

#### Challenge 1: The "Slow" Re-ranker

**Scenario:** Re-ranking 50 docs takes 2 seconds. TTFT suffers.
**Root Cause:** Cross-Encoders are heavy.
**Solution:**
- **ColBERT:** Use Late Interaction models (faster than Cross-Encoders).
- **Parallel:** Re-rank in parallel (if possible).
- **Reduction:** Re-rank only top 20 instead of 50.

#### Challenge 2: Evaluation is Expensive

**Scenario:** Running RAGAS on 1000 questions costs $50 in GPT-4 calls.
**Root Cause:** LLM-as-a-Judge is pricey.
**Solution:**
- **Small Judge:** Use a fine-tuned Llama-3-70B as the judge.
- **Sampling:** Eval only a random 5% of traffic daily.

#### Challenge 3: Streaming & Tool Use

**Scenario:** Agent decides to call a tool. You can't stream "tool calling" tokens to the user.
**Root Cause:** UX complexity.
**Solution:**
- **Loading State:** Show "Searching..." UI state while tool runs.
- **Stream Text:** Only stream the *final answer* tokens.

#### Challenge 4: Secret Leakage

**Scenario:** Developer commits `.env` with OpenAI Key.
**Root Cause:** Human error.
**Solution:**
- **Pre-commit Hooks:** Scan for secrets before commit.
- **GitGuardian:** Monitor repo for leaks.
- **Rotation:** Auto-rotate keys.

#### Challenge 5: Scaling Vector Search

**Scenario:** 10M vectors. Latency increases.
**Root Cause:** Brute force search.
**Solution:**
- **HNSW:** Ensure HNSW index is tuned (M, ef_construct).
- **Sharding:** Distribute vectors across multiple nodes.
- **Quantization:** Use Scalar Quantization (INT8) to reduce RAM usage by 4x.

### System Design Scenario: Scaling to 1M Users

**Requirement:** Global deployment.
**Design:**
1.  **CDN:** Cloudflare for static assets.
2.  **Edge:** Deploy API to Edge locations (if possible).
3.  **Read Replicas:** Read-only Postgres replicas for history.
4.  **Vector Sharding:** Shard Qdrant by User ID.
5.  **Cache:** Aggressive Redis caching of common queries.

### Summary Checklist for Production
- [ ] **Eval:** Run **RAGAS** before every deploy.
- [ ] **UX:** Implement **Streaming**.
- [ ] **Ops:** Use **Kubernetes** for scaling.
- [ ] **Security:** **Pin Model Versions**.
- [ ] **Cost:** Monitor **Token Usage** per user.
