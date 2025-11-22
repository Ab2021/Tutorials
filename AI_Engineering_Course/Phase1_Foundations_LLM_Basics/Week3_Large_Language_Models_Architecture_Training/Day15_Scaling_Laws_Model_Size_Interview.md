# Day 15: Scaling Laws & Model Size
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Explain the Chinchilla Scaling Law. How did it change the way we train LLMs?

**Answer:**
- **Concept:** Chinchilla states that for a compute-optimal model, model size ($N$) and training data ($D$) should scale equally ($N \propto D \propto C^{0.5}$). Specifically, you need about **20 tokens per parameter**.
- **Change:** Before Chinchilla (Kaplan era), people trained huge models on small data (GPT-3: 175B on 300B tokens). After Chinchilla, people train smaller models on much more data (LLaMA: 65B on 1.4T tokens). This produces models that are better and cheaper to run.

#### Q2: What is the difference between Compute-Optimal and Inference-Optimal?

**Answer:**
- **Compute-Optimal:** Minimizes the cost to *train* the model to a certain performance level. (Result: Large model, medium data).
- **Inference-Optimal:** Minimizes the total cost of ownership (Training + Inference). Since inference cost depends only on model size, we prefer smaller models trained on massive data (over-trained relative to Chinchilla).
- **Example:** LLaMA-3 8B is trained on 15T tokens (Ratio 1875:1). It is far from compute-optimal but extremely inference-optimal.

#### Q3: Why do we see "Emergent Abilities" in large models?

**Answer:**
- **Phenomenon:** Capabilities (like arithmetic, coding, reasoning) that are near-zero in small models suddenly spike to high performance once the model passes a certain scale threshold (e.g., 10B or 100B params).
- **Hypothesis:**
    1.  **Phase Transition:** The model transitions from memorizing heuristics to learning general algorithms/circuits.
    2.  **Metric Artifact:** Sometimes the metric (e.g., exact match) is discontinuous. The model might be getting "closer" linearly, but the score stays 0 until it gets it perfectly right.

#### Q4: How do you estimate the training cost (in FLOPs) of a Transformer?

**Answer:**
- **Formula:** $FLOPs \approx 6 \cdot N \cdot D$
    - $N$: Number of parameters.
    - $D$: Number of training tokens.
    - Factor 6 comes from: 2 FLOPs/param (forward) + 4 FLOPs/param (backward).
- **Example:** 1B param model on 1B tokens = $6 \times 10^{18}$ FLOPs (6 ExaFLOPs).

#### Q5: We are running out of high-quality text data. What are the strategies to continue scaling?

**Answer:**
1.  **Multi-Epoch Training:** Train on the same data for 4-5 epochs. (Beyond that, returns diminish).
2.  **Synthetic Data:** Use strong models (GPT-4) to generate high-quality textbooks/code for smaller models (Phi-1, Orca).
3.  **Multimodal Data:** Transcribe YouTube videos, podcasts, and meetings to convert audio/video into text.

---

### Production Challenges

#### Challenge 1: Choosing the Right Model Size for Deployment

**Scenario:** You have a budget of $5000/month for inference. You need to process 1M requests/day.
**Analysis:**
1.  **Latency Constraint:** If you need <200ms latency, you might be limited to 7B or 13B models (or need H100s).
2.  **Throughput:** Calculate tokens/sec required.
3.  **Cost:**
    - 70B model: Requires 2x A100 (expensive).
    - 8B model: Fits on 1x A10G (cheap).
**Decision:** If the 8B model meets the quality bar (maybe with fine-tuning), it is vastly superior economically. Always start small and scale up only if necessary.

#### Challenge 2: Training Instability at Scale

**Scenario:** Training a 7B model. Loss spikes and diverges at 20% progress.
**Root Cause:**
- **Batch Size:** Too small for the scale. Large models need large batch sizes (millions of tokens) to smooth out gradient noise.
- **Learning Rate:** Too high.
- **Precision:** BF16 is essential. FP16 often overflows.
**Solution:**
- Restart from last checkpoint.
- Skip the bad batch.
- Reduce LR slightly.
- Ensure Global Batch Size is large (e.g., 4M tokens).

#### Challenge 3: The "Data Wall" in Domain Adaptation

**Scenario:** You want to train a "Legal LLM". You only have 10B tokens of legal text.
**Issue:** 10B tokens is not enough to train a 7B model from scratch (needs ~140B for Chinchilla).
**Solution:**
- **Don't Pre-train:** Continual Pre-training (CPT) or Fine-tuning (SFT) on top of a strong base model (Mistral/LLaMA).
- **Mix Data:** Mix your 10B legal tokens with 100B general tokens (replay buffer) to prevent catastrophic forgetting.

#### Challenge 4: Estimating Hardware Requirements

**Scenario:** Boss asks: "How many GPUs do we need to train a 65B model on 1T tokens in 1 month?"
**Calculation:**
1.  Total FLOPs: $6 \cdot 65e9 \cdot 1e12 = 3.9e23$.
2.  Time: 30 days = $2.6e6$ seconds.
3.  Required FLOPs/sec: $3.9e23 / 2.6e6 = 1.5e17$ (150 PFLOPS).
4.  A100 Performance: ~150 TFLOPS effective.
5.  GPUs needed: $1.5e17 / 1.5e14 = 1000$ GPUs.
**Answer:** "We need about 1000 A100s, boss. Do we have \$2M?"

### Summary Checklist for Production
- [ ] **Model Selection:** Prefer inference-optimal models (LLaMA-3, Mistral) over older compute-optimal ones.
- [ ] **Hardware:** Use BF16-capable hardware (A100/H100) for training >7B models.
- [ ] **Data:** Quality > Quantity. 1T high-quality tokens > 10T Common Crawl.
- [ ] **Scaling:** Don't train from scratch unless you have >100 GPUs. Fine-tune instead.
