# Day 57: Scaling Laws & Compute Optimization
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between Kaplan and Chinchilla scaling laws?

**Answer:**
**Kaplan (2020):**
- Loss ∝ N^(-0.076), D^(-0.095)
- Recommended larger models, less data
- **Example:** GPT-3 (175B params, 300B tokens)

**Chinchilla (2022):**
- **20 tokens per parameter** optimal
- Kaplan under-estimated importance of data
- **Example:** Chinchilla (70B params, 1.4T tokens) outperforms GPT-3

**Key Insight:** Data matters more than Kaplan suggested.

#### Q2: What are emergent abilities and at what scale do they appear?

**Answer:**
**Emergent Abilities:** Capabilities that appear suddenly at scale

**Thresholds:**
- **Few-shot learning:** ~10B parameters
- **Chain-of-thought reasoning:** ~60B parameters
- **Instruction following:** ~100B parameters

**Implication:** Some capabilities require minimum scale, can't be achieved with smaller models + more training.

#### Q3: How do you calculate training compute for an LLM?

**Answer:**
**Formula:** `C = 6 × N × D`
- **6:** FLOPs per token per parameter (forward + backward)
- **N:** Number of parameters
- **D:** Training tokens

**Example (LLaMA 70B):**
- C = 6 × 70B × 1.4T = 588 × 10^21 FLOPs
- On 2048 A100 GPUs: ~21 days

#### Q4: What is the Chinchilla optimal allocation?

**Answer:**
**Rule:** For fixed compute budget C:
- N_optimal ∝ C^0.5
- D_optimal ∝ C^0.5
- **Ratio:** 20 tokens per parameter

**Example:**
- 70B model → 1.4T tokens (20 × 70B)
- 175B model → 3.5T tokens (20 × 175B)

#### Q5: How does MoE improve scaling efficiency?

**Answer:**
**Sparse Activation:** Only use 10-20% of parameters per token

**Benefits:**
- **10x parameters, 2x compute:** Can scale to 1T+ params
- **Same inference cost:** Only active params used
- **Better performance:** More capacity without proportional cost

**Example:** Mixtral 8x7B has 56B total params, 14B active (similar cost to 14B dense model)

---

### Production Challenges

#### Challenge 1: Under-Trained Model

**Scenario:** Trained 70B model on 300B tokens (GPT-3 style). Performance worse than expected.
**Root Cause:** Under-trained by Chinchilla standards (need 1.4T tokens).
**Solution:**
- **Continue Training:** Train for 1.1T more tokens.
- **Over-train:** Train to 2T tokens for even better performance.
- **Smaller Model:** Train 30B model on 600B tokens instead (same compute, better performance).

#### Challenge 2: Data Scarcity

**Scenario:** Need 3.5T tokens for 175B model but only have 1T high-quality tokens.
**Root Cause:** Insufficient high-quality data.
**Solution:**
- **Smaller Model:** Train 50B model on 1T tokens (Chinchilla optimal).
- **Data Augmentation:** Generate synthetic data with LLMs.
- **Lower Quality Data:** Use web data (lower quality but more available).
- **Multi-epoch:** Train on same data multiple times (diminishing returns).

#### Challenge 3: Compute Budget Exceeded

**Scenario:** Training taking 2x longer than planned.
**Root Cause:** Underestimated compute or inefficient implementation.
**Solution:**
- **Reduce Model Size:** 70B → 50B (save 30% compute).
- **Reduce Data:** 1.4T → 1T tokens (save 30% compute).
- **Optimize:** Use Flash Attention, mixed precision (2-3x speedup).
- **More GPUs:** Scale to 4096 GPUs (2x faster wall-clock time).

#### Challenge 4: Emergent Abilities Not Appearing

**Scenario:** Trained 50B model but no chain-of-thought reasoning.
**Root Cause:** Below threshold (~60B parameters).
**Solution:**
- **Scale Up:** Train 70B model instead.
- **Accept Limitation:** 50B won't have this capability.
- **Prompt Engineering:** Use techniques that don't require chain-of-thought.
- **Fine-tuning:** Fine-tune on chain-of-thought examples (may help).

#### Challenge 5: Inference Cost Too High

**Scenario:** 175B model costs $0.10 per request (too expensive).
**Root Cause:** Large model, many output tokens.
**Solution:**
- **Smaller Model:** Use 70B model (2.5x cheaper, 90% quality).
- **Quantization:** INT8 quantization (2x cheaper).
- **MoE:** Use MoE architecture (same quality, lower cost).
- **Caching:** Cache common responses (free for cache hits).

### Summary Checklist for Production
- [ ] **Follow Chinchilla:** Use **20 tokens per parameter** for optimal training.
- [ ] **Compute Budget:** Calculate `C = 6 × N × D` before starting.
- [ ] **Data Planning:** Ensure sufficient high-quality data.
- [ ] **Emergent Abilities:** Check if target capabilities require larger scale.
- [ ] **Efficiency:** Use **MoE** for scaling, **quantization** for inference.
- [ ] **Monitor:** Track loss vs compute to verify scaling laws.
- [ ] **Over-train:** Consider training beyond optimal for better performance.
