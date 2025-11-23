# Day 57: Scaling Laws & Compute Optimization
## Core Concepts & Theory

### Scaling Laws Fundamentals

**Observation:** Model performance improves predictably with scale

**Key Variables:**
- **N:** Number of parameters
- **D:** Dataset size (tokens)
- **C:** Compute budget (FLOPs)

### 1. Kaplan Scaling Laws (2020)

**Power Law Relationship:**
```
Loss ∝ N^(-α)
Loss ∝ D^(-β)
Loss ∝ C^(-γ)
```

**Findings:**
- **α ≈ 0.076:** Doubling parameters reduces loss by 5%
- **β ≈ 0.095:** Doubling data reduces loss by 6%
- **Compute-Optimal:** For fixed compute, balance model size and data

**Example:**
- 10x compute → 3x larger model + 3x more data

### 2. Chinchilla Scaling Laws (2022)

**Revision:** Kaplan underestimated importance of data

**Optimal Allocation:**
```
N_optimal ∝ C^0.5
D_optimal ∝ C^0.5
```

**Rule of Thumb:**
- **20 tokens per parameter** for optimal training
- **Example:** 70B model → 1.4T tokens

**Comparison:**
- **GPT-3 (175B):** 300B tokens (under-trained)
- **Chinchilla (70B):** 1.4T tokens (optimal)
- **Chinchilla outperforms GPT-3** with fewer parameters

### 3. Compute Budget Allocation

**Training Compute:**
```
C = 6 × N × D
```
- **6:** Approximate FLOPs per token per parameter
- **N:** Parameters
- **D:** Training tokens

**Example:**
- **LLaMA 70B:** 70B params × 1.4T tokens = 588 × 10^21 FLOPs
- **Training time:** ~21 days on 2048 A100 GPUs

### 4. Inference Compute

**Per-Token Cost:**
```
FLOPs_per_token = 2 × N
```

**Total Inference:**
```
C_inference = 2 × N × T_output
```

**Example:**
- **70B model, 100 tokens:** 14 × 10^12 FLOPs

### 5. Emergent Abilities

**Phenomenon:** Capabilities appear suddenly at scale

**Examples:**
- **Few-shot learning:** Emerges at ~10B parameters
- **Chain-of-thought:** Emerges at ~60B parameters
- **Instruction following:** Emerges at ~100B parameters

**Implication:** Some capabilities require minimum scale

### 6. Compute-Optimal Training

**Chinchilla Optimal:**
- **70B model:** 1.4T tokens
- **175B model:** 3.5T tokens

**Over-training:**
- Train beyond optimal for better performance
- **Example:** LLaMA 2 trained on 2T tokens (>optimal)

### 7. Data Scaling

**Data Requirements:**
- **1B model:** 20B tokens
- **10B model:** 200B tokens
- **100B model:** 2T tokens

**Data Availability:**
- **High-quality text:** ~10T tokens available
- **Web data:** ~100T tokens (lower quality)
- **Implication:** Data will become bottleneck

### 8. Efficiency Improvements

**MoE (Mixture of Experts):**
- **Sparse activation:** Only use 10-20% of parameters
- **Effective scaling:** 10x parameters, 2x compute

**Quantization:**
- **INT8:** 2x memory reduction, 1.5x speedup
- **INT4:** 4x memory reduction, 2x speedup

### 9. Real-World Examples

**GPT-3 (2020):**
- **175B parameters**
- **300B tokens** (under-trained by Chinchilla standards)
- **3.14 × 10^23 FLOPs**

**Chinchilla (2022):**
- **70B parameters**
- **1.4T tokens** (optimal)
- **5.76 × 10^23 FLOPs**
- **Outperforms GPT-3**

**LLaMA 2 (2023):**
- **70B parameters**
- **2T tokens** (over-trained)
- **Better than Chinchilla**

**GPT-4 (2023):**
- **Rumored 1.8T parameters** (MoE)
- **~13T tokens**
- **Estimated 10^25 FLOPs**

### 10. Future Trends

**Scaling Limits:**
- **Compute:** Growing 10x every 2 years
- **Data:** High-quality data running out
- **Energy:** Training costs millions of dollars

**Solutions:**
- **Synthetic data:** Generate training data with LLMs
- **Data efficiency:** Better data curation, deduplication
- **Algorithmic improvements:** Better architectures, training methods

### Summary

**Scaling Laws:**
- **Kaplan:** N^(-0.076), D^(-0.095)
- **Chinchilla:** 20 tokens per parameter optimal
- **Emergent abilities:** Appear at 10B, 60B, 100B+ parameters

**Compute Budget:**
- **Training:** C = 6 × N × D
- **Inference:** C = 2 × N × T_output

**Best Practices:**
- Follow Chinchilla scaling (20 tokens/param)
- Over-train if inference cost matters
- Use MoE for efficient scaling

### Next Steps
In the Deep Dive, we will implement scaling law predictions and compute budget planning with complete code examples.
