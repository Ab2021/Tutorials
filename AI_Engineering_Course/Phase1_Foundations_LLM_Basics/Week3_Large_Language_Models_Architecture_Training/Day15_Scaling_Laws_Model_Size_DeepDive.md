# Day 15: Scaling Laws & Model Size
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Deriving the Chinchilla Optimal Point

**The Loss Function:**
Hoffmann et al. modeled the loss $L$ as a function of parameters $N$ and tokens $D$:
$$ L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} $$
- $E$: Irreducible loss (entropy of natural language).
- $\frac{A}{N^\alpha}$: Error due to finite model size.
- $\frac{B}{D^\beta}$: Error due to finite data size.

**The Constraint:**
We have a fixed compute budget $C$ (in FLOPs).
Approximation: $C \approx 6 \cdot N \cdot D$.
(6 FLOPs per parameter per token for training).

**Optimization Problem:**
Minimize $L(N, D)$ subject to $C = 6ND$.
Substitute $D = \frac{C}{6N}$:
$$ L(N) = E + \frac{A}{N^\alpha} + \frac{B}{(C/6N)^\beta} $$

Differentiate with respect to $N$ and set to 0 to find the minimum.
The empirical values found by DeepMind were $\alpha \approx 0.5$ and $\beta \approx 0.5$.
Since $\alpha \approx \beta$, the optimal allocation implies equal scaling:
$$ N_{opt} \propto C^{0.5} $$
$$ D_{opt} \propto C^{0.5} $$

**Result:**
For every 10x increase in compute:
- Increase model size by $\approx 3.16x$.
- Increase data size by $\approx 3.16x$.

### 2. The Economics of Training (FLOPs Calculation)

**Estimating Cost:**
Training Cost $\approx$ Total FLOPs / GPU FLOPs per second.

**Total FLOPs:**
$$ FLOPs \approx 6 \cdot N \cdot D $$
- Forward pass: $2N$ FLOPs per token.
- Backward pass: $4N$ FLOPs per token.

**Example: LLaMA-65B**
- $N = 65 \times 10^9$
- $D = 1.4 \times 10^{12}$ (1.4T tokens)
- Total FLOPs $= 6 \cdot 65e9 \cdot 1.4e12 \approx 5.46 \times 10^{23}$ FLOPs.

**Time on A100s:**
- A100 (80GB) peak BF16 performance $\approx 312$ TFLOPS.
- Real-world utilization (MFU) $\approx 50\%$ (good optimization).
- Effective FLOPs/sec $\approx 150 \times 10^{12}$.
- GPU-seconds $= \frac{5.46 \times 10^{23}}{1.5 \times 10^{14}} \approx 3.64 \times 10^9$ seconds.
- GPU-hours $\approx 1,000,000$ hours.
- With 2048 GPUs: $\approx 500$ hours ($\approx 21$ days).

**Cost:**
- Cloud price $\approx \$2$/hour/GPU.
- Total Cost $\approx 1,000,000 \times \$2 = \$2,000,000$.

### 3. Inference Economics (The Case for Over-Training)

**Inference Cost:**
$$ Cost_{inf} \propto N $$
(Depends only on model size, not training data).

**Trade-off:**
- **Chinchilla Optimal:** Minimizes *Training* Compute.
- **Inference Optimal:** Minimizes *Total Cost* (Training + Lifetime Inference).

If you expect to run the model millions of times, you should **over-train** a smaller model.
- Instead of 65B params on 1.4T tokens (Chinchilla).
- Train 8B params on 11T tokens (same compute budget?).
- **Result:** Training cost is similar, but inference is 8x cheaper and faster forever.
- This is the **LLaMA / Mistral strategy**.

### 4. Scaling Data: The Bottleneck

We are running out of high-quality text data.
- **Common Crawl:** Petabytes, but noisy.
- **High Quality (Books, Papers, Code):** Limited.
- **Projected "Data Wall":** 2026-2028.

**Solutions:**
1.  **Synthetic Data:** Use GPT-4 to generate training data for smaller models (Phi-1/2, Orca).
2.  **Multi-Epoch Training:** Train on the same data multiple times. (Muennighoff et al. showed up to 4 epochs is fine, beyond that overfitting starts).
3.  **Multimodal Data:** Train on video/audio transcripts.

### Summary of Mechanics

| Concept | Formula | Implication |
| :--- | :--- | :--- |
| **Training FLOPs** | $6ND$ | Cost scales linearly with params and data. |
| **Inference FLOPs** | $2N$ | Cost scales linearly with params only. |
| **Chinchilla** | $N \approx D/20$ | Optimal for training budget. |
| **LLaMA** | $D \gg 20N$ | Optimal for inference budget. |

### Code: FLOPs Estimator

```python
def estimate_training_cost(params_billions, tokens_trillions, gpu_type='A100', num_gpus=1024, utilization=0.5):
    N = params_billions * 1e9
    D = tokens_trillions * 1e12
    
    total_flops = 6 * N * D
    
    gpu_flops = {
        'A100': 312e12,
        'H100': 989e12,  # FP16/BF16 Tensor Core
        'V100': 125e12
    }[gpu_type]
    
    effective_flops = gpu_flops * utilization
    total_gpu_seconds = total_flops / effective_flops
    
    wall_time_seconds = total_gpu_seconds / num_gpus
    wall_time_days = wall_time_seconds / (24 * 3600)
    
    print(f"Total FLOPs: {total_flops:.2e}")
    print(f"Training Time: {wall_time_days:.2f} days on {num_gpus} {gpu_type}s")
    
# Example: LLaMA-3 70B on 15T tokens
estimate_training_cost(70, 15, 'H100', 4096)
```
