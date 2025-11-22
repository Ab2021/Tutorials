# Day 20: Training Stability & Convergence Issues
## Core Concepts & Theory

### The Nightmare of Divergence

Training LLMs is not "fire and forget". Large models are fragile.
**Divergence:** The loss suddenly increases (spikes) and then either becomes NaN (Not a Number) or stays high, refusing to decrease.

### 1. Common Instability Symptoms

1.  **Loss Spikes:** Sudden jumps in loss. Often recoverable, but sometimes fatal.
2.  **NaN Loss:** Loss becomes `nan`. Usually caused by gradient overflow (FP16) or division by zero.
3.  **Slow Convergence:** Loss decreases much slower than expected scaling laws predict.
4.  **Collapse:** Model outputs repetitive text or empty strings.

### 2. Root Causes & Fixes

**A. Learning Rate (LR) Too High:**
- **Symptom:** Oscillating loss or immediate divergence.
- **Fix:** Use a Warmup period (linear increase). Use Cosine Decay. If spiking, lower max LR.

**B. Gradient Clipping:**
- **Symptom:** Exploding gradients in deep layers.
- **Fix:** Clip global gradient norm to 1.0. `torch.nn.utils.clip_grad_norm_`.

**C. Precision Issues (FP16):**
- **Symptom:** NaN loss.
- **Fix:** Switch to **BF16** (Brain Float 16). It has the same dynamic range as FP32, preventing overflow. If HW doesn't support BF16, tune Loss Scaling carefully.

**D. Architecture: Post-LN vs Pre-LN:**
- **Post-LN (BERT):** LayerNorm *after* residual add. Hard to train deep models.
- **Pre-LN (GPT-2/3):** LayerNorm *before* attention/FFN. Much more stable gradients. **Standard for LLMs.**

### 3. The "Loss Spike" Phenomenon

Often caused by "Bad Data" in a specific batch.
- **Example:** A document with 100,000 repeated spaces, or binary garbage interpreted as text.
- **Mechanism:** Generates massive activations -> massive gradients -> updates weights too much -> destroys model.
- **Mitigation:**
    - **Data Filtering:** Remove outliers.
    - **Skip Batch:** If gradient norm is > 10x average, skip the update.

### 4. Weight Decay & Regularization

- **Weight Decay:** Prevents weights from growing too large.
- **Independent Weight Decay (AdamW):** Critical. Standard Adam couples weight decay with gradient updates, which is wrong.
- **Z-Loss (PaLM):** Auxiliary loss to keep logit sums close to zero. Improves stability in large models.
    - $L_z = 10^{-4} \cdot \log^2(\sum e^{logits})$

### Summary of Stability Checklist

| Feature | Recommendation | Why? |
| :--- | :--- | :--- |
| **Architecture** | Pre-LN | Stable gradients |
| **Precision** | BF16 | Prevents overflow |
| **Optimizer** | AdamW | Correct regularization |
| **Clipping** | Norm = 1.0 | Prevents explosion |
| **Warmup** | 1-5% of steps | Safe initialization |

### Next Steps
In the Deep Dive, we will analyze the mathematics of Pre-LN vs Post-LN and implement a "Spike Detection" training loop.
