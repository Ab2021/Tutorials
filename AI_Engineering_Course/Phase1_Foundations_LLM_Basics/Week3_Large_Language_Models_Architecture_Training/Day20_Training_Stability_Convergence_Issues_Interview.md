# Day 20: Training Stability & Convergence Issues
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why is Pre-LayerNorm preferred over Post-LayerNorm for LLMs?

**Answer:**
- **Post-LN (BERT):** LayerNorm is applied after the residual addition. Gradients must pass through the LayerNorm to reach earlier layers. This can cause gradient vanishing/exploding, requiring a "Warmup" phase to stabilize training.
- **Pre-LN (GPT):** LayerNorm is applied inside the residual block (before Attention/FFN). The residual path is clean ($x + F(x)$). Gradients flow directly through the residual connection (Identity path) to earlier layers. This improves stability significantly, allowing training of much deeper models.

#### Q2: What causes "Loss Spikes" during training?

**Answer:**
- **Data:** A batch containing "poisoned" data (e.g., extremely long sequence of garbage characters, binary data read as text) produces massive activations.
- **Optimization:** These activations lead to massive gradients.
- **Update:** The optimizer takes a huge step, pushing weights into a bad region of the loss landscape.
- **Result:** Loss spikes up. If the update was too large, the model might not recover (divergence).

#### Q3: How does Weight Decay differ in Adam vs. AdamW?

**Answer:**
- **Adam (L2 Regularization):** Adds $\frac{1}{2} \lambda ||w||^2$ to the loss. The gradient is $\lambda w$. In Adam, this gradient is scaled by the adaptive factor $1/\sqrt{v_t}$. This means weights with large gradients get *less* decay.
- **AdamW (Decoupled Weight Decay):** Applies decay directly to the weights: $w_{t+1} = w_t - \eta (\dots) - \eta \lambda w_t$. The decay is uniform for all weights, regardless of their gradient magnitude. This leads to better generalization.

#### Q4: Why do we clip gradients?

**Answer:**
- **Exploding Gradients:** In deep networks (especially RNNs or deep Transformers), gradients can multiply through layers and become huge (NaN/Inf).
- **Clipping:** We calculate the global L2 norm of the gradient vector. If it exceeds a threshold (e.g., 1.0), we scale the entire vector down so its norm equals 1.0.
- **Benefit:** Prevents a single bad batch from destroying the model weights.

#### Q5: What is the "Warmup" phase in learning rate scheduling?

**Answer:**
- **Definition:** Increasing the learning rate linearly from 0 to `max_lr` over the first few thousand steps.
- **Purpose:** At initialization, weights are random. Gradients are noisy and high variance. A large LR would cause chaotic updates. Warmup allows the optimizer to estimate the first and second moments ($m_t, v_t$) of the gradients accurately before taking large steps.

---

### Production Challenges

#### Challenge 1: NaN Loss after 1 week of training

**Scenario:** You trained a 7B model for 7 days. Suddenly, Loss = NaN.
**Diagnosis:**
- **Check Logs:** Did the loss spike before NaN? (Likely data issue).
- **Check Precision:** Are you using FP16? (Likely overflow).
**Solution:**
- **Resume:** Load the checkpoint from 1 hour ago.
- **Skip:** Skip the data shard that was being processed when it crashed.
- **Precision:** Switch to BF16 if possible. If stuck on V100 (FP16), increase the epsilon in LayerNorm or reduce Loss Scale.

#### Challenge 2: Model generates empty strings

**Scenario:** Loss converged, but inference outputs empty strings or `\n\n\n`.
**Root Cause:**
- **EOS Token:** The model learned that predicting `<EOS>` immediately minimizes loss (maybe due to padding issues in training data).
- **Tokenizer:** `<EOS>` token ID mismatch between training and inference.
**Solution:**
- **Check Data:** Ensure training samples are not just empty strings or padding.
- **Penalty:** Apply a length penalty during generation to discourage early stopping.

#### Challenge 3: Training Diverges at the start

**Scenario:** Loss goes up instead of down immediately.
**Root Cause:**
- **LR:** Learning rate is too high (e.g., 1e-3 instead of 3e-4).
- **Init:** Weights initialized with too large variance.
- **Labels:** Labels are shifted? (e.g., predicting current token instead of next token).
**Solution:**
- **Sanity Check:** Try to overfit a single batch. If that fails, it's a bug in code.
- **Lower LR:** Reduce LR by 10x.

#### Challenge 4: "Oscillating" Loss

**Scenario:** Loss goes down, then up, then down...
**Root Cause:**
- **Batch Size:** Too small. Gradients are too noisy.
- **LR:** Too high. Bouncing between walls of the valley.
**Solution:**
- **Accumulation:** Increase gradient accumulation steps to simulate larger batch size.
- **Scheduler:** Ensure LR decay is working (should decrease over time).

### Summary Checklist for Production
- [ ] **Precision:** Use **BF16**.
- [ ] **Norm:** Use **Pre-LN**.
- [ ] **Clip:** Set **Grad Clip = 1.0**.
- [ ] **Data:** Filter out outliers (length, perplexity).
- [ ] **Checkpoint:** Save every 1-2 hours.
