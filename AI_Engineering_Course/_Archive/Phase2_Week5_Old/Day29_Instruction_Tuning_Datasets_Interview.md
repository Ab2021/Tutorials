# Day 29: Efficient Fine-Tuning (PEFT, LoRA, QLoRA)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why does LoRA use Low-Rank matrices?

**Answer:**
Hypothesis: The "intrinsic dimension" of the change required to adapt a pre-trained model is low.
The model already knows English and Logic. It just needs a small "nudge" to learn a new style or format. We don't need to change the full rank of the weight matrix to achieve this.

#### Q2: What is the difference between LoRA and QLoRA?

**Answer:**
*   **LoRA:** Efficient training method (Adapters). Can be done on fp16 or fp32 models.
*   **QLoRA:** LoRA applied to a **4-bit Quantized** base model. It adds specific innovations (NF4, Double Quant, Paged Optimizers) to make training stable in 4-bit.

#### Q3: Can you train multiple LoRAs?

**Answer:**
Yes. You can have one base model and swap LoRA adapters on the fly (Multi-LoRA serving).
*   User A -> Adapter A (Medical)
*   User B -> Adapter B (Legal)
*   This saves huge VRAM compared to hosting 2 full models.

#### Q4: What happens if you set Rank too high?

**Answer:**
*   **Overfitting:** The adapter has too many parameters and memorizes the small training set.
*   **Cost:** VRAM usage increases.
*   **Diminishing Returns:** Usually $r=64$ is enough. $r=1024$ rarely helps.

### Production Challenges

#### Challenge 1: Catastrophic Forgetting

**Scenario:** You fine-tune Llama-3 on coding. It forgets how to speak English properly.
**Root Cause:** Overwriting knowledge.
**Solution:**
*   **LoRA:** Less prone to this than full finetuning, but still possible.
*   **Replay Buffer:** Mix in 10% of the original pre-training data (general English) into your fine-tuning dataset.

#### Challenge 2: Inference Latency with Adapters

**Scenario:** You load adapters dynamically at runtime. It adds overhead.
**Root Cause:** Matrix multiplication $W x + B A x$.
**Solution:**
*   **Merge:** Merge adapters into weights offline ($W' = W + BA$). Zero overhead.
*   **Optimized Kernels:** Use `punica` or `S-LoRA` kernels for fast multi-LoRA serving.

#### Challenge 3: Quantization Artifacts

**Scenario:** The 4-bit QLoRA model is slightly dumber than the 16-bit LoRA model.
**Root Cause:** Precision loss.
**Solution:**
*   **Evaluation:** Always benchmark the 4-bit model against the 16-bit baseline. If the drop is >2%, consider 8-bit or full precision if budget allows.

### System Design Scenario: Multi-Tenant LLM Platform

**Requirement:** Serve 100 different fine-tuned models for 100 customers.
**Design:**
1.  **Base Model:** Load one Llama-3-70B (4-bit) into GPU VRAM.
2.  **Adapters:** Store 100 LoRA adapters (100MB each) in RAM/Disk.
3.  **Routing:** When Request comes for Customer A, load Adapter A into the GPU (Hot-swap).
4.  **Batching:** Use S-LoRA to batch requests for different adapters together.

### Summary Checklist for Production
*   [ ] **Merge:** Always merge for single-model deployments.
*   [ ] **Rank:** Start with $r=16$.
*   [ ] **Alpha:** Set $\alpha = 2r$.
*   [ ] **Target:** Target `all-linear` modules for best quality.
