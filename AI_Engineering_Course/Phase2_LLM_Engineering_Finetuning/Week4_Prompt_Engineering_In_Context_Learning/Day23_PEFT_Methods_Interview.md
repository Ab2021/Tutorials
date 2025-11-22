# Day 23: Parameter-Efficient Fine-tuning (PEFT)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why is LoRA more efficient than Full Fine-Tuning (FFT)?

**Answer:**
- **Memory:** FFT requires storing optimizer states (momentum, variance) for all $N$ parameters. LoRA only stores them for the low-rank matrices ($<1\%$ of $N$).
- **Storage:** FFT checkpoints are the size of the full model (e.g., 140GB). LoRA checkpoints are tiny (e.g., 100MB).
- **Compute:** LoRA has slightly fewer FLOPs during backward pass (gradients only flow to A and B), though forward pass is similar.

#### Q2: Can you merge LoRA weights back into the base model? Why would you do that?

**Answer:**
- **Yes.** Since LoRA is a linear operation ($W' = W + BA$), we can compute $BA$ and add it to $W$.
- **Why:** To eliminate inference latency. If we keep them separate, we have to perform two matrix multiplications ($Wx$ and $BAx$) and sum them. Merging results in a standard linear layer with zero overhead.

#### Q3: What is the difference between LoRA and Adapters (Houlsby)?

**Answer:**
- **Architecture:** LoRA adds a parallel branch ($W + \Delta W$). Adapters insert sequential layers (Linear -> ReLU -> Linear) *between* transformer blocks.
- **Latency:** LoRA has zero added latency after merging. Adapters add latency because they increase the depth of the network and cannot be merged into the original weights (due to non-linearity).

#### Q4: Why do we initialize LoRA matrix B to zero?

**Answer:**
- To ensure that at step 0, the model output is identical to the pre-trained base model.
- $\Delta W = B A$. If $B=0$, then $\Delta W = 0$.
- If we initialized both A and B randomly, the initial $\Delta W$ would be non-zero random noise, which would degrade the model's performance immediately and destabilize training.

#### Q5: Explain the "Rank" hyperparameter in LoRA.

**Answer:**
- **Rank ($r$):** The inner dimension of matrices $A$ ($r \times d$) and $B$ ($d \times r$).
- Controls the capacity of the adapter.
- **Low Rank (8-16):** Sufficient for most tasks (Instruction Following, Summarization).
- **High Rank (64-128):** Needed for complex tasks learning new knowledge (e.g., learning a new language or complex coding patterns).

---

### Production Challenges

#### Challenge 1: Serving Multiple LoRA Adapters

**Scenario:** You have one base model (LLaMA-70B) and 50 different customers, each with a custom fine-tuned LoRA adapter.
**Solution:**
- **Multi-LoRA Serving (vLLM / LoRAX):** Load the base model once into VRAM. Load all 50 tiny adapters into CPU/VRAM.
- **Request Routing:** When a request comes in for Customer A, apply Adapter A's weights on-the-fly during the forward pass.
- **Benefit:** Massive cost savings compared to hosting 50 full 70B models.

#### Challenge 2: QLoRA Training Instability

**Scenario:** Training loss spikes or doesn't converge with 4-bit QLoRA.
**Root Cause:**
- **Precision:** 4-bit quantization adds noise.
- **BF16:** Ensure computation type is BF16, not FP16 (overflows).
- **Norm:** LayerNorm should be in float32 for stability.
**Solution:**
- Use `bnb_4bit_compute_dtype="bfloat16"`.
- Use `paged_adamw_32bit`.
- Increase LoRA rank or alpha.

#### Challenge 3: "Catastrophic Forgetting" in LoRA?

**Scenario:** Even with LoRA, the model forgets basic facts.
**Analysis:**
- LoRA updates fewer parameters, so it forgets *less* than FFT, but still forgets.
- **Solution:** Use a lower learning rate or mix in replay data.
- **Rank:** Lower rank restricts the model from changing too much, acting as a regularizer.

#### Challenge 4: Debugging Silent Failures (Adapter not working)

**Scenario:** You trained an adapter, loaded it, but the model output looks exactly like the base model.
**Root Cause:**
- **Forgot to Merge/Enable:** In HuggingFace PEFT, you must call `model.enable_adapter_layers()`.
- **Zero Init:** Did you accidentally initialize A to zero too? (Then gradients are zero).
- **Alpha:** Is alpha set to 0?
**Solution:**
- Print `model.print_trainable_parameters()` before training.
- Check `adapter_config.json`.

### Summary Checklist for Production
- [ ] **Method:** Use **QLoRA** for training (save VRAM), **LoRA** for inference (merge).
- [ ] **Rank:** Start with $r=16$, $\alpha=32$.
- [ ] **Modules:** Target `q_proj`, `v_proj` (minimum) or `all_linear` (best performance).
- [ ] **Serving:** Use **vLLM** with Multi-LoRA support.
