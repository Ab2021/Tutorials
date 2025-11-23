# Day 53: Model Compression & Quantization
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between dynamic and static quantization?

**Answer:**
**Dynamic Quantization:**
- Quantize weights statically, activations dynamically at runtime.
- **No calibration** needed.
- **Use Case:** LSTM, Transformer (variable input sizes).
- **Speedup:** 2-3x.

**Static Quantization:**
- Quantize both weights and activations statically.
- **Requires calibration** dataset.
- **Use Case:** CNN, production (fixed input sizes).
- **Speedup:** 3-4x (better than dynamic).

#### Q2: How does GPTQ work?

**Answer:**
- **Layer-by-layer quantization** for LLMs.
- **Algorithm:**
  1. Compute Hessian (second-order information).
  2. Quantize weights to minimize reconstruction error.
  3. Propagate error to next layer.
- **Benefits:** 1-2% accuracy loss at INT4, fast quantization.
- **Use Case:** Quantize 70B models to INT4 (8x compression).

#### Q3: What is knowledge distillation?

**Answer:**
- **Concept:** Train small model (student) to mimic large model (teacher).
- **Loss:** `α × hard_loss + (1-α) × soft_loss`
  - **Hard loss:** Cross-entropy with true labels.
  - **Soft loss:** KL divergence with teacher's soft labels.
- **Temperature:** Scale logits before softmax (higher = softer).
- **Benefits:** Student achieves 80-90% of teacher performance with 10x fewer parameters.

#### Q4: What is the difference between structured and unstructured pruning?

**Answer:**
**Unstructured Pruning:**
- Remove individual weights.
- **Sparsity:** 50-90%.
- **Speedup:** Requires sparse kernels (not always faster).

**Structured Pruning:**
- Remove entire neurons/channels.
- **Sparsity:** 20-50%.
- **Speedup:** Works with standard kernels (always faster).

**Example:** Unstructured removes random weights, structured removes entire rows/columns.

#### Q5: How does AWQ differ from GPTQ?

**Answer:**
**AWQ (Activation-Aware Weight Quantization):**
- Protects important weights (high activation magnitude).
- **Better accuracy** than GPTQ at same bit-width.
- **Especially good for INT4**.

**GPTQ:**
- Minimizes reconstruction error uniformly.
- **Faster quantization**.

**When:** AWQ for best accuracy, GPTQ for speed.

---

### Production Challenges

#### Challenge 1: Quantization Accuracy Drop

**Scenario:** Quantized model to INT8. Accuracy dropped 10% (unacceptable).
**Root Cause:** Some layers are sensitive to quantization.
**Solution:**
- **Mixed Precision:** Keep sensitive layers in FP16, quantize rest to INT8.
- **QAT:** Use quantization-aware training instead of PTQ.
- **AWQ:** Use activation-aware quantization.
- **Higher Bits:** Use INT8 instead of INT4.

#### Challenge 2: Pruned Model Not Faster

**Scenario:** Pruned 50% of weights but model is same speed.
**Root Cause:** Unstructured pruning without sparse kernels.
**Solution:**
- **Structured Pruning:** Remove entire neurons/channels.
- **Sparse Kernels:** Use libraries that support sparse operations (e.g., torch.sparse).
- **Recompile:** Ensure model is recompiled to take advantage of sparsity.

#### Challenge 3: Distillation Student Underperforms

**Scenario:** Student model only achieves 60% of teacher performance (expected 80-90%).
**Root Cause:** Student too small or training issues.
**Solution:**
- **Larger Student:** Use bigger student model.
- **Temperature:** Tune temperature (try 2.0, 3.0, 5.0).
- **Alpha:** Adjust hard/soft loss balance (try α=0.3 instead of 0.5).
- **More Data:** Train on more data or longer.

#### Challenge 4: GPTQ Quantization Too Slow

**Scenario:** GPTQ quantization takes 12 hours for 70B model.
**Root Cause:** Large calibration dataset or inefficient implementation.
**Solution:**
- **Smaller Calibration:** Use 128-512 samples instead of 1000+.
- **Batch Processing:** Process multiple samples in parallel.
- **GPU:** Use GPU for Hessian computation.
- **AutoGPTQ:** Use optimized library instead of manual implementation.

#### Challenge 5: Quantized Model Inference Still Slow

**Scenario:** Quantized to INT8 but inference is only 1.2x faster (expected 2-3x).
**Root Cause:** Not using optimized kernels or hardware doesn't support INT8.
**Solution:**
- **Optimized Backend:** Use TensorRT, ONNX Runtime, or vLLM.
- **Hardware:** Ensure GPU supports INT8 (Turing+, A100, H100).
- **Kernel Fusion:** Fuse operations (Conv+BN+ReLU).
- **Batch Size:** Increase batch size to amortize overhead.

### Summary Checklist for Production
- [ ] **Quantization:** Use **INT8** for 4x compression, **INT4** for 8x.
- [ ] **PTQ vs QAT:** Use **PTQ** for quick results, **QAT** for best accuracy.
- [ ] **GPTQ/AWQ:** Use for **LLM quantization** (AWQ for best accuracy).
- [ ] **Pruning:** Use **structured pruning** for guaranteed speedup.
- [ ] **Distillation:** Use for **training smaller models** (80-90% quality).
- [ ] **Calibration:** Use **128-512 samples** for calibration.
- [ ] **Benchmark:** Measure **actual speedup** on target hardware.
