# Day 53: Model Compression & Quantization
## Core Concepts & Theory

### Model Compression Goals

**Objectives:**
- **Reduce Size:** Fit larger models in memory
- **Increase Speed:** Faster inference
- **Lower Cost:** Cheaper deployment

**Techniques:**
- Quantization
- Pruning
- Knowledge distillation
- Low-rank factorization

### 1. Quantization Fundamentals

**Concept:** Reduce precision of weights/activations

**Precision Levels:**
- **FP32:** 32-bit floating point (standard)
- **FP16:** 16-bit floating point (2x compression)
- **INT8:** 8-bit integer (4x compression)
- **INT4:** 4-bit integer (8x compression)

**Benefits:**
- **Memory:** 2-8x reduction
- **Speed:** 1.5-4x faster
- **Cost:** Proportional savings

### 2. Post-Training Quantization (PTQ)

**Process:**
```
1. Train model in FP32
2. Calibrate on small dataset
3. Quantize weights and activations
4. Deploy quantized model
```

**Methods:**

**Dynamic Quantization:**
- Quantize weights statically
- Quantize activations dynamically at runtime
- **Use Case:** LSTM, Transformer inference

**Static Quantization:**
- Quantize both weights and activations statically
- Requires calibration dataset
- **Use Case:** CNN, production deployment

### 3. Quantization-Aware Training (QAT)

**Concept:** Train with quantization in mind

**Process:**
```
1. Insert fake quantization ops during training
2. Model learns to be robust to quantization
3. Convert to actual quantized model
```

**Benefits:**
- Better accuracy than PTQ
- Model adapts to quantization noise

### 4. GPTQ (GPT Quantization)

**Concept:** Layer-by-layer quantization for LLMs

**Algorithm:**
```
For each layer:
  1. Compute Hessian (second-order info)
  2. Quantize weights to minimize reconstruction error
  3. Propagate error to next layer
```

**Benefits:**
- **Accuracy:** 1-2% loss at INT4
- **Speed:** Fast quantization (<1 hour for 70B)

### 5. AWQ (Activation-Aware Weight Quantization)

**Concept:** Protect important weights from quantization

**Process:**
```
1. Identify important weights (high activation magnitude)
2. Keep important weights in higher precision
3. Quantize remaining weights aggressively
```

**Benefits:**
- Better accuracy than GPTQ at same bit-width
- Especially good for INT4

### 6. Pruning

**Concept:** Remove unimportant weights

**Types:**

**Unstructured Pruning:**
- Remove individual weights
- **Sparsity:** 50-90%
- **Speedup:** Requires sparse kernels

**Structured Pruning:**
- Remove entire neurons/channels
- **Sparsity:** 20-50%
- **Speedup:** Works with standard kernels

**Magnitude Pruning:**
```python
# Remove smallest weights
mask = abs(weights) > threshold
pruned_weights = weights * mask
```

### 7. Knowledge Distillation

**Concept:** Train small model (student) to mimic large model (teacher)

**Process:**
```
1. Train large teacher model
2. Generate soft labels with teacher
3. Train small student on soft labels
4. Student learns to mimic teacher
```

**Loss:**
```
L = α * L_hard(student, true_labels) + 
    (1-α) * L_soft(student, teacher_logits)
```

### 8. Low-Rank Factorization

**Concept:** Decompose weight matrix into low-rank factors

**Example:**
```
W (m×n) ≈ A (m×r) × B (r×n)
where r << min(m, n)
```

**Parameters:** `m×n → m×r + r×n`

**Compression:** `r/(m+n)` ratio

### 9. Quantization Formats

**INT8:**
- **Range:** -128 to 127
- **Quantization:** `q = round(x / scale) + zero_point`
- **Dequantization:** `x = (q - zero_point) * scale`

**INT4:**
- **Range:** -8 to 7
- **Storage:** 2 values per byte
- **Use Case:** Extreme compression

**FP8 (H100):**
- **E4M3:** 4-bit exponent, 3-bit mantissa
- **E5M2:** 5-bit exponent, 2-bit mantissa
- **Benefit:** Better than INT8 for some models

### 10. Real-World Examples

**LLaMA 2 70B:**
- **FP16:** 140 GB
- **INT8 (GPTQ):** 35 GB (4x compression)
- **INT4 (GPTQ):** 17.5 GB (8x compression)
- **Accuracy Loss:** <2% at INT4

**BERT:**
- **FP32:** 440 MB
- **INT8 (Dynamic):** 110 MB
- **Speedup:** 2-3x faster

### Summary

**Compression Techniques:**
- **Quantization:** 2-8x compression (INT8, INT4)
- **Pruning:** 2-10x compression (structured/unstructured)
- **Distillation:** Train smaller model
- **Low-Rank:** Factorize weight matrices

**Best Practices:**
- **PTQ:** Quick, 2-3% accuracy loss
- **QAT:** Better accuracy, slower
- **GPTQ/AWQ:** Best for LLMs
- **Pruning:** Combine with quantization

### Next Steps
In the Deep Dive, we will implement quantization, pruning, and distillation with complete code examples.
