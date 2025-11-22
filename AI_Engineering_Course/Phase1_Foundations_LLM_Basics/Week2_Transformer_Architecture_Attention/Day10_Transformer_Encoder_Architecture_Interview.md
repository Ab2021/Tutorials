# Day 10: Transformer Encoder Architecture
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Explain the role of residual connections in Transformers. Why are they essential?

**Answer:**

**Without Residuals (Deep Network Problem):**

```python
x_out = layer12(layer11(...layer1(x_in)))

Gradient: ∂L/∂x_in = ∂L/∂x_out × ∂layer12/∂layer11 × ... × ∂layer1/∂x_in
# Product of 12 terms → Vanishing or exploding!
```

**With Residuals:**

```python
x = x + layer(x)  # Add input to output

Gradient: ∂x_out/∂x_in = 1 + ∂layer/∂x_in
# Always has identity (1) path!
```

**Key Benefits:**

1. **Gradient Flow:** Direct path for gradients (no vanishing)
2. **Easier Optimization:** Network learns residuals (easier than full transformation)
3. **Ensemble Effect:** Can view as ensemble of shallow paths

**Production Impact:**

```
Without residuals: Can't train beyond 6 layers
With residuals: Successfully train 100+ layer Transformers
```

---

#### Q2: Why use Layer Normalization instead of Batch Normalization in Transformers?

**Answer:**

**Batch Normalization:**

```python
# Normalize across batch dimension
mean = x.mean(dim=0)  # Across batch
std = x.std(dim=0)

normalized = (x - mean) / std
```

**Problems for Sequences:**

1. **Variable Lengths:**
   ```
   Batch = ["The cat", "A very long sentence"]
   # Need padding → <PAD> tokens affect statistics
   ```

2. **Batch Size Dependency:**
   ```
   Training: batch=32, statistics from 32 samples
   Inference: batch=1, different statistics!
   ```

3. **Sequence Length Dependency:**
   ```
   Different sequence lengths → Different normalization
   ```

**Layer Normalization:**

```python
# Normalize across features (per sample, per position)
mean = x.mean(dim=-1)  # Across features
std = x.std(dim=-1)

normalized = (x - mean) / std
```

**Advantages:**

1. **Independent of Batch:** Works with batch=1
2. **Independent of Sequence Length:** Stable for any length
3. **Position-Specific:** Each position normalized independently

**Interview Follow-up:**
*Q: When would you use Batch Norm in NLP?*

**A:** Rarely. Only for fixed-size inputs (e.g., sentence embeddings after pooling). For sequences, always Layer Norm.

---

#### Q3: The FFN in Transformers uses a 4× expansion (d_ff = 4 × d_model). Why this specific ratio?

**Answer:**

**Configuration:**

```python
d_model = 512
d_ff = 2048  # 4× expansion

FFN(x) = W2(ReLU(W1(x)))
where W1: (512, 2048), W2: (2048, 512)
```

**Reasoning:**

**1. Parameter Capacity:**

```
No expansion (1×): 512² = 262K params
4× expansion: 512×2048 + 2048×512 = 2.1M params

8× more parameters → More representational power
```

**2. Empirical Optimal:**

Ablation studies (from original Transformer paper):

```
1× expansion: Underfitting (insufficient capacity)
2× expansion: Better but not optimal
4× expansion: Best performance/cost trade-off
8× expansion: Marginal improvement, 2× more compute
```

**3. Non-Linearity:**

```python
# Projection to higher dimension
h = ReLU(W1 @ x)  # (2048-dim space)
# More dimensions → More expressiveness

# Project back
y = W2 @ h  # (512-dim)
```

Higher-dimensional intermediate space allows learning complex transformations.

**4. Most Model Parameters:**

```
Attention: 4 × 512² = 1M params (Q,K,V,O)
FFN: 2 × 512 × 2048 = 2.1M params

67% of layer parameters are inFFN!
```

---

#### Q4: Compare pre-norm and post-norm BERT. Which would you use in production?

**Answer:**

**Post-Norm (Original Transformer, BERT):**

```python
x = norm(x + attention(x))
x = norm(x + ffn(x))
```

**Pre-Norm (Modern, e.g., GPT-2):**

```python
x = x + attention(norm(x))
x = x + ffn(norm(x))
```

**Comparison:**

| Aspect | Post-Norm | Pre-Norm |
|--------|-----------|----------|
| Training Stability | Less stable | More stable |
| Warmup Needed | Yes (10K steps) | Less (1K steps) |
| Final Performance | Slightly better | Slightly worse (~1%) |
| Learning Rate Sensitivity | High | Lower |
| Deep Models (>24 layers) | Difficult | Easier |

**Production Choice:**

**For Fine-Tuning Existing Models:**
- Use what model was pre-trained with (usually post-norm for BERT)

**For Training New Models:**
- **Pre-norm** (modern standard)
- Reasons:
  - Faster to train (less warmup)
  - More stable (easier hyperparameter tuning)
  - Scales better to very deep models
  - Performance difference minimal (<1%)

**Real Example:**

```
GPT-2, GPT-3, T5, LLaMA: All use pre-norm
BERT, RoBERTa: Use post-norm (legacy)

New models (2023+): Almost exclusively pre-norm
```

---

#### Q5: Your BERT model has 12 layers but you notice layers 10-12 have very low gradient magnitudes. What could be wrong?

**Answer:**

**Diagnosis:**

```python
# Check gradient norms by layer
for i, layer in enumerate(model.encoder.layers):
    grad_norm = layer.attention.weight.grad.norm()
    print(f"Layer {i}: {grad_norm:.4f}")

# Output:
# Layer 0: 0.8
# Layer 5: 0.6
# Layer 10: 0.1  # Very low!
# Layer 11: 0.05
# Layer 12: 0.02
```

**Possible Causes:**

**1. Vanishing Gradients (Missing/Broken Residuals):**

```python
# Check if residuals properly implemented
class BrokenEncoder(nn.Module):
    def forward(self, x):
        x = self.attention(x)  # Missing: x = x + ...
        x = self.layer_norm(x)
        return x

# Fix: Add residual connections
x = x + self.attention(x)
```

**2. Poor Initialization:**

```python
# If weights initialized too small
nn.init.normal_(layer.weight, std=0.001)  # Too small!

# Upper layers get tiny gradients
# Fix: Use proper initialization
nn.init.xavier_uniform_(layer.weight)
```

**3. Learning Rate Too High for Upper Layers:**

```python
# Solution: Layer-wise learning rate decay
optimizer = AdamW([
    {'params': model.encoder.layers[:6].parameters(), 'lr': 1e-4},
    {'params': model.encoder.layers[6:12].parameters(), 'lr': 5e-5},
])
```

**4. Dead ReLUs in FFN:**

```python
# If many neurons have negative pre-activations
# ReLU outputs 0 → No gradient!

# Check:
activations_pre_relu = []
with torch.no_grad():
    for layer in model.encoder.layers:
        act = layer.ffn[0](x)  # Before ReLU
        dead_ratio = (act < 0).float().mean()
        print(f"Dead neurons: {dead_ratio:.2%}")

# If >50% dead: Initialization or learning rate problem
```

**5. Gradient Clipping Too Aggressive:**

```python
# If clipping norm too low
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Too low!

# Disproportionately affects upper layers
# Fix: Increase max_norm or remove clipping
```

**Debugging Steps:**

1. Verify residual connections exist
2. Check weight initialization
3. Monitor activations (look for dead neurons)
4. Try disabling gradient clipping temporarily
5. Use layer-wise learning rates

---

### Production Challenges

**Challenge: BERT Inference Latency**

**Scenario:**
- BERT-base: 110M parameters
- Inference: 50ms per sample (too slow for real-time)
- Need: <10ms latency

**Solutions:**

**1. Distillation (DistilBERT):**
```
Teacher: BERT-base (12 layers, 110M params)
Student: DistilBERT (6 layers, 66M params)

Result: 2× faster, 60% smaller, 97% performance
```

**2. Quantization:**
```
FP32 → INT8
4× smaller, 2-3× faster
< 1% accuracy loss
```

**3. Layer Pruning:**
```
Remove layers 10-12 (least important)
30% faster, 2-3% accuracy loss
```

**4. ONNX + TensorRT:**
```
Export to ONNX, optimize with TensorRT
2-3× speedup from kernel fusion
```

**Combined:**
```
Distillation + Quantization + ONNX:
50ms → 8ms (6× faster!)
```

---

### Key Takeaways

1. **Residuals enable depth**: Critical for training 12+ layer models
2. **Layer Norm > Batch Norm**: For sequences, always use Layer Norm
3. **4× FFN expansion**: Empirically optimal trade-off
4. **Pre-norm for new models**: More stable, standard for modern Transformers
5. **Gradient flow monitoring**: Essential for debugging deep models
