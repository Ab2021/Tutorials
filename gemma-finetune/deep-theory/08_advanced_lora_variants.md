# 8. Advanced LoRA Variants — Beyond Standard LoRA

## Table of Contents
- [The LoRA Landscape](#the-lora-landscape)
- [AdaLoRA: Adaptive Rank Allocation](#adalora-adaptive-rank-allocation)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](#dora-weight-decomposed-low-rank-adaptation)
- [LoRA+: Differential Learning Rates](#lora-differential-learning-rates)
- [QA-LoRA: Quantization-Aware LoRA](#qa-lora-quantization-aware-lora)
- [LongLoRA: Efficient Long-Context LoRA](#longlora-efficient-long-context-lora)
- [GaLore: Gradient Low-Rank Projection](#galore-gradient-low-rank-projection)
- [LoRA Merging and Composition](#lora-merging-and-composition)
- [Comparison Table](#comparison-table)
- [Which Variant Should You Use?](#which-variant-should-you-use)

---

## The LoRA Landscape

```
2021: LoRA (original)     → 2B+ citations, industry standard
2023: QLoRA               → 4-bit + LoRA, consumer GPU access
2024: DoRA, LoRA+, GaLore → pushing quality and efficiency further

The field moves fast! Here's what each variant adds:
```

---

## AdaLoRA: Adaptive Rank Allocation

### The Problem with Fixed Rank

```
Standard LoRA: Every adapted layer gets the SAME rank (e.g., r=16).

But not all layers are equally important!
  Layer 1 (early):  Learns basic features → might need rank 4
  Layer 10 (middle): Learns complex patterns → might need rank 32
  Layer 18 (late):   Learns task-specific features → might need rank 24

Fixed rank wastes parameters on unimportant layers
and starves important layers.
```

### AdaLoRA's Solution

```
AdaLoRA (Adaptive LoRA) DYNAMICALLY adjusts rank per layer
during training based on importance scores.

Algorithm:
  1. Start with higher rank (e.g., r=32) for all layers
  2. Parameterize ΔW = P × Λ × Q (SVD-like decomposition)
     where Λ = diag(λ₁, λ₂, ..., λᵣ) are singular values
  3. During training, compute importance score for each singular value:
     importance(λᵢ) = |λᵢ| × sensitivity_to_loss
  4. Gradually prune (zero out) low-importance singular values
  5. Each layer ends up with different effective rank

Result:
  Layer 1: r=4  (simple features, few directions needed)
  Layer 5: r=8
  Layer 10: r=24 (complex patterns, many directions needed)
  Layer 18: r=12

Same total parameter budget, but BETTER allocated!
```

### When to Use AdaLoRA

```
✅ When you're not sure what rank each layer needs
✅ When you want to squeeze more quality from the same parameter budget
❌ More complex to implement (not as well supported in PEFT)
❌ Slightly slower training (importance computation overhead)
```

---

## DoRA: Weight-Decomposed Low-Rank Adaptation

### The Key Insight

```
Standard LoRA modifies weights as:
  W' = W + ΔW = W + BA

DoRA separates weight updates into MAGNITUDE and DIRECTION:
  W = m × (W / ||W||)     (decompose into magnitude × unit direction)

  Fine-tuned weight:
  W' = (m + Δm) × (W + BA) / ||W + BA||

Where:
  m = magnitude scalar (learns how "strong" this weight is)
  W/||W|| = direction (normalized to unit length)
  BA = directional update (same as standard LoRA)
  Δm = magnitude update (additional scalar per output neuron)
```

### Why This Helps

```
Analysis from the DoRA paper:
  Full fine-tuning tends to update both magnitude AND direction of weights.
  Standard LoRA primarily updates direction only.
  
  By explicitly allowing magnitude changes, DoRA closes the gap
  between LoRA and full fine-tuning.

Performance (from the paper):
  Task         │ LoRA  │ DoRA  │ Full FT │ Gap Closed
  ─────────────┼───────┼───────┼─────────┼───────────
  Commonsense  │ 80.4  │ 83.7  │ 84.2    │ 87%
  Math         │ 67.8  │ 71.2  │ 72.1    │ 79%
  Code         │ 45.3  │ 48.9  │ 49.5    │ 86%
  
  DoRA closes ~80-87% of the gap between LoRA and full fine-tuning!
```

### Implementation

```python
from peft import LoraConfig

config = LoraConfig(
    r=16,
    use_dora=True,  # Enable DoRA!
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # DoRA adds a magnitude parameter per output neuron per adapted layer
    # Extra parameters: ~2048 per layer × 18 layers = 37K (negligible)
)
```

---

## LoRA+: Differential Learning Rates

### The Problem

```
Standard LoRA: Matrix A and matrix B have the SAME learning rate.

But A and B have very different roles:
  A (down-projection): d×r, initialized randomly
  B (up-projection):   r×d, initialized to ZERO

At the start of training:
  A already has non-zero values → gradients are meaningful
  B is all zeros → gradients are small (the output is zero)

Result: B needs a HIGHER learning rate than A to catch up!
```

### LoRA+'s Solution

```
Use different learning rates for A and B:

  lr_A = base_lr           (standard rate for A)
  lr_B = base_lr × ratio   (higher rate for B, typically ratio=16)

Example:
  lr_A = 2e-4
  lr_B = 2e-4 × 16 = 3.2e-3

Result:
  B learns faster → catches up to A → better convergence → ~2% improvement
  Almost no extra cost (just different LR, same computation)
```

### When to Use LoRA+

```
✅ Free improvement — no extra memory, minimal code change
✅ Especially helpful for smaller ranks (r=4, r=8)
✅ Works with any LoRA-based method

Implementation (manual, as PEFT doesn't natively support this yet):
  param_groups = [
      {"params": [p for n, p in model.named_parameters() if "lora_A" in n],
       "lr": 2e-4},
      {"params": [p for n, p in model.named_parameters() if "lora_B" in n],
       "lr": 3.2e-3},
  ]
  optimizer = AdamW(param_groups)
```

---

## QA-LoRA: Quantization-Aware LoRA

### The Deployment Problem

```
Standard QLoRA workflow:
  1. Load model in 4-bit
  2. Train LoRA in fp16
  3. For deployment: merge LoRA into model
  
  Problem: Merging fp16 LoRA into 4-bit model requires dequantizing,
  merging, then re-quantizing. The re-quantization adds error!
  
  W_4bit + LoRA_fp16 → dequant(W_4bit) + LoRA_fp16 → W_merged_fp16 → requant → W_merged_4bit
                                                                        ↑ QUALITY LOSS!
```

### QA-LoRA's Solution

```
Train LoRA knowing that it will be quantized after merging.

Key idea: Structure LoRA's output to be QUANTIZATION-FRIENDLY:
  - Use group-wise operations that align with quantization block boundaries
  - The merged weight naturally quantizes well

Result: Merged, quantized model retains nearly full fine-tuned quality.
```

---

## LongLoRA: Efficient Long-Context LoRA

### The Problem

```
Gemma-2B max context: 8192 tokens
Attention cost: O(n²) = O(8192²) = 67 million attention scores per head!

What if you want to fine-tune on LONGER contexts (32K, 64K, 128K)?
  Standard attention for 128K tokens:
  128K² = 16.4 BILLION attention scores per head → IMPOSSIBLE on consumer GPU
```

### LongLoRA's Solution

```
Shifted Sparse Attention (S²A):

Instead of every token attending to every other token (O(n²)):
  Split the sequence into groups of G tokens.
  Each token only attends within its group (O(n × G)).
  
  But! Shift the groups by G/2 in alternating attention heads
  to ensure information flows across group boundaries.

┌──────────┬──────────┬──────────┐  Head 1: groups aligned
│  Group 1  │  Group 2  │  Group 3  │
└──────────┴──────────┴──────────┘

   ┌──────────┬──────────┬──────────┐  Head 2: groups SHIFTED
   │  Group 1  │  Group 2  │  Group 3  │
   └──────────┴──────────┴──────────┘

Combined: Information flows across the entire sequence!

Memory: O(n × G) instead of O(n²)
  For 128K tokens, G=4096: 128K × 4K = 524M (vs 16.4B) → 31× cheaper!
```

---

## GaLore: Gradient Low-Rank Projection

### A Different Approach

```
LoRA: Add low-rank adapters to the MODEL (modify W with A×B)
GaLore: Project the GRADIENT to low-rank (compress ∇W, not W itself)

Key insight: If the weight update ΔW is low-rank,
then the GRADIENT ∇W is ALSO approximately low-rank!

Algorithm:
  1. Compute full gradient ∇W ∈ ℝ^(d×d)      (expensive, full-size)
  2. Project: ∇W_projected = P × ∇W × Q       (compress to r×r)
  3. Update optimizer states using ∇W_projected (cheap, only r×r)
  4. Reconstruct: ∇W_approx = Pᵀ × ∇W_projected × Qᵀ
  5. Update: W = W - lr × ∇W_approx

Periodically update P, Q (every T steps) via SVD of ∇W.
```

### GaLore vs LoRA

```
                    │ LoRA          │ GaLore
────────────────────┼───────────────┼──────────────
Changes model arch? │ Yes (adapters)│ No (only optimizer)
Inference overhead  │ None (merged) │ None
Memory savings      │ 70-80%        │ 60-70%
Quality             │ Very good     │ Slightly better
Can do full FT?     │ No (0.13%)    │ Yes! (full rank update, low-rank optimizer)
```

---

## LoRA Merging and Composition

### Merging Multiple LoRA Adapters

```
Scenario: You have two fine-tuned adapters:
  LoRA_review: Trained on product reviews
  LoRA_support: Trained on customer support

Can you merge them?

1. LINEAR MERGE (Simple Average):
   W' = W + α₁ × A₁B₁ + α₂ × A₂B₂
   
   Example: α₁ = 0.7, α₂ = 0.3
   "70% review expert, 30% support expert"
   
   ✅ Simple
   ❌ Quality often degrades (task interference)

2. TIES (Trim, Elect, Sign Merge):
   - Trim low-magnitude changes (< threshold)
   - For conflicts, use majority sign
   - Disjoint changes are simply combined
   
   ✅ Better quality than linear merging
   ❌ More complex

3. DARE (Drop And Rescale):
   - Randomly drop some adapter weights
   - Rescale remaining weights
   - Merge the sparse adapters
   
   ✅ Good empirical results
   ❌ Randomness introduces variance
```

### LoRA Adapter Switching

```python
# Load base model once, swap LoRA adapters as needed
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

# Load review adapter
model = PeftModel.from_pretrained(model, "path/to/lora_review")
output = model.generate(review_prompt)  # Product recommendation

# Switch to support adapter (no model reload!)
model.load_adapter("path/to/lora_support", adapter_name="support")
model.set_adapter("support")
output = model.generate(support_prompt)  # Customer support

# Memory: ONE base model + lightweight adapters
# Much cheaper than loading multiple full models!
```

---

## Comparison Table

| Variant | Key Innovation | Extra Cost | Quality vs LoRA | Maturity |
|---------|---------------|------------|-----------------|----------|
| **LoRA** | Low-rank adapters | Baseline | Baseline | ⭐⭐⭐⭐⭐ |
| **QLoRA** | 4-bit quantization | -70% memory | ~Same | ⭐⭐⭐⭐⭐ |
| **DoRA** | Magnitude + direction | +1% params | +2-4% | ⭐⭐⭐⭐ |
| **LoRA+** | Differential LR | None | +1-2% | ⭐⭐⭐ |
| **AdaLoRA** | Adaptive rank | +10% compute | +1-3% | ⭐⭐⭐ |
| **LongLoRA** | Sparse attention | -O(n²) attention | Same (long ctx) | ⭐⭐⭐ |
| **GaLore** | Gradient projection | +SVD compute | +1-2% | ⭐⭐ |

---

## Which Variant Should You Use?

```
Start with:
  Standard QLoRA (our default) → well-tested, widely supported

If you need better quality:
  Try DoRA (use_dora=True in PEFT config) → easy upgrade

If you want free improvement:
  Try LoRA+ (different LR for A and B) → no extra memory

If you need longer contexts:
  Try LongLoRA → necessary for > 8K contexts

If you're doing research:
  Try GaLore → full-rank updates with LoRA-like memory

For production:
  Standard QLoRA → proven, reliable, well-supported
```
