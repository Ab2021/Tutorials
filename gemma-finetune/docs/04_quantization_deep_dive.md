# 4. Quantization Deep Dive — How 4-Bit Compression Works

## Table of Contents
- [What Is Quantization?](#what-is-quantization)
- [Number Representations](#number-representations)
- [How 4-Bit Quantization Works](#how-4-bit-quantization-works)
- [NF4 vs FP4](#nf4-vs-fp4)
- [Double Quantization](#double-quantization)
- [Quantization During Training](#quantization-during-training)
- [When to Use Which Precision](#when-to-use-which-precision)
- [bitsandbytes Library](#bitsandbytes-library)

---

## What Is Quantization?

**Quantization** = representing numbers with fewer bits, trading precision for memory.

### Real-World Analogy

Imagine you're recording music:
- **32-bit float (studio quality):** Captures every subtle detail, huge files
- **16-bit float (CD quality):** Indistinguishable from studio to most ears
- **8-bit integer (telephone quality):** Noticeable loss, but understandable
- **4-bit (compressed voice):** Recognizable, loses detail

For neural networks, we're quantizing model **weights** (the learned parameters):

```
   float32         float16         int8           4-bit
┌──────────┐   ┌──────────┐   ┌──────┐     ┌────────┐
│ 4 bytes  │   │ 2 bytes  │   │1 byte│     │0.5 byte│
│ per param│   │ per param│   │ each │     │  each  │
│ 32 bits  │   │ 16 bits  │   │8 bits│     │ 4 bits │
└──────────┘   └──────────┘   └──────┘     └────────┘
  Highest         Good          Decent        Lowest
  memory         memory        memory        memory

2B params:
  8 GB            4 GB           2 GB          1 GB
```

---

## Number Representations

### Float32 (Full Precision)
```
┌───┬──────────┬───────────────────────┐
│ S │ Exponent │       Mantissa        │
│ 1 │  8 bits  │       23 bits         │
└───┴──────────┴───────────────────────┘
  Total: 32 bits = 4 bytes

Value = (-1)^S × 2^(Exponent-127) × (1 + Mantissa)

Range: ±3.4 × 10^38
Precision: ~7 decimal digits
Example: 3.14159265 → stored perfectly
```

### Float16 (Half Precision)
```
┌───┬──────────┬──────────┐
│ S │ Exponent │ Mantissa │
│ 1 │  5 bits  │ 10 bits  │
└───┴──────────┴──────────┘
  Total: 16 bits = 2 bytes

Range: ±65,504
Precision: ~3 decimal digits
Example: 3.14159265 → stored as 3.14 (loses detail)
```

### BFloat16 (Brain Float)
```
┌───┬──────────┬────────┐
│ S │ Exponent │Mantissa│
│ 1 │  8 bits  │ 7 bits │
└───┴──────────┴────────┘
  Total: 16 bits = 2 bytes

SAME RANGE as float32 (same exponent bits!)
LESS PRECISION (7 vs 23 mantissa bits)

Why it exists:
  Google Brain (hence "b"float) found that neural networks
  care more about RANGE (not overflowing) than PRECISION
  (exact decimal places). BFloat16 keeps float32's range
  while using half the memory.
```

### Float16 vs BFloat16 — When to Use Which

```
                    float16           bfloat16
Range:              ±65,504           ±3.4 × 10^38
Precision:          ~3 decimal        ~2 decimal
GPU support:        All NVIDIA        Ampere+ (RTX 3000+)
Overflow risk:      Higher            Lower
Training stability: Can overflow      Very stable

RECOMMENDATION:
  If you have Ampere+ GPU → Use bfloat16 (our default)
  If you have older GPU   → Use float16
```

---

## How 4-Bit Quantization Works

### Block Quantization

Instead of quantizing each weight independently, weights are grouped into **blocks** (default: 64 weights per block).

```
Original weights (float16):
[0.32, -0.15, 0.87, -0.42, ..., 0.23]  ← 64 values

Step 1: Find the absolute maximum in the block
  absmax = max(|0.32|, |-0.15|, |0.87|, ...) = 0.87

Step 2: Normalize to [-1, 1] range
  [0.37, -0.17, 1.00, -0.48, ..., 0.26]   divide by absmax

Step 3: Map to 4-bit values (16 possible values)
  4 bits = 2^4 = 16 discrete values
  
  For NF4: [-1.0, -0.68, -0.44, -0.28, -0.16, -0.06, 0.00, 0.04,
             0.10, 0.17, 0.25, 0.34, 0.45, 0.58, 0.73, 1.00]

  0.37 → nearest is 0.34 (index 11)   → stored as 1011 (4 bits)
  -0.17 → nearest is -0.16 (index 4)  → stored as 0100 (4 bits)
  1.00 → nearest is 1.00 (index 15)   → stored as 1111 (4 bits)

Step 4: Store:
  - 64 × 4-bit indices = 32 bytes (vs. 64 × 2 bytes = 128 bytes in fp16)
  - 1 × fp16 scaling factor (absmax) = 2 bytes
  - Total: 34 bytes vs. 128 bytes = 73% compression!
```

### Dequantization (During Computation)

When the model needs to use a weight for computation:

```
Stored: index=11, scaling_factor=0.87

Step 1: Look up NF4 value for index 11 → 0.34
Step 2: Multiply by scaling factor: 0.34 × 0.87 = 0.296
Step 3: Cast to compute dtype (bfloat16)
Step 4: Use 0.296 in the matrix multiplication

Original value was 0.32 → reconstructed as 0.296
Error: |0.32 - 0.296| = 0.024 (small!)
```

---

## NF4 vs FP4

### FP4 (Float Point 4)
Uses a standard floating-point layout:
```
┌───┬───────┬──────────┐
│ S │  Exp  │ Mantissa │
│ 1 │ 2 bit │  1 bit   │
└───┴───────┴──────────┘

Values: {-6, -4, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 4, 6, ...}
Linearly spaced? No — exponentially spaced
```

### NF4 (Normal Float 4)
Designed specifically for neural network weights:
```
Neural network weights follow a Normal distribution:
  Most weights are near 0
  Fewer weights are far from 0

NF4 places more quantization bins near 0 where most weights live:
  Near 0:  [..., -0.06, 0.00, 0.04, 0.10, ...]  ← densely packed
  Far from 0: [-1.0, -0.68, ..., 0.73, 1.00]      ← sparse

This minimizes the average quantization error!
```

### Comparison

```
Weight distribution:  ██████████████
                      ██████████████████
                    ████████████████████████
                  ████████████████████████████
              ████████████████████████████████████
         ──────────────────────────────────────────── → value
         -1.0      -0.5      0.0      0.5      1.0

FP4 bins:  |     |     |  | || |  |     |     |
                              ← evenly distributed around and near zero

NF4 bins:  |    |   |  | |||| |  |   |    |
                         ← MORE bins near 0 where most weights are!
```

**NF4 wins** because it matches the distribution of neural network weights. The error is smaller on average, leading to better quality.

**Our choice: NF4** (configured as `bnb_4bit_quant_type="nf4"` in config.py)

---

## Double Quantization

### The Problem

Each block of 64 weights has a scaling factor stored in fp16 (2 bytes).

```
2B weights / 64 per block = 31.25M scaling factors
31.25M × 2 bytes = 62.5 MB

That's 62.5 MB JUST for scaling factors!
```

### The Solution

Quantize the scaling factors themselves!

```
Step 1: Group scaling factors into blocks of 256
Step 2: Quantize each scaling factor from fp16 to int8 (1 byte)
Step 3: Each group of 256 has its own fp32 meta-scaling factor

31.25M × 1 byte = 31.25 MB                 (int8 scaling factors)
31.25M / 256 × 4 bytes = 488 KB            (meta-scaling factors)
Total: ~31.7 MB vs. 62.5 MB                 (saved ~30.8 MB)
```

### Is It Worth It?

For Gemma-2B: saves ~30 MB. Small but completely free — no quality impact.
For Gemma-7B: saves ~100 MB. More meaningful.
For a 70B model: saves ~1 GB. Significant!

**Our choice: Enabled** (`use_double_quant=True` in config.py)

---

## Quantization During Training

### How QLoRA Handles the Forward Pass

```
┌──────────────────────────────────────────────────────┐
│                    FORWARD PASS                       │
│                                                      │
│  Input tensor (bfloat16)                             │
│       │                                              │
│       ├──────────────────┬───────────────────┐       │
│       │                  │                   │       │
│       ▼                  ▼                   ▼       │
│  ┌─────────┐      ┌───────────┐      ┌──────────┐   │
│  │ Frozen W │      │  LoRA A   │      │  LoRA B  │   │
│  │ (4-bit)  │      │  (fp16)   │      │  (fp16)  │   │
│  └────┬─────┘      └─────┬─────┘      └────┬─────┘   │
│       │                  │                  │        │
│       │ Dequantize       │                  │        │
│       │ 4-bit → bf16     │                  │        │
│       ▼                  ▼                  │        │
│  ┌──────────┐      ┌──────────┐             │        │
│  │ MatMul   │      │ MatMul   │─────────────┘        │
│  │ bf16     │      │ bf16     │                      │
│  └────┬─────┘      └────┬─────┘                      │
│       │                  │                           │
│       │                  │ × (alpha/rank)            │
│       │                  │                           │
│       └──────────┬───────┘                           │
│                  │ ADD                                │
│                  ▼                                   │
│           Output (bfloat16)                          │
└──────────────────────────────────────────────────────┘
```

### Key Points

1. **Frozen weights stay in 4-bit** on GPU → saves memory
2. **Dequantization happens on-the-fly** during each forward pass → small compute cost
3. **LoRA weights are in fp16/bf16** → full precision for trainable params
4. **Gradients only flow through LoRA** → no gradients for 4-bit weights

---

## When to Use Which Precision

### Decision Tree

```
             Do you have an Ampere+ GPU?
             (RTX 3000+, A100, H100)
                      │
              ┌───────┴───────┐
              │               │
             YES             NO
              │               │
              ▼               ▼
        Use bfloat16      Use float16
        for compute       for compute
              │               │
              └───────┬───────┘
                      │
         Do you have enough VRAM
         for full-precision model?
                      │
              ┌───────┴───────┐
              │               │
         > 40 GB         < 40 GB
              │               │
              ▼               ▼
         Full precision    Use 4-bit
         (no quantization) quantization
                               │
                               ▼
                          NF4 + Double Quant
                          (our default)
```

### Summary Table

| Precision | Bits | Memory per 2B params | Quality | Use When |
|-----------|------|---------------------|---------|----------|
| float32 | 32 | 8 GB | ⭐⭐⭐⭐⭐ | Research, debugging |
| bfloat16 | 16 | 4 GB | ⭐⭐⭐⭐⭐ | Training compute |
| float16 | 16 | 4 GB | ⭐⭐⭐⭐ | Older GPU training |
| int8 | 8 | 2 GB | ⭐⭐⭐⭐ | Inference optimization |
| **NF4** | **4** | **1 GB** | **⭐⭐⭐⭐** | **QLoRA training (ours)** |
| FP4 | 4 | 1 GB | ⭐⭐⭐ | Alternative to NF4 |

---

## bitsandbytes Library

### What It Is

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes) is a CUDA library by Tim Dettmers (the QLoRA paper author) that implements efficient quantization operations on NVIDIA GPUs.

### What It Provides

1. **`Linear4bit`**: A quantized linear layer that stores weights in 4-bit
2. **`Linear8bitLt`**: 8-bit linear layer with outlier handling
3. **Optimizers**: 8-bit Adam, AdaGrad, LAMB for memory savings
4. **Quantization functions**: `quantize_4bit()`, `dequantize_4bit()`

### How It's Used in Our Code

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,               # Store weights in 4-bit
    bnb_4bit_quant_type="nf4",       # Use NF4 algorithm
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16
    bnb_4bit_use_double_quant=True,  # Double quantize scaling factors
)

# When we load the model with this config, HuggingFace Transformers
# automatically replaces all linear layers with bitsandbytes Linear4bit
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    quantization_config=bnb_config,
)
```

### Common Issues

| Issue | Solution |
|-------|----------|
| `CUDA Setup failed` | `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` |
| `No GPU found` | bitsandbytes requires NVIDIA GPU with CUDA |
| Windows not supported | Use WSL2 on Windows |
| CUDA version mismatch | Reinstall: `pip install bitsandbytes --force-reinstall` |
