# Quantization Fundamentals
## The Mathematical and Practical Foundation of Neural Network Quantization

> **Part of**: AIMET Deep Dive Series | Document 2 of 15

---

## Table of Contents
1. [Why Quantization?](#1-why-quantization)
2. [Number Representation: From FP32 to INT8](#2-number-representation-from-fp32-to-int8)
3. [Quantization Mathematics](#3-quantization-mathematics)
4. [Symmetric vs Asymmetric Quantization](#4-symmetric-vs-asymmetric-quantization)
5. [Per-Tensor vs Per-Channel Quantization](#5-per-tensor-vs-per-channel-quantization)
6. [Static vs Dynamic Quantization](#6-static-vs-dynamic-quantization)
7. [Quantization Schemes in AIMET](#7-quantization-schemes-in-aimet)
8. [Calibration Methods](#8-calibration-methods)
9. [Quantization Error Analysis](#9-quantization-error-analysis)
10. [Hardware Implications of Quantization](#10-hardware-implications-of-quantization)
11. [Bit-Width Trade-offs (INT8, INT4, FP16, BF16)](#11-bit-width-trade-offs-int8-int4-fp16-bf16)
12. [Mixed Precision Quantization](#12-mixed-precision-quantization)
13. [Quantization of Different Layer Types](#13-quantization-of-different-layer-types)
14. [Quantization Noise Theory](#14-quantization-noise-theory)
15. [Common Quantization Pitfalls](#15-common-quantization-pitfalls)

---

## 1. Why Quantization?

### The Computational Reality of Deep Learning

Modern neural networks are parameter-heavy:
- **ResNet-50**: 25.6 million parameters, 97.8 MB (FP32)
- **BERT-Base**: 110 million parameters, 440 MB (FP32)
- **LLaMA-2 7B**: 7 billion parameters, 28 GB (FP32)

For edge deployment, the constraints are severe:

```
Edge Device Constraints (Typical Mobile SoC, 2024):
├── Available RAM: 4–12 GB (shared with OS and apps)
├── On-chip SRAM (NPU cache): 8–32 MB
├── DRAM Bandwidth: 50–100 GB/s
├── NPU Peak Performance: 10–45 TOPS (INT8)
├── Power Budget (sustained AI): 2–5W
└── Latency Budget: 10–100ms per inference
```

**Key Insight**: The bottleneck is not computation speed — it's **memory bandwidth**. Moving a 28 GB model from DRAM to NPU would take:
```
28 GB / 50 GB/s bandwidth = 0.56 seconds per model load
= Completely unusable for real-time inference
```

After INT8 quantization:
```
7 GB / 50 GB/s = 0.14 seconds
With INT4: 3.5 GB / 50 GB/s = 0.07 seconds
```

---

## 2. Number Representation: From FP32 to INT8

### Floating-Point 32 (FP32)

IEEE 754 standard:
```
Bit Layout (32 bits total):
 31  30-23  22-0
[S] [EXP8] [MAN23]

S   = Sign bit (1 bit)
EXP = Exponent (8 bits, biased by 127)
MAN = Mantissa/Fraction (23 bits)

Value = (-1)^S × 2^(EXP-127) × (1 + MAN/2^23)

Range: ~±3.4 × 10^38
Precision: ~7 significant decimal digits
Special values: ±inf, NaN, denormals

Example: -0.347812
Binary: 1 01111101 01100100000010101110001
```

### Floating-Point 16 (FP16)

Half-precision, a common training format:
```
Bit Layout (16 bits):
[S1][EXP5][MAN10]

Range: ±65,504
Precision: ~3.3 decimal digits
Risk: Overflow for gradients > 65,504
```

### Brain Float 16 (BF16)

Google/Intel format, better range than FP16:
```
Bit Layout (16 bits):
[S1][EXP8][MAN7]

Range: Same as FP32 (~±3.4 × 10^38)
Precision: Only ~2 decimal digits
Advantage: Same exponent range as FP32 → no overflow
           Easy to convert FP32 ↔ BF16 (just truncate mantissa)
```

### Integer 8 (INT8)

Signed or Unsigned:
```
Signed INT8:
Range: -128 to 127
Representation: Two's complement
Zero: 0 (exact)
Storage: 1 byte

Unsigned INT8:
Range: 0 to 255
Storage: 1 byte

Memory savings vs FP32: 4x
```

### Integer 4 (INT4)

Extreme compression:
```
Signed INT4:
Range: -8 to 7
Storage: Half a byte (packed, 2 values per byte)

Unsigned INT4:
Range: 0 to 15

Memory savings vs FP32: 8x
Risk: Very limited precision, accuracy cliff
```

### Comparison Table

| Format | Bits | Range | Precision | Memory vs FP32 | HW Support |
|--------|------|-------|-----------|----------------|-----------|
| FP64 | 64 | ±1.8×10^308 | ~15 decimal | 0.5x (2x larger) | Limited |
| FP32 | 32 | ±3.4×10^38 | ~7 decimal | 1x (baseline) | Universal |
| BF16 | 16 | ±3.4×10^38 | ~2 decimal | 2x smaller | GPU, TPU, Gaudi |
| FP16 | 16 | ±65,504 | ~3 decimal | 2x smaller | GPU, NPU |
| INT8 | 8 | -128 to 127 | integer only | 4x smaller | DSP, NPU, CPU |
| INT4 | 4 | -8 to 7 | integer only | 8x smaller | LLM accelerators |
| INT2 | 2 | -2 to 1 | integer only | 16x smaller | Research |
| Binary | 1 | {-1, +1} | 1 bit | 32x smaller | Research |

---

## 3. Quantization Mathematics

### The Core Linear Quantization Formula

The mapping from floating-point to integer:

```
Quantize:
  q = clip( round(r / S) + Z, q_min, q_max )

Dequantize:
  r̃ = S × (q - Z)

Where:
  r  = original floating-point value
  q  = quantized integer value
  S  = scale factor (positive float)
  Z  = zero point (integer, shifts representation)
  q_min, q_max = integer range bounds
  r̃  = reconstructed floating-point (approximation of r)

For INT8 (symmetric, q_min=-127, q_max=127):
  S = max(|r_min|, |r_max|) / 127
  Z = 0

For INT8 (asymmetric, q_min=-128, q_max=127):
  S = (r_max - r_min) / (q_max - q_min) = (r_max - r_min) / 255
  Z = round(-r_min / S) + q_min
```

### Worked Example

```
Real values to quantize: [-2.1, 0.0, 1.5, 3.8, -1.2]
Target: INT8 (unsigned, [0, 255])

Step 1: Find range
  r_min = -2.1,  r_max = 3.8

Step 2: Compute scale and zero point
  S = (3.8 - (-2.1)) / (255 - 0) = 5.9 / 255 ≈ 0.02314
  Z = round(-(−2.1) / 0.02314) + 0 = round(90.75) = 91

Step 3: Quantize
  q(-2.1) = clip(round(-2.1/0.02314) + 91, 0, 255)
          = clip(round(-90.74) + 91, 0, 255)
          = clip(-91 + 91, 0, 255) = 0
  
  q(0.0)  = clip(round(0.0/0.02314) + 91, 0, 255) = 91
  q(1.5)  = clip(round(64.82) + 91, 0, 255) = clip(156, 0, 255) = 156
  q(3.8)  = clip(round(164.22) + 91, 0, 255) = clip(255, 0, 255) = 255
  q(-1.2) = clip(round(-51.86) + 91, 0, 255) = clip(39, 0, 255) = 39

Step 4: Dequantize (verify roundtrip)
  r̃(-2.1) = 0.02314 × (0 - 91) = -2.1057 ≈ -2.1 ✓
  r̃(0.0)  = 0.02314 × (91 - 91) = 0.0 ✓
  r̃(3.8)  = 0.02314 × (255 - 91) = 3.7950 ≈ 3.8 ✓
```

### Quantization Error

```
Error = r̃ - r

For uniform quantization:
  Maximum error = S/2 (half a step)
  
  With S = 0.02314:
    Max error = 0.01157 (tiny for INT8)

Signal-to-Quantization-Noise Ratio (SQNR):
  SQNR = 6.02 × N + 1.76 dB
  
  N = 8 bits → SQNR ≈ 49.9 dB (excellent)
  N = 4 bits → SQNR ≈ 25.8 dB (limited, noticeable)
  N = 1 bit  → SQNR ≈ 7.8 dB (very low quality)
```

---

## 4. Symmetric vs Asymmetric Quantization

### Symmetric Quantization

The quantization range is centered around zero:

```
Symmetric (Signed INT8, range: -127 to 127):

  Floating Point Range: [-R, +R]
  Scale: S = R / 127
  Zero Point: Z = 0 (always)

  Advantages:
  ✓ No zero-point computation → simpler hardware
  ✓ Zero maps exactly to zero (no bias in zero representation)
  ✓ Faster MAC (Multiply-Accumulate) operations
  ✓ Standard for weight quantization

  Disadvantages:
  ✗ Wastes range if distribution is asymmetric (e.g., post-ReLU)
  ✗ For activations after ReLU (all positive), wastes negative half

Visualization:
  |----[-127...0...+127]----|
  FP range: [-R ... 0 ... +R]
  (Symmetric: both sides equal length)
```

### Asymmetric Quantization

The quantization range can be shifted:

```
Asymmetric (Unsigned INT8, range: 0 to 255):

  Floating Point Range: [min_val, max_val]
  Scale: S = (max_val - min_val) / 255
  Zero Point: Z = round(-min_val / S)

  Advantages:
  ✓ More efficient for non-negative activations (post-ReLU)
  ✓ Full use of the integer range
  ✓ Better accuracy for asymmetric distributions

  Disadvantages:
  ✗ Non-zero "zero point" adds overhead in hardware
  ✗ Zero point must be subtracted before each multiplication
  ✗ Slightly more complex hardware implementation

Visualization (Post-ReLU activations, range [0, R]):
  |----[0...255]----|
  FP range: [0 ... R]
  (All quantization bits used for positive values)
```

### AIMET Configuration for Symmetric vs Asymmetric

```python
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel

sim = QuantizationSimModel(
    model=model,
    dummy_input=dummy_input,
    default_output_bw=8,          # 8-bit activations
    default_param_bw=8,           # 8-bit weights
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    config_file='path/to/config.json'
)

# Config file controls symmetric vs asymmetric per layer:
# {
#   "defaults": {
#     "is_symmetric": true  ← symmetric by default
#   },
#   "params": {
#     "is_symmetric": true  ← weights: symmetric
#   },
#   "op_type": {
#     "Conv": {
#       "is_input_quantized": true,
#       "is_symmetric": false  ← activations: asymmetric
#     }
#   }
# }
```

### Industry Practice

| Tensor Type | Common Scheme | Reason |
|-------------|---------------|--------|
| Weights (Conv/Linear) | Symmetric | Zero = 0 (biases don't need shift) |
| Weights (LLM) | Symmetric or Asymmetric | Depends on architecture |
| Activations (post-ReLU) | Asymmetric | Range is [0, max], not symmetric |
| Activations (post-Sigmoid/Tanh) | Asymmetric | Bounded range |
| Biases | FP32 (often not quantized) | Small impact, sensitive |
| Embedding vectors | Asymmetric | Different per-row distributions |

---

## 5. Per-Tensor vs Per-Channel Quantization

### Per-Tensor (Per-Layer) Quantization

One scale and zero-point for the **entire weight tensor**:

```
Weight tensor shape: [out_ch=64, in_ch=3, kH=3, kW=3]
= 1728 values total

Per-Tensor: 
  1 scale S, 1 zero-point Z for all 1728 values
  
  Problem:
  Channel 0 weights range: [-0.01, 0.01]  (fine-grained)
  Channel 60 weights range: [-5.2, 5.1]   (wide range)
  
  Shared scale must accommodate [-5.2, 5.1]
  → Channel 0's small variations get "crushed" into very few INT8 values
  → High quantization error for small-range channels
```

### Per-Channel Quantization

One scale and zero-point **per output channel**:

```
Weight tensor shape: [out_ch=64, in_ch=3, kH=3, kW=3]

Per-Channel:
  64 scales S[0..63], 64 zero-points Z[0..63]
  Each channel gets its own optimal range
  
  Channel 0: S = 0.01/127 ≈ 7.87e-5 (fine-grained for small range)
  Channel 60: S = 5.2/127 ≈ 0.0409 (wider for large range)
  
  Result: Each channel uses its full INT8 precision
  → Much lower quantization error
  → Typically 1–2% better accuracy than per-tensor
```

### Hardware Implications

```
Per-Tensor: Simple, fast
├── 1 scale lookup per layer
├── Works on all quantized hardware
└── Lower accuracy but maximum speed

Per-Channel: More accurate, slight overhead
├── N scale lookups per layer (N = out_channels)
├── Requires hardware support for channel-wise dequantization
├── Fully supported on Qualcomm Hexagon, NVIDIA Tensor Cores
└── Standard for weight quantization in production
```

### AIMET Per-Channel Configuration

```python
# Enable per-channel quantization in AIMET config file:
{
    "defaults": {
        "per_channel_quantization": false  # Default: per-tensor
    },
    "params": {
        "per_channel_quantization": true   # Weights: per-channel
    }
}

# Programmatic approach:
sim = QuantizationSimModel(model, dummy_input)

# Enable per-channel for specific layer
layer = sim.model.conv1
layer.param_quantizers['weight'].enable_per_channel_quantization()
```

### Per-Token Quantization (for LLMs)

A variant for transformer models:

```
Attention pattern: [batch, seq_len, hidden_dim]

Per-Tensor: 1 scale for entire activation tensor
Per-Channel: 1 scale per hidden_dim slice
Per-Token: 1 scale per token (seq_len position)

For LLMs with activation outliers:
  Standard per-tensor INT8 fails (outliers dominate scale)
  Per-token captures token-specific distributions
  → Enables practical W8A8 quantization for LLMs
```

---

## 6. Static vs Dynamic Quantization

### Static Quantization

Scale and zero-point computed **offline** using calibration data:

```python
# AIMET Static Quantization Workflow:

# Step 1: Create simulation model
sim = QuantizationSimModel(model, dummy_input)

# Step 2: Calibrate with representative data (OFFLINE)
def calibrate(model, args):
    for batch in calibration_loader:
        model(batch)  # Run forward pass; AIMET observes ranges

sim.compute_encodings(
    forward_pass_callback=calibrate,
    forward_pass_callback_args=None
)

# At this point: all scales/zero-points are FIXED

# Step 3: Deploy (no scale computation at runtime)
output = sim.model(test_input)  # Fast! Pre-computed encodings
```

**Properties**:
- ✅ Fast inference (scale is pre-computed)
- ✅ Suitable for hardware with fixed quantization parameters
- ✅ Standard for edge deployment
- ⚠️ Requires calibration dataset to be representative
- ⚠️ May fail if real-world distribution differs from calibration

### Dynamic Quantization

Scale and zero-point computed **at runtime** for each input:

```python
# PyTorch Dynamic Quantization (simpler case)
import torch

model_dyn = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Only quantize Linear layers
    dtype=torch.qint8
)

# Scales computed fresh for each input during inference
# Works best for models with highly variable activation ranges (LSTMs, etc.)
```

**Properties**:
- ✅ Better accuracy for variable-range activations
- ✅ No calibration dataset needed
- ❌ Scale computation overhead at runtime
- ❌ Not suitable for DSP/NPU (which expect static encodings)
- ❌ AIMET focuses on static quantization for hardware deployment

### Weight-Only Quantization

Weights quantized statically; activations remain floating-point:

```
Execution Model:
  Weights stored as INT4/INT8 (compressed in memory)
  At runtime: INT → FP dequantization before compute
  Computation: FP16 or FP32

Use Case: LLM deployment where:
  - Memory bottleneck dominates (weight size)
  - Activation quantization is too inaccurate
  - Hardware supports FP16 compute but limited INT8 engine

Performance Profile:
  Memory: ~4-8x reduction (INT4 weights)
  Compute: No speedup vs FP16 (dequantize before MAC)
  Bandwidth: 4-8x improvement (critical for LLM token generation)
```

---

## 7. Quantization Schemes in AIMET

AIMET's `QuantScheme` enum defines how min/max ranges are determined:

### TF Scheme (Min-Max / Absolute Min-Max)

```python
QuantScheme.post_training_tf

# Algorithm:
# 1. Run calibration data through model
# 2. Track absolute minimum and maximum observed values
# 3. Set range = [observed_min, observed_max]

# Properties:
# + Simple, deterministic
# + Guarantees no clipping of calibration data
# - Highly sensitive to outliers
# - A single extreme value widens the range for all values

# Mathematical formula:
# scale = (max_val - min_val) / (2^bits - 1)
# zero_point = round(-min_val / scale) + qmin

# Example problem:
# Activation values: [0, 0.1, 0.2, ..., 1.0, ..., 50.0] (outlier at 50)
# TF range: [0, 50] → all values in [0,1] get only 2-3 INT8 levels
```

### TF-Enhanced Scheme (Signal-to-Quantization-Noise Ratio / MSE)

```python
QuantScheme.post_training_tf_enhanced  # Default and recommended

# Algorithm:
# 1. Run calibration data through model
# 2. Collect histogram of observed values
# 3. Search over candidate min/max thresholds
# 4. Select thresholds that minimize Mean Squared Error (MSE) between
#    original and quantized representations

# Properties:
# + Robust to outliers (clips them, allows denser quantization of main distribution)
# + Typically 1-2% better accuracy than TF scheme
# + Recommended default for most models
# - Slightly slower calibration (histogram analysis)

# Example benefit:
# Activation values: [0, 0.1, 0.2, ..., 1.0, ..., 50.0] (outlier at 50)
# TF-Enhanced: Clips outlier, range set to [0, 1.05]
#              → Main distribution [0,1] gets full INT8 resolution
```

### Percentile Scheme

```python
# Not a distinct QuantScheme in AIMET but achievable via custom config
# Conceptually: set range to [P_low percentile, P_high percentile]

# Common choice: [0.001%, 99.999%] percentiles
# Effect: Clips top and bottom 0.001% of values as outliers

# In practice: TF-Enhanced achieves similar results
# AIMET's TF-Enhanced internally uses MSE minimization which
# effectively clips outlier percentiles
```

### Range Learning (for QAT)

```python
QuantScheme.training_range_learning_with_tf_init
QuantScheme.training_range_learning_with_tf_enhanced_init

# Used during Quantization-Aware Training (QAT)
# Initial ranges from TF or TF-Enhanced calibration
# Then ranges are LEARNED via backpropagation during training

# Scale parameters become differentiable:
# dL/d(scale) computed via straight-through estimator
# → Ranges adapt to minimize task loss, not just reconstruction error
```

---

## 8. Calibration Methods

### What is Calibration?

Calibration is the process of running representative data through a model to determine optimal quantization scales. The quality of calibration data directly impacts quantized accuracy.

### Calibration Data Requirements

```python
# AIMET Calibration Best Practices:

# 1. Size: 512–2048 samples is typically sufficient
#    Diminishing returns beyond ~1000 samples

# 2. Distribution: Must match deployment distribution
#    - Use training set validation split
#    - Include edge cases if they matter (e.g., night images for self-driving)
#    - Not the test set (leads to overfitting)

# 3. Batch size: 
#    - For per-tensor: Any batch size
#    - For per-channel/per-token: Match expected inference batch size

# Example calibration setup:
calibration_dataset = Subset(train_dataset, indices=range(1024))
calibration_loader = DataLoader(
    calibration_dataset,
    batch_size=64,
    shuffle=True,  # Shuffle for diversity
    num_workers=4
)

def calibration_fn(model, args):
    model.eval()
    with torch.no_grad():
        for images, _ in calibration_loader:
            model(images.cuda())

sim.compute_encodings(
    forward_pass_callback=calibration_fn,
    forward_pass_callback_args=None
)
```

### Calibration Data-Free Approaches

Some AIMET techniques (CLE, Analytical Bias Correction) require NO calibration data:

```python
# Cross-Layer Equalization - Data-Free
from aimet_torch.cross_layer_equalization import equalize_model

equalized_model = equalize_model(model, input_shapes=[(1, 3, 224, 224)])
# No calibration data needed! Works purely on weight statistics.

# Analytical Bias Correction - Uses BatchNorm statistics (not input data)
from aimet_torch.bias_correction import correct_bias, AnalyticalBiasCorrectParams

params = AnalyticalBiasCorrectParams()
correct_bias(model, params, num_quant_samples=None)  # No data needed
```

### Calibration Quality Metrics

```python
# After calibration, evaluate quality:

# Method 1: Direct accuracy
int8_accuracy = evaluate(sim.model, val_loader)
fp32_accuracy = evaluate(model, val_loader)
accuracy_drop = fp32_accuracy - int8_accuracy
print(f"Accuracy drop: {accuracy_drop:.3f}%")

# Method 2: Per-layer quantization error (QuantAnalyzer)
from aimet_torch.quant_analyzer import QuantAnalyzer, CallbackFunc

eval_cb = CallbackFunc(evaluate, val_loader)
qanalyzer = QuantAnalyzer(sim.model, dummy_input, eval_cb)
qanalyzer.analyze()  # Generates per-layer sensitivity reports
```

---

## 9. Quantization Error Analysis

### Sources of Quantization Error

```
1. Rounding Error (Unavoidable)
   Cause: Discrete integer levels cannot represent all FP32 values
   Formula: ε_round = r - round(r/S) × S, where |ε| ≤ S/2
   Mitigation: Smaller scale S (higher precision)

2. Clipping Error (Avoidable)
   Cause: Values outside [q_min×S + offset, q_max×S + offset] are clipped
   Impact: Catastrophic for outlier-heavy distributions
   Mitigation: TF-Enhanced scheme, CLE, SmoothQuant

3. Accumulated Propagation Error
   Cause: Small layer-wise errors compound through deep networks
   Impact: Final layer errors >> individual layer errors
   Mitigation: Mixed precision, AdaRound, QAT

4. Covariance Shift
   Cause: Batch norm statistics collected with FP32, but INT8 introduces bias
   Impact: Batch norm layers become misaligned post-quantization
   Mitigation: BN Re-estimation (AIMET)
```

### Layer Sensitivity Analysis

Not all layers are equally sensitive to quantization:

```
Typical Sensitivity Pattern (ResNet-50):

Layer            | Sensitivity | Notes
-----------------|-------------|------------------------
conv1 (first)    | HIGH        | Input distribution critical
layer1.conv1     | MEDIUM      | Early features
layer2.conv2     | LOW         | Mid-level features
layer3.conv2     | LOW         | High-level features
layer4.conv1     | HIGH        | Final feature extraction
fc (last)        | VERY HIGH   | Direct output impact

Key insight: First and last layers are most sensitive
→ Common practice: Keep first/last layers at FP32 or INT16
```

```python
# AIMET QuantAnalyzer: Find sensitive layers automatically
from aimet_torch.quant_analyzer import QuantAnalyzer

analyzer = QuantAnalyzer(
    model=model,
    dummy_input=dummy_input,
    forward_pass_callback=eval_callback
)

# Find layer with most impact on accuracy
sensitivity_results = analyzer.check_per_layer_quantization_encodings(
    sim, num_batches=64
)

# Results: DataFrame with per-layer accuracy impact
print(sensitivity_results)
# Layer          | FP32 Acc | INT8 Acc | Drop
# conv1          | 76.1%    | 73.2%    | 2.9%  ← SENSITIVE
# layer1.conv1   | 76.1%    | 75.8%    | 0.3%
# layer2.conv2   | 76.1%    | 76.0%    | 0.1%
```

---

## 10. Hardware Implications of Quantization

### How INT8 Multiplication Works on Hardware

```
FP32 Matrix Multiply (GEMM):
  C[i,j] = Σ_k A[i,k] × B[k,j]
  Hardware: 32-bit FPU, ~1 cycle per op, high energy

INT8 Matrix Multiply:
  C_int[i,j] = Σ_k A_int[i,k] × B_int[k,j]
  Hardware: 8-bit integer multiplier → 32-bit accumulator
  Energy: ~4x less than FP32 per operation
  Throughput: 4x more ops per clock cycle (4× INT8 per FP32 width)

Dequantize output:
  C[i,j] = S_A × S_B × (C_int[i,j] - Z corrections)
  This is done once per output, not per MAC
```

### Memory Bandwidth Analysis

```
Memory Access Pattern for ResNet-50 Inference:

FP32 Model:
  Weights: 97.8 MB DRAM access
  Activations: ~200 MB per inference (layer-by-layer)
  Total bandwidth: ~300 MB per inference

INT8 Model:
  Weights: 24.4 MB DRAM access (4x reduction)
  Activations: ~50 MB per inference (4x reduction)
  Total bandwidth: ~75 MB per inference

At 50 GB/s DRAM bandwidth:
  FP32: 300 MB / 50 GB/s = 6ms memory time
  INT8: 75 MB / 50 GB/s = 1.5ms memory time
  → 4x faster even before compute speedup!
```

### Quantcomm Hexagon DSP Execution Model

```
Hexagon DSP (Snapdragon 8 Gen 3):

┌────────────────────────────────────────┐
│           Hexagon Tensor Unit          │
│                                        │
│  ┌──────────┐    ┌──────────────────┐  │
│  │ INT8 MAC │    │   Weight Buffer  │  │
│  │  Engine  │    │  (Scratchpad)    │  │
│  │          │    │  (2MB on-chip)   │  │
│  └──────────┘    └──────────────────┘  │
│                                        │
│  ┌────────────────────────────────┐    │
│  │  Activation Buffer (1MB SRAM)  │    │
│  └────────────────────────────────┘    │
│                                        │
│  Peak: 16 TOPS INT8                    │
└────────────────────────────────────────┘
        ↕ DRAM Interface (50 GB/s)
┌────────────────────────────────────────┐
│          LPDDR5 Memory (12GB)           │
│          (Model weights stored here)    │
└────────────────────────────────────────┘

Critical constraint: 
  Weight buffer = 2MB → only weights fitting in 2MB compute fast
  Larger models must page weights from DRAM → slower
  → INT4 allows 2x more parameters in same SRAM → massive speedup
```

---

## 11. Bit-Width Trade-offs (INT8, INT4, FP16, BF16)

### Accuracy vs Compression Comparison

```
Model: ResNet-50, ImageNet, FP32 baseline = 76.13%

| Precision | Top-1 Acc | Acc Drop | Model Size | Speedup (Hexagon) |
|-----------|-----------|----------|------------|-------------------|
| FP32      | 76.13%    | 0.00%    | 97.8 MB    | 1.0x              |
| FP16      | 76.10%    | 0.03%    | 48.9 MB    | 1.5x              |
| BF16      | 76.08%    | 0.05%    | 48.9 MB    | 1.5x              |
| INT8 (PTQ)| 75.80%    | 0.33%    | 24.4 MB    | 5–8x              |
| INT8 (QAT)| 76.05%    | 0.08%    | 24.4 MB    | 5–8x              |
| INT4 (PTQ)| 71.20%    | 4.93%    | 12.2 MB    | 8–12x             |
| INT4 (QAT)| 74.80%    | 1.33%    | 12.2 MB    | 8–12x             |

Note: INT4 without careful optimization (GPTQ/AdaRound) can catastrophically fail
```

### The INT4 Accuracy Cliff

```
Problem: Rounding error scales as O(1/2^bits)

INT8: S ≈ dynamic_range/255  → 256 discrete levels
INT4: S ≈ dynamic_range/15   → 16 discrete levels (16x coarser!)

For a weight with range [-1, 1]:
  INT8: step size = 2/255 ≈ 0.0078 (fine)
  INT4: step size = 2/15  ≈ 0.133  (16x coarser)

For complex models with many channels competing for bit levels,
this coarseness causes severe accuracy degradation.

Solutions for INT4:
  1. Per-channel / per-group quantization (reduces range per group)
  2. AdaRound (optimizes which integer to round to)
  3. GPTQ (uses Hessian information to compensate)
  4. AWQ (identifies and protects salient weights)
  5. QAT (model learns to tolerate the coarser representation)
```

### When to Use Which Precision

```
Decision Guide:
├── FP16/BF16: When accuracy is critical, size not the bottleneck
│             (GPU serving, cloud inference)
├── INT8: ✅ The sweet spot for most edge deployments
│         Best balance of accuracy, speed, memory
│         Supported by virtually all edge NPUs
├── INT4 weights + INT8 activations: 
│         LLM inference where weight size dominates
│         (e.g., running 7B models on mobile devices)
└── INT4 weights + INT4 activations:
          Extreme edge cases (microcontrollers, wearables)
          Requires careful architecture design
```

---

## 12. Mixed Precision Quantization

### Concept

Not all layers need the same bit-width. Mixed precision assigns different precisions to different layers based on sensitivity:

```
Example Mixed Precision Assignment:
┌─────────────────────────────────────────────────────────┐
│ Layer          │ Precision │ Rationale                  │
├────────────────┼───────────┼────────────────────────────┤
│ input_stem     │ INT8      │ Less sensitive, good comp.  │
│ layer1         │ INT4      │ High compression desired    │
│ layer2         │ INT4      │ High compression desired    │
│ layer3         │ INT8      │ Higher sensitivity observed │
│ final_conv     │ INT8      │ Critical for output quality │
│ classifier     │ INT8      │ Output layer, very sensitive│
└─────────────────────────────────────────────────────────┘

Overall: ~6-bit average while maximizing accuracy
```

### Hardware-Aware Quantization (HAQ)

Reinforcement learning approach to automatically find optimal mixed precision:

```python
# Conceptual HAQ workflow (not direct AIMET API but principle):

# Environment: Model + target hardware
# State: Current layer precision assignment
# Action: Change a layer's bit-width
# Reward: -latency (minimize) subject to accuracy constraint

# AIMET-compatible approach via QuantAnalyzer + manual override:
from aimet_torch.quant_analyzer import QuantAnalyzer

# 1. Run sensitivity analysis
analyzer.check_per_layer_quantization_encodings(sim)

# 2. Identify sensitive layers
sensitive_layers = ['conv1', 'fc']

# 3. Override to INT8 for sensitive, INT4 for others
for name, module in sim.model.named_modules():
    if name in sensitive_layers:
        module.output_quantizers[0].bitwidth = 8
    else:
        module.output_quantizers[0].bitwidth = 4
```

---

## 13. Quantization of Different Layer Types

### Convolutional Layers

```
Conv2d Weight Tensor: [out_ch, in_ch, kH, kW]
  Quantization: Per-channel (per out_ch) or per-tensor
  Typical setting: INT8, symmetric

Conv2d Activation (output): [batch, out_ch, H, W]
  Quantization: Per-tensor
  Typical setting: INT8, asymmetric (post-ReLU, all positive)
```

### Fully Connected / Linear Layers

```
Linear Weight: [out_features, in_features]
  Quantization: Per-channel (per out_features)
  Typical setting: INT8 or INT4 (for LLMs)

Linear Activation:
  Often more varied than Conv (no spatial structure)
  Typical setting: INT8 asymmetric
```

### Attention Mechanisms (Transformers)

```
Multi-Head Attention has unique challenges:
  
  Q, K, V projections:
    Weight: INT8 per-channel
    Activation: INT8 per-tensor or per-token
  
  Attention scores (Q×K^T / √d):
    Has extreme outliers for long sequences
    May need: per-token quantization or higher precision
  
  Softmax output:
    Range [0, 1] - naturally bounded
    Asymmetric INT8 works well
  
  V projection output:
    Standard INT8
  
  AIMET specific considerations for attention:
    - Quantize Q, K, V projections independently
    - Attention score quantization requires custom configuration
    - Consider keeping softmax + attention weight multiply in FP32
```

### Batch Normalization

```
BN in Training:
  y = γ × (x - μ) / √(σ² + ε) + β
  Parameters: γ (scale), β (bias), running_mean (μ), running_var (σ²)

BN in Inference (Folded):
  After BN folding into preceding Conv:
  The BN layers disappear from the graph
  → No quantization needed for folded BN

BN in Inference (Un-folded):
  Scale (γ) and bias (β) can be INT8 or FP32
  Running statistics: FP32 (statistical, not compute-heavy)
  
AIMET recommendation: Always fold BN before quantization
```

### Activation Functions

```
ReLU:
  Range: [0, ∞) in theory, [0, R] in practice
  Quantization: Asymmetric INT8 (natural fit)
  Scale set by calibration max

ReLU6:
  Range: [0, 6] — bounded!
  Quantization: Perfect for INT8, very accurate
  Scale = 6/255 = 0.02353 (fixed, no calibration needed)
  Common in MobileNet architectures

Sigmoid:
  Range: (0, 1) — bounded
  Quantization: Very accurate with INT8 or even INT4
  
GeLU/SiLU (LLM activations):
  Range: Unbounded, non-monotonic
  Quantization: More challenging, per-token quantization often needed
  
Softmax:
  Range: (0, 1), sums to 1
  Quantization: INT8 asymmetric works, but attention scores harder
```

---

## 14. Quantization Noise Theory

### Statistical Model of Quantization

Quantization can be modeled as adding noise:

```
Uniform Quantization Noise Model:
  r̃ = r + η
  
Where η (quantization noise) is approximately:
  Uniform: η ~ U(-S/2, S/2)
  Mean: E[η] = 0 (no bias for symmetric quantization)
  Variance: Var[η] = S²/12

For INT8 with scale S:
  Noise power = S²/12

Signal-to-Noise Ratio:
  SQNR = E[r²] / Var[η] = 12 × E[r²] / S²

For a uniform signal in range [-R, R]:
  E[r²] = R²/3
  S = 2R/255 (INT8, symmetric)
  SQNR = 12 × (R²/3) / ((2R/255)²) = 12/3 × (255/2)² ≈ 31,000 ≈ 45 dB
  
This is the theoretical maximum; practical SNR is lower due to:
  - Non-uniform distributions
  - Outliers causing clipping
  - Correlated errors across layers
```

### The Accuracy Cliff Visualization

```
Top-1 Accuracy vs Bit-Width (Conceptual):

100% |
 95% |         ●──● FP32 (76.1%) baseline
 90% |      ●──                     
 85% |   ●──         QAT           
 80% |●──          ●──● PTQ INT8 (75.8%)
 75% |           ●──                  
 70% |         ●── PTQ INT4 (71.2%)  
 65% |       ●──                     
     |_______________________________
        16   12    8    4    2    1
               Bit Width

Key observation: 
  - Minimal drop from FP32 → INT8 (with AIMET)
  - Significant drop at INT4 without specialized techniques
  - Catastrophic failure below 4 bits without architecture changes
```

---

## 15. Common Quantization Pitfalls

### Pitfall 1: Not Folding Batch Normalization

```
Problem: 
  BN parameters have different statistical properties than conv weights
  Running them through QuantSim without folding causes:
  - Scale mismatches between BN γ/β and conv weights
  - ~0.5–2% unnecessary accuracy loss

Solution:
  Always fold BN before quantization
  AIMET does this automatically in AutoQuant
  Manual: fold_all_batch_norms(model, input_shapes)
```

### Pitfall 2: Using Non-Representative Calibration Data

```
Problem:
  Calibrating with training data from one domain, deploying on another
  
  Example: Calibrating face detection with indoor studio photos,
           deploying on outdoor surveillance cameras
  
  Result: Scale factors optimized for wrong activation distributions
          → 3–10% accuracy drop on deployment data

Solution:
  Use deployment-representative calibration data
  Include temporal diversity, edge cases
  Test: calibration accuracy ≈ full validation accuracy (within 0.5%)
```

### Pitfall 3: Ignoring Outlier Activations

```
Problem:
  LLMs and some CNNs have activation outliers that appear sporadically
  
  Example (BERT activation): 
    99% of values in range [-1, 1]
    But occasional values at 50, -47, 103...
  
  With TF (min-max) scheme:
    scale = 103/127 = 0.811
    Regular values [-1, 1] → only ~2-3 INT8 levels
    → Catastrophic accuracy loss

Solutions:
  1. Use TF-Enhanced scheme (AIMET default)
  2. Apply CLE before quantization
  3. Use per-token quantization for transformers
  4. Consider SmoothQuant preprocessing
```

### Pitfall 4: Quantizing Batch Normalization During QAT

```
Problem:
  During QAT, BN running statistics drift due to quantization noise
  If BN is quantized during QAT, the statistics become doubly corrupted

Solution:
  During QAT: Keep BN in floating-point
  After QAT: Apply BN Re-estimation
  Then: Fold BN into preceding layers

AIMET workflow:
  1. QAT with BN unfrozen (FP32)
  2. BN re-estimation pass
  3. BN folding
  4. Export
```

### Pitfall 5: Skipping Layer Sensitivity Analysis

```
Problem:
  Applying uniform INT8 quantization to all layers
  Missing that a single sensitive layer is causing 90% of accuracy drop
  
  Example: 
    Single depthwise separable conv layer with 8x scale imbalance
    → 2% accuracy drop from just this one layer
  
  But: Keeping this layer at FP32 and quantizing everything else:
    → Only 0.1% accuracy drop

Solution:
  Always run QuantAnalyzer before finializing quantization settings
  Use mixed precision for sensitive layers
  
  AIMET Code:
  analyzer = QuantAnalyzer(model, dummy_input, eval_callback)
  analyzer.check_per_layer_quantization_encodings(sim)
  # Returns layer-wise accuracy drop table
```

---

## Summary

Understanding quantization fundamentals is essential for effective use of AIMET. Key takeaways:

1. **Quantization is primarily a memory bandwidth optimization**, especially for edge deployment
2. **Symmetric for weights, asymmetric for activations** is the standard configuration
3. **Per-channel quantization** significantly improves accuracy with minimal overhead
4. **TF-Enhanced (MSE) scheme** is AIMET's recommended default — it handles outliers gracefully
5. **Layer sensitivity varies dramatically** — always run QuantAnalyzer to identify bottlenecks
6. **INT8 is the sweet spot** — near-lossless with AIMET techniques, hardware-accelerated everywhere
7. **INT4 requires specialized algorithms** (AdaRound, GPTQ, AWQ) to maintain acceptable accuracy

---

*Next: [03_Post_Training_Quantization_PTQ.md](./03_Post_Training_Quantization_PTQ.md)*
*Prev: [01_AIMET_Overview_and_Architecture.md](./01_AIMET_Overview_and_Architecture.md)*

---

## 16. Quantization for Sparse Networks (Mixed Density + Precision)

While AIMET focuses heavily on precision reduction (quantization), combining it with density reduction (sparsity/pruning) yields the ultimate efficiency. 

### The Interaction of Sparsity and Quantization
When a network is heavily pruned (e.g., 80% sparsity), the remaining non-zero weights become critically important. 
- **Challenge**: Pruned networks are generally *more sensitive* to quantization noise because there is less redundancy to absorb the error.
- **Strategy**: 
  1. Prune the FP32 model first (e.g., using Channel Pruning).
  2. Fine-tune the FP32 sparse model to recover accuracy.
  3. Apply QAT (Quantization-Aware Training) rather than just PTQ. The sparse structure must learn to adapt to the discrete quantization levels.

### Sparse Quantization on Hardware
Hardware accelerators handle sparse quantized matrices differently. On NVIDIA Ampere (2:4 structured sparsity), the hardware literally skips the multiplication of the zeroed weights. On Qualcomm Hexagon DSPs, structured sparsity (channel pruning) results in smaller dense matrices, which are then quantized efficiently using standard dense INT8 MACs.

---

## 17. Integer Arithmetic Beyond Multiplication: Quantized Softmax, LayerNorm

While convolutions and linear layers (MAC operations) consume the bulk of compute, non-linear operations must also be quantized to keep the entire network on the NPU/DSP. Moving data back to the CPU to compute FP32 Softmax destroys latency.

### Quantized Softmax
The mathematical definition of Softmax involves exponentiation: $\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$.
- Computing $e^x$ directly in INT8 is impossible.
- **Solution**: Use integer polynomial approximations or lookup tables (LUTs). 
- **Implementation trick**: Subtract the maximum value before exponentiation: $e^{x_i - x_{max}}$. In integer math, this shifts the range to negative numbers, preventing overflow and allowing the use of highly optimized negative-exponent LUTs.

### Quantized LayerNorm
LayerNorm requires computing the mean and variance of a tensor slice.
- **Challenge**: Computing variance requires squaring values, which rapidly overflows 8-bit or even 16-bit integers.
- **Solution**: The mean and variance are typically computed using higher precision accumulators (INT32). The division by standard deviation is implemented using a reciprocal square root LUT (RSQRT LUT) combined with INT32 multipliers, before scaling the result back down to INT8 for the output.

---

## 18. Fixed-Point vs Floating-Point Quantization Comparison

The terms "Integer Quantization" and "Fixed-Point Quantization" are often used interchangeably, but there is a nuanced difference.

### Fixed-Point Arithmetic (Q-Format)
In fixed-point math, an integer represents a fractional number based on an implied decimal point position.
- **Format**: Q$m.n$ (e.g., Q3.4 means 3 integer bits, 4 fractional bits, 1 sign bit = 8 bits).
- **Scale**: The scale factor is implicitly $2^{-n}$.
- **Hardware**: DSPs natively support Q-format arithmetic with built-in shift operations.

### Floating-Point Quantization (AIMET's Approach)
AIMET models quantization as an affine transform: $r = S(q - Z)$.
- **Scale (S)**: Is a full FP32 number, not restricted to powers of 2.
- **Implementation**: The hardware performs the INT8 MAC, and then multiplies the INT32 accumulator by the FP32 scale (often using an FP16 or simulated FP multiplier) at the end of the operation. 
- **Advantage**: Affine quantization (with FP scales) provides significantly better accuracy than strict Q-format fixed-point because the scale can exactly match the dynamic range of the tensor, rather than being rounded to the nearest power of 2.

---

## 19. Block Floating Point (BFP) Format Deep-Dive

Block Floating Point (BFP) is a hybrid between floating-point and integer representation, gaining immense popularity in AI hardware.

### How BFP Works
Instead of every number having its own exponent (like FP32) or a single scale for the whole tensor (like INT8 per-tensor), BFP groups numbers into "blocks" (e.g., blocks of 16 or 32 values).
- **The Block**: A single shared exponent is stored for the entire block.
- **The Values**: The individual values in the block are stored as low-bit mantissas (e.g., 8-bit or 4-bit integers).
- **Compute**: Hardware aligns the mantissas using the shared exponent, performs integer MACs, and applies the exponent to the output.

### Advantages
- **Accuracy**: Almost identical to FP16 because the shared exponent captures the dynamic range of that specific small block.
- **Hardware Efficiency**: MAC operations are performed on integers, keeping the silicon area and power consumption low.

---

## 20. NVIDIA's E4M3 and E5M2 FP8 Formats Explained

The FP8 standard introduces two specialized 8-bit floating-point formats, primarily driven by NVIDIA (Hopper architecture) and now widely adopted.

### E4M3 (4 Exponent bits, 3 Mantissa bits)
- **Range**: Smaller dynamic range, but higher precision (3 bits of mantissa).
- **Use Case**: Used for **Activations** and **Weights** during inference and the forward pass of training. Activations generally need more precision than dynamic range once normalized.

### E5M2 (5 Exponent bits, 2 Mantissa bits)
- **Range**: Massive dynamic range (matches FP16 exponent range), but very low precision.
- **Use Case**: Used for **Gradients** during training. Gradients often contain extreme outliers and require a vast dynamic range, but neural networks are surprisingly robust to low-precision gradient updates.

AIMET supports simulating these FP8 formats for QAT pipelines targeting modern data center hardware.

---

## 21. Quantization Error Visualization: Activation Histograms

Visualizing histograms is the most powerful debugging tool for quantization.

### The "Good" Histogram (Bell Curve)
If your activation tensor looks like a normal distribution centered at 0 (or centered at some positive mean for post-ReLU):
- TF-Enhanced easily finds a clipping threshold that chops off the long tails (which contain <0.1% of the data).
- The remaining data is evenly distributed across the 256 INT8 bins.

### The "Bad" Histogram (Bimodal or Extreme Outliers)
If your tensor has a massive spike at 0, a cluster at 0.5, and a single outlier value at 120.0:
- Using absolute min-max (TF scheme) forces the scale to accommodate 120.0.
- The values at 0.5 are quantized into the exact same INT8 bin as the values at 0.0.
- **Result**: Complete loss of information for 99.9% of the data. 
- **Solution**: Visualizing this immediately tells you to use a percentile clipping scheme, TF-Enhanced, or to insert a LayerNorm to tame the outlier.

---

## 22. Information-Theoretic Perspective on Quantization (Rate-Distortion)

Quantization can be framed as a Rate-Distortion optimization problem from Information Theory.
- **Rate (R)**: The number of bits used (e.g., 4 bits, 8 bits).
- **Distortion (D)**: The loss in model accuracy or the Mean Squared Error (MSE) of the tensors.

### Shannon's Rate-Distortion Theory
The goal of AIMET algorithms like AdaRound and AutoQuant is to find the point on the Rate-Distortion curve that minimizes D for a given R. 
- A uniform quantizer (min-max) is often far from the optimal Shannon bound.
- Vector Quantization (quantizing groups of weights as a single vector index into a codebook) approaches the theoretical limit much closer than scalar quantization. This is why techniques like QuIP and extreme LLM compression rely on vector quantization or lattice quantization, pushing the boundaries of the Rate-Distortion frontier.

---

## 23. Channel-Wise Quantization Implementation in CUDA

Understanding how channel-wise quantization operates at the hardware level demystifies its performance.

### CUDA Kernel Logic
For a matrix multiplication $C = A \times W$, where $W$ is per-channel quantized:
1. **Load**: Load the INT8 activation tile $A_{int}$ and INT8 weight tile $W_{int}$ into Tensor Cores or Shared Memory.
2. **Compute**: Perform the dense INT8 GEMM (General Matrix Multiply). The accumulator is INT32: $C_{int32}$.
3. **Dequantize (The Per-Channel part)**:
   - The kernel loads an array of floating-point scales: `float scales_W[num_channels]`.
   - The kernel loads the activation scale: `float scale_A`.
   - Iterate over the output dimension $j$:
     $C[i, j] = C_{int32}[i, j] \times \text{scale\_A} \times \text{scales\_W}[j]$

Because the multiplication by `scales_W[j]` happens in the outer loop (applied to the final accumulated output, not inside the dot product loop), the overhead of per-channel quantization is practically zero.

---

## 24. Quantization Unit Tests: How to Write Tests for Quantized Models

When integrating AIMET into a production CI/CD pipeline, you must write unit tests that verify the quantized model's integrity.

### Example Unit Test Constraints
1. **Encoding Verification**: Assert that the generated `encodings.json` contains valid ranges (min < max, scales > 0).
2. **Hardware Constraints**: Assert that specific layers (e.g., Softmax) remain in FP16 if the target NPU does not support INT8 Softmax.
3. **Signal-to-Noise Ratio (SNR) Test**: Pass a fixed, known input tensor through both the FP32 and QuantSim model. Compute the MSE between the outputs. Assert that the SNR is > 30 dB.
4. **Zero-Point Sanity**: If using symmetric quantization for weights, assert that `zero_point == 0` for all weight encodings.

```python
def test_quantized_snr():
    sim = QuantizationSimModel(...)
    sim.compute_encodings(...)
    
    fp32_out = model(test_input)
    int8_out = sim.model(test_input)
    
    noise_power = torch.mean((fp32_out - int8_out)**2)
    signal_power = torch.mean(fp32_out**2)
    snr_db = 10 * torch.log10(signal_power / noise_power)
    
    assert snr_db > 30.0, f"Quantization noise too high! SNR: {snr_db} dB"
```

---

## 25. Academic References: Seminal Quantization Papers with Summaries

To truly master quantization, reading the foundational literature is recommended.

1. **"Quantizing deep convolutional networks for efficient inference: A whitepaper" (Jacob et al., Google, 2018)**
   - *Summary*: The definitive guide to integer-only arithmetic for inference. Introduced the affine quantization math and the standard INT8/INT32 accumulator hardware model that TFLite and most NPUs use today.
2. **"Up or Down? Adaptive Rounding for Post-Training Quantization" (Nagel et al., Qualcomm, 2020)**
   - *Summary*: The paper behind AIMET's AdaRound. Proves that rounding to the nearest integer is mathematically suboptimal. Formulates rounding as a quadratic unconstrained binary optimization (QUBO) problem.
3. **"Data-Free Quantization Through Weight Equalization and Bias Correction" (Nagel et al., Qualcomm, 2019)**
   - *Summary*: The foundation of Cross-Layer Equalization (CLE). Shows how to shift scaling factors between consecutive layers (e.g., Conv -> DepthwiseConv) to balance the dynamic ranges, eliminating the need for calibration data.
4. **"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (Xiao et al., MIT/NVIDIA, 2022)**
   - *Summary*: Solves the activation outlier problem in LLMs by mathematically migrating the difficulty of quantization from activations to weights, enabling W8A8 quantization for 100B+ parameter models.
