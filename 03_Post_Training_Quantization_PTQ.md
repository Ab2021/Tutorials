# Post-Training Quantization (PTQ) in AIMET
## Deep Dive: CLE, Bias Correction, AdaRound, AutoQuant, and More

> **Part of**: AIMET Deep Dive Series | Document 3 of 15

---

## Table of Contents
1. [PTQ Overview and Philosophy](#1-ptq-overview-and-philosophy)
2. [QuantizationSimModel (QuantSim)](#2-quantizationsimmodel-quantsim)
3. [Model Preparation: Validator and Preparer](#3-model-preparation-validator-and-preparer)
4. [Batch Normalization Folding](#4-batch-normalization-folding)
5. [Cross-Layer Equalization (CLE)](#5-cross-layer-equalization-cle)
6. [Bias Correction](#6-bias-correction)
7. [AdaRound (Adaptive Rounding)](#7-adaround-adaptive-rounding)
8. [Sequential MSE (SequentialMSE)](#8-sequential-mse-sequentialmse)
9. [AutoQuant: The Automated PTQ Pipeline](#9-autoquant-the-automated-ptq-pipeline)
10. [QuantAnalyzer: Diagnostics and Visualization](#10-quantanalyzer-diagnostics-and-visualization)
11. [Complete PTQ Workflow with Code](#11-complete-ptq-workflow-with-code)
12. [PTQ Configuration Reference](#12-ptq-configuration-reference)
13. [Troubleshooting PTQ Issues](#13-troubleshooting-ptq-issues)
14. [Performance Benchmarks](#14-performance-benchmarks)

---

## 1. PTQ Overview and Philosophy

### What is Post-Training Quantization?

PTQ quantizes a pre-trained model **without any retraining**. It uses:
- The pre-trained model weights (frozen)
- A small calibration dataset (typically 512–2048 samples)
- Mathematical techniques to compensate for quantization error

### Why PTQ?

```
Training Compute Requirements:
  FP32 Training ResNet-50: ~100 GPU-hours
  QAT Fine-tuning:         ~10 GPU-hours
  PTQ Calibration:         ~0.1 GPU-hours (100x cheaper!)

When PTQ is Sufficient:
  ✓ Target bit-width: INT8 (8 bits)
  ✓ Model is "quantization friendly" (balanced weight ranges)
  ✓ Accuracy target: <1% drop from FP32
  ✓ No labeled training data available (only unlabeled samples)
  
When PTQ May Not Be Sufficient:
  ✗ Target bit-width: INT4 or lower
  ✗ Complex models with extreme activation outliers
  ✗ Accuracy target: <0.1% drop (demanding use case)
  → Use QAT as fallback
```

### AIMET PTQ Techniques Stack

```
Recommended PTQ Stack (in order of application):

1. Model Preparer           → Fix graph structural issues
2. Batch Norm Folding       → Remove BN, simplify model
3. Cross-Layer Equalization → Balance weight ranges across layers  
4. Bias Correction          → Fix output bias from quantization
5. QuantSim Setup           → Insert fake quantization nodes
6. Calibration              → Compute optimal scale/offset per layer
7. AdaRound                 → Optimize weight rounding decisions
8. Evaluate                 → Measure accuracy
9. Export                   → Generate model + encodings.json

Alternatively: Use AutoQuant to automate steps 3–8
```

---

## 2. QuantizationSimModel (QuantSim)

### What is QuantSim?

QuantSim is the **core AIMET class** that transforms a floating-point model into a quantization-aware model by inserting "fake quantization" nodes. These nodes simulate quantization during forward passes while keeping backpropagation gradients intact.

### How Fake Quantization Works

```
Original FP32 Forward Pass:
  conv_output = F.conv2d(input, weight, bias)
  relu_output = F.relu(conv_output)

QuantSim Wrapped Forward Pass:
  # Weight quantization (before conv)
  weight_q = QuantizeDequantize(weight, scale_w, zp_w)  ← fake quant
  
  # Convolution with quantized weights
  conv_output = F.conv2d(input, weight_q, bias)
  
  # Activation quantization (after op)
  conv_output_q = QuantizeDequantize(conv_output, scale_a, zp_a)  ← fake quant
  
  relu_output = F.relu(conv_output_q)
  relu_output_q = QuantizeDequantize(relu_output, scale_r, zp_r)  ← fake quant

Where QuantizeDequantize:
  x → round(x/scale + zero_point) → clamp → (x - zero_point) × scale
  
Key insight: The VALUES become quantization-affected (simulates hardware)
             But GRADIENTS can still flow through (straight-through estimator)
```

### QuantSim API

```python
from aimet_torch.quantsim import QuantizationSimModel, QuantScheme
from aimet_common.defs import QuantizationDataType

# Full QuantSim constructor:
sim = QuantizationSimModel(
    # Model to quantize
    model = model,
    
    # Example input (defines input shapes for tracing)
    dummy_input = torch.rand(1, 3, 224, 224),
    
    # Quantization scheme for determining min/max ranges
    quant_scheme = QuantScheme.post_training_tf_enhanced,  # Default, recommended
    # Options:
    # - QuantScheme.post_training_tf         (absolute min/max)
    # - QuantScheme.post_training_tf_enhanced (MSE-optimized, clips outliers)
    # - QuantScheme.training_range_learning_with_tf_init
    # - QuantScheme.training_range_learning_with_tf_enhanced_init
    
    # Bit-widths
    default_output_bw = 8,   # Activation quantization bit-width
    default_param_bw  = 8,   # Weight quantization bit-width
    
    # Data type (INT or FLOAT quantization)
    default_data_type = QuantizationDataType.int,
    # Options: QuantizationDataType.int, QuantizationDataType.float
    
    # Path to JSON configuration file (optional)
    config_file = None,  # Uses default config if None
    
    # In-place modification? (False: creates new model copy)
    in_place = False,
)

# The sim.model attribute is the quantized model
print(type(sim.model))  # <class 'ModelWithFakeQuant'>
```

### Accessing Quantizers

```python
# List all quantizers in the model
for name, module in sim.model.named_modules():
    if hasattr(module, 'param_quantizers'):
        print(f"\nLayer: {name}")
        for param_name, quantizer in module.param_quantizers.items():
            print(f"  Param '{param_name}': enabled={quantizer.enabled}, "
                  f"bw={quantizer.bitwidth}")
    if hasattr(module, 'output_quantizers'):
        for i, quantizer in enumerate(module.output_quantizers):
            print(f"  Output[{i}]: enabled={quantizer.enabled}, "
                  f"bw={quantizer.bitwidth}")

# Modify a specific quantizer:
# Get the first conv layer's weight quantizer
conv1_module = sim.model.conv1
conv1_module.param_quantizers['weight'].bitwidth = 4  # Change to 4-bit
conv1_module.param_quantizers['weight'].enabled = True

# Disable quantization for a sensitive layer:
sim.model.last_fc.output_quantizers[0].enabled = False
```

### Computing Encodings (Calibration)

```python
# Define the calibration callback
def calibrate(model, args):
    """Run calibration data through model to compute scale/offset"""
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(calibration_loader):
            images = images.cuda()
            _ = model(images)
            
            # Optional: limit calibration to N batches for speed
            if batch_idx >= 50:  # 50 batches × 32 imgs = 1600 images
                break

# Compute encodings (this is where calibration happens)
sim.compute_encodings(
    forward_pass_callback = calibrate,
    forward_pass_callback_args = None
)

# After this call, all quantizers have computed scale/zero_point:
print(sim.model.conv1.param_quantizers['weight'].encoding)
# TfEncoding(min=-0.34, max=0.33, delta=0.00260, offset=-128, bw=8)
```

### Exporting the Model

```python
import os

# Export generates two files:
# 1. model file (PyTorch .pth or ONNX .onnx)
# 2. encodings.json with all scale/offset values

sim.export(
    path     = './exported_models',
    filename_prefix = 'resnet50_int8',
    dummy_input = torch.rand(1, 3, 224, 224)
)

# Files created:
# ./exported_models/resnet50_int8.onnx
# ./exported_models/resnet50_int8.encodings.json

# The encodings.json looks like:
# {
#   "activation_encodings": {
#     "conv1": [{"bitwidth": 8, "dtype": "int", "is_symmetric": "False",
#                "max": 2.843, "min": 0.0, "offset": 0, "scale": 0.01115}]
#   },
#   "param_encodings": {
#     "conv1.weight": [{"bitwidth": 8, "dtype": "int", "is_symmetric": "True",
#                       "max": 0.3281, "min": -0.3281, "offset": -127, "scale": 0.002583}]
#   }
# }
```

---

## 3. Model Preparation: Validator and Preparer

### Why Preparation is Needed

AIMET inserts quantization nodes by tracing the model's computational graph using `torch.fx` (for PyTorch). Some model constructs break this tracing:

```python
# PROBLEMATIC CODE (breaks AIMET graph tracing):

class BadModel(nn.Module):
    def forward(self, x):
        # Functional ops without module wrapper → hard to insert quantizers
        x = F.relu(x)           # Should be nn.ReLU module
        x = F.dropout(x, 0.5)  # Should be nn.Dropout module
        
        # Data-dependent control flow → breaks tracing
        if x.sum() > 0:
            x = self.branch_a(x)
        else:
            x = self.branch_b(x)
        
        # In-place operations → quantization incompatible
        x += self.shortcut(x)   # In-place add
        
        return x
```

### Model Validator

```python
from aimet_torch.model_validator.model_validator import ModelValidator

# Check if model is AIMET-compatible
validator = ModelValidator(model, dummy_input)
is_valid = validator.validate_model()

if is_valid:
    print("Model passes all AIMET compatibility checks ✓")
else:
    print("Model has issues:")
    for issue in validator.model_validation_issues:
        print(f"  - {issue}")
    # Example issues:
    # - "Layer 'relu' uses functional F.relu instead of nn.ReLU module"
    # - "In-place addition detected in 'layer2'"
    # - "Model contains non-traceable control flow"
```

### Model Preparer (Automatic Fixer)

```python
from aimet_torch.model_preparer import prepare_model

# Automatically fix common issues
prepared_model = prepare_model(model)

# What prepare_model does:
# 1. Replaces F.relu → nn.ReLU()
# 2. Replaces F.avg_pool2d → nn.AvgPool2d()
# 3. Replaces in-place ops with out-of-place versions
# 4. Handles torch.add(), operator+ → proper module wrappers
# 5. Restructures BN operations

# Verify the fixed model works
output_original = model(dummy_input)
output_prepared = prepared_model(dummy_input)
assert torch.allclose(output_original, output_prepared, atol=1e-5), \
    "Model preparer changed output!"
print("Prepared model is functionally equivalent ✓")
```

---

## 4. Batch Normalization Folding

### The Mathematics of BN Folding

During inference, Batch Normalization computes:
```
BN forward (inference mode):
  y = γ × (x - running_mean) / √(running_var + ε) + β
  
  Expanding:
  y = (γ / √(running_var + ε)) × x + (β - γ × running_mean / √(running_var + ε))
  
  This is a linear operation: y = A × x + B
  where A = γ / √(running_var + ε)
        B = β - γ × running_mean / √(running_var + ε)
```

A linear operation can be **absorbed into the preceding Conv/Linear layer**:

```
Before folding:
  Conv: out = W × input + b
  BN:   y = A × out + B
         = A × (W × input + b) + B
         = (A × W) × input + (A × b + B)

After folding:
  Conv': W' = A × W,  b' = A × b + B
  out' = W' × input + b'

Result: BN layer removed! Same computation, simpler graph.
```

### AIMET BN Folding API

```python
from aimet_torch.batch_norm_fold import fold_all_batch_norms

# Fold all BN layers in the model
# Returns list of (conv_or_linear, bn) pairs that were folded
folded_pairs = fold_all_batch_norms(
    model = model,
    input_shapes = [(1, 3, 224, 224)],
    # Optional: also works with:
    # dummy_input = torch.rand(1, 3, 224, 224)
)

print(f"Folded {len(folded_pairs)} Conv-BN pairs")

# Verify: BN layers should now be identity operations or removed
# The conv layers should have updated weights incorporating BN params
```

### When to Fold BN

```
Timing:
  ✓ ALWAYS fold before creating QuantSim
  ✓ Fold before applying CLE (CLE relies on folded model)
  ✓ Fold before AdaRound calibration

  ✗ Do NOT fold during QAT (BN re-estimation needed first)
  ✗ Do NOT fold before BN re-estimation pass

Correct order:
  1. (Optional) BN Re-estimation [if doing QAT cleanup]
  2. BN Folding
  3. QuantSim creation
  4. Calibration / AdaRound
  5. Export
```

### What Happens to BN Parameters

```python
# Before folding: ResNet50 has 53 BatchNorm2d layers
count_before = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
print(f"BN layers before folding: {count_before}")  # → 53

# After folding:
fold_all_batch_norms(model, [(1, 3, 224, 224)])
count_after = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
print(f"BN layers after folding: {count_after}")  # → 0 (or stubs)

# The Conv layers now have modified weights:
# Original: conv.weight shape [64, 3, 7, 7], conv.bias = None
# After BN fold: conv.weight shape [64, 3, 7, 7] (same shape, new values!)
#                conv.bias shape [64] (bias added to absorb BN beta)
```

---

## 5. Cross-Layer Equalization (CLE)

### The Problem CLE Solves

```
Consider a ResNet-50 bottleneck block:
  Conv_1 → BN → ReLU → Conv_2

After BN folding:
  Conv_1 weights per-channel ranges:
    Channel 0:  [-0.005, 0.007]  (very small range)
    Channel 32: [-4.2,   4.1]   (very large range)
    Channel 63: [-0.023, 0.019] (small range)

Problem: With per-tensor quantization of weights:
  Scale = max_range / 127 = 4.2/127 ≈ 0.033
  
  Channel 0 (range 0.007):
    0.007 / 0.033 = 0.21 → rounds to 0 (integer!)
    → Channel 0 weights effectively become ZERO
    → Catastrophic: entire feature channel lost!
  
  Channel 32 (range 4.2):  
    4.2 / 0.033 = 127 → full range, accurate
```

### CLE Mathematical Principle

CLE exploits the **scale equivariance** of ReLU activation functions:

```
Key observation:
  ReLU(s × x) = s × ReLU(x)   for any s > 0

This means we can rescale weights between adjacent layers:
  y = Conv_2(ReLU(Conv_1(x)))
    = Conv_2(ReLU(S⁻¹ × S × Conv_1(x)))
    = Conv_2(S⁻¹ × ReLU(S × Conv_1(x)))    [scale equivariance]
    
  = (Conv_2 × S⁻¹)(ReLU((S × Conv_1)(x)))
  
  Where:
    Conv_1_new = S × Conv_1  (scale up first layer)
    Conv_2_new = Conv_2 × S⁻¹ (scale down second layer)
    
  Mathematical equivalence: EXACT SAME OUTPUT

CLE sets S to equalize the per-channel ranges:
  S[c] = √(range_1[c] × range_2[c])  (geometric mean)
  
  Conv_1_new[:, c, :, :] = Conv_1[:, c, :, :] × S[c]
  Conv_2_new[c, :, :, :] = Conv_2[c, :, :, :] / S[c]
```

### CLE After Equalization

```
After CLE:
  Channel 0:  range scaled from 0.007 to ~0.17 (balanced)
  Channel 32: range scaled from 4.2 to ~0.83 (balanced)
  Channel 63: range scaled from 0.019 to ~0.13 (balanced)

New per-tensor scale: max_range/127 ≈ 0.83/127 ≈ 0.0065
Much more balanced — all channels get adequate representation!
```

### AIMET CLE API

```python
from aimet_torch.cross_layer_equalization import (
    equalize_model,
    HighBiasFold,
    CrossLayerScaling
)

# Simple API: equalize entire model
# This handles CLE + High Bias Folding automatically
equalized_model = equalize_model(
    model = model,
    input_shapes = [(1, 3, 224, 224)]
)

# Note: equalize_model automatically:
# 1. Detects consecutive Conv-BN-ReLU patterns
# 2. Folds BN if not already done
# 3. Applies Cross-Layer Scaling
# 4. Applies High Bias Folding

# Advanced API (step by step):
from aimet_torch.cross_layer_equalization import (
    get_cross_layer_scaling_params,
    scale_model_params_for_cle
)

# Step 1: Find equalization parameters
cls_params = get_cross_layer_scaling_params(model, [(1, 3, 224, 224)])

# Step 2: Apply scaling
scale_model_params_for_cle(model, cls_params)

# Step 3: High bias folding
from aimet_torch.cross_layer_equalization import fold_all_batch_norms_to_scale
# (Applied internally by equalize_model)
```

### High Bias Folding

CLE rescaling can create large bias values in Conv_1_new. These large biases cause separate quantization issues:

```
Problem: After CLE scaling up Conv_1:
  Conv_1_new weights: [-0.17, 0.17]  (good, balanced)
  Conv_1_new bias: increased from [-0.1] to [-14.3] ← HUGE!
  
  Large bias → quantizing bias requires wide range
  → Low precision for bias → accuracy loss

Solution: High Bias Folding
  Absorb part of the large bias into the NEXT layer's bias:
  
  If Conv_1_new has bias b and Conv_2_new has bias b2:
    Estimate activation statistics (using calibration data or analytically)
    Transfer portion of b into b2 via activation statistics
    
  Result: Large bias redistributed, both layers have manageable bias values
```

---

## 6. Bias Correction

### Why Bias Correction is Needed

Even after CLE, quantization introduces a systematic bias (mean error) in layer outputs:

```
Example:
  True Conv output for 1000 samples: mean = 0.05, std = 2.3
  Quantized Conv output: mean = 0.43, std = 2.3  ← mean shifted!

  The 0.38 mean shift accumulates through layers:
  Layer 1 shift: 0.38
  Layer 2 shift: 0.52  (error propagates)
  ...
  Final layer shift: causes classification error!
```

### Two Approaches

#### Analytical Bias Correction (Data-Free)

Uses BatchNorm statistics (mean, variance) to estimate the expected shift:

```python
from aimet_torch.bias_correction import (
    correct_bias,
    AnalyticalBiasCorrectParams,
    StopCriterion
)

params = AnalyticalBiasCorrectParams(
    # Use BatchNorm mean/variance to compute correction analytically
    # No data needed!
    num_quant_samples = 0,  # 0 = analytical only
)

correct_bias(
    model = quantized_sim.model,
    quant_params = params,
    num_quant_samples = 0,
    data_loader = None,
    correct_only_bias = True
)

# Mathematics:
# Expected quantization error = E[Q(W)×input - W×input]
# = E[Q(W) - W] × E[input]    (approximately)
# = ΔW × μ_input
# where μ_input is from BN running_mean statistics

# Correction: bias -= ΔW × μ_input  (cancels the shift)
```

#### Empirical Bias Correction (Uses Small Dataset)

More accurate: measures the actual shift using real data:

```python
from aimet_torch.bias_correction import (
    correct_bias,
    BiasCorrectionParams
)

params = BiasCorrectionParams(
    batch_size = 64,
    num_quant_samples = 1000,      # Number of calibration samples
    num_bias_correct_samples = 512, # Samples for bias correction computation
)

# Requires a small calibration dataset
correct_bias(
    model = quantized_sim.model,
    quant_params = params,
    num_quant_samples = 1000,
    data_loader = calibration_loader
)

# Process:
# 1. For each layer:
#    a. Run FP32 and quantized versions on calibration data
#    b. Compute output difference: δ = mean(out_fp32 - out_quantized)
#    c. Adjust bias: bias -= δ
# 2. This directly cancels the mean shift

# Typical accuracy improvement: 0.3–1.0% over uncorrected INT8
```

---

## 7. AdaRound (Adaptive Rounding)

### The Problem with Nearest Rounding

Standard quantization uses nearest-integer rounding:
```
Weight w = 2.7, scale S = 1.0:
  Nearest round: round(2.7) = 3
  → Always rounds up, no matter the context
```

But this is **locally optimal, not globally optimal**. Rounding a weight down might allow better overall layer output accuracy.

### AdaRound's Key Insight

> "The rounding decision for each weight should be made to minimize the **reconstruction error of the layer output**, not the rounding error of the individual weight."

```
Mathematical Formulation (from ICML 2020 paper):

Minimize: ||Wx - W_q × x||²  (layer output error)

Where W_q[i,j] = S × (round(W[i,j]/S) + v[i,j])
      v[i,j] ∈ {0, 1}  (binary: round up or down)

This is a Quadratic Unconstrained Binary Optimization (QUBO) problem!
```

### AdaRound Algorithm

```
Step 1: Initialize
  For each weight w: start with v = round(w/S) - floor(w/S)
  (continuous relaxation between 0 and 1)

Step 2: Soft Relaxation
  Use differentiable surrogate for the binary constraint:
  v_soft = sigmoid(β × (h_v - 0.5))
  where h_v is a learned parameter, β is annealing coefficient

Step 3: Optimization
  Minimize L = ||layer_output_fp32 - layer_output_quantized||²
             + λ × regularizer(v)
  
  using gradient descent on h_v
  (straight-through estimator allows gradient through quantization)

Step 4: Finalize
  v_final[i,j] = round(v_soft[i,j])  → {0 or 1}
  W_q[i,j] = S × (floor(W[i,j]/S) + v_final[i,j])

Key: The optimization is LOCAL (per-layer), making it fast!
     Each layer is optimized independently using a small calibration dataset.
```

### AIMET AdaRound API

```python
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_common.defs import QuantScheme

# Step 1: Define calibration dataset
adaround_data_loader = DataLoader(
    calibration_dataset,
    batch_size = 32,
    num_workers = 4
)

# Step 2: Configure AdaRound parameters
params = AdaroundParameters(
    # Data loader for calibration
    data_loader = adaround_data_loader,
    
    # Number of batches to use for calibration
    num_batches = 4,  # 4 batches × 32 = 128 samples (sufficient)
    
    # Default quantization parameters
    default_num_iterations = 10000,  # Optimization iterations per layer
    default_reg_param = 0.01,        # Regularization strength (λ)
    default_beta_range = (20, 2),    # Annealing range for β
    default_warm_start = 0.2,        # Fraction of iterations for warm-up
)

# Step 3: Apply AdaRound (CPU or GPU)
quantized_model = Adaround.apply_adaround(
    model = model,
    dummy_input = dummy_input,
    params = params,
    
    # Export path for AdaRound encodings
    path = './adaround_models',
    filename_prefix = 'resnet50_adaround',
    
    # Quantization scheme for QuantSim
    default_param_bw = 4,          # AdaRound especially useful for 4-bit!
    default_quant_scheme = QuantScheme.post_training_tf_enhanced,
)

# Step 4: The returned model has AdaRound-optimized weights
# Continue with activation calibration using QuantSim
sim = QuantizationSimModel(
    quantized_model, dummy_input,
    default_param_bw = 4,
    default_output_bw = 8
)
sim.set_and_freeze_param_encodings('./adaround_models/resnet50_adaround.encodings')
sim.compute_encodings(calibrate, None)  # Calibrate activations only
```

### AdaRound Performance

```
Results on ResNet-18 (ImageNet, FP32 baseline = 71.1%):

Method              | INT4 Accuracy | Notes
--------------------|--------------|------------------------
Nearest Rounding    | 51.2%        | Catastrophic failure
BRECQ (2021)        | 64.1%        | Data-free advanced method
AdaRound (ICML 2020)| 68.0%        | AIMET implementation
AdaRound + BN fold  | 70.1%        | With proper preprocessing
AdaRound + QAT      | 70.9%        | Fine-tuning after AdaRound

Computation time per layer: ~5 minutes on single GPU
Total for ResNet-50 (~50 weight layers): ~4 hours (GPU), can be parallelized
```

---

## 8. Sequential MSE (SequentialMSE)

### What is Sequential MSE?

Sequential MSE is a layer-by-layer quantization optimization that minimizes the Mean Squared Error of each layer's output sequentially:

```
Process (greedy, left-to-right through network):

Layer 0 (first layer):
  - Try all candidate scale values for weights AND activations
  - Select scale that minimizes MSE(output_fp32, output_quantized)
  - Fix these scales permanently

Layer 1 (second layer):
  - Input is now the quantized output from Layer 0
  - Try all candidate scales for Layer 1
  - Select scales minimizing MSE for Layer 1's output
  - Fix these scales permanently

...repeat for all layers...

Key: Each layer's optimization accounts for quantization error from previous layers
```

### Sequential MSE vs AdaRound

```
Feature Comparison:
                    | Sequential MSE    | AdaRound
--------------------|-------------------|------------------
Scope               | Weights + Activations | Weights only
Optimization type   | Search/grid-based | Gradient descent
Compute cost        | Medium            | Higher (gradient opt)
Data requirement    | Small calibration | Small calibration
Best for            | 8-bit general opt | 4-bit weight-only
Theoretical basis   | MSE minimization  | QUBO reformulation
AIMET integration   | AutoQuant uses it | Explicit API
```

---

## 9. AutoQuant: The Automated PTQ Pipeline

### Philosophy

AutoQuant eliminates the need for manual exploration of PTQ techniques. It:
1. Applies techniques sequentially
2. Evaluates accuracy after each step
3. Stops when the accuracy target is met

```
AutoQuant Internal Pipeline:

Phase 1: Baseline
  → Apply BN Folding only
  → Evaluate accuracy
  → If meets target: DONE
  
Phase 2: CLE
  → Add Cross-Layer Equalization
  → Evaluate accuracy
  → If meets target: DONE
  
Phase 3: AdaRound
  → Add Adaptive Rounding
  → Evaluate accuracy
  → If meets target: DONE
  
Phase 4: Manual fallback
  → Report failure, recommend QAT
  → Return best model found so far
```

### AutoQuant API

```python
from aimet_torch.auto_quant import AutoQuant, AutoQuantWithoutAdaptiveRounding

# Define evaluation function
def eval_callback(model, num_samples=500):
    """Returns accuracy (0.0 to 1.0)"""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i * images.size(0) >= num_samples:
                break
            outputs = model(images.cuda())
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels.cuda()).sum().item()
            total += labels.size(0)
    return correct / total

# Create AutoQuant object
auto_quant = AutoQuant(
    model = model,
    dummy_input = dummy_input,
    
    # Data loader for calibration AND evaluation
    data_loader = calibration_loader,
    
    # Evaluation callback
    eval_callback = eval_callback,
    
    # Quantization parameters
    param_bw = 8,           # Weight bit-width
    output_bw = 8,          # Activation bit-width
    quant_scheme = QuantScheme.post_training_tf_enhanced,
    
    # Optional: AdaRound config
    adaround_params = AdaroundParameters(
        data_loader = calibration_loader,
        num_batches = 4
    ),
)

# Set accuracy target
auto_quant.set_eval_callback(eval_callback)

# Run AutoQuant
# allowed_accuracy_drop: Stop when accuracy >= FP32_accuracy - allowed_drop
optimized_model, optimized_accuracy, encoding_path = auto_quant.optimize(
    allowed_accuracy_drop = 0.01  # Accept up to 1% accuracy drop
)

print(f"Optimized INT8 Accuracy: {optimized_accuracy:.4f}")
print(f"Encodings saved to: {encoding_path}")

# Export the final model
sim_final = QuantizationSimModel(optimized_model, dummy_input)
sim_final.load_encodings(encoding_path)
sim_final.export('./final_models', 'resnet50_autoquant', dummy_input)
```

### AutoQuant Result Analysis

```python
# Check what techniques were applied
results = auto_quant.get_results()
print(results)

# Expected output:
# AutoQuant Results:
# ├── BN Folding only: 75.4% (FAILED - need 75.1%)
# ├── + CLE:           75.6% (FAILED - but improved)
# ├── + AdaRound:      75.9% (SUCCESS - meets target)
# └── Final technique: BN Fold + CLE + AdaRound
```

---

## 10. QuantAnalyzer: Diagnostics and Visualization

### What QuantAnalyzer Does

```
QuantAnalyzer Outputs:
├── Per-Layer Sensitivity Analysis
│   Shows: Which layers contribute most to accuracy drop
│   Format: Bar chart + CSV with [layer_name, fp32_acc, int8_acc, drop]
│
├── Per-Layer Encoding Analysis
│   Shows: Scale, offset, min, max per layer
│   Format: JSON + histogram plots
│
├── Min-Max Range Analysis
│   Shows: How calibration data affected encoding ranges
│   Useful for: Identifying outlier-dominated layers
│
└── Enabled/Disabled Quantizer Analysis
    Shows: Impact of individually enabling/disabling each quantizer
    Useful for: Finding the "worst" single quantizer to address
```

### QuantAnalyzer API

```python
from aimet_torch.quant_analyzer import QuantAnalyzer, CallbackFunc

# Setup the analyzer
analyzer = QuantAnalyzer(
    model = model,
    dummy_input = dummy_input,
    forward_pass_callback = CallbackFunc(
        func = eval_with_forward_pass,
        args = calibration_loader
    ),
    eval_callback = CallbackFunc(
        func = evaluate_accuracy,
        args = val_loader
    ),
)

# Run full analysis
analyzer.analyze(
    quant_scheme = QuantScheme.post_training_tf_enhanced,
    default_param_bw = 8,
    default_output_bw = 8,
    config_file = None,
    results_dir = './quant_analysis_results'
)

# Results are saved to ./quant_analysis_results/:
# ├── per_layer_quant_disabled.csv    ← Layer sensitivity analysis
# ├── per_layer_encoding_ranges.json  ← Encoding statistics
# └── activation_histograms/          ← Visual distribution plots
```

### Interpreting Sensitivity Results

```
Sample per_layer_quant_disabled.csv:
layer_name          | fp32_acc | disabled_acc | sensitivity
--------------------|----------|--------------|------------
conv1               | 76.13%   | 76.01%       | LOW (0.12%)
layer1.0.conv1      | 76.13%   | 75.88%       | MEDIUM (0.25%)
layer4.2.conv3      | 76.13%   | 74.21%       | HIGH (1.92%) ← Problem!
fc                  | 76.13%   | 73.10%       | VERY HIGH (3.03%)

Action from this data:
  1. Keep 'fc' (final layer) at higher precision (FP32 or 16-bit)
  2. Keep 'layer4.2.conv3' at INT8 with per-channel quantization
  3. All other layers: standard INT8 is fine
```

---

## 11. Complete PTQ Workflow with Code

### Full Production PTQ Pipeline

```python
"""
Complete AIMET Post-Training Quantization Pipeline
for ResNet-50 on ImageNet
"""
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.model_preparer import prepare_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantizationSimModel, QuantScheme
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.quant_analyzer import QuantAnalyzer, CallbackFunc

# ─────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained FP32 model
model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

dummy_input = torch.rand(1, 3, 224, 224, device=device)

# Dataset setup
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
val_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), normalize
])

imagenet_val = datasets.ImageFolder('/data/imagenet/val', val_transform)
calibration_subset = Subset(imagenet_val, range(1024))
calibration_loader = DataLoader(calibration_subset, batch_size=64, num_workers=4)
val_loader = DataLoader(imagenet_val, batch_size=64, num_workers=4)

# ─────────────────────────────────────────
# 2. MODEL PREPARATION
# ─────────────────────────────────────────
# Step 2a: Validate model
validator = ModelValidator(model, dummy_input)
if not validator.validate_model():
    print("Model has issues, running model preparer...")
    model = prepare_model(model)
    model = model.to(device)

# Step 2b: Equalize model (CLE + BN fold)
print("Applying Cross-Layer Equalization...")
model = equalize_model(model, [(1, 3, 224, 224)])
model = model.to(device)
model.eval()

# ─────────────────────────────────────────
# 3. ADAROUND
# ─────────────────────────────────────────
print("Running AdaRound...")
adaround_params = AdaroundParameters(
    data_loader = calibration_loader,
    num_batches = 4,
    default_num_iterations = 10000,
)

model = Adaround.apply_adaround(
    model = model,
    dummy_input = dummy_input,
    params = adaround_params,
    path = './ptq_output',
    filename_prefix = 'resnet50',
    default_param_bw = 8,
    default_quant_scheme = QuantScheme.post_training_tf_enhanced,
)

# ─────────────────────────────────────────
# 4. QUANT SIM + CALIBRATION
# ─────────────────────────────────────────
print("Creating QuantSim and calibrating activations...")

sim = QuantizationSimModel(
    model = model,
    dummy_input = dummy_input,
    default_output_bw = 8,
    default_param_bw = 8,
    quant_scheme = QuantScheme.post_training_tf_enhanced,
)

# Load AdaRound-computed weight encodings
sim.set_and_freeze_param_encodings(
    './ptq_output/resnet50.encodings'
)

# Calibrate activation encodings
def calibrate(model, _):
    model.eval()
    with torch.no_grad():
        for images, _ in calibration_loader:
            model(images.to(device))

sim.compute_encodings(calibrate, None)

# ─────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────
def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

fp32_acc = evaluate(model)
int8_acc = evaluate(sim.model)
print(f"FP32 Accuracy: {fp32_acc:.2f}%")
print(f"INT8 Accuracy: {int8_acc:.2f}%")
print(f"Accuracy Drop: {fp32_acc - int8_acc:.2f}%")

# ─────────────────────────────────────────
# 6. EXPORT
# ─────────────────────────────────────────
print("Exporting quantized model...")
sim.export(
    path = './ptq_output',
    filename_prefix = 'resnet50_int8',
    dummy_input = dummy_input
)

print("Done!")
print("Output files:")
print("  ./ptq_output/resnet50_int8.onnx")
print("  ./ptq_output/resnet50_int8.encodings.json")
```

---

## 12. PTQ Configuration Reference

### AIMET Quantization Config File (JSON)

```json
{
    "defaults": {
        "ops": {
            "is_output_quantized": "True",
            "is_symmetric": "False"
        },
        "params": {
            "is_quantized": "True",
            "is_symmetric": "True"
        },
        "strict_symmetric": "False",
        "unsigned_symmetric": "True",
        "per_channel_quantization": "False"
    },
    "params": {
        "bias": {
            "is_quantized": "False"
        }
    },
    "op_type": {
        "Conv": {
            "is_input_quantized": "True",
            "is_symmetric": "False",
            "params": {
                "weight": {
                    "is_quantized": "True",
                    "is_symmetric": "True"
                },
                "bias": {
                    "is_quantized": "False"
                }
            }
        },
        "Linear": {
            "is_input_quantized": "True",
            "params": {
                "weight": {
                    "is_quantized": "True",
                    "is_symmetric": "True"
                }
            }
        },
        "Gemm": {
            "is_input_quantized": "True"
        }
    },
    "supergroups": [
        {"op_list": ["Conv", "Relu"]},
        {"op_list": ["Conv", "Clip"]},
        {"op_list": ["Conv", "BatchNormalization", "Relu"]},
        {"op_list": ["Add", "Relu"]},
        {"op_list": ["Gemm", "Relu"]}
    ],
    "model_input": {
        "is_input_quantized": "True"
    },
    "model_output": {}
}
```

### Key Configuration Options Explained

```
"is_output_quantized": "True"
  → Add a quantizer after this op's output
  
"is_input_quantized": "True"
  → Add a quantizer before this op's input

"is_symmetric": "True"
  → Symmetric quantization (zero-point = 0)
  → Default for weights

"per_channel_quantization": "True"
  → One scale per output channel (for Conv/Linear weights)
  → Improves accuracy at slight overhead

"supergroups": [{"op_list": ["Conv", "Relu"]}]
  → Treat Conv+Relu as a single fused op
  → Only quantize the OUTPUT of Relu (not intermediate Conv output)
  → Matches hardware fusion behavior

"model_input": {"is_input_quantized": "True"}
  → Add quantizer for the network's input tensor
  → Often needed for 8-bit end-to-end deployment
```

---

## 13. Troubleshooting PTQ Issues

### Issue 1: Large Accuracy Drop (>2%) After Simple Quantization

```
Diagnosis:
  1. Check for outlier activations:
     - Run QuantAnalyzer encoding analysis
     - Look for layers with (max - min) >> most other layers
  
  2. Run layer sensitivity:
     - QuantAnalyzer per_layer_quant_disabled
     - Find which layers cause >1% individual drop
  
Solutions:
  A. Apply CLE (if not done): equalize_model()
  B. Apply AdaRound: Adaround.apply_adaround()
  C. Use per-channel quantization: config "per_channel_quantization": True
  D. Set first and last layers to FP32:
     sim.model.conv1.output_quantizers[0].enabled = False
     sim.model.fc.output_quantizers[0].enabled = False
  E. Switch to QAT if PTQ insufficient
```

### Issue 2: BN Folding Breaks Model

```
Symptoms: 
  After fold_all_batch_norms, model output differs significantly

Cause:
  BN layers are in training mode (using batch statistics, not running stats)

Fix:
  model.eval()  # CRITICAL: must be in eval mode before BN fold
  fold_all_batch_norms(model, [(1, 3, 224, 224)])
```

### Issue 3: AdaRound OOM (Out of Memory)

```
Cause: AdaRound loads entire model + calibration batch per layer

Fixes:
  1. Reduce batch size in AdaroundParameters
  2. Reduce num_batches
  3. Use half-precision for intermediate computation
  4. Apply layer-by-layer with manual memory management
```

---

## 14. Performance Benchmarks

### AIMET PTQ Results (Official Benchmarks)

| Model | FP32 Top-1 | INT8 PTQ | Drop | Method |
|-------|-----------|---------|------|--------|
| ResNet-50 | 76.1% | 75.8% | 0.3% | CLE + AdaRound |
| MobileNetV2 | 71.8% | 71.1% | 0.7% | CLE + AdaRound |
| EfficientNet-B0 | 77.1% | 76.5% | 0.6% | CLE + AdaRound |
| InceptionV3 | 77.9% | 77.2% | 0.7% | CLE + AdaRound |
| BERT-Base SQUAD | 88.5 F1 | 87.9 F1 | 0.6 F1 | AdaRound |
| YOLOv5s | 56.8 mAP | 56.1 mAP | 0.7% | CLE + AdaRound |

### INT4 PTQ Results

| Model | FP32 Top-1 | INT4 PTQ (Nearest) | INT4 PTQ (AdaRound) | INT4 QAT |
|-------|-----------|-------------------|---------------------|---------|
| ResNet-18 | 71.1% | 51.2% | 68.0% | 70.8% |
| ResNet-50 | 76.1% | 62.4% | 71.5% | 75.2% |
| MobileNetV2 | 71.8% | 38.1% | 60.2% | 68.5% |

### Latency Improvements (Snapdragon 888, Hexagon DSP)

| Model | FP32 CPU (ms) | INT8 DSP (ms) | Speedup |
|-------|-------------|--------------|---------|
| ResNet-50 | 285 | 38 | **7.5x** |
| MobileNetV2 | 55 | 8 | **6.9x** |
| BERT-Base | 450 | 62 | **7.3x** |

---

*Next: [04_Quantization_Aware_Training_QAT.md](./04_Quantization_Aware_Training_QAT.md)*
*Prev: [02_Quantization_Fundamentals.md](./02_Quantization_Fundamentals.md)*
