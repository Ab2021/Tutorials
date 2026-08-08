# Edge Deployment and Qualcomm Hardware for AI Models
## Complete Guide: Snapdragon, AI 100, QAIRT, and the AIMET Deployment Pipeline

> **Part of**: AIMET Deep Dive Series | Document 7 of 15

---

## Table of Contents
1. [Qualcomm AI Hardware Ecosystem Overview](#1-qualcomm-ai-hardware-ecosystem-overview)
2. [Snapdragon Mobile SoC Architecture (Hexagon)](#2-snapdragon-mobile-soc-architecture-hexagon)
3. [Hexagon NPU Deep Dive](#3-hexagon-npu-deep-dive)
4. [Qualcomm Cloud AI 100](#4-qualcomm-cloud-ai-100)
5. [QAIRT SDK: Qualcomm AI Runtime](#5-qairt-sdk-qualcomm-ai-runtime)
6. [QNN SDK: Qualcomm Neural Network SDK](#6-qnn-sdk-qualcomm-neural-network-sdk)
7. [Complete AIMET → Qualcomm Deployment Pipeline](#7-complete-aimet--qualcomm-deployment-pipeline)
8. [Qualcomm AI Hub](#8-qualcomm-ai-hub)
9. [Edge AI Constraints: Memory, Thermal, Power](#9-edge-ai-constraints-memory-thermal-power)
10. [INT8 vs INT4 vs FP16 on Qualcomm Hardware](#10-int8-vs-int4-vs-fp16-on-qualcomm-hardware)
11. [Mixed Precision Deployment Strategies](#11-mixed-precision-deployment-strategies)
12. [Security: TrustZone, Secure Boot, Model Encryption](#12-security-trustzone-secure-boot-model-encryption)
13. [Multi-Model Concurrent Execution](#13-multi-model-concurrent-execution)
14. [Power Profiling and Thermal Management](#14-power-profiling-and-thermal-management)
15. [Competitive Analysis: Qualcomm vs NVIDIA vs Apple vs Google](#15-competitive-analysis-qualcomm-vs-nvidia-vs-apple-vs-google)
16. [Domain-Specific Deployment: Mobile, Auto, XR, IoT](#16-domain-specific-deployment-mobile-auto-xr-iot)

---

## 1. Qualcomm AI Hardware Ecosystem Overview

### The Qualcomm AI Portfolio

```
Qualcomm AI Hardware Family:

┌────────────────────────────────────────────────────────┐
│                 QUALCOMM AI ECOSYSTEM                  │
├──────────────────┬─────────────────┬───────────────────┤
│   MOBILE/EDGE    │   AUTOMOTIVE    │   INFRASTRUCTURE  │
│                  │                 │                   │
│ Snapdragon 8     │ Snapdragon Ride │ Cloud AI 100      │
│ Snapdragon 7/6   │ SA8775P         │ (Inference Card)  │
│ Snapdragon X     │ SA8650P         │                   │
│ (Windows ARM)    │ ASIL-B/D Rated  │                   │
├──────────────────┴─────────────────┴───────────────────┤
│              COMMON AI RUNTIME LAYER                   │
│  QAIRT (Qualcomm AI Runtime) / QNN SDK                │
│  Supports: DLP, DSP, GPU, CPU backends                │
├────────────────────────────────────────────────────────┤
│              MODEL OPTIMIZATION LAYER                  │
│  AIMET (AI Model Efficiency Toolkit)                  │
│  PTQ, QAT, Compression → Encodings JSON               │
└────────────────────────────────────────────────────────┘
```

### Qualcomm AI Engine: The Heterogeneous Compute Approach

Qualcomm's AI Engine is NOT a single processor — it's an orchestrated combination of specialized compute units:

```
Qualcomm AI Engine Components:
  
  CPU (Kryo):
    - Arm Cortex-X4 (1 core, high performance, ~3.3 GHz)
    - Arm Cortex-A720 (5 cores, efficiency, ~3.2 GHz)
    - Arm Cortex-A520 (2 cores, ultra-efficiency, ~2.3 GHz)
    - Role: Preprocessing, postprocessing, orchestration
    - AI ops: int8 via Arm NEON SIMD, ~10 TOPS
    
  GPU (Adreno 750):
    - Compute shaders for ML: ~20 TOPS FP16
    - Good for: Batch inference, FP16 attention
    - Memory: Shares system LPDDR5X
    
  DSP/NPU (Hexagon 798):
    - Dedicated AI accelerator
    - INT8 performance: ~45 TOPS
    - INT4 performance: ~75 TOPS (on Gen 3)
    - Has its own TCM (Tightly Coupled Memory): 4–8 MB
    - Sub-1ms wakeup latency
    
  Total SoC AI performance (marketing): ~98 TOPS
  (Sum of CPU + GPU + NPU; actual sustained depends on workload)
```

---

## 2. Snapdragon Mobile SoC Architecture (Hexagon)

### Generation Evolution

| SoC | Year | Hexagon Gen | INT8 TOPS (NPU) | INT4 | Key AI Features |
|-----|------|-------------|-----------------|------|-----------------|
| Snapdragon 845 | 2018 | Hexagon 685 | 3 TOPS | ❌ | First integrated AI Engine |
| Snapdragon 865 | 2020 | Hexagon 698 | 15 TOPS | ❌ | 5G + AI |
| Snapdragon 888 | 2021 | Hexagon 780 | 26 TOPS | ❌ | 3rd Gen AI Engine |
| Snapdragon 8 Gen 1 | 2022 | Hexagon 790 | 32 TOPS | ❌ | 4th Gen AI Engine |
| Snapdragon 8 Gen 2 | 2022 | Hexagon 8 Gen 2 | 38 TOPS | ✅ | INT4 debut |
| Snapdragon 8 Gen 3 | 2023 | Hexagon 8 Gen 3 | 45 TOPS | ✅ | On-device Gen AI |
| Snapdragon X Elite | 2024 | Hexagon X | 45 TOPS | ✅ | Windows ARM, Copilot+ |

### Snapdragon 8 Gen 3 Full Specifications

```
Snapdragon 8 Gen 3 (SM8650)
├── CPU Subsystem:
│   ├── 1× Cortex-X4 @ 3.3 GHz (Prime core)
│   ├── 5× Cortex-A720 @ 3.2 GHz (Performance)
│   └── 2× Cortex-A520 @ 2.27 GHz (Efficiency)
│
├── GPU: Adreno 750
│   ├── ~46% faster than Adreno 740
│   ├── Supports hardware ray tracing
│   └── Vulkan 1.3, OpenCL 3.0
│
├── Hexagon NPU (Neural Processing Unit):
│   ├── INT8 performance: 45 TOPS (dedicated NPU)
│   ├── INT4 performance: ~75 TOPS (2× INT8)
│   ├── FP16 performance: ~22 TOPS
│   ├── TCM (L1 cache): 8 MB
│   ├── Shared L2: 32 MB
│   └── HMX: 8×8 INT8 matrix units
│
├── Memory:
│   ├── LPDDR5X @ 4800 MHz
│   ├── Bandwidth: 77 GB/s
│   └── Max capacity: 24 GB (phones)
│
├── ISP: Spectra ISP (18-bit, 3×18MP cameras simultaneously)
├── Modem: Snapdragon X75 (5G, 10 Gbps peak)
└── Process: TSMC 4nm
```

### Mid-range and Entry Snapdragon AI

```
Snapdragon 7s Gen 2 (Mid-range, 2023):
  NPU: 15 TOPS INT8
  Use cases: On-device keyword spotting, simple CV tasks

Snapdragon 6 Gen 3 (Upper entry, 2024):  
  NPU: 25 TOPS INT8
  Use cases: Real-time translation, camera AI

Snapdragon 4 Gen 2 (Entry, 2023):
  NPU: 5 TOPS INT8
  Use cases: Wake word detection, basic face unlock
```

---

## 3. Hexagon NPU Deep Dive

### Hexagon Architecture Components

```
Hexagon Processor Architecture:

┌───────────────────────────────────────────────────────┐
│                    HEXAGON CORE                       │
│                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐ │
│  │   SCALAR    │  │   VECTOR    │  │    TENSOR     │ │
│  │    UNIT     │  │    UNIT     │  │     UNIT      │ │
│  │  (HVX base) │  │  (HVX/HVX2) │  │   (HMX)      │ │
│  │             │  │             │  │               │ │
│  │ 32-bit ops  │  │ 512/1024-bit│  │ MAC arrays    │ │
│  │ control flow│  │ SIMD vectors│  │ 8×8 INT8 mat  │ │
│  └─────────────┘  └─────────────┘  └───────────────┘ │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │              MEMORY SUBSYSTEM                   │  │
│  │  TCM: 4-8 MB (low latency, scratchpad)         │  │
│  │  L2 Cache: 32 MB (shared with CPU cluster)     │  │
│  │  L3 Cache: 8 MB (system cache)                 │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │                DMA ENGINE                       │  │
│  │  Async data movement: DRAM → TCM                │  │
│  │  Double buffering: Load next while computing    │  │
│  └─────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

### HVX: Hexagon Vector eXtensions

HVX is Qualcomm's SIMD engine, optimized for activation functions and element-wise operations:

```
HVX Operation Width: 1024 bits (128 bytes)

INT8 throughput: 128 operations/cycle
INT16 throughput: 64 operations/cycle  
INT32 throughput: 32 operations/cycle

Example: ReLU on 128 INT8 values in 1 cycle!

HVX is ideal for:
  - ReLU, PReLU, sigmoid activation functions
  - Depthwise convolutions (high parallelism, low reuse)
  - Normalization (LayerNorm, BatchNorm)
  - Transpose, reshape operations
  - Element-wise add, multiply (residual connections)
```

### HMX: Hexagon Matrix eXtensions

HMX is the tensor core unit for matrix multiplication:

```
HMX: 8×8 INT8 Systolic Array

Each cycle: 8×8 = 64 INT8 MACs
Frequency: ~1.1 GHz → 64 × 10^9 MACs/sec = ~64 TOPS*

*Theoretical peak; actual depends on data reuse and memory bandwidth

HMX is ideal for:
  - Standard 2D convolutions (large kernels)
  - Fully connected / Linear layers (matrix-vector multiply)
  - Attention key-query products (Q×K^T)
  - Transformer feed-forward layers

NOT well suited for:
  - Depthwise convolutions (use HVX instead)
  - Small kernel convolutions (1×1 limited by latency)
```

### Memory Hierarchy and Data Flow

```
Execution Model for a Conv Layer on Hexagon:

Step 1: DMA prefetch (async, background)
  DRAM → TCM: Load weight tile (4MB slice)
  DRAM → TCM: Load input tile
  
Step 2: HMX computation (while next DMA is queued)
  TCM → HMX registers
  Systolic array: Input × Weight → INT32 accumulator
  
Step 3: Post-processing (HVX)
  INT32 → Requantize to INT8 (scale, shift)
  Apply bias
  Apply activation (ReLU via HVX)
  
Step 4: Write back
  TCM → DRAM (output feature map)

Double buffering: Steps 2+3 overlap with DMA for next tile
→ Memory latency effectively hidden!
```

### Operator Support on Hexagon DSP

| Operation | HVX | HMX | Notes |
|-----------|-----|-----|-------|
| Conv2D (3×3, 5×5) | ✅ | ✅ | HMX preferred for large kernels |
| Depthwise Conv2D | ✅ | ❌ | HVX optimal |
| Linear/FC | ❌ | ✅ | Matrix multiply |
| BatchNorm | ✅ | ❌ | Element-wise |
| ReLU/ReLU6 | ✅ | ❌ | 1 cycle per 128 INT8 |
| Sigmoid/Tanh | ✅ | ❌ | Lookup table approach |
| Softmax | ✅ | ❌ | Requires exp approximation |
| Avg/Max Pool | ✅ | ❌ | Sliding window |
| Transpose | ✅ | ❌ | Shuffle instructions |
| LSTM/GRU cell | ❌ | ✅ | Via linear layer decomposition |
| Multi-Head Attention | ❌ | ✅ | Q×K, softmax(A)×V |
| Embedding lookup | ✅ | ❌ | Table gather |

---

## 4. Qualcomm Cloud AI 100

### Hardware Architecture

The Qualcomm Cloud AI 100 is a PCIe inference accelerator targeting cloud and edge-server workloads:

```
Cloud AI 100 Ultra (Top SKU):
  ┌──────────────────────────────────────────┐
  │           CLOUD AI 100 ULTRA             │
  │                                          │
  │  ┌─────────┐ ┌─────────┐ ┌──────────┐  │
  │  │ AI Core │ │ AI Core │ │ AI Core  │  │
  │  │  (×8)   │ │  (×8)   │ │  ...×16  │  │
  │  └─────────┘ └─────────┘ └──────────┘  │
  │                                          │
  │  INT8: 400 TOPS                         │
  │  FP16: 200 TFLOPS                       │
  │  MX formats: MXFP8, MXFP6, MXFP4      │
  │                                          │
  │  Memory: 128 GB LPDDR5X @ 3.2 TB/s     │
  │  PCIe: Gen 5 x16 (128 GB/s)            │
  │  TDP: 150W                              │
  └──────────────────────────────────────────┘

SKUs:
  AI 100 Standard: 8 AI Cores, 16 GB, 75W, 100 TOPS INT8
  AI 100 Pro:      14 AI Cores, 32 GB, 110W, 200 TOPS INT8
  AI 100 Ultra:    16 AI Cores, 128 GB, 150W, 400 TOPS INT8
```

### MicroScaling (MX) Formats

AI 100 Ultra uniquely supports MicroScaling formats — a middle ground between block floating point and per-channel quantization:

```
MX Format Structure:
  - Block size: typically 32 elements share one exponent (scale)
  - Individual elements use compact mantissa only

MX Format Comparison:
  ┌──────────────┬────────────┬──────────────┬────────────┐
  │ Format       │ Exponent   │ Mantissa     │ Bits/elem  │
  ├──────────────┼────────────┼──────────────┼────────────┤
  │ MXFP8 E4M3  │ 4 shared   │ 3 per elem   │ ~4.0 eff.  │
  │ MXFP8 E5M2  │ 5 shared   │ 2 per elem   │ ~3.5 eff.  │
  │ MXFP6 E3M2  │ 3 shared   │ 2 per elem   │ ~3.0 eff.  │
  │ MXFP6 E2M3  │ 2 shared   │ 3 per elem   │ ~3.5 eff.  │
  │ MXFP4 E2M1  │ 2 shared   │ 1 per elem   │ ~2.0 eff.  │
  │ MXINT8       │ 8 shared   │ 8 per elem   │ 8.0        │
  └──────────────┴────────────┴──────────────┴────────────┘

Advantages over pure INT4:
  - Better handling of outliers (shared exponent covers wide range)
  - Higher accuracy than INT4 nearest-round
  - Hardware-efficient: lookup tables + small adders
```

### AI 100 Deployment: Ahead-of-Time Compilation

```
Deployment to Cloud AI 100 (QPC workflow):

┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────┐
│ ONNX     │───▶│  qaic-runner │───▶│ Compilation │───▶│  .qpc  │
│ model    │    │  (AoT comp.) │    │ optimization│    │ binary │
└──────────┘    └──────────────┘    └─────────────┘    └────────┘
                      │                                     │
                      │                                     ▼
               quantization.yaml                   ┌─────────────┐
               (encodings from AIMET)              │ AI 100 Card │
                                                   │  runtime    │
                                                   └─────────────┘

AoT Compilation steps:
  1. Graph optimization: op fusion, constant folding
  2. Memory planning: static allocation for zero fragmentation
  3. Kernel selection: best kernel for each op on AI Core
  4. Schedule generation: pipeline stages across 16 cores
  5. Binary generation: .qpc file (standalone, no JIT)

Benefits of AoT:
  - Zero compilation overhead at runtime
  - Deterministic latency (no JIT variability)
  - Smaller memory footprint at inference time
```

---

## 5. QAIRT SDK: Qualcomm AI Runtime

### QAIRT Overview

QAIRT (formerly SNPE - Snapdragon Neural Processing Engine) is the unified runtime that converts and runs AI models on any Qualcomm hardware.

```
QAIRT vs QNN (historical):
  SNPE (old) → Replaced by QAIRT (unified)
  QNN SDK → Lower-level C++ API (used by QAIRT internally)
  
  Recommendation: Use QAIRT for new projects (higher-level)
                  Use QNN directly only for custom ops/kernels
```

### qairt-converter: Complete Command Reference

```bash
# Basic conversion (FP32 ONNX → DLC)
qairt-converter \
  --input_network model.onnx \
  --output_path model.dlc

# With AIMET quantization encodings (recommended)
qairt-converter \
  --input_network model.onnx \
  --output_path model_int8.dlc \
  --quantization_overrides encodings.json \  # AIMET-generated
  --input_dim input_1 "1,3,224,224"

# Full production conversion command:
qairt-converter \
  --input_network model.onnx \
  --output_path model_int8.dlc \
  --quantization_overrides ./aimet_output/model.encodings.json \
  --input_dim "input" "1,3,224,224" \
  --input_dtype "input" "float32" \
  --out_node "output" \
  --disable_batchnorm_folding \     # If AIMET already folded BN
  --keep_int64_inputs \             # For NLP models with token IDs
  --op_package_lib "./custom_ops.so:CustomOpPackage" \  # Custom ops
  --show_unconsumed_nodes           # Debug: show unused nodes

# Key flags:
# --input_network: ONNX/TF/TFLite model path
# --output_path: Output DLC path
# --quantization_overrides: AIMET encodings JSON
# --input_dim: Input tensor name and shape
# --backend: "htp" (Hexagon), "cpu", "gpu" (for validation)
# --float_fallback: Layers to keep in float
# --act_bw 8/16: Activation bit-width override
# --weights_bitwidth 8/4: Weight bit-width override
```

### DLC File Format

The DLC (Deep Learning Container) is Qualcomm's proprietary model format:

```
DLC Binary Structure:
┌─────────────────────────────────────────┐
│ DLC Header                              │
│   - Magic number: 0x494E444C ("INDL")  │
│   - Version: uint32                     │
│   - Model hash: SHA-256                 │
├─────────────────────────────────────────┤
│ Graph Descriptor (Protobuf)             │
│   - Layer sequence with connections     │
│   - Input/output tensor shapes          │
│   - Quantization parameters per tensor  │
├─────────────────────────────────────────┤
│ Weight Data (Blob)                      │
│   - All weights concatenated            │
│   - Each layer: offset + size           │
│   - INT8 weights: direct byte array     │
│   - INT4 weights: nibble-packed         │
├─────────────────────────────────────────┤
│ Encoding Table                          │
│   - Per-tensor: scale, offset, bw       │
│   - Per-channel tables (optional)       │
└─────────────────────────────────────────┘
```

### DLC Execution Backends

```python
# Python API for running DLC
from qti.aisw.dlc_utils import snpe_utils
import numpy as np

# Load DLC
dlc_container = snpe_utils.load_dlc("model_int8.dlc")

# Check supported backends
print("HTP available:", dlc_container.is_backend_supported("HTP"))
print("GPU available:", dlc_container.is_backend_supported("GPU"))
print("CPU available:", dlc_container.is_backend_supported("CPU"))

# Run inference on HTP (Hexagon Tensor Processor)
input_tensor = np.random.rand(1, 3, 224, 224).astype(np.float32)

result = dlc_container.execute(
    inputs={"input": input_tensor},
    runtime="HTP"   # or "GPU", "CPU", "AIP"
)
output = result["output"]
```

---

## 6. QNN SDK: Qualcomm Neural Network SDK

### QNN Architecture

QNN is the lower-level SDK used when QAIRT's abstractions are insufficient:

```
QNN SDK Layer:
  Application
      │
  QNN API (C interface)
      │
      ├── QnnHtp (Hexagon backend)
      ├── QnnGpu (GPU backend)  
      ├── QnnDsp (DSP backend, legacy)
      └── QnnCpu (Reference implementation)
```

### QNN Python API (Inference)

```python
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qacc_c
import qnn_sdk

# Initialize QNN context
qnn_context = qnn_sdk.QnnContext(
    backend="libQnnHtp.so",           # Hexagon backend
    model="model_int8.serialized",    # QNN serialized model
    system_lib="libQnnSystem.so"
)

# Create inference session
session = qnn_sdk.QnnSession(qnn_context)

# Run inference
import numpy as np
inputs = {
    "input_0": np.random.rand(1, 3, 224, 224).astype(np.float32)
}
outputs = session.execute(inputs)
predictions = outputs["output_0"]
print(f"Predicted class: {predictions.argmax()}")

# Cleanup
session.close()
qnn_context.destroy()
```

### QNN C++ API for Embedded Deployment

```cpp
// Minimal QNN C++ deployment example
#include "QnnInterface.h"
#include "QnnHtp.h"
#include "System/QnnSystemInterface.h"

// Step 1: Load backend library
void* backendHandle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL);
QNN_INTERFACE_VER_TYPE interface;
QnnInterface_getProviders(&interface, 1);

// Step 2: Create backend
Qnn_BackendHandle_t backendHandle;
interface.backendCreate(nullptr, &backendHandle);

// Step 3: Load context from .bin file (compiled model)
Qnn_ContextHandle_t contextHandle;
interface.contextCreateFromBinary(
    backendHandle, nullptr,
    modelBinaryData, modelBinarySize,  // Loaded from .bin
    &contextHandle, nullptr
);

// Step 4: Create graph execution
Qnn_GraphHandle_t graphHandle;
interface.graphRetrieve(contextHandle, "main_graph", &graphHandle);

// Step 5: Execute
Qnn_Tensor_t inputs[1] = { /* input tensor descriptor */ };
Qnn_Tensor_t outputs[1] = { /* output tensor descriptor */ };
interface.graphExecute(graphHandle, inputs, 1, outputs, 1, nullptr, nullptr);
```

---

## 7. Complete AIMET → Qualcomm Deployment Pipeline

### End-to-End Workflow

```
STAGE 1: Model Training (PyTorch)
══════════════════════════════════════════════
  import torch
  model = MyModel()
  train(model, train_loader)    # Standard FP32 training
  torch.save(model.state_dict(), 'model_fp32.pth')

STAGE 2: AIMET Optimization
══════════════════════════════════════════════
  from aimet_torch.quantsim import QuantizationSimModel, QuantScheme
  from aimet_torch.cross_layer_equalization import equalize_model
  from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
  
  # 2a. Prepare model
  model = equalize_model(model, [(1, 3, 224, 224)])
  
  # 2b. AdaRound (optional, recommended for INT8)
  adaround_params = AdaroundParameters(data_loader, num_batches=4)
  model = Adaround.apply_adaround(model, dummy_input, adaround_params,
                                  path='./output', filename_prefix='mymodel',
                                  default_param_bw=8)
  
  # 2c. QuantSim + Calibration
  sim = QuantizationSimModel(model, dummy_input, default_output_bw=8, 
                             default_param_bw=8)
  sim.compute_encodings(calibrate_fn, None)
  
  # 2d. Evaluate
  acc = evaluate(sim.model)
  print(f"INT8 accuracy: {acc:.2f}%")

STAGE 3: Export (ONNX + Encodings JSON)
══════════════════════════════════════════════
  sim.export(
      path='./output',
      filename_prefix='mymodel_int8',
      dummy_input=dummy_input,
      onnx_export_args={'opset_version': 13}
  )
  # → ./output/mymodel_int8.onnx
  # → ./output/mymodel_int8.encodings.json

STAGE 4: Qualcomm Conversion (DLC)
══════════════════════════════════════════════
  qairt-converter \
    --input_network ./output/mymodel_int8.onnx \
    --output_path ./output/mymodel_int8.dlc \
    --quantization_overrides ./output/mymodel_int8.encodings.json \
    --input_dim "input" "1,3,224,224"

STAGE 5: Validation (On-host simulation)
══════════════════════════════════════════════
  qairt-net-run \
    --container mymodel_int8.dlc \
    --input_list inputs.txt \
    --output_dir ./validation_outputs \
    --use_htp         # Simulate HTP behavior on host

STAGE 6: On-Device Deployment
══════════════════════════════════════════════
  # Push DLC to device (via ADB)
  adb push mymodel_int8.dlc /data/local/tmp/
  
  # Run on device
  adb shell /data/local/tmp/qairt-net-run \
    --container /data/local/tmp/mymodel_int8.dlc \
    --input_list /data/local/tmp/inputs.txt \
    --use_htp \
    --output_dir /data/local/tmp/outputs
```

### Encodings JSON → QAIRT Quantization Overrides

The AIMET-generated `encodings.json` is directly consumed by `qairt-converter` via `--quantization_overrides`:

```json
{
  "activation_encodings": {
    "conv1/output:0": [
      {
        "bitwidth": 8,
        "dtype": "int",
        "is_symmetric": "False",
        "max": 6.142,
        "min": 0.0,
        "offset": 0,
        "scale": 0.02408
      }
    ]
  },
  "param_encodings": {
    "conv1/kernel:0": [
      {
        "bitwidth": 8,
        "dtype": "int",
        "is_symmetric": "True",
        "max": 0.3142,
        "min": -0.3142,
        "offset": -128,
        "scale": 0.002466
      }
    ]
  }
}
```

---

## 8. Qualcomm AI Hub

### What is AI Hub?

[aihub.qualcomm.com](https://aihub.qualcomm.com) provides:
1. **Remote profiling** on real Snapdragon devices (no hardware purchase needed)
2. **Pre-optimized model zoo** (100+ models already compiled for Snapdragon)
3. **Performance estimation** across device fleet

### AI Hub Python SDK

```python
import qai_hub as hub
import torch

# Step 1: Authenticate (one-time setup)
# qai-hub configure --api_token YOUR_TOKEN

# Step 2: Compile your model
my_model = torch.jit.trace(model, dummy_input)
compile_job = hub.submit_compile_job(
    model=my_model,
    device=hub.Device("Samsung Galaxy S24"),    # Target device
    input_specs={"input": ((1, 3, 224, 224), torch.float32)},
    options="--target_runtime tflite",          # or "qnn", "onnx"
)
compile_job.wait()
compiled_model = compile_job.get_target_model()

# Step 3: Profile performance
profile_job = hub.submit_profile_job(
    model=compiled_model,
    device=hub.Device("Samsung Galaxy S24"),
)
profile_job.wait()
profile_result = profile_job.get_result()

print(f"Inference latency: {profile_result.inference_time:.2f} ms")
print(f"Peak memory: {profile_result.peak_memory_bytes / 1024:.1f} KB")

# Step 4: Validate accuracy
validation_job = hub.submit_inference_job(
    model=compiled_model,
    device=hub.Device("Samsung Galaxy S24"),
    inputs={"input": sample_inputs},
)
validation_job.wait()
predictions = validation_job.get_output_data()
```

### AI Hub Pre-Optimized Model Zoo

```
Available Models (subset, as of 2024):
  
  Computer Vision:
    ├── MobileNet-v2 (INT8, optimized for Snapdragon)
    ├── EfficientNet-B0 through B4 (INT8)
    ├── YOLOv8n/s/m/l/x (INT8, real-time detection)
    ├── ViT-B/16 (INT8 transformer)
    ├── SAM (Segment Anything, INT8)
    ├── Stable Diffusion v1.5 (INT8, 30s on SD8Gen3)
    └── ControlNet (INT8)
    
  NLP/LLM:
    ├── Whisper-Tiny/Small/Medium (INT8, ASR)
    ├── LLaMA-2-7B (INT4, ~10 tokens/sec on SD8Gen3)
    ├── Mistral-7B (INT4)
    ├── Baichuan-7B (INT4, Chinese LLM)
    └── BERT-Base/Large (INT8)
    
  Audio:
    ├── OpenAI Whisper (INT8)
    ├── Encodec (neural audio codec)
    └── MusicGen-Small (INT8)
```

---

## 9. Edge AI Constraints: Memory, Thermal, Power

### Memory Budget on Mobile Devices

```
Smartphone Memory Budget (typical, 12 GB RAM device):
  OS and system services:     3.0 GB (reserved)
  Active apps:                2.0 GB
  Camera/ISP buffers:         0.5 GB
  Graphics/GPU:               0.5 GB
  ─────────────────────────────────────
  Available for AI model:     ~6 GB maximum
  Practical safe limit:       ~2-3 GB
  
  Therefore:
  ├── LLaMA-2 7B FP16 (14 GB): NOT feasible
  ├── LLaMA-2 7B INT8 (7 GB): Barely feasible, risky
  ├── LLaMA-2 7B INT4 (3.5 GB): Feasible with careful management
  └── Phi-2 2.7B INT4 (1.5 GB): Comfortable
  
Peak activation memory:
  ResNet-50 INT8: ~6 MB activations
  BERT-Base INT8: ~24 MB activations
  ViT-L/16 INT8: ~180 MB activations
  → Must fit in Hexagon TCM (8 MB) or L2 (32 MB) for best performance
```

### Thermal Constraints

```
Thermal Design Power (TDP) constraints:

Device Type:          TDP Budget   Sustained AI TOPS
──────────────────────────────────────────────────────
Smartwatch:           < 0.5W       ~1 TOPS INT8
True wireless earbuds:< 100mW      <0.1 TOPS
Smartphone:           < 5W         ~10-15 TOPS sustained
Tablet:               < 10W        ~20 TOPS sustained
Laptop (ARM):         < 15W        ~30 TOPS sustained
Edge server card:     < 75W        ~100 TOPS sustained
Data center card:     < 150W       ~400 TOPS sustained

Thermal throttling effect:
  Snapdragon 8 Gen 3 peak: 45 TOPS (for ~5 seconds)
  After thermal saturation: drops to ~25 TOPS sustained
  With active cooling (tablet): ~35 TOPS sustained
```

### Power Budget Analysis

```python
# Simple power estimation for inference
def estimate_inference_power(model_macs, frequency_mhz, tops_efficiency):
    """
    Rough power estimation for edge inference
    
    Args:
        model_macs: MACs for one inference pass
        frequency_mhz: NPU clock frequency
        tops_efficiency: fraction of peak TOPS achieved (0.0-1.0)
    """
    # Peak TOPS from hardware spec
    peak_tops = frequency_mhz * 1e6 * ARRAY_SIZE * ARRAY_SIZE * 2  # 2 ops per MAC
    effective_tops = peak_tops * tops_efficiency
    
    # Time per inference
    inference_time_sec = (model_macs * 2) / (effective_tops * 1e12)
    
    # Energy per inference (TDP × time)
    tdp_watts = 5  # Typical smartphone NPU TDP
    energy_joules = tdp_watts * inference_time_sec
    
    return {
        "inference_time_ms": inference_time_sec * 1000,
        "energy_mJ": energy_joules * 1000,
        "inferences_per_second": 1 / inference_time_sec
    }

# Example: MobileNetV2 INT8 on Snapdragon 8 Gen 3
result = estimate_inference_power(
    model_macs=300e6,      # 300M MACs
    frequency_mhz=1100,    # 1.1 GHz Hexagon clock
    tops_efficiency=0.6    # 60% of peak (realistic)
)
# → ~8ms, ~40mJ, ~125 inferences/sec
```

---

## 10. INT8 vs INT4 vs FP16 on Qualcomm Hardware

### Quantization Format Performance Table

| Format | Memory | Bandwidth Savings | Compute Speedup | Accuracy Impact |
|--------|--------|-------------------|-----------------|-----------------|
| FP32 | 4× | 1× | 1× | Baseline |
| FP16 | 2× | 2× | 2–4× (GPU) | Negligible |
| BF16 | 2× | 2× | 2–4× (some HW) | Negligible |
| INT8 | 1× | 4× | **6–8×** (NPU) | <1% drop |
| INT4 | 0.5× | 8× | **10–15×** (NPU) | 1–5% drop |
| INT2 | 0.25× | 16× | Experimental | 10–30% drop |

### On Hexagon Specifically

```
Why INT8 is the sweet spot for Hexagon HMX:
  HMX is natively INT8 — it's designed for 8-bit arithmetic
  INT8 → Full HMX utilization
  INT4 → HMX handles nibble-packed pairs (2× throughput)
  FP16 → Processed by Adreno GPU (slower than HMX for batch inference)
  FP32 → CPU only (much slower than NPU)

Key insight: Model must be INT8 to use the NPU at all!
  FP32 model on Snapdragon → runs on CPU → 10–50× SLOWER
```

### Activation vs Weight Quantization Strategies

```
W8A8: Weights INT8, Activations INT8
  Best for: Most CNNs, standard deployment
  Accuracy: <1% drop with proper calibration
  
W4A8: Weights INT4, Activations INT8
  Best for: Memory-limited models (LLMs, large ViTs)
  Accuracy: 1–3% drop with GPTQ/AWQ
  Throughput: ~1.5× faster than W8A8 (memory-bound models)
  
W4A4: Weights INT4, Activations INT4
  Best for: Extreme memory pressure, research
  Accuracy: 3–10% drop (model-dependent)
  Challenges: Activation outliers cause severe issues
  
W16A8 (Weight-only INT4, activations FP16):
  Common for LLMs: decode phase is memory-bandwidth bottleneck
  Weights dequantized on-the-fly before compute
  Effective where bandwidth > compute
```

---

## 11. Mixed Precision Deployment Strategies

### Per-Layer Bit-Width Allocation

```python
# AIMET Mixed Precision Configuration

from aimet_torch.quantsim import QuantizationSimModel, QuantScheme

sim = QuantizationSimModel(model, dummy_input, 
                           default_output_bw=8, default_param_bw=8)

# Keep sensitive layers at higher precision
sensitive_layers = {
    'conv1': 8,          # First conv: sensitive to input noise
    'layer4.2.conv3': 8, # Last layer before classifier: sensitive
    'fc': 16,            # Classifier: critical for accuracy
}

for name, module in sim.model.named_modules():
    # Set first and last layers to higher precision
    if 'conv1' == name:
        module.output_quantizers[0].bitwidth = 8
        module.param_quantizers['weight'].bitwidth = 8
    elif 'fc' == name or 'classifier' in name:
        module.output_quantizers[0].bitwidth = 16
        module.output_quantizers[0].enabled = True
    elif hasattr(module, 'param_quantizers'):
        # Middle layers: 4-bit weights, 8-bit activations
        if 'weight' in module.param_quantizers:
            module.param_quantizers['weight'].bitwidth = 4
```

### Automatic Mixed Precision with QuantAnalyzer

```python
from aimet_torch.quant_analyzer import QuantAnalyzer, CallbackFunc

# Run sensitivity analysis to guide precision assignment
analyzer = QuantAnalyzer(
    model=model,
    dummy_input=dummy_input,
    forward_pass_callback=CallbackFunc(calibrate_fn, calibration_loader),
    eval_callback=CallbackFunc(evaluate_fn, val_loader),
)

analyzer.analyze(
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    default_param_bw=4,   # Start with 4-bit
    default_output_bw=8,
    results_dir='./sensitivity_analysis'
)

# Results in ./sensitivity_analysis/per_layer_quant_disabled.csv
# → Identify which layers cause >0.5% accuracy drop when quantized to 4-bit
# → Keep those at 8-bit, leave rest at 4-bit
```

---

## 12. Security: TrustZone, Secure Boot, Model Encryption

### Qualcomm TrustZone for Model Protection

```
ARM TrustZone on Qualcomm Devices:
  Normal World (Android/Linux):
    ├── Applications
    ├── QAIRT runtime
    └── Model weights (unencrypted, visible to OS)
  
  Secure World (QSEE - Qualcomm Secure Execution Environment):
    ├── Key management
    ├── Decryption engine
    └── Secure storage for encrypted model weights

Protected Model Loading:
  1. OEM generates encryption key (AES-256)
  2. Model weights encrypted offline: AES-GCM(key, DLC_data)
  3. Encrypted DLC shipped to device
  4. At runtime: QSEE decrypts model in secure memory
  5. HTP processes model without weights appearing in normal world RAM
```

### Secure Boot Chain for AI Models

```
Device Boot Sequence with AI Model Verification:
  
  ROM bootloader (immutable, factory)
       │ verifies signature
       ▼
  Primary bootloader (signed by Qualcomm)
       │ verifies signature  
       ▼
  Android Bootloader (ABOOT, OEM-signed)
       │ verifies dm-verity
       ▼
  Android OS (verified partition)
       │
       ▼
  QAIRT loads DLC file
       │ Verifies model hash against signed manifest
       │ Checks model certificate chain
       ▼
  Hexagon processes model
```

### Model Watermarking for IP Protection

```python
# Embedding invisible watermark in model weights
# (Detectable post-deployment without degrading accuracy)

def embed_watermark(model, watermark_bits: str):
    """Embed watermark in LSBs of specific weight matrices"""
    watermark_bytes = watermark_bits.encode()
    
    for name, param in model.named_parameters():
        if 'fc.weight' in name:  # Target specific layer
            # Convert to int representation
            weights_int = (param.data * 127).clamp(-127, 127).round().int()
            
            # Embed watermark in LSBs (1-bit per weight, imperceptible change)
            for i, bit in enumerate(format(hash(watermark_bytes), 'b')):
                if i >= weights_int.numel(): break
                weights_int.view(-1)[i] = (weights_int.view(-1)[i] & ~1) | int(bit)
            
            # Convert back
            param.data = weights_int.float() / 127
    
    return model
```

---

## 13. Multi-Model Concurrent Execution

### NPU Resource Sharing

Modern Qualcomm devices can run multiple AI models concurrently:

```
Multi-Model Execution on Hexagon:

Scenario: Smartphone camera pipeline
  ├── Face detection model: 5ms, HIGH priority
  ├── Scene classification: 15ms, MEDIUM priority
  ├── Depth estimation: 20ms, LOW priority
  └── Portrait segmentation: 10ms, HIGH priority

RTOS Scheduler on Hexagon:
  Time slot 0-5ms:   Face detection (HIGH) → runs on HMX
  Time slot 5-10ms:  Portrait segmentation (HIGH) → runs on HMX
  Time slot 10-15ms: Depth estimation (LOW) → runs on HVX in parallel
  Time slot 15-20ms: Scene classification (MEDIUM) → runs on HMX

Key: HVX and HMX can run simultaneously!
  → Face detection (HMX) + normalization preprocessing (HVX) in parallel
```

### Setting QoS Priorities in QAIRT

```python
# Python API for QoS priority setting (conceptual)
from qti.aisw.dlc_utils import QairtRuntime

runtime = QairtRuntime("model.dlc", backend="HTP")

# Set execution priority
runtime.set_performance_profile(
    profile="burst",           # "burst", "balanced", "sustained_high_performance"
    priority="high",           # "high", "normal", "low"
    dcvs_enable=True,          # Dynamic Clock and Voltage Scaling
    dcvs_policy="performance"  # "performance", "power_saver"
)

output = runtime.execute({"input": input_tensor})
```

---

## 14. Power Profiling and Thermal Management

### PMIC-Based Power Measurement

```
Power Measurement Setup (Development):

Hardware:
  ├── PMIC (Power Management IC): built into Qualcomm SoC
  │   └── Monitors current/voltage per power rail
  ├── USB Power Monitor (Monsoon Solutions FPA400)
  │   └── External measurement, ground truth
  └── Snapdragon Profiler (software, estimates)

Software Power Profiling:
  adb shell cat /sys/class/power_supply/bms/current_now
  # Returns microamps drawn from battery

  For per-component profiling:
  adb shell cat /sys/bus/iio/devices/iio:device0/in_current0_input
  # DSP power rail (Hexagon)
  
  Snapdragon Profiler GUI:
  → Shows real-time per-component power breakdown
  → Correlates with GPU/NPU utilization %
```

### DTPC: Dynamic Thermal and Power Control

```
DTPC Algorithm (Qualcomm's thermal governor):

Every 100ms tick:
  1. Read temperature sensors (CPU, GPU, NPU, skin)
  2. Compare against thermal zones:
     Zone 0 (<37°C): No throttling
     Zone 1 (37-39°C): CPU down-clock by 10%
     Zone 2 (39-41°C): NPU power cap to 70%
     Zone 3 (41-43°C): GPU + NPU throttle to 50%
     Zone 4 (>43°C): Emergency: CPU+GPU+NPU to minimum
  
  3. Apply frequency/voltage scaling
  4. Report to kernel thermal framework

AI Impact:
  At 20°C ambient: full 45 TOPS for 10+ minutes
  At 35°C ambient: ~30 TOPS sustained (throttled after 30s)
  In pocket (thermal mass): ~25 TOPS sustained

Mitigation in Production Apps:
  - Schedule AI inference in short bursts (not continuous)
  - Use duty-cycling: 100ms burst, 100ms idle
  - Monitor skin temperature and proactively throttle workload
```

---

## 15. Competitive Analysis: Qualcomm vs NVIDIA vs Apple vs Google

### Edge AI SoC Comparison Table

| Spec | Snapdragon 8 Gen 3 | Apple A17 Pro | Google Tensor G3 | Samsung Exynos 2400 |
|------|---------------------|---------------|-----------------|---------------------|
| Process Node | TSMC 4nm | TSMC 3nm | Samsung 4nm | Samsung 4nm |
| NPU TOPS (claimed) | 98 TOPS | 35 TOPS | 15 TOPS | 34.4 TOPS |
| NPU TOPS (measured) | ~45 TOPS | ~28 TOPS | ~10 TOPS | ~20 TOPS |
| INT4 Support | ✅ | ✅ (in CoreML) | Limited | Limited |
| FP16 NPU | ✅ (Adreno GPU) | ✅ | ✅ | ✅ |
| Optimization SDK | AIMET + QAIRT | Core ML Tools | TFLite + XNNPACK | Xclipse + OneUI |
| LLM (7B INT4) | ✅ ~10 tok/s | ✅ ~12 tok/s | Limited | Limited |
| Power at peak AI | ~5W | ~5W | ~6W | ~6W |
| ML Framework | ONNX/PyTorch/TF | Core ML | TensorFlow | TensorFlow |
| Open Source SDK? | Yes (AIMET) | Partial | Yes (TFLite) | Limited |

### NVIDIA Jetson vs Qualcomm AI 100 (Edge Server)

| Spec | Jetson AGX Orin 64G | Qualcomm AI 100 Ultra |
|------|--------------------|-----------------------|
| AI Performance | 275 TOPS | 400 TOPS |
| Memory | 64 GB LPDDR5 | 128 GB LPDDR5X |
| Memory BW | 204 GB/s | 3.2 TB/s (MCC) |
| TDP | 60W | 150W |
| Quantization | TensorRT INT8/FP8 | AIMET + QPC (INT8/MX) |
| LLM (LLaMA-2 7B) | 35 tok/s (INT4) | 80 tok/s (MXFP4) |
| Software Maturity | High (TensorRT) | Growing (QAIRT) |
| Price (est.) | ~$2,000 dev kit | ~$3,500 PCIe card |

---

## 16. Domain-Specific Deployment: Mobile, Auto, XR, IoT

### Mobile Consumer Apps

```
Key Requirements: <100ms UX response, <100mW sustained power

Typical AI App Models:
  Face unlock:      MobileFaceNet INT8, ~2ms, 5mW
  Photo enhance:    SR-ResNet INT8, ~80ms, 30mW
  Text prediction:  DistilBERT INT8, ~20ms, 15mW
  AR effects:       BlazeFace + body pose INT8, ~10ms, 20mW
  Voice assistant:  Whisper-Tiny INT8, ~200ms for 5s audio, 50mW

AIMET Recipe for Mobile:
  1. PTQ with CLE + AdaRound (INT8)
  2. Target: <1% accuracy drop
  3. Export with AIMET → DLC
  4. Use Qualcomm AI Hub to validate on target device
```

### Automotive (Snapdragon Ride)

```
Snapdragon Ride SA8775P (ASIL-B):
  Compute: 30 TOPS (dual Hexagon + Adreno)
  Safety: IEC 61508 SIL-2 / ISO 26262 ASIL-B
  Temperature: -40°C to 125°C (AEC-Q100)
  
  Typical models:
  - Lane detection: UltraFast INT8, 5ms
  - Object detection: YOLOX INT8, 15ms
  - Driver monitoring: EfficientNet INT8, 10ms
  
  AIMET for Auto:
  - Functional safety margin: keep >10% accuracy above safety threshold
  - Per-layer sensitivity mandatory before INT4
  - Hardware-in-loop testing required
```

### XR/AR (Qualcomm Snapdragon XR2+ Gen 2)

```
XR Constraints:
  Motion-to-photon latency: <20ms (VR), <5ms (passthrough AR)
  Power budget: ~5W (self-contained headset, 2-hour battery)
  Thermal: No fan, passive cooling only
  
  AI workloads:
  - Eye tracking: ~2ms, 1mW
  - Hand tracking: ~5ms, 3mW
  - Scene understanding: ~15ms, 8mW
  - Avatar animation: ~10ms, 5mW
  
  All must run simultaneously!
  Total AI budget: ~32ms (within one 60Hz frame)
  
  Strategy: INT4 for all models, staggered execution across frames
```

### IoT / Industrial Edge Gateway

```
Example: Industrial Quality Inspection Gateway
  Hardware: Qualcomm QCS6490 (Snapdragon 778G variant, fanless)
  Form factor: DIN-rail mount, 24V industrial supply
  
  Compute: 12 TOPS, 8GB LPDDR5
  TDP: 15W (no fan)
  
  Models running simultaneously:
  ├── Defect detector (YOLOv8s INT8): 8ms, 3W
  ├── Measurement model (regression INT8): 2ms, 0.5W
  ├── Barcode reader (classical): 1ms, 0.2W
  └── Anomaly detector (AutoEncoder INT8): 3ms, 1W
  Total: 14ms per image cycle → 71 FPS throughput
  Power: ~5W AI, 10W total system
  
  AIMET Pipeline:
  1. Train on factory defect dataset
  2. AIMET PTQ (INT8, no labeled data needed)
  3. Validate on-device via AI Hub
  4. Deploy via OTA update system
```

---

*Prev: [06_AIMET_APIs_and_Configuration.md](./06_AIMET_APIs_and_Configuration.md)*
*Next: [08_Comparable_Libraries_and_Ecosystem.md](./08_Comparable_Libraries_and_Ecosystem.md)*
