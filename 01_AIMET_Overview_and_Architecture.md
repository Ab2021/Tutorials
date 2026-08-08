# AIMET Overview and Architecture
## AI Model Efficiency Toolkit — Complete Reference Guide

> **Source**: Qualcomm Innovation Center | https://github.com/qualcomm/aimet
> **Version Coverage**: AIMET 1.31+ (PyTorch, TensorFlow, ONNX backends)

---

## Table of Contents
1. [What is AIMET?](#1-what-is-aimet)
2. [History and Origins](#2-history-and-origins)
3. [The Edge AI Problem Statement](#3-the-edge-ai-problem-statement)
4. [AIMET Architecture Overview](#4-aimet-architecture-overview)
5. [Core Pillars: Quantization and Compression](#5-core-pillars-quantization-and-compression)
6. [Supported Frameworks and Backends](#6-supported-frameworks-and-backends)
7. [Hardware Targets](#7-hardware-targets)
8. [Installation and Setup](#8-installation-and-setup)
9. [Repository Structure](#9-repository-structure)
10. [AIMET Ecosystem and Toolchain](#10-aimet-ecosystem-and-toolchain)
11. [The Qualcomm AI Stack](#11-the-qualcomm-ai-stack)
12. [AIMET Model Zoo](#12-aimet-model-zoo)
13. [Licensing and Community](#13-licensing-and-community)
14. [When to Use AIMET](#14-when-to-use-aimet)
15. [Limitations and Considerations](#15-limitations-and-considerations)

---

## 1. What is AIMET?

**AIMET (AI Model Efficiency Toolkit)** is a free, open-source Python library developed and maintained by the **Qualcomm Innovation Center (QuIC)**. It provides state-of-the-art algorithms to optimize neural network models for efficient deployment — particularly on resource-constrained edge devices such as mobile phones, IoT devices, autonomous vehicles, and embedded systems.

### Core Value Proposition

Modern deep learning models are trained in 32-bit floating-point (FP32) precision, which provides excellent numerical range and accuracy during training. However, deploying such models on hardware with limited:
- **Memory** (RAM measured in megabytes, not gigabytes)
- **Compute** (no high-end GPU, only NPU/DSP units)
- **Power Budget** (battery-powered, strict thermal envelopes)
- **Bandwidth** (limited DRAM bandwidth between memory and compute)

...requires fundamentally different representation. AIMET bridges this gap.

### In One Sentence
> AIMET converts floating-point neural networks into integer-precision models that maintain near-original accuracy while running **5x–15x faster** and consuming **4x less memory** on specialized edge hardware.

### Key Numbers

| Metric | FP32 Baseline | INT8 with AIMET | Improvement |
|--------|--------------|-----------------|-------------|
| Model Size | 100 MB | ~25 MB | **4x reduction** |
| Inference Latency (DSP) | 100ms | 10–20ms | **5–10x faster** |
| Power Consumption | Baseline | 20–40% | **2–4x reduction** |
| Accuracy Drop (typical) | — | <1% | Near-lossless |

---

## 2. History and Origins

### Timeline

| Year | Milestone |
|------|-----------|
| 2019 | Qualcomm Innovation Center (QuIC) begins internal development of AIMET |
| 2020 | AIMET open-sourced on GitHub under the BSD 3-Clause License |
| 2020 | AdaRound paper presented at **ICML 2020** — "Up or Down? Adaptive Rounding for Post-Training Quantization" (Nagel et al.) |
| 2021 | AIMET v1.x released with PyTorch and TensorFlow support |
| 2022 | AutoQuant introduced for automated PTQ pipelines |
| 2023 | ONNX backend support added; expanded LLM support |
| 2024 | AIMET integrated into Qualcomm AI Hub; support for INT4 and LLM compression |
| 2025 | AIMET 2.0 announced with unified API and MicroScaling format support |

### Research Foundation

AIMET is not just an engineering toolkit — it is grounded in **peer-reviewed research** published by Qualcomm AI Research:

1. **"Data-Free Quantization Through Weight Equalization and Bias Correction"** (Nagel et al., ICCV 2019) — Basis for CLE and Bias Correction
2. **"Up or Down? Adaptive Rounding for Post-Training Quantization"** (Nagel et al., ICML 2020) — Foundation of AdaRound
3. **"ADAROUND: ADAptive ROUNDing"** — Implementation details of weight rounding optimization
4. **"Improving Post-Training Neural Quantization"** — Sequential MSE techniques

### Organization
- **Creator**: Qualcomm Innovation Center (QUIC), a subsidiary of Qualcomm Technologies, Inc.
- **GitHub**: `qualcomm/aimet` (previously also `quic/aimet`)
- **License**: BSD 3-Clause (permissive open-source)
- **Language**: Python (primary), C++ (performance-critical components)

---

## 3. The Edge AI Problem Statement

### The Gap Between Training and Deployment

Neural network training happens in floating-point world:
```
FP32 Weight Example:
-0.347812345678901  (stored as 4 bytes, 32 bits)
Range: ~±3.4 × 10^38
Precision: ~7 decimal digits
```

But edge hardware operates in integer world:
```
INT8 Weight Example:
-44  (stored as 1 byte, 8 bits)
Range: -128 to 127
Precision: integers only
```

### Naive Quantization Fails

Simply rounding FP32 → INT8 using nearest-integer rounding often causes **catastrophic accuracy degradation**:

- A model with 76.1% Top-1 accuracy (FP32) might drop to 55% or lower with naive INT8
- This happens because:
  - Weight distributions are non-uniform (long tails, outliers)
  - Different layers have wildly different dynamic ranges
  - Simple rounding introduces correlated errors that cascade

### Why Edge Devices Demand Quantization

```
Battery Life Impact:
├── FP32 computation on CPU: ~500mW average
├── INT8 on DSP/NPU: ~50-100mW
└── Result: 5-10x better battery efficiency

Memory Pressure:
├── ResNet-50 (FP32): 97.8 MB
├── ResNet-50 (INT8): ~24.4 MB
└── Embedded device RAM budget: often 512MB total

Real-Time Latency Requirements:
├── Autonomous driving: <50ms perception loop
├── Face unlock: <100ms total pipeline
├── Keyword spotting: <10ms
└── FP32 on CPU cannot meet these at acceptable power
```

### The Solution Space

AIMET addresses this through three categories:
1. **Pre-Quantization Optimization**: Making the FP32 model more "quantization-friendly"
2. **Quantization Simulation**: Accurately predicting on-hardware accuracy before deployment
3. **Post-Quantization Recovery**: Fine-tuning to recover accuracy lost during quantization

---

## 4. AIMET Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER MODEL (FP32)                     │
│          (PyTorch / TensorFlow / ONNX)                   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  MODEL PREPARATION LAYER                  │
│  ┌─────────────────┐  ┌──────────────────┐              │
│  │  Model Validator │  │  Model Preparer  │              │
│  │  (Graph check)  │  │  (Auto-refactor) │              │
│  └─────────────────┘  └──────────────────┘              │
│  ┌────────────────────────────────────────┐              │
│  │   Connected Graph (Computational DAG)  │              │
│  └────────────────────────────────────────┘              │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              QUANTIZATION ENGINE                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │    QuantizationSimModel (QuantSim)                │   │
│  │    ┌──────────────┐  ┌────────────────────────┐  │   │
│  │    │ Fake Quant   │  │ Encoding Computation   │  │   │
│  │    │ Node Insert  │  │ (TF / TF-Enhanced /    │  │   │
│  │    │              │  │  MSE / Percentile)     │  │   │
│  │    └──────────────┘  └────────────────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  PTQ Techniques:          QAT:                          │
│  ┌─────────────┐          ┌──────────────────────────┐  │
│  │ CLE         │          │ Standard QAT             │  │
│  │ Bias Corr.  │          │ QAT with Range Learning  │  │
│  │ AdaRound    │          └──────────────────────────┘  │
│  │ AutoQuant   │                                         │
│  └─────────────┘                                         │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              COMPRESSION ENGINE                           │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐  │
│  │ Spatial SVD  │ │ Weight SVD   │ │ Channel Pruning │  │
│  └──────────────┘ └──────────────┘ └─────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │    Greedy Compression Ratio Selection Algorithm    │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              EXPORT AND VISUALIZATION                     │
│  ┌──────────────────────┐  ┌────────────────────────┐   │
│  │ sim.export()         │  │ QuantAnalyzer          │   │
│  │ → model.onnx         │  │ Visualization Tools    │   │
│  │ → encodings.json     │  │                        │   │
│  └──────────────────────┘  └────────────────────────┘   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              HARDWARE DEPLOYMENT                          │
│  ┌────────────────┐  ┌──────────────────────────────┐   │
│  │ Qualcomm QAIRT │  │ Snapdragon NPU (Hexagon)     │   │
│  │ (DLC format)   │  │ Qualcomm Cloud AI 100        │   │
│  └────────────────┘  └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Internal Module Structure

AIMET is organized into several key modules:

| Module | Purpose |
|--------|---------|
| `aimet_common` | Shared utilities: quantization definitions, graph traversal, logging |
| `aimet_torch` | PyTorch-specific APIs for QuantSim, QAT, compression |
| `aimet_tensorflow` | TensorFlow/Keras-specific APIs |
| `aimet_onnx` | ONNX-specific quantization support |
| `aimet_torch.quantsim` | Core QuantizationSimModel class |
| `aimet_torch.adaround` | AdaRound implementation |
| `aimet_torch.auto_quant` | AutoQuant automated pipeline |
| `aimet_torch.compress` | SVD and channel pruning |
| `aimet_torch.qat` | QAT with range learning |

---

## 5. Core Pillars: Quantization and Compression

### Pillar 1: Quantization

Quantization reduces the **bit-width** of numerical representations:

```
Floating Point (FP32) → Integer (INT8, INT4)

Quantization Formula:
  x_q = round(x / scale) + zero_point
  x_dequant = (x_q - zero_point) × scale

Where:
  scale = (max_val - min_val) / (2^bits - 1)
  zero_point = round(-min_val / scale)
```

**Two major strategies**:
1. **Post-Training Quantization (PTQ)**: Optimize a pre-trained model without retraining
2. **Quantization-Aware Training (QAT)**: Simulate quantization during fine-tuning

### Pillar 2: Compression

Compression reduces the **number of parameters and computations**:

```
Model Compression Strategies in AIMET:

1. Spatial SVD:
   Conv(m, n, h, w) → Conv(m, k, h, 1) + Conv(k, n, 1, w)
   Rank k << min(m*h, n*w) → Fewer MACs

2. Weight SVD:
   Linear(m, n) → Linear(m, k) + Linear(k, n)
   Rank k << min(m, n) → Fewer parameters

3. Channel Pruning:
   Conv with C_in input channels → Conv with C_in' < C_in channels
   Least significant channels removed based on activation magnitude
```

### Combining Both

Many production deployments use **both** quantization and compression:

```
Typical Pipeline:
1. Start with FP32 model (100% accuracy baseline)
2. Apply Channel Pruning (target: 2x MACs reduction)
   → Accuracy: 99.2% of baseline
3. Apply Spatial SVD (target: 1.5x additional speedup)
   → Accuracy: 98.8% of baseline
4. Apply PTQ with AdaRound (INT8)
   → Accuracy: 98.5% of baseline, 4x memory, 8x compute speedup
5. QAT fine-tuning if needed
   → Accuracy: 99.0%+ of baseline
```

---

## 6. Supported Frameworks and Backends

### PyTorch Backend (`aimet-torch`)

**Most Feature-Complete Backend**

```python
# Installation
pip install aimet-torch

# Key imports
from aimet_torch.quantsim import QuantizationSimModel, QuantScheme
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.auto_quant import AutoQuant
from aimet_torch.compress import ModelCompressor
from aimet_torch.defs import SpatialSvdParameters, ChannelPruningParameters
```

**Supported Features (PyTorch)**:
- ✅ QuantizationSimModel (QuantSim)
- ✅ PTQ: CLE, Bias Correction, AdaRound, AutoQuant
- ✅ QAT (Standard + Range Learning)
- ✅ Spatial SVD, Weight SVD, Channel Pruning
- ✅ Greedy Ratio Selection
- ✅ Model Validator / Preparer
- ✅ Batch Norm Folding and Re-estimation
- ✅ QuantAnalyzer Visualization
- ✅ ONNX Export with encodings

### TensorFlow Backend (`aimet-tensorflow`)

```python
# Installation
pip install aimet-tensorflow

# Key imports
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.cross_layer_equalization import equalize_model
from aimet_tensorflow.adaround.adaround_weight import Adaround
```

**Supported Features (TensorFlow)**:
- ✅ QuantizationSimModel
- ✅ CLE, Bias Correction, AdaRound
- ✅ AutoQuant
- ✅ Spatial SVD, Channel Pruning
- ✅ Model Validator / Preparer (Keras-focused)
- ⚠️ Some features may lag PyTorch support

### ONNX Backend (`aimet-onnx`)

```python
# Installation
pip install aimet-onnx

# Key imports
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.adaround.adaround_weight import Adaround
```

**Supported Features (ONNX)**:
- ✅ QuantizationSimModel (PTQ)
- ✅ AdaRound
- ✅ CLE
- ⚠️ Limited compression support
- ✅ Industry-standard ONNX format compatibility

### Backend Feature Comparison

| Feature | PyTorch | TensorFlow | ONNX |
|---------|---------|------------|------|
| QuantSim | ✅ Full | ✅ Full | ✅ |
| PTQ CLE | ✅ | ✅ | ✅ |
| AdaRound | ✅ | ✅ | ✅ |
| AutoQuant | ✅ | ✅ | ⚠️ Partial |
| Standard QAT | ✅ | ✅ | ❌ |
| QAT Range Learning | ✅ | ⚠️ | ❌ |
| Spatial SVD | ✅ | ✅ | ❌ |
| Channel Pruning | ✅ | ✅ | ❌ |
| QuantAnalyzer | ✅ | ✅ | ✅ |
| Model Preparer | ✅ | ✅ | N/A |

---

## 7. Hardware Targets

### Primary Targets: Qualcomm Silicon

#### Qualcomm Snapdragon (Mobile/Edge)

The Snapdragon family powers billions of edge devices:

| Platform | NPU | Key Models |
|----------|-----|-----------|
| Snapdragon 8 Gen 3 | Hexagon NPU (14 TOPS peak) | Galaxy S24, OnePlus 12 |
| Snapdragon 8 Gen 2 | Hexagon NPU (10.7 TOPS) | Galaxy S23, ASUS ROG 7 |
| Snapdragon 888 | Hexagon 780 (26 TOPS) | Galaxy S21 |
| Snapdragon X Elite | Hexagon NPU (45 TOPS) | Copilot+ PCs |
| Snapdragon Ride | ADAS NPU | Automotive |

**Hexagon NPU Architecture**:
- Dedicated tensor cores for INT8 matrix multiplication
- Hardware-accelerated quantized convolutions
- On-chip SRAM for activations (limited, critical bottleneck)
- Heterogeneous compute: CPU + GPU + DSP + NPU

#### Qualcomm Cloud AI 100

Designed for edge-cloud and data-center inference at scale:

```
Cloud AI 100 Specs:
├── Architecture: 16 AI Core processor
├── Peak Performance: ~400 TOPS (INT8)
├── Memory: 32–100 GB LPDDR5
├── Bandwidth: 200+ GB/s
├── Power: 75W (Ultra), 35W (Standard)
└── Supported formats: FP16, BF16, INT8, FP8, MXFP6
```

The AI 100 uniquely supports **MicroScaling (MX) formats** for LLM deployment:
- **MXFP6**: 6-bit floating point — better accuracy than INT6, higher throughput than FP16
- **MXFP4**: 4-bit floating point — for extreme compression of LLM weights
- **MXINT8**: Microscaling INT8 — group quantization for improved accuracy

### Secondary Targets (via ONNX export)

AIMET-quantized models (exported as ONNX with encodings) can also target:
- **ARM Ethos NPUs** (Cortex series embedded)
- **MediaTek APUs** (via ONNX Runtime delegation)
- **Intel CPUs/NPUs** (via OpenVINO conversion)
- **NVIDIA GPUs** (via TensorRT, though TensorRT has own quantization)
- **Google Edge TPU** (via TFLite conversion chain)

---

## 8. Installation and Setup

### System Requirements

| Requirement | Ubuntu 22.04 | Ubuntu 20.04 |
|-------------|-------------|-------------|
| Python | 3.10 | 3.8 |
| CUDA | 11.x–12.x | 11.x |
| cuDNN | 8.x | 8.x |
| GPU (for QAT) | NVIDIA (recommended) | NVIDIA (recommended) |
| RAM | ≥16 GB | ≥16 GB |
| Disk | ≥20 GB | ≥20 GB |

> **Note**: GPU is optional for PTQ (which can run on CPU) but strongly recommended for QAT (which requires training-like compute).

### Step 1: Create Conda Environment

```bash
# Create isolated Python environment
conda create -n aimet python=3.10 -y
conda activate aimet

# Verify Python version
python --version  # Should output: Python 3.10.x
```

### Step 2: Install AIMET Package

Choose based on your framework:

```bash
# Option A: PyTorch backend
pip install aimet-torch

# Option B: TensorFlow backend
pip install aimet-tensorflow

# Option C: ONNX backend
pip install aimet-onnx

# Option D: All backends (full installation)
pip install aimet-torch aimet-tensorflow aimet-onnx
```

### Step 3: Install Jupyter for Tutorials

```bash
pip install jupyter notebook
jupyter notebook  # Opens browser at localhost:8888
```

### Step 4: Clone Repository for Examples

```bash
git clone https://github.com/qualcomm/aimet.git
cd aimet

# Navigate to examples
ls Examples/torch/quantization/
# → adaround.py, bn_re_estimation.py, cle_bc.py, qat.py, ...
```

### Step 5: Docker-Based Setup (Alternative)

Qualcomm provides pre-built Docker images for AIMET:

```bash
# Pull the official Docker image
docker pull releases.ubuntu.com/aimet-torch:latest

# Run container with GPU support
docker run --gpus all -it \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  releases.ubuntu.com/aimet-torch:latest \
  /bin/bash
```

### Verification

```python
# Test installation
import aimet_torch
from aimet_torch.quantsim import QuantizationSimModel
print("AIMET PyTorch backend installed successfully!")
print(f"AIMET version: {aimet_torch.__version__}")
```

### Common Installation Issues

| Issue | Solution |
|-------|---------|
| `libcuda.so not found` | Install CUDA toolkit and update `LD_LIBRARY_PATH` |
| `torch version mismatch` | Pin torch version: `pip install torch==2.x.x` |
| `numpy compatibility` | Use numpy<2.0: `pip install "numpy<2.0"` |
| `GLIBCXX version error` | Update libstdc++: `conda install -c conda-forge libstdcxx-ng` |

---

## 9. Repository Structure

```
qualcomm/aimet/
├── Docs/                          # Documentation source (Sphinx)
│   ├── api_docs/                  # Auto-generated API reference
│   │   ├── torch_docs/
│   │   ├── tensorflow_docs/
│   │   └── onnx_docs/
│   └── user_guide/               # Conceptual guides
├── Examples/                      # Jupyter notebooks and scripts
│   ├── torch/
│   │   ├── quantization/
│   │   │   ├── ptq/              # Post-Training Quantization examples
│   │   │   ├── qat/              # QAT examples
│   │   │   └── adaround/         # AdaRound examples
│   │   └── compression/
│   │       ├── spatial_svd.py
│   │       ├── channel_pruning.py
│   │       └── weight_svd.py
│   ├── tensorflow/
│   └── onnx/
├── TrainingExtensions/            # Core implementation
│   ├── torch/                     # aimet_torch module
│   │   ├── src/
│   │   │   ├── AimetTorch/
│   │   │   │   ├── quantization/
│   │   │   │   ├── adaround/
│   │   │   │   ├── auto_quant.py
│   │   │   │   └── compress/
│   │   └── tests/
│   ├── tensorflow/
│   └── onnx/
├── NightlyTests/                  # Automated integration tests
├── ModelOptimizations/            # C++ performance-critical code
│   └── ops/                      # Custom CUDA ops for quantization sim
├── Jenkins/                       # CI/CD configuration
├── requirements/                  # Python dependencies per backend
│   ├── reqs_torch.txt
│   ├── reqs_tensorflow.txt
│   └── reqs_onnx.txt
├── setup.py                       # Package build configuration
└── README.md                      # Project overview
```

---

## 10. AIMET Ecosystem and Toolchain

### The Complete Optimization Workflow

```
Training Environment                Optimization (AIMET)
┌─────────────────┐                ┌──────────────────────┐
│ PyTorch Training │                │ 1. Model Preparation  │
│                 │ ─── FP32 ────► │    (Validator +       │
│ TF/Keras Train  │   model        │     Preparer)         │
└─────────────────┘                │                       │
                                   │ 2. PTQ Pipeline       │
                                   │    (CLE → AdaRound)   │
                                   │                       │
                                   │ 3. QuantSim + Eval   │
                                   │    (accuracy check)   │
                                   │                       │
                                   │ 4. QAT (if needed)   │
                                   │    (fine-tune)        │
                                   │                       │
                                   │ 5. Export             │
                                   │    (ONNX + JSON)      │
                                   └──────────┬────────────┘
                                              │
                                              ▼
Deployment Tools                      Hardware Runtime
┌─────────────────────┐              ┌─────────────────────┐
│ qairt-converter     │              │ Snapdragon NPU       │
│  .onnx → .dlc       │ ──────────► │ (Hexagon DSP)        │
│                     │   QPC/DLC   │                       │
│ QAIRT SDK           │              │ Cloud AI 100         │
│ QNN SDK             │              │                       │
└─────────────────────┘              └─────────────────────┘
```

### Integration with MLOps Pipelines

AIMET integrates naturally with existing MLOps tools:

```python
# Example: AIMET in an MLflow pipeline
import mlflow
from aimet_torch.auto_quant import AutoQuant

with mlflow.start_run():
    # Log original model metrics
    mlflow.log_metric("fp32_accuracy", eval_fp32(model))
    
    # Run AutoQuant
    auto_quant = AutoQuant(model, dummy_input, data_loader, eval_fn)
    optimized_model, accuracy, encoding_path = auto_quant.optimize(
        allowed_accuracy_drop=0.01
    )
    
    # Log optimized metrics
    mlflow.log_metric("int8_accuracy", accuracy)
    mlflow.log_artifact(encoding_path, "encodings")
    mlflow.pytorch.log_model(optimized_model, "quantized_model")
```

---

## 11. The Qualcomm AI Stack

AIMET is one layer in Qualcomm's comprehensive AI software stack:

```
┌─────────────────────────────────────────────────────────┐
│                  Applications / Models                   │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│              Qualcomm AI Hub                             │
│    (Pre-optimized model repository + cloud tools)        │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                     AIMET                                │
│    (Model quantization and compression toolkit)          │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                QAIRT / QNN SDK                           │
│    (Runtime SDK for cross-platform inference)            │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│         Hexagon NN / Hexagon SDK (DSP Driver layer)      │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│          Snapdragon Silicon / Cloud AI 100               │
│    (Physical NPU, DSP, GPU, CPU hardware)                │
└─────────────────────────────────────────────────────────┘
```

### Qualcomm AI Runtime (QAIRT)

QAIRT is the deployment runtime that consumes AIMET's output:

```bash
# Convert AIMET-quantized ONNX model to Qualcomm DLC
qairt-converter \
  --input_network model.onnx \
  --output_path ./deployed/ \
  --output_model quantized_model \
  --quantization_overrides encodings.json \
  --input_dtype float32

# Generated output:
# ./deployed/quantized_model.dlc  ← Deployable format
```

---

## 12. AIMET Model Zoo

The AIMET Model Zoo (`github.com/quic/aimet-model-zoo`) is a curated collection of:
- Pre-trained FP32 baseline models
- Quantized INT8 optimized versions
- Exact quantization recipes and scripts

### Available Model Categories

| Domain | Models Available |
|--------|----------------|
| **Image Classification** | ResNet-50, MobileNetV2, EfficientNet-B0, InceptionV3, RegNet |
| **Object Detection** | YOLOv5, SSD-MobileNet, RetinaNet |
| **Semantic Segmentation** | DeepLabV3+, FCN, MobileNetV3-Seg |
| **Pose Estimation** | HRNet, MobileNet Pose |
| **Super Resolution** | EDSR, RDN |
| **Speech Recognition** | Wav2Letter, DeepSpeech |
| **NLP** | BERT-Base, RoBERTa, DistilBERT |
| **Generative AI** | Stable Diffusion components |

### Using a Model Zoo Entry

```python
# Example: Loading a pre-quantized ResNet-50
from aimet_model_zoo.torch.classification import ResNet50

# Get FP32 baseline
fp32_model = ResNet50.get_model()

# Get INT8 quantized model (with encodings)
int8_model, encodings = ResNet50.get_quantized_model()

# Verify accuracy
accuracy = evaluate(int8_model, val_dataloader)
print(f"INT8 Top-1 Accuracy: {accuracy:.2f}%")
```

### Qualcomm AI Hub (Extended Zoo)

Beyond AIMET Model Zoo, **Qualcomm AI Hub** (`aihub.qualcomm.com`) provides:
- 500+ pre-optimized, hardware-validated models
- Direct download in device-ready format
- Performance benchmarks per-device
- One-click deployment to connected Qualcomm devices

```python
# Qualcomm AI Hub Python API
import qai_hub as hub

# Submit model for on-device profiling
profile_job = hub.submit_profile_job(
    model=my_quantized_model,
    device=hub.Device("Samsung Galaxy S24"),
)

# Get real hardware latency results
profile = profile_job.download_profile()
print(f"Latency on S24: {profile.layers[0].execution_time_ms:.1f}ms")
```

---

## 13. Licensing and Community

### License
AIMET is released under the **BSD 3-Clause License** — a permissive open-source license that allows commercial use, modification, and distribution.

```
Copyright (c) 2018, Qualcomm Innovation Center, Inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
```

### Community and Support

| Resource | URL |
|----------|-----|
| GitHub Repository | https://github.com/qualcomm/aimet |
| Documentation | https://qualcomm.github.io/aimet-pages/ |
| Issues / Bug Reports | GitHub Issues |
| Discussion Forum | GitHub Discussions |
| Model Zoo | https://github.com/quic/aimet-model-zoo |
| Qualcomm Dev Network | https://developer.qualcomm.com |
| AI Hub | https://aihub.qualcomm.com |

### Contribution Guidelines
AIMET welcomes contributions via Pull Requests:
1. Fork the repository
2. Create a feature branch
3. Follow the existing code style (PEP8 for Python)
4. Add unit tests in `TrainingExtensions/*/tests/`
5. Submit PR with detailed description

---

## 14. When to Use AIMET

### Decision Tree

```
Do you need to deploy a deep learning model on edge/mobile?
├── YES → Is your target device Qualcomm-based?
│   ├── YES → AIMET is the primary choice
│   │   ├── PyTorch model → use aimet-torch
│   │   ├── TensorFlow model → use aimet-tensorflow
│   │   └── ONNX model → use aimet-onnx
│   └── NO → Which hardware?
│       ├── Intel CPU/GPU → Consider Intel Neural Compressor
│       ├── NVIDIA GPU → Consider TensorRT/ModelOpt
│       ├── Apple Silicon → Consider Core ML Tools
│       └── ARM Cortex → AIMET + TFLite delegate
└── NO → Cloud inference needs optimization?
    ├── High accuracy priority → Start with FP16 (no AIMET needed)
    └── Cost/throughput optimization → AIMET INT8 + GPU runtime
```

### Ideal Use Cases for AIMET

1. **Mobile applications** requiring on-device AI (face recognition, on-device NLP)
2. **Automotive ADAS** on Snapdragon Ride platforms
3. **IoT and wearables** with extreme battery constraints
4. **XR devices** (AR/VR headsets with thermal limits)
5. **Edge servers** using Cloud AI 100 for smart cameras/retail
6. **Research** into quantization and compression algorithms

---

## 15. Limitations and Considerations

### Current Limitations

| Limitation | Details |
|-----------|---------|
| OS Support | Ubuntu-centric (not native Windows/macOS) |
| ONNX Support | Compression not yet available |
| LLM Coverage | Still maturing for very large transformer models |
| Custom Ops | Non-standard layers may need manual configuration |
| Training Compute | QAT requires GPU resources |

### Quantization Limitations

- **Activation outliers**: Some models (especially LLMs) have extreme activation outliers that break INT8 quantization. Requires specialized handling (SmoothQuant, AWQ approach)
- **Recurrent networks**: LSTMs/GRUs require specific handling due to time-step dependencies
- **Sparse models**: AIMET compression focuses on structured sparsity; unstructured pruning has limited native support

### When AIMET May Not Be Sufficient

1. **Sub-4-bit quantization** for complex models: May need specialized techniques like GPTQ/AWQ for LLMs
2. **Deployment on non-Qualcomm hardware**: The quantization simulation is calibrated for Qualcomm hardware behavior; other hardware may differ slightly
3. **Custom hardware accelerators**: Very specialized ASICs may require custom quantization schemes not supported by AIMET defaults

---

## Summary

AIMET represents the state of the art in production-grade model optimization for edge deployment. Its combination of:
- Research-backed algorithms (CLE, AdaRound, AutoQuant)
- Accurate hardware simulation (QuantSim)
- Multiple framework support (PyTorch, TF, ONNX)
- End-to-end toolchain integration (QAIRT, AI Hub)

...makes it an essential tool for any team deploying AI on Qualcomm-powered edge devices. The following documents in this series dive deep into each component, providing both conceptual understanding and practical implementation guidance.

---

*Next: [02_Quantization_Fundamentals.md](./02_Quantization_Fundamentals.md)*

---

## 16. AIMET 2.0 Changes: Unified API and New Abstractions

The release of AIMET 2.0 marks a significant paradigm shift in how developers interact with the toolkit. Previously, AIMET required different API calls and entirely different mental models depending on whether you were using PyTorch, TensorFlow, or ONNX. AIMET 2.0 introduces a Unified API that abstracts away the underlying framework, allowing for a write-once, run-anywhere approach to model compression and quantization.

### The Unified API
The core of the Unified API is the `aimet.Model` class. Instead of wrapping a PyTorch `nn.Module` or a TensorFlow `tf.keras.Model` directly in framework-specific QuantSim wrappers, you wrap them in `aimet.Model`.

```python
# Old AIMET 1.x (PyTorch)
from aimet_torch.quantsim import QuantizationSimModel
sim = QuantizationSimModel(model, dummy_input)

# New AIMET 2.0 (Unified)
import aimet
aimet_model = aimet.Model(model, framework='pytorch') # or 'tensorflow', 'onnx'
sim = aimet.QuantSim(aimet_model)
```

### New Abstractions
- **Quantizer Abstraction**: A single, unified `aimet.quantization.Quantizer` class now handles all quantization logic, replacing the disparate `aimet_common.defs.QuantScheme` implementations.
- **Backend Delegates**: AIMET 2.0 introduces the concept of delegates, which allow it to simulate the exact hardware behavior of specific deployment targets (e.g., Qualcomm Hexagon DSP, Cloud AI 100) dynamically, rather than relying on static configuration files.
- **Declarative Configuration**: Configuration of PTQ and QAT pipelines is now entirely declarative via YAML files, moving away from complex Python configuration dictionaries.

---

## 17. AIMET vs BitsAndBytes vs AutoGPTQ Feature Matrix

When choosing a quantization library, it's essential to understand how AIMET compares to other popular open-source options like BitsAndBytes (huggingface) and AutoGPTQ.

| Feature | AIMET | BitsAndBytes | AutoGPTQ |
| :--- | :--- | :--- | :--- |
| **Primary Use Case** | Edge Deployment (NPU/DSP) | Cloud Training/Inference | Cloud LLM Inference |
| **Target Hardware** | Qualcomm Snapdragon, AI 100 | NVIDIA GPUs | NVIDIA GPUs |
| **Supported Precision** | FP16, INT8, INT4, MX | INT8, FP4, NF4 | INT4, INT3, INT2 |
| **PTQ Algorithms** | AdaRound, CLE, AutoQuant | LLM.int8(), zero-point | GPTQ |
| **QAT Support** | Comprehensive (Range Learning) | QLoRA (Fine-tuning) | Limited |
| **Model Compression** | SVD, Channel Pruning | None | None |
| **Frameworks** | PyTorch, TF, ONNX | PyTorch (via HF) | PyTorch (via HF) |
| **AutoML/Auto-Tune** | AutoQuant, Greedy Selection | No | No |

**Summary**: 
- Use **BitsAndBytes** if you are fine-tuning large language models (LLMs) on NVIDIA GPUs using QLoRA.
- Use **AutoGPTQ** if you are deploying LLMs on NVIDIA GPUs and need aggressive 4-bit quantization.
- Use **AIMET** if you are deploying any DNN (Vision, Audio, NLP) to edge devices, mobile phones, or specialized AI accelerators, as it provides hardware-accurate simulation and advanced compression techniques like SVD.

---

## 18. AIMET Contribution Guide: How to Add Custom Quantizers

Contributing to AIMET is highly encouraged. A common requirement for researchers is adding a novel quantization scheme (e.g., a new logarithmic quantizer or a specialized outlier-handling quantizer).

### Step-by-Step Guide

1. **Environment Setup**:
   Fork the repository and set up the development environment using the provided Dockerfiles in the `Jenkins/` directory to ensure all C++ dependencies are met.

2. **C++ Implementation (Backend)**:
   The core performance-critical quantization logic resides in C++.
   - Navigate to `ModelOptimizations/ops/`.
   - Implement your custom quantizer logic in a new `.cpp` file, inheriting from the base `Quantizer` class.
   - Example: Implement `CustomLogQuantizer::quantize(float* input, int8_t* output, int size)`.
   - Expose the C++ function to Python using Pybind11 in the corresponding binding files.

3. **Python Wrapper (Frontend)**:
   - Navigate to `TrainingExtensions/torch/src/AimetTorch/quantization/`.
   - Create a Python wrapper class that inherits from `aimet_torch.tensor_quantizer.TensorQuantizer`.
   - Map your Python configuration parameters to the Pybind11 C++ calls.

4. **Add to QuantScheme Enum**:
   - Update `aimet_common.defs.QuantScheme` to include your new scheme (e.g., `QuantScheme.custom_log`).

5. **Unit Testing**:
   - Add comprehensive tests in `TrainingExtensions/torch/tests/` verifying that the forward pass matches your theoretical mathematical model and that gradients (if QAT) are computed correctly.

---

## 19. AIMET Roadmap: Announced and Planned Features

The Qualcomm Innovation Center actively develops AIMET. Here is a look at the announced roadmap for the upcoming year:

### Q3 2026: Enhanced LLM Support
- **Native AWQ (Activation-aware Weight Quantization)**: Built-in support for AWQ, which has proven highly effective for LLaMA and Mistral models.
- **KV Cache Quantization**: Native APIs to quantize the Transformer Key-Value cache to INT8 and INT4, drastically reducing memory footprint during long-context generation.

### Q4 2026: Advanced Hardware Modeling
- **Cycle-Accurate Latency Estimation**: Currently, AIMET uses MAC reduction as a proxy for latency. The new feature will provide cycle-accurate latency predictions for specific Snapdragon SoCs directly within the Python API.
- **Memory Bandwidth Modeling**: Profiling tools to estimate DRAM read/write bandwidth utilization after compression.

### Q1 2027: Neural Architecture Search (NAS)
- Integration of a lightweight NAS engine that works in tandem with the Greedy Compression algorithm to not just prune, but fundamentally alter the topology of the network for the target hardware.

---

## 20. Debugging AIMET Issues: Common Errors with Solutions

Working with quantization can be tricky. Here are the most common errors and how to resolve them.

### Error 1: `RuntimeError: Quantization Sim Model requires a valid dummy input`
- **Cause**: The `dummy_input` provided to `QuantizationSimModel` does not match the expected input shape or type of the model.
- **Solution**: Ensure the dummy input has the correct batch size, channels, height, and width. If the model takes multiple inputs, provide a tuple of tensors: `dummy_input = (torch.randn(1, 3, 224, 224), torch.randn(1, 10))`.

### Error 2: `ValueError: Batch Norm layer not folded`
- **Cause**: Attempting to run certain PTQ techniques (like CLE) when Batch Normalization layers are still present as separate nodes in the computational graph.
- **Solution**: Explicitly call `aimet_torch.batch_norm_fold.fold_all_batch_norms(model, dummy_input)` before passing the model to AIMET optimization functions.

### Error 3: Accuracy drops to 0.1% after INT4 Quantization
- **Cause**: Naive INT4 quantization causes massive precision loss.
- **Solution**: You must use AdaRound or QAT for INT4. Do not rely solely on the default `QuantScheme.post_training_tf_enhanced` for 4-bit weights.

### Error 4: `OSError: libaimet_ops.so: cannot open shared object file`
- **Cause**: The C++ extensions were not compiled correctly, or you are running in an environment without the required CUDA/cuDNN libraries.
- **Solution**: Use the official AIMET Docker image, or ensure your `LD_LIBRARY_PATH` includes the directory containing the compiled AIMET shared objects.

---

## 21. AIMET with Docker: Complete Containerized Workflow

To avoid environment and dependency hell (especially with C++ bindings and CUDA), using Docker is the recommended approach for AIMET.

### Building the Image
Qualcomm provides base Dockerfiles.
```bash
git clone https://github.com/qualcomm/aimet.git
cd aimet
# Build the PyTorch GPU image
docker build -t aimet-torch-gpu -f Jenkins/Dockerfile.pytorch-gpu .
```

### Running the Container
```bash
docker run --rm -it --gpus all \
  -v /path/to/your/code:/workspace \
  -v /path/to/your/data:/data \
  -p 8888:8888 \
  aimet-torch-gpu /bin/bash
```

### Docker Compose Workflow
For team environments, a `docker-compose.yml` is useful:
```yaml
version: '3.8'
services:
  aimet-env:
    image: aimet-torch-gpu:latest
    volumes:
      - ./:/workspace
      - /mnt/data/datasets:/datasets
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: jupyter notebook --ip 0.0.0.0 --allow-root
```

---

## 22. AIMET CI/CD Integration with GitHub Actions

Automating model compression ensures that every new model iteration is profiled and quantized automatically.

### Example GitHub Action Workflow (`.github/workflows/aimet.yml`)
```yaml
name: AIMET Optimization Pipeline

on:
  push:
    branches: [ main ]

jobs:
  optimize:
    runs-on: ubuntu-latest-gpu
    container:
      image: releases.ubuntu.com/aimet-torch:latest
      options: --gpus all

    steps:
    - uses: actions/checkout@v3
    
    - name: Run AutoQuant
      run: |
        python scripts/run_autoquant.py --model_path weights/model_latest.pth --output_dir ./optimized/
        
    - name: Upload Artifacts
      uses: actions/upload-artifact@v3
      with:
        name: quantized_model
        path: |
          ./optimized/model.onnx
          ./optimized/encodings.json
          
    - name: Report Metrics
      run: |
        python scripts/generate_report.py ./optimized/metrics.json >> $GITHUB_STEP_SUMMARY
```

---

## 23. AIMET in Research Papers: Citation Guide and Related Works

If you use AIMET in academic research, proper citation is required. AIMET implements several seminal papers from Qualcomm AI Research.

### Primary Citation
If you use the toolkit generally, cite the official whitepaper/repo:
```bibtex
@misc{aimet_repository,
  title={AI Model Efficiency Toolkit (AIMET)},
  author={Qualcomm Innovation Center},
  year={2020},
  url={https://github.com/qualcomm/aimet}
}
```

### Citing Specific Algorithms
- **AdaRound**:
  ```bibtex
  @inproceedings{nagel2020up,
    title={Up or down? Adaptive rounding for post-training quantization},
    author={Nagel, Markus and Amjad, Rana Ali and Van Baalen, Mart and Louizos, Christos and Blankevoort, Tijmen},
    booktitle={International Conference on Machine Learning},
    pages={7197--7206},
    year={2020},
    organization={PMLR}
  }
  ```
- **Cross-Layer Equalization (CLE)**:
  ```bibtex
  @inproceedings{nagel2019data,
    title={Data-free quantization through weight equalization and bias correction},
    author={Nagel, Markus and Baalen, Mart van and Blankevoort, Tijmen and Welling, Max},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={1325--1334},
    year={2019}
  }
  ```

---

## 24. Community Resources, Tutorials, YouTube Content

To further your understanding of AIMET, leverage these community resources:

### Official Tutorials
- **AIMET GitHub Examples**: The `Examples/` directory in the repository contains deeply commented Jupyter Notebooks. Start with `Examples/torch/quantization/ptq/adaround.ipynb`.
- **Qualcomm Developer Network**: Features long-form articles and use-case studies (e.g., "Deploying YOLOv7 on Snapdragon using AIMET").

### Video Content
- **Qualcomm AI Research YouTube Channel**: Look for the "AI Model Efficiency" playlist. Key videos include:
  - *Tech Talk: Post-Training Quantization with AIMET* (1 hour deep dive)
  - *NeurIPS 2020 Tutorial: Quantization and Neural Network Compression* (Presented by Tijmen Blankevoort, heavily features AIMET concepts).
- **Community Tutorials**: Search YouTube for "AIMET PyTorch Tutorial" for community-created walkthroughs on exporting models to ONNX and deploying via QAIRT.

### Forums
- **GitHub Discussions**: The primary place for troubleshooting and feature requests.
- **StackOverflow**: Tag questions with `[aimet]` and `[qualcomm]`.

