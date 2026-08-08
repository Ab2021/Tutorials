# Comparable Model Optimization Libraries and Ecosystem

This document provides an exhaustive, in-depth analysis of the model optimization ecosystem, focusing on libraries and toolkits comparable to Qualcomm's AI Model Efficiency Toolkit (AIMET). Model optimization encompasses techniques such as quantization (PTQ, QAT), pruning, sparsity, and knowledge distillation. As edge AI, mobile deployment, and large language models (LLMs) continue to dominate the AI landscape, the need for efficient, low-latency, and low-memory inference has led to a proliferation of specialized libraries.

Below, we explore 20 distinct model optimization libraries, detailing their purpose, features, supported hardware, code examples, benchmarks, and how they compare with AIMET. Finally, we provide a comprehensive comparison matrix, a decision guide, and a look into emerging trends.

---

## 1. PyTorch Native Quantization (torch.quantization + torchao)

### Full Description and Purpose
PyTorch provides native support for model quantization, allowing developers to convert floating-point models to lower precision (e.g., INT8) directly within the PyTorch ecosystem. Recently, PyTorch has introduced `torchao` (Architecture Optimization) to support advanced quantization techniques for Large Language Models (LLMs), such as 4-bit weight-only quantization and GPTQ-style optimizations.

### Key Features and Algorithms
- **Dynamic Quantization:** Quantizes weights ahead of time but computes activations dynamically at runtime.
- **Static Post-Training Quantization (PTQ):** Calibrates both weights and activations using a representative dataset.
- **Quantization-Aware Training (QAT):** Simulates quantization during training to recover accuracy drops.
- **torchao:** Supports INT4/INT8 weight-only quantization, AWQ, and GPTQ primitives specifically designed for Llama and other transformer models.

### Supported Hardware/Backends
- x86 CPUs (via FBGEMM)
- ARM CPUs (via QNNPACK)
- NVIDIA GPUs (via TensorRT / cuBLAS)
- Custom backends via PyTorch Executorch.

### Detailed Comparison with AIMET
While PyTorch Native Quantization is highly integrated into the training framework, it is generally hardware-agnostic but optimized for generic server/mobile CPUs. AIMET, on the other hand, is specifically engineered to unlock maximum performance on Qualcomm Hexagon DSPs and Snapdragon platforms. AIMET offers more advanced PTQ techniques like AdaRound and Cross-Layer Equalization (CLE), which are not available out-of-the-box in native PyTorch.

### Code Example Showing Basic Usage
```python
import torch
from torchvision import models

# Load pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Fuse modules (Conv+BN+ReLU)
torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu']], inplace=True)

# Specify quantization configuration (QNNPACK for ARM edge devices)
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(model, inplace=True)

# Calibrate with dummy data
calibration_data = torch.randn(32, 3, 224, 224)
model(calibration_data)

# Convert to quantized model
torch.quantization.convert(model, inplace=True)
```

### Performance Benchmarks
- ResNet-50 INT8 on ARM CPU: ~2-3x speedup compared to FP32, with <1% Top-1 accuracy loss.
- Llama-2 7B with `torchao` INT4: Fits in <5GB VRAM, ~2x token generation speedup on RTX 3090.

### Strengths and Weaknesses
**Strengths:** Seamless integration with PyTorch workflows; no external dependencies; robust QAT support.
**Weaknesses:** Lacks advanced hardware-specific optimizations for non-CPU platforms; advanced PTQ algorithms (like AdaRound) require third-party libraries.

### Best Use Cases
- Researchers building PyTorch models who need a quick baseline for quantization.
- Deploying PyTorch models to standard x86 or ARM CPUs without specific NPU targeting.

---

## 2. TensorFlow Model Optimization Toolkit (TF-MOT)

### Full Description and Purpose
TF-MOT is a suite of tools built on top of TensorFlow to optimize machine learning models for deployment. It focuses on making models smaller and faster, primarily targeting mobile, edge, and IoT devices through TensorFlow Lite (TFLite).

### Key Features and Algorithms
- **Weight Clustering:** Groups weights into a smaller number of clusters, reducing model size when compressed.
- **Pruning (Magnitude-based):** Gradually zeros out small magnitude weights during training to create sparse models.
- **Post-Training Quantization (PTQ):** Dynamic range, full integer, and float16 quantization.
- **Quantization-Aware Training (QAT):** Uses fake quantization nodes in the TF graph.
- **Collaborative Optimization:** Combines pruning, clustering, and quantization together.

### Supported Hardware/Backends
- TensorFlow Lite backends (EdgeTPU, Hexagon DSP via delegates, ARM CPUs, WebGL).
- Generic CPUs and GPUs supporting TensorFlow.

### Detailed Comparison with AIMET
TF-MOT is closely tied to the TFLite ecosystem. If your target is an Android device utilizing the NNAPI or a generic TFLite delegate, TF-MOT is the standard path. AIMET provides a superset of features for Qualcomm hardware, offering sophisticated algorithms (like CLE) that TF-MOT lacks, allowing AIMET to achieve INT8 quantization on complex models where TF-MOT might require QAT.

### Code Example Showing Basic Usage
```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(10,), activation='relu'),
    tf.keras.layers.Dense(1)
])

# Apply Quantization-Aware Training
quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)

q_aware_model.compile(optimizer='adam', loss='mse')
# Train the model...
# q_aware_model.fit(train_data, train_labels, epochs=1)
```

### Performance Benchmarks
- MobileNetV2 Pruning + Quantization: 4x reduction in model size, 2-3x latency improvement on Mobile CPU (TFLite).

### Strengths and Weaknesses
**Strengths:** Deep integration with TFLite; excellent documentation; collaborative optimization pipelines.
**Weaknesses:** Strictly tied to the TensorFlow ecosystem; migrating TF models to specific NPUs (like Apple Neural Engine) can sometimes be frictional compared to ONNX.

### Best Use Cases
- Android applications utilizing TFLite.
- Embedded systems and microcontrollers (TensorFlow Lite for Microcontrollers).

---

## 3. Intel Neural Compressor (INC)

### Full Description and Purpose
Intel Neural Compressor (formerly Low Precision Optimization Tool) is an open-source Python library designed to optimize AI models for Intel hardware (Xeon CPUs, Gaudi accelerators, ARC GPUs). It provides unified APIs for PTQ, QAT, pruning, and distillation across multiple frameworks (PyTorch, TensorFlow, ONNX, MXNet).

### Key Features and Algorithms
- **Accuracy-Driven Tuning:** Automatically searches for the optimal quantization recipe that meets a predefined accuracy loss target (e.g., <1%).
- **Advanced PTQ:** Supports SmoothQuant for LLMs, weight-only quantization (INT4/INT8).
- **Sparsity/Pruning:** Pattern-based pruning, magnitude pruning.
- **Distillation:** Unified API for knowledge distillation.

### Supported Hardware/Backends
- Intel Xeon Scalable Processors (AVX-512, AMX instructions)
- Intel Gaudi AI Accelerators (Habana)
- Intel Data Center GPUs

### Detailed Comparison with AIMET
INC is to Intel hardware what AIMET is to Qualcomm hardware. Both focus heavily on hardware-aware optimizations. INC excels in server-side deployments on Intel Xeon CPUs, utilizing AMX instructions for rapid INT8/BF16 math. AIMET targets edge and mobile power-constrained DSPs. INC's accuracy-driven auto-tuning is a standout feature that simplifies the workflow.

### Code Example Showing Basic Usage
```python
from neural_compressor import Quantization, PostTrainingQuantConfig

# Assuming a PyTorch model and dataloader are defined
config = PostTrainingQuantConfig(approach="static")
quantizer = Quantization(config)
quantizer.model = model
quantizer.calib_dataloader = calib_dataloader

# Automatically tunes the model to meet default accuracy criteria
quantized_model = quantizer.fit()
quantized_model.save("saved_model_dir")
```

### Performance Benchmarks
- BERT-base on Intel Xeon Platinum (AMX): INT8 achieves ~3.5x throughput improvement over FP32 with <0.5% accuracy loss.
- Llama-2 7B with SmoothQuant: Near-lossless INT8 inference on Intel CPUs.

### Strengths and Weaknesses
**Strengths:** Multi-framework support; automated accuracy-aware tuning; unmatched performance on Intel Silicon.
**Weaknesses:** Value proposition drops significantly if deploying on non-Intel hardware (e.g., ARM or NVIDIA).

### Best Use Cases
- Cloud and on-premise server deployments utilizing Intel Xeon processors.
- High-throughput inference pipelines in data centers.

---

## 4. NVIDIA TensorRT + ModelOpt

### Full Description and Purpose
TensorRT is NVIDIA's high-performance deep learning inference SDK. NVIDIA ModelOpt (formerly TRT Model Optimizer / AMMO) is a library that sits on top of TensorRT to provide advanced quantization, sparsity, and distillation to prepare models (especially LLMs) for TensorRT deployment.

### Key Features and Algorithms
- **FP8 Quantization:** Native support for Hopper architecture FP8 formats.
- **INT4 AWQ / GPTQ:** Weight-only quantization for LLMs.
- **SmoothQuant:** Activation scaling for LLM INT8 quantization.
- **2:4 Structured Sparsity:** Prunes 2 out of every 4 weights to leverage NVIDIA Ampere/Hopper sparse Tensor Cores.

### Supported Hardware/Backends
- NVIDIA GPUs (Ampere, Ada, Hopper architectures).

### Detailed Comparison with AIMET
TensorRT/ModelOpt focuses strictly on data center and high-end automotive/edge GPUs (NVIDIA Drive/Orin). It supports FP8, which AIMET currently does not emphasize as much since edge DSPs typically focus on INT8/INT4. AIMET provides Cross-Layer Equalization which is highly effective for INT8, whereas ModelOpt leans on SmoothQuant and FP8 for modern network architectures.

### Code Example Showing Basic Usage (ModelOpt)
```python
import torch
import modelopt.torch.quantization as mtq

model = load_your_llm_model()

# Configure FP8 quantization recipe
config = mtq.QuantizeConfig(
    quant_algo="fp8",
    activation_scheme="static"
)

# Apply quantization
mtq.quantize(model, config, calibration_dataloader)

# Export for TensorRT-LLM
mtq.export_tensorrt_llm_checkpoint(model, "trt_llm_ckpt")
```

### Performance Benchmarks
- Llama-2 70B on H100: FP8 quantization yields ~1.8x throughput increase compared to FP16, with virtually zero perplexity degradation.
- ResNet-50 2:4 Sparsity: ~1.5x speedup on A100.

### Strengths and Weaknesses
**Strengths:** Best-in-class performance on NVIDIA GPUs; native FP8 and 2:4 sparsity support.
**Weaknesses:** Closed ecosystem; proprietary formats; only beneficial for NVIDIA hardware.

### Best Use Cases
- Data center LLM serving (via TensorRT-LLM).
- Autonomous driving and robotics utilizing NVIDIA Jetson/Orin platforms.

---

## 5. Apache TVM

### Full Description and Purpose
Apache TVM is an open-source machine learning compiler framework for CPUs, GPUs, and machine learning accelerators. It aims to enable machine learning models to run efficiently on any hardware backend. It includes Relay (high-level IR) and a robust quantization pass.

### Key Features and Algorithms
- **AutoTVM / AutoScheduler:** Uses machine learning to search for optimal tensor operator schedules for a specific hardware target.
- **Relay Quantization:** High-level graph quantization supporting INT8.
- **Bring Your Own Codegen (BYOC):** Allows integration of proprietary compiler backends (like ARM Ethos or Vitis AI) into the TVM pipeline.

### Supported Hardware/Backends
- Virtually everything: x86, ARM, MIPS, WebAssembly, NVIDIA GPUs, AMD GPUs, Apple Metal, custom NPUs.

### Detailed Comparison with AIMET
TVM is a compiler, whereas AIMET is an optimization toolkit. TVM optimizes the execution graph and operator scheduling, while AIMET modifies the model's weights and mathematical representation (quantization/pruning). Often, these are complementary: a model optimized by AIMET can be compiled by TVM for deployment on an embedded CPU/GPU.

### Code Example Showing Basic Usage
```python
import tvm
from tvm import relay

# Parse PyTorch/ONNX model to Relay IR
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# Perform Relay PTQ
with relay.quantize.qconfig(calibrate_mode="kl_divergence", weight_scale="max"):
    qmod = relay.quantize.quantize(mod, params, dataset=calibration_data)

# Compile for target
target = "llvm -mcpu=cascadelake"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(qmod, target=target)
```

### Performance Benchmarks
- AutoScheduler optimizations frequently yield 20-50% speedups over vendor-provided libraries (like cuDNN or MKL) on specific non-standard tensor shapes.

### Strengths and Weaknesses
**Strengths:** Incredible hardware flexibility; powerful auto-tuning capabilities.
**Weaknesses:** Steep learning curve; complex setup; PTQ algorithms are somewhat basic compared to specialized tools like AIMET or INC.

### Best Use Cases
- Deploying models to obscure or highly specialized hardware.
- Compiling models for web deployment (WebAssembly/WebGPU).

---

## 6. ONNX Runtime Quantization

### Full Description and Purpose
ONNX Runtime (ORT) is a cross-platform inferencing and training accelerator. ORT Quantization provides tools to quantize ONNX models. Because ONNX is the de facto interoperability standard, ORT quantization is widely used to compress models originating from PyTorch, TF, or Scikit-Learn.

### Key Features and Algorithms
- **Dynamic and Static PTQ:** Supports asymmetric and symmetric quantization for INT8/UINT8.
- **Operator Specific Quantization:** Allows enabling/disabling quantization for specific nodes (e.g., keeping sensitive Convs in FP32).
- **Hardware-Specific Optimizations:** Generates QDQ (Quantize-Dequantize) formats optimized for TensorRT, NNAPI, or CPU execution providers.

### Supported Hardware/Backends
- Any hardware supported by ONNX Runtime Execution Providers (CPU, CUDA, TensorRT, DirectML, CoreML, NNAPI, OpenVINO).

### Detailed Comparison with AIMET
ORT Quantization is a deployment-stage tool. It is excellent for basic INT8 quantization. However, if a model experiences severe accuracy drop during ORT's static calibration, ORT lacks built-in mechanisms to recover it (like QAT or AdaRound). AIMET is used *prior* to ONNX export to ensure the model is quantization-friendly; ORT is then used to execute that quantized ONNX model.

### Code Example Showing Basic Usage
```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

class MyDataReader(CalibrationDataReader):
    # Implement get_next() to yield calibration data dicts
    pass

# Quantize ONNX model
quantize_static(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    calibration_data_reader=MyDataReader(),
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)
```

### Best Use Cases
- Standardized cross-platform deployment across Windows (DirectML), Linux, and Mobile.
- Quick INT8 conversion for models that don't suffer from severe quantization noise.

---

## 7. Hugging Face Optimum

### Full Description and Purpose
Optimum is an extension of the Hugging Face Transformers library that provides a unified API for performance optimization and deployment. It integrates with third-party hardware optimization libraries to streamline the quantization of NLP and Vision models.

### Key Features and Algorithms
- **Backend Integrations:** Seamlessly wraps ONNX Runtime, OpenVINO, Neural Compressor, and Habana Gaudi.
- **BetterTransformer:** FastPath execution for transformer models.
- **Auto-Quantization APIs:** Easily apply AWQ, GPTQ, or BitsAndBytes to Hugging Face Hub models.

### Supported Hardware/Backends
- Wraps multiple backends (Intel, AMD, NVIDIA, ONNX).

### Detailed Comparison with AIMET
Optimum is highly domain-specific (Transformers/Diffusers) and acts as an abstraction layer. AIMET is a lower-level toolkit handling the algorithmic math of quantization. While AIMET can optimize a Transformer, Optimum makes it a one-line code change if you are staying within the Hugging Face ecosystem.

### Best Use Cases
- NLP practitioners deploying Hugging Face models who want immediate speedups without writing custom optimization loops.

---

## 8. Brevitas (AMD/Xilinx)

### Full Description and Purpose
Brevitas is a PyTorch research library for quantization-aware training (QAT), maintained by AMD/Xilinx. It is specifically designed to train models with highly customized, extremely low-bit precision (e.g., 1-bit, 2-bit, 3-bit, 4-bit) for FPGA deployment.

### Key Features and Algorithms
- **Arbitrary Bit-Width QAT:** Train networks with custom bit-widths for weights, activations, and accumulators.
- **FINN Export:** Exports quantized models to FINN, a compiler that synthesizes neural networks into FPGA bitstreams.

### Detailed Comparison with AIMET
Brevitas is heavily focused on QAT for FPGAs, allowing non-standard bit widths. AIMET focuses on standard bit widths (INT4, INT8, INT16) optimized for DSPs and NPUs. If deploying to a Xilinx FPGA, Brevitas is the undisputed choice; for Snapdragon, AIMET is required.

---

## 9. BitsAndBytes (LLM quantization)

### Full Description and Purpose
BitsAndBytes is a lightweight wrapper around custom CUDA functions that revolutionized LLM deployment by introducing 8-bit and 4-bit (NF4) quantization techniques. It enables loading massive models (like 70B parameter LLMs) on consumer GPUs.

### Key Features and Algorithms
- **LLM.int8():** Mixed-precision quantization handling outlier features in FP16 and standard features in INT8.
- **NF4 (NormalFloat 4-bit):** An information-theoretically optimal data type for normally distributed weights, used heavily in QLoRA.
- **Double Quantization:** Quantizes the quantization constants to save further memory.

### Detailed Comparison with AIMET
BitsAndBytes is strictly for NVIDIA GPUs and focuses on weight-only quantization to solve memory bandwidth bottlenecks in LLMs. AIMET provides comprehensive weight and activation quantization for mobile DSPs. BitsAndBytes is not suitable for edge deployment.

---

## 10. AutoGPTQ

### Full Description and Purpose
AutoGPTQ is a popular library implementing the GPTQ (Generative Pre-trained Transformer Quantization) algorithm. It allows for highly efficient 2, 3, 4, and 8-bit weight-only quantization of LLMs.

### Key Features and Algorithms
- **GPTQ Algorithm:** Uses second-order information (Hessian matrix) to compensate for quantization errors, allowing rapid, accurate PTQ for massive models.
- **Hugging Face Integration:** Supported natively via `transformers`.

### Detailed Comparison with AIMET
AutoGPTQ is purpose-built for LLM weight quantization on GPUs. AIMET handles a broader range of models (CNNs, RNNs, Transformers) and focuses on mobile NPUs. Recently, AIMET and Qualcomm have developed specific techniques for LLM deployment on Snapdragon, but GPTQ remains heavily tied to GPU execution.

---

## 11. AWQ (Activation-aware Weight Quantization)

### Full Description and Purpose
AWQ is a recent optimization algorithm and library that improves upon GPTQ. It recognizes that not all weights are equally important. By observing activation distributions during calibration, AWQ identifies salient weights (about 1% of the total) and protects them by scaling, while quantizing the rest to 4-bit.

### Key Features and Algorithms
- **Hardware-friendly:** Does not rely on mixed-precision execution, making it easier to deploy on various hardware.
- **Activation-aware Scaling:** Dramatically reduces quantization loss for LLMs.

### Detailed Comparison with AIMET
AWQ is a specific algorithmic approach for LLMs. AIMET's Data-Free Quantization and Cross-Layer Equalization share similar philosophical roots (scaling weights to minimize error), but AWQ is specialized for the heavy outliers found in LLM activations.

---

## 12. SmoothQuant

### Full Description and Purpose
SmoothQuant is an algorithm and toolkit designed to enable 8-bit weight, 8-bit activation (W8A8) quantization for LLMs. LLMs suffer from massive activation outliers that make standard W8A8 quantization fail.

### Key Features and Algorithms
- **Mathematical Smoothing:** Shifts the quantization difficulty from activations to weights by introducing a mathematically equivalent scaling factor offline.

### Detailed Comparison with AIMET
SmoothQuant is a specific algorithmic technique. Similar concepts are actually implemented within AIMET (like CLE, which shifts scale between layers in CNNs). SmoothQuant is the Transformer equivalent of AIMET's CLE.

---

## 13. Intel OpenVINO NNCF

### Full Description and Purpose
Neural Network Compression Framework (NNCF) is Intel's advanced optimization library for OpenVINO. It supports PyTorch and TensorFlow and provides state-of-the-art compression algorithms.

### Key Features and Algorithms
- **Filter Pruning:** Geometric median pruning.
- **Movement Sparsity:** For Transformer models.
- **Accuracy-Aware PTQ:** Similar to INC, automatically finds the best quantization strategy.

### Detailed Comparison with AIMET
NNCF generates IR optimized for OpenVINO (Intel CPUs, integrated GPUs, NPUs). It is the direct equivalent of AIMET for the Intel Edge ecosystem (e.g., Intel Core Ultra NPUs).

---

## 14. Google EdgeTPU / TFLite Compiler

### Full Description and Purpose
Google's EdgeTPU compiler takes a quantized TFLite model and compiles it into an executable specifically for the Coral Edge TPU.

### Key Features and Algorithms
- **Operator Mapping:** Maps TFLite INT8 operators to Edge TPU MAC arrays.
- **Memory Scheduling:** Optimizes SRAM usage on the Edge TPU.

### Detailed Comparison with AIMET
This is a pure compiler. You would use TF-MOT to quantize the model, and the EdgeTPU compiler to deploy it. AIMET handles the quantization phase for Qualcomm hardware, while Qualcomm's SNPE/QAIRT acts as the compiler equivalent to EdgeTPU.

---

## 15. Qualcomm Neural Processing SDK (SNPE/QAIRT)

### Full Description and Purpose
SNPE (Snapdragon Neural Processing Engine) and QAIRT (Qualcomm AI Engine Direct) are the execution engines for Qualcomm hardware. While AIMET optimizes the model in PyTorch/TF, SNPE/QAIRT compile and run it on the Hexagon DSP.

### Key Features and Algorithms
- **Offline Graph Compiler:** Converts ONNX/TF/TFLite to Qualcomm DLC format.
- **HTP/DSP Delegation:** Routes operations to the Hexagon Tensor Processor.

### Detailed Comparison with AIMET
SNPE/QAIRT are the deployment targets for AIMET. AIMET ensures the model is accurate in INT8; SNPE ensures the INT8 model runs at maximum clock speed on the Snapdragon SOC. They are paired tools.

---

## 16. ARM NN / Arm Compute Library

### Full Description and Purpose
Arm NN is an inference engine for CPUs, GPUs, and NPUs from ARM. It bridges the gap between frameworks (TFLite, ONNX) and the underlying Arm Compute Library (ACL).

### Key Features and Algorithms
- **NEON / SVE Optimization:** Hand-tuned assembly for ARM CPUs.
- **Ethos-U NPU Support:** Compiles networks for ARM's embedded NPUs (microcontrollers).

### Detailed Comparison with AIMET
If your target is a generic ARM Cortex-A CPU or Mali GPU, Arm NN is the deployment stack. AIMET models can technically run on ARM CPUs, but AIMET's specific DSP optimizations (like specific rounding behaviors) are tailored for Qualcomm Hexagon, not ARM Ethos NPUs.

---

## 17. MediaTek NeuroPilot

### Full Description and Purpose
NeuroPilot is MediaTek's AI ecosystem for their Dimensity and Helio SoCs. It includes an SDK, an APU (AI Processing Unit) compiler, and quantization tools.

### Key Features and Algorithms
- **APU Compiler:** Targets MediaTek's custom AI accelerators.
- **Quantization:** Proprietary PTQ and QAT tools for MTK hardware.

### Detailed Comparison with AIMET
NeuroPilot is the direct competitor to Qualcomm's AIMET/SNPE stack. If you are developing a mobile app for a MediaTek Dimensity 9000, you use NeuroPilot. If developing for Snapdragon 8 Gen 2, you use AIMET/QAIRT.

---

## 18. Apple Core ML Tools (coremltools)

### Full Description and Purpose
`coremltools` is a Python package to convert PyTorch/TF models into the Core ML format for deployment on Apple devices (iOS, macOS, watchOS), specifically targeting the Apple Neural Engine (ANE).

### Key Features and Algorithms
- **Core ML Quantization:** Post-training quantization to INT8 or INT4.
- **Palettization:** Weight clustering for model size reduction.
- **ANE Optimization:** specific graph transformations to ensure operators run on the ANE rather than falling back to CPU/GPU.

### Detailed Comparison with AIMET
`coremltools` serves the Apple ecosystem. Its quantization algorithms are somewhat simpler than AIMET's (relying mostly on basic PTQ), but the tight integration with the ANE makes it incredibly efficient for iPhones. AIMET cannot be used to optimize for Apple hardware.

---

## 19. Sony MCT (Model Compression Toolkit)

### Full Description and Purpose
Sony MCT is an open-source library for neural network quantization, primarily designed to target edge devices like Sony's IMX500 intelligent vision sensors.

### Key Features and Algorithms
- **Hardware-Aware Quantization:** Uses a Target Platform Model (TPM) to define the specific constraints of the target hardware.
- **Gradient-based PTQ:** Similar to AdaRound, optimizes rounding parameters.

### Detailed Comparison with AIMET
Sony MCT's architecture (using a TPM) is very modern and similar to AIMET's philosophy of hardware-aware optimization. It is highly specialized for ultra-low-power vision sensors, whereas AIMET targets powerful mobile SoCs.

---

## 20. SpQR (Sparse-Quantized Representation for LLMs)

### Full Description and Purpose
SpQR is an advanced research algorithm/library for LLM compression. It solves the outlier problem by keeping the ~1% outlier weights in dense FP16, and compressing the remaining 99% of weights to 3-bit or 4-bit.

### Key Features and Algorithms
- **Bilevel Quantization:** Uses a highly compressed format for non-outliers.
- **Exact Accuracy:** Achieves near-zero perplexity degradation even at 3-bit.

### Detailed Comparison with AIMET
SpQR is an algorithmic implementation for LLMs. It requires custom GPU kernels to run efficiently (due to the sparse FP16 matrix). AIMET currently focuses on dense quantization (W4A8 or W8A8) because mobile DSPs handle dense matrix multiplications much more efficiently than sparse + dense mixed formats.

---

## Detailed Comparison Table: All 20 Libraries across 15 Dimensions
| Library | Target Hardware | Precision | Algorithms | Ease of Use | Ecosystem | Open Source | Community Support | Auto-Tuning | LLM Focus | Frameworks | Custom Backends | Memory Optimization | Speedup | License |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AIMET | Qualcomm DSP | INT4, INT8, INT16 | AdaRound, CLE, QAT | Medium | Qualcomm | Yes | Medium | No | Emerging | PyTorch, TF, ONNX | Yes | High | High | BSD |
| PyTorch Native | CPU, Generic | INT8 | PTQ, QAT | High | PyTorch | Yes | High | No | Medium | PyTorch | No | Medium | Medium | BSD |
| TF-MOT | EdgeTPU, CPU | INT8, FP16 | PTQ, QAT | High | TensorFlow | Yes | High | No | Low | TensorFlow | No | High | Medium | Apache 2.0 |
| INC | Intel CPUs/GPUs | INT8, FP8 | PTQ, QAT, AWQ | Medium | Intel | Yes | Medium | Yes | High | PyTorch, TF, ONNX | Yes | High | High | Apache 2.0 |
| TensorRT | NVIDIA GPUs | INT8, FP8, INT4 | PTQ, AWQ | Low | NVIDIA | No | Medium | Yes | High | PyTorch, ONNX | No | High | Very High | Proprietary |
| Apache TVM | Agnostic | INT8 | Auto-scheduling | Low | Agnostic | Yes | Medium | Yes | Medium | Any | Yes | Medium | Medium | Apache 2.0 |
| ONNX Runtime | Agnostic | INT8 | QDQ, PTQ | High | ONNX | Yes | High | No | Medium | ONNX | Yes | Medium | Medium | MIT |
| HF Optimum | Agnostic | INT8, INT4 | AWQ, GPTQ | High | Hugging Face| Yes | High | No | High | PyTorch | Yes | High | Medium | Apache 2.0 |
| Brevitas | FPGAs | 1-8 bit | QAT | Low | AMD/Xilinx | Yes | Low | No | Low | PyTorch | No | High | Medium | BSD |
| BitsAndBytes | NVIDIA GPUs | 8-bit, 4-bit | NF4, LLM.int8 | High | PyTorch | Yes | High | No | Very High| PyTorch | No | Very High | Medium | MIT |
| AutoGPTQ | GPUs | 2-8 bit | GPTQ | High | PyTorch | Yes | Medium | No | Very High| PyTorch | No | High | Medium | MIT |
| AWQ | GPUs | 4-bit | Activation-aware | High | PyTorch | Yes | Medium | No | Very High| PyTorch | No | High | Medium | MIT |
| SmoothQuant | GPUs, CPUs | 8-bit | W8A8 | Medium | Agnostic | Yes | Low | No | High | PyTorch | No | Medium | Medium | MIT |
| Intel NNCF | Intel Hardware| INT8 | PTQ, QAT | Medium | OpenVINO | Yes | Medium | Yes | Medium | PyTorch, TF | No | Medium | Medium | Apache 2.0 |
| EdgeTPU Compiler| Coral Devices | INT8 | TFLite Mapping | Medium | Google | No | Low | No | Low | TensorFlow | No | High | High | Proprietary |
| SNPE/QAIRT | Qualcomm DSP | INT8 | Compilation | Low | Qualcomm | No | Medium | No | Medium | ONNX, TFLite | No | High | Very High | Proprietary |
| Arm NN | ARM CPUs/NPUs | INT8 | NEON | Medium | ARM | Yes | Medium | No | Low | TFLite, ONNX | No | Medium | Medium | MIT |
| MediaTek NeuroPilot| MediaTek APU | INT8 | PTQ, QAT | Low | MediaTek | No | Low | No | Low | PyTorch, TF | No | High | High | Proprietary |
| Core ML Tools | Apple ANE | INT8, INT4 | PTQ, Palettization| High | Apple | Yes | Medium | No | Medium | PyTorch, TF | No | Medium | High | BSD |
| Sony MCT | Sony Sensors | INT8 | TPM-based PTQ | Medium | Sony | Yes | Low | Yes | Low | PyTorch, TF | No | High | Medium | Apache 2.0 |
| SpQR | GPUs | 3-4 bit | Bilevel | Low | PyTorch | Yes | Low | No | Very High| PyTorch | No | Very High | Medium | MIT |

## When to use AIMET specifically (Decision Tree with 20 branches)
1. Are you targeting a Qualcomm Snapdragon SoC? (Yes -> 2, No -> 10)
2. Is the model memory bound (like LLMs) or compute bound (CNNs)? (Compute -> 3, Memory -> 4)
3. Are you using standard Convolutions/Linear layers? (Yes -> Use AIMET AdaRound + CLE, No -> 5)
4. Use AIMET weight-only quantization (W4A8).
5. Does the model have custom operators? (Yes -> Write custom AIMET quantizer, No -> 6)
6. Do you need INT8 accuracy? (Yes -> 7, No -> 8)
7. Run AIMET Cross-Layer Equalization (CLE). Did accuracy recover? (Yes -> Deploy, No -> 9)
8. Evaluate FP16 inference on Hexagon HTP instead of INT8.
9. Use AIMET Quantization-Aware Training (QAT).
10. Are you targeting Apple devices? (Yes -> Use Core ML Tools, No -> 11)
11. Are you targeting NVIDIA GPUs? (Yes -> Use TensorRT/ModelOpt, No -> 12)
12. Are you targeting Intel CPUs? (Yes -> Use INC, No -> 13)
13. Are you targeting generic ARM CPUs? (Yes -> Use Arm NN, No -> 14)
14. Are you targeting FPGAs? (Yes -> Use Brevitas, No -> 15)
15. Is it a Large Language Model? (Yes -> 16, No -> 18)
16. Do you need easy deployment via HF? (Yes -> Use BitsAndBytes/Optimum, No -> 17)
17. Use AutoGPTQ, AWQ, SpQR, or QuIP#.
18. Are you running on the Web (WASM)? (Yes -> Use Apache TVM, No -> 19)
19. Are you targeting Coral Edge TPU? (Yes -> Use TF-MOT + EdgeTPU Compiler, No -> 20)
20. Fallback: Use ONNX Runtime or TFLite generic quantization.

## Combining multiple tools: AIMET + TensorRT workflow
In some complex edge-to-cloud environments, a model might be prepared using AIMET to ensure structural quantization robustness, exported to ONNX, and then optimized by TensorRT for cloud deployment. AIMET’s CLE modifies the weights directly to make them more quantization-friendly. These updated weights can then be exported in FP32 and fed into TensorRT’s PTQ calibrator, which often achieves higher accuracy than running TensorRT on the original un-equalized weights.

## AIMET + TFLite + Core ML chain for multi-platform deployment
To deploy a single architecture across Android, iOS, and Web, developers create a unified pipeline:
1. Train in PyTorch.
2. Run AIMET CLE and AdaRound.
3. Export the "quantization-friendly" FP32 model.
4. Convert to TFLite (for Android/NNAPI) using standard TF tools.
5. Convert to Core ML (for iOS/ANE) using coremltools.
Since AIMET has already smoothed the weight distributions, the basic PTQ in TFLite and Core ML succeeds without needing framework-specific QAT.

## Framework migration: TF model -> PyTorch -> AIMET pipeline
Many legacy models in TensorFlow 1.x/2.x must be migrated to PyTorch to utilize AIMET's most advanced features. The pipeline involves exporting TF to ONNX, then using `onnx2torch` to rebuild the PyTorch graph. Once in PyTorch, AIMET’s `prepare_model` function can fuse batch norms and insert fake quantization nodes seamlessly.

## Quantization format interoperability: ONNX QDQ, AIMET encodings, TFLite flatbuffers
Interoperability is a major challenge. ONNX uses Quantize-Dequantize (QDQ) nodes. AIMET uses an internal encodings format (JSON based) mapping layers to scale/offset. TFLite bakes these into flatbuffer tensors. A critical deployment step involves translating AIMET's JSON encodings into ONNX QDQ nodes, which is supported by AIMET's `export_to_onnx` utilities, allowing cross-compatibility with ONNX Runtime.

## Vendor lock-in analysis: which tools are portable
- **Portable:** ONNX Runtime, Apache TVM, PyTorch Native, HF Optimum.
- **Locked-in:** AIMET (Qualcomm), TensorRT (NVIDIA), Core ML (Apple), INC (Intel), NeuroPilot (MediaTek).
Using AIMET strongly ties your best performance to Qualcomm hardware, as its algorithms explicitly model Hexagon DSP behaviors.

## Community and support comparison
- PyTorch Native, HF Optimum, and BitsAndBytes enjoy massive open-source communities.
- AIMET has a strong open-source presence on GitHub but relies heavily on official Qualcomm support for complex DSP issues.
- Proprietary tools like TensorRT and Core ML have excellent official documentation but lack open-source code visibility.

## License compatibility (commercial vs research use)
- AIMET: BSD 3-Clause (Highly permissive for commercial).
- PyTorch/ONNX: BSD/MIT (Permissive).
- TF-MOT / TVM: Apache 2.0 (Permissive, requires patent notices).
- TensorRT / SNPE: Proprietary (Requires vendor agreements for redistribution).

## SpQR detailed analysis (Sparse-Quantized Representation)
SpQR isolates the top 1% of weights (outliers) and stores them in sparse FP16 format, while quantizing the remaining 99% to 3-bit or 4-bit. This dual-representation approach allows LLMs to fit into significantly less VRAM without the perplexity degradation typically seen in pure 4-bit quantization.

## QuIP# (Incoherence-processed quantization for LLMs)
QuIP# applies randomized orthogonal matrices (like the Hadamard transform) to the weights and activations before quantization. This spreads the outlier information across all weights, creating "incoherence." This smoothed distribution allows for highly accurate 2-bit quantization, pushing the limits of LLM compression.

## AQLM (Additive Quantization of Language Models)
AQLM extends product quantization (PQ) by jointly optimizing multiple codebooks. Instead of rounding weights to the nearest scalar, AQLM approximates weight vectors as the sum of vectors from multiple codebooks, achieving extreme compression ratios (e.g., 2-bit) with minimal accuracy loss.

## FLUTE (Flexible lookup table quantization)
FLUTE provides a generalized framework for non-linear quantization. Instead of standard integer steps, it uses a lookup table of values. This allows the quantization bins to precisely match the empirical distribution of the weights, maximizing information retention.

## Emerging 2025 tools: QMoE, ZipLM, LLM-Shearing
- **QMoE:** Specialized tools for quantizing Mixture of Experts (MoE) models, addressing the unique memory access patterns of expert routing.
- **ZipLM:** A framework for structurally pruning and quantizing LLMs simultaneously to achieve optimal speedups on specific hardware constraints.
- **LLM-Shearing:** Advanced structured pruning techniques to downsize massive LLMs (e.g., 70B to 10B) while retaining capabilities, followed by aggressive PTQ.

## Benchmark comparison across all libraries
| Model | Hardware | AIMET (INT8) | TensorRT (INT8) | Core ML (INT8) | TFLite (INT8) |
|---|---|---|---|---|---|
| ResNet-50 | NPU/DSP | 99.8% orig acc | 99.7% orig acc | 99.5% orig acc | 99.2% orig acc |
| MobileNetV2 | NPU/DSP | 99.0% orig acc | 98.5% orig acc | 98.8% orig acc | 98.0% orig acc |
| Llama-2 7B | GPU/DSP | (W4A8) 95% | (AWQ) 98% | N/A | N/A |
*(Note: AIMET consistently achieves the highest retention of original accuracy on mobile architectures due to CLE and AdaRound).*

## Decision Guide: Which to use when?

Choosing the right optimization library depends almost entirely on your **Deployment Target** and **Model Type**.

1. **Targeting Qualcomm Snapdragon Devices (Android/Automotive):**
   - **Use AIMET**. It is the only toolkit that deeply understands the constraints and math of the Hexagon DSP. Combine it with QAIRT for deployment.

2. **Targeting Apple Devices (iOS/macOS):**
   - **Use coremltools**. Native integration ensures operators run on the Apple Neural Engine.

3. **Targeting Server/Cloud NVIDIA GPUs:**
   - **Use NVIDIA ModelOpt + TensorRT**. Essential for leveraging Tensor Cores and FP8 formats.
   - For rapid LLM prototyping on consumer GPUs, use **BitsAndBytes** or **AutoGPTQ/AWQ** via Hugging Face.

4. **Targeting Server/Cloud Intel CPUs:**
   - **Use Intel Neural Compressor (INC)**. It leverages AMX instructions for massive speedups on Xeon processors.

5. **Cross-Platform Mobile/Edge (Unknown Hardware):**
   - **Use ONNX Runtime Quantization** or **TF-MOT (TFLite)**. These provide the best generic fallbacks (running optimized CPU paths via XNNPACK or NNAPI) when a specific NPU is not targeted.

6. **Deploying Large Language Models (LLMs):**
   - For GPUs: **BitsAndBytes (QLoRA)**, **AWQ**, or **AutoGPTQ**.
   - For Edge (Snapdragon): **AIMET** (using W4A8 techniques currently being developed for mobile).

---

## Emerging Trends in Model Optimization

1. **Shift from QAT to Advanced PTQ:** Historically, aggressive quantization (like INT4) required expensive Quantization-Aware Training. Algorithms like AIMET's AdaRound, GPTQ, and AWQ have made Post-Training Quantization so accurate that QAT is becoming a last resort, saving massive compute costs.
2. **Weight-Only Quantization for LLMs:** Because LLM inference is memory-bandwidth bound (reading weights from VRAM), quantizing weights to 4-bit while keeping activations in FP16/INT8 has become the standard approach.
3. **Sub-4-bit and Non-Linear Quantization:** Techniques like NF4 (NormalFloat) and SpQR are exploring 2-bit and 3-bit spaces by using non-linear binning and handling outliers sparsely.
4. **Hardware-Compiler-Optimization Co-design:** Libraries are increasingly using a Target Platform Model (TPM), as seen in AIMET and Sony MCT, where the optimization algorithm queries the compiler to understand exactly how the hardware handles specific math, creating a perfect quantization recipe.
5. **FP8 as the New Standard for Data Centers:** While mobile edge (Qualcomm/Apple) relies on INT8/INT4 integer math, data center GPUs (NVIDIA Hopper) and AI accelerators are moving to FP8 (E4M3/E5M2), which requires no zero-point calibration and handles dynamic ranges better than INT8.

This document serves as a comprehensive landscape overview, illustrating that while optimization algorithms share common mathematical foundations, the implementation is fiercely fragmented by hardware ecosystems. AIMET remains the apex tool for the Qualcomm ecosystem, just as TensorRT, INC, and CoreML rule their respective domains.

## Advanced Library Deep Dives and Comparisons

### SpQR: Sparse-Quantized Representation for LLMs

**Paper**: 'SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression' (2023)

**Key Innovation**:
SpQR identifies a small set of outlier weights (1-5%) and keeps them in FP16, while quantizing the rest to INT3/INT4.

```
SpQR Weight Decomposition:
  W = W_dense (INT3 quantized, 95-99%) 
    + W_sparse (FP16, 1-5% outliers)

Benefits:
  Near-lossless quality even at 3-bit
  Perplexity comparable to FP16 baseline
  Memory: ~4x smaller than FP16
```

### QuIP# (Quantization with Incoherence Processing)

**Core Idea**: Pre-multiply weights and Hessians by random orthogonal matrices to reduce outlier severity, enabling 2-bit quantization.

```python
# QuIP# quantization flow (conceptual)
def quip_quantize(W, H):  # W=weights, H=Hessian
    # Step 1: Incoherence processing
    E, R = random_hadamard_rotation()
    W_rot = R @ W @ R.T  # Rotate weights
    H_rot = R @ H @ R.T  # Rotate Hessian
    
    # Step 2: Lattice quantization (E8 lattice)
    W_q = lattice_quantize(W_rot, codebook='E8')
    
    # Step 3: De-rotate
    return R.T @ W_q @ R
```

**Results** (LLaMA-2 70B):
| Method | Bits/Weight | Perplexity | Size |
|--------|------------|-----------|------|
| FP16   | 16         | 3.32      | 140GB|
| GPTQ   | 4          | 3.84      | 35GB |
| AWQ    | 4          | 3.72      | 35GB |
| QuIP#  | 2          | 4.15      | 17.5GB|

### AQLM (Additive Quantization of Language Models)

Uses **product quantization** (multiple codebooks) for extreme compression:

```
AQLM: Each weight row represented as sum of codebook vectors
  W[i] = Σ_j codebook_j[code_j[i]]

  With 2 codebooks of size 256, 8-bit codes:
  Effective bits = 8 + 8 = 16 bits per row, but rows have many elements
  Actual bits per weight ≈ 2 bits!

Benefits:
  Near 2-bit quantization with good accuracy
  Fast lookup decoding on GPU

Results on LLaMA-2 7B:
  FP16 baseline: 5.47 perplexity
  AQLM 2-bit:    6.94 perplexity
  AQLM 3-bit:    5.71 perplexity
```

### FLUTE (Flexible Lookup Table Engine)

Hardware-efficient lookup table based quantization:

```
FLUTE Concept:
  Store quantized weights as indices into small codebooks
  Codebooks learned via k-means or gradient methods
  
  During inference:
  1. Load INT4 index
  2. Look up FP16 value in codebook
  3. Multiply with FP16 activation
  
  Hardware: Custom CUDA kernel exploiting shared memory
```

### Emerging 2025 Tools

| Tool | Organization | Key Feature |
|------|-------------|-------------|
| QMoE | ETH Zurich | Mixture-of-Experts quantization |
| ZipLM | Stanford | Structured pruning for LLMs |
| LLM-Shearing | Princeton | Structured pruning via shearing |
| SqueezeLLM | UC Berkeley | Sparse+quantized hybrid |
| ZeroQuant-V2 | Microsoft | FP8 quantization for large models |
| GPTQ-for-LLaMA | Community | GPTQ applied to LLaMA family |

## Complete 20-Library Feature Matrix

| Library | PTQ | QAT | Pruning | Distillation | LLM Support | Edge HW | Framework | License |
|---------|-----|-----|---------|-------------|-------------|---------|-----------|--------|
| AIMET | ✅ | ✅ | ✅ | ❌ | Partial | Qualcomm | PyTorch/TF/ONNX | BSD-3 |
| PyTorch Native | ✅ | ✅ | ❌ | ❌ | ❌ | CPU/ARM | PyTorch | BSD |
| TF-MOT | ✅ | ✅ | ✅ | ❌ | ❌ | ARM/EdgeTPU | TensorFlow | Apache |
| Intel INC | ✅ | ✅ | ✅ | ✅ | ✅ | Intel x86 | PyTorch/TF/ONNX | Apache |
| TensorRT | ✅ | ✅ | ❌ | ❌ | ✅ | NVIDIA GPU | PyTorch/ONNX | Proprietary |
| Apache TVM | ✅ | ❌ | ❌ | ❌ | Partial | Universal | Any | Apache |
| ONNX Runtime | ✅ | ❌ | ❌ | ❌ | Partial | Universal | ONNX | MIT |
| Optimum | ✅ | ✅ | ✅ | ✅ | ✅ | Multiple | HF/PyTorch | Apache |
| Brevitas | ❌ | ✅ | ❌ | ❌ | ❌ | FPGA/CPU | PyTorch | BSD |
| BitsAndBytes | Partial | ✅ | ❌ | ❌ | ✅ | NVIDIA GPU | PyTorch | MIT |
| AutoGPTQ | ✅ | ❌ | ❌ | ❌ | ✅ | NVIDIA/AMD | PyTorch | MIT |
| AWQ | ✅ | ❌ | ❌ | ❌ | ✅ | NVIDIA GPU | PyTorch | MIT |
| SmoothQuant | ✅ | ❌ | ❌ | ❌ | ✅ | GPU | PyTorch | Apache |
| OpenVINO NNCF | ✅ | ✅ | ✅ | ✅ | ✅ | Intel | PyTorch/TF | Apache |
| EdgeTPU TFLite | ✅ | ✅ | ✅ | ❌ | ❌ | Google Coral | TensorFlow | Apache |
| Qualcomm SNPE | ✅ | ❌ | ❌ | ❌ | Partial | Qualcomm | Any | Proprietary |
| ARM NN | ✅ | ❌ | ❌ | ❌ | ❌ | ARM | TFLite/ONNX | MIT |
| NeuroPilot | ✅ | ❌ | ❌ | ❌ | ❌ | MediaTek | TFLite | Proprietary |
| Core ML Tools | ✅ | ✅ | ✅ | ❌ | ✅ | Apple ANE | PyTorch/TF | BSD |
| MCT (Sony) | ✅ | ✅ | ✅ | ❌ | Partial | Sony IMX500 | PyTorch/Keras | Apache |

## Decision Guide: Choosing the Right Quantization Tool

```
Decision Tree:

Q1: What is your target hardware?
├── Qualcomm Snapdragon/AI 100 → AIMET + QAIRT
├── NVIDIA GPU → TensorRT + ModelOpt
├── Intel CPU/GPU/Gaudi → OpenVINO NNCF or Intel INC
├── Apple Silicon → Core ML Tools
├── ARM Cortex-M (MCU) → CMSIS-NN + TFLite
├── ARM Cortex-A → TFLite Delegate + ARM NN
├── Google Coral Edge TPU → TFLite with EdgeTPU compiler
├── Sony IMX500 → Model Compression Toolkit (MCT)
└── Multiple targets → ONNX Runtime + target-specific backend

Q2: What type of model?
├── LLM (>1B params)
│   ├── Weight-only INT4 on GPU → GPTQ/AWQ/BitsAndBytes
│   ├── W8A8 on GPU → SmoothQuant + TensorRT-LLM
│   └── On mobile edge → AIMET INT4 + QAIRT
├── Vision CNN
│   ├── For Qualcomm → AIMET PTQ/QAT
│   ├── For Intel → OpenVINO NNCF
│   └── For General → TF-MOT or PyTorch native
└── Transformer (not LLM)
    ├── BERT-class → AIMET or Optimum
    └── ViT → AIMET or TensorRT

Q3: What accuracy requirement?
├── <0.5% drop from FP32 → QAT (any framework)
├── <1% drop → PTQ with AdaRound/GPTQ
└── <2% drop → Basic PTQ sufficient
```

## Format Interoperability Map

```
Model Format Flow:

PyTorch .pth ←→ ONNX .onnx ←→ TFLite .tflite
                    ↓
              AIMET .encodings.json
                    ↓
              QAIRT .dlc (Qualcomm)
              TensorRT .engine (NVIDIA)
              OpenVINO .xml/.bin (Intel)
              Core ML .mlpackage (Apple)

Universal checkpoint: ONNX + QDQ nodes
  Supported by: ONNX Runtime, TensorRT, OpenVINO, AIMET
```
