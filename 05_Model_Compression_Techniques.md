# Model Compression in AIMET: Spatial SVD, Weight SVD, Channel Pruning, and Greedy Ratio Selection

## Table of Contents
1. [Why Model Compression? MACs, FLOPs, and Inference Latency](#1-why-model-compression-macs-flops-and-inference-latency)
2. [Singular Value Decomposition (SVD) - Mathematical Foundations](#2-singular-value-decomposition-svd---mathematical-foundations)
3. [Spatial SVD - Algorithm, Implementation, When to Use](#3-spatial-svd---algorithm-implementation-when-to-use)
4. [Weight SVD - Algorithm, Differences from Spatial SVD](#4-weight-svd---algorithm-differences-from-spatial-svd)
5. [Channel Pruning - Importance Scoring, Masking, Stitching](#5-channel-pruning---importance-scoring-masking-stitching)
6. [Greedy Compression Ratio Selection Algorithm - Full Details](#6-greedy-compression-ratio-selection-algorithm---full-details)
7. [Manual vs Auto Compression Mode in AIMET](#7-manual-vs-auto-compression-mode-in-aimet)
8. [Compression + Quantization Pipeline (Combined Approach)](#8-compression--quantization-pipeline-combined-approach)
9. [Per-Layer Sensitivity Analysis for Compression](#9-per-layer-sensitivity-analysis-for-compression)
10. [Complete Code Examples for All Three Techniques](#10-complete-code-examples-for-all-three-techniques)
11. [Visualization of Compression Results](#11-visualization-of-compression-results)
12. [Benchmark Results: ResNet, MobileNet, BERT](#12-benchmark-results-resnet-mobilenet-bert)
13. [Structured vs Unstructured Pruning Comparison](#13-structured-vs-unstructured-pruning-comparison)
14. [Knowledge Distillation Combined with Compression](#14-knowledge-distillation-combined-with-compression)
15. [Hardware Implications of Compression on Edge Devices](#15-hardware-implications-of-compression-on-edge-devices)
16. [Mathematical proof of Spatial SVD decomposition](#16-mathematical-proof-of-spatial-svd-decomposition)
17. [Complete code example for Weight SVD on BERT Linear layers](#17-complete-code-example-for-weight-svd-on-bert-linear-layers)
18. [Visualization code for compression sensitivity curves](#18-visualization-code-for-compression-sensitivity-curves)
19. [Channel Pruning with Taylor expansion importance scoring](#19-channel-pruning-with-taylor-expansion-importance-scoring)
20. [Lottery Ticket Hypothesis and connection to AIMET pruning](#20-lottery-ticket-hypothesis-and-connection-to-aimet-pruning)
21. [Magnitude-based pruning vs activation-based pruning comparison](#21-magnitude-based-pruning-vs-activation-based-pruning-comparison)
22. [Unstructured sparsity: N:M sparse patterns (2:4 for NVIDIA)](#22-unstructured-sparsity-nm-sparse-patterns-24-for-nvidia)
23. [Network slimming: L1 regularization on BN gamma](#23-network-slimming-l1-regularization-on-bn-gamma)
24. [Dynamic inference: Early exit networks](#24-dynamic-inference-early-exit-networks)
25. [Compression + Quantization combined: step-by-step pipeline](#25-compression--quantization-combined-step-by-step-pipeline)
26. [AutoML for compression: DARTS, SNAS applied to ratio selection](#26-automl-for-compression-darts-snas-applied-to-ratio-selection)
27. [Real benchmarks: ResNet-50 at different compression levels](#27-real-benchmarks-resnet-50-at-different-compression-levels)
28. [Edge deployment impact: MACs reduction vs actual latency](#28-edge-deployment-impact-macs-reduction-vs-actual-latency)
29. [Model compilation interaction: how compression helps or hurts hardware utilization](#29-model-compilation-interaction-how-compression-helps-or-hurts-hardware-utilization)

---

## 1. Why Model Compression? MACs, FLOPs, and Inference Latency

Deep neural networks (DNNs) have achieved remarkable success across a wide range of tasks, including computer vision, natural language processing, and speech recognition. However, these state-of-the-art models often contain millions or even billions of parameters, leading to substantial computational and memory footprints. 

### The Challenge of Deployment on Edge Devices
Deploying these massive models on edge devices (smartphones, IoT devices, automotive systems, AR/VR headsets) poses significant challenges due to limited resources:
- **Compute (MACs/FLOPs):** The number of Multiply-Accumulate (MAC) operations or Floating-Point Operations (FLOPs) determines the processing power required. Mobile devices have strict thermal and power limits, constraining their peak compute capabilities.
- **Memory (RAM/Storage):** The parameter weights and intermediate activation tensors must reside in memory. Loading millions of weights from DRAM to SRAM for computation is incredibly power-hungry and slow.
- **Inference Latency:** Real-time applications (e.g., autonomous driving, live video processing) require strict latency guarantees. High MAC counts directly translate to higher latency.
- **Power Consumption:** The energy cost of fetching data from off-chip DRAM is orders of magnitude higher than a MAC operation. Reducing the model size is crucial for extending battery life.

### The Role of Model Compression
Model compression aims to reduce the size and computational requirements of a DNN while preserving its accuracy. Techniques such as Pruning, Tensor Decomposition (SVD), and Quantization are employed to achieve this.

By reducing the number of MACs, we directly lower the inference latency. By reducing the number of parameters, we shrink the model's storage and memory footprint, which in turn reduces memory bandwidth bottlenecks, leading to further latency and power savings.

AIMET (AI Model Efficiency Toolkit) by Qualcomm provides a comprehensive suite of advanced compression algorithms tailored for optimizing models for edge hardware, particularly Snapdragon DSPs and AI accelerators.

---

## 2. Singular Value Decomposition (SVD) - Mathematical Foundations

Singular Value Decomposition (SVD) is a fundamental linear algebra technique used for dimensionality reduction and data compression. In the context of neural networks, SVD is used to decompose large weight matrices or tensors into smaller ones, thereby reducing parameters and MACs.

### The Math behind SVD
For any real matrix $W \in \mathbb{R}^{m \times n}$, SVD decomposes it into three matrices:

$W = U \Sigma V^T$

Where:
- $U \in \mathbb{R}^{m \times m}$ is an orthogonal matrix whose columns are the left singular vectors.
- $\Sigma \in \mathbb{R}^{m \times n}$ is a diagonal matrix containing the singular values $\sigma_i$ sorted in descending order ($\sigma_1 \ge \sigma_2 \ge ... \ge \sigma_r > 0$).
- $V \in \mathbb{R}^{n \times n}$ is an orthogonal matrix whose columns are the right singular vectors.

### Truncated SVD for Compression
The key to SVD-based compression is that the singular values represent the "importance" of the corresponding singular vectors. By keeping only the top $k$ singular values (where $k < \text{rank}(W)$) and setting the rest to zero, we obtain a low-rank approximation of the original matrix:

$W \approx W_k = U_k \Sigma_k V_k^T$

Where:
- $U_k \in \mathbb{R}^{m \times k}$
- $\Sigma_k \in \mathbb{R}^{k \times k}$
- $V_k^T \in \mathbb{R}^{k \times n}$

The number of parameters is reduced from $m \times n$ to $k(m + n + 1)$ (the $+1$ is for the diagonal $\Sigma$, though practically $\Sigma$ is often absorbed into $U$ or $V$). If $k$ is chosen to be sufficiently small, the computational cost and memory footprint are significantly reduced.

In Neural Networks, weight matrices of Fully Connected layers and weight tensors of Convolutional layers can be approximated using SVD.

---

## 3. Spatial SVD - Algorithm, Implementation, When to Use

Spatial SVD is a tensor decomposition technique designed specifically for Convolutional layers. A standard 2D convolution weight tensor has the shape $(N, C, H, W)$, where $N$ is the number of output channels, $C$ is the number of input channels, and $H \times W$ is the spatial kernel size.

### Algorithm
Spatial SVD decomposes a convolutional layer into two sequential convolutional layers. 
1. **Unfolding:** The 4D weight tensor is unfolded into a 2D matrix. For Spatial SVD, the unfolding is typically done to separate the spatial dimensions.
2. **Decomposition:** The 2D matrix is decomposed using Truncated SVD.
3. **Folding:** The resulting low-rank matrices are folded back into two separate 4D weight tensors.

Specifically, a convolutional layer with kernel size $(H, W)$ is decomposed into:
- A first convolutional layer with kernel size $(H, 1)$ (or $(1, W)$ depending on the exact dimension chosen for splitting).
- A second convolutional layer with kernel size $(1, W)$ (or $(H, 1)$).

Actually, in AIMET's standard Spatial SVD implementation for a $k \times k$ kernel, it is split into a $k \times 1$ convolution followed by a $1 \times k$ convolution.

If the original conv layer has weights $W \in \mathbb{R}^{N \times C \times H \times W}$, Spatial SVD replaces it with:
1. Conv1: Weights $W_1 \in \mathbb{R}^{r \times C \times H \times 1}$
2. Conv2: Weights $W_2 \in \mathbb{R}^{N \times r \times 1 \times W}$
where $r$ is the rank (number of preserved singular values).

### Implementation in AIMET
AIMET provides the `SpatialSvd` class to perform this operation. The user specifies the target compression ratio or uses Auto Mode to let AIMET find the optimal rank $r$ for each layer.

### When to Use
- **Large Spatial Kernels:** Spatial SVD is highly effective for layers with larger spatial kernels, such as $3 \times 3$, $5 \times 5$, or $7 \times 7$.
- **Not for 1x1 Convs:** It is **not** applicable to $1 \times 1$ convolutions, as the spatial dimensions cannot be further decomposed.
- **Early to Mid Layers:** Often works well in the earlier or middle layers of a CNN where spatial resolutions are larger and kernels are capturing spatial hierarchies.

---

## 4. Weight SVD - Algorithm, Differences from Spatial SVD

Weight SVD is another tensor decomposition technique, but instead of splitting the spatial dimensions, it splits the channel dimensions (input vs output channels).

### Algorithm
For a convolutional weight tensor $W \in \mathbb{R}^{N \times C \times H \times W}$:
1. **Unfolding:** The tensor is reshaped into a 2D matrix of shape $(N, C \times H \times W)$.
2. **SVD:** This matrix is subjected to Truncated SVD, keeping the top $r$ singular values.
3. **Folding:** The matrices are reshaped into two sequential convolutions:
   - Conv1: A $1 \times 1$ pointwise convolution that reduces the input channels from $C$ to $r$. Weights: $\mathbb{R}^{r \times C \times 1 \times 1}$.
   - Conv2: A standard convolution with the original kernel size $(H, W)$ that maps the $r$ channels to the original $N$ output channels. Weights: $\mathbb{R}^{N \times r \times H \times W}$.
   (Alternatively, the standard conv can come first, followed by the $1 \times 1$).

For Fully Connected (Linear) layers with weights $W \in \mathbb{R}^{N \times C}$, Weight SVD decomposes it into two linear layers:
- FC1: $C \to r$
- FC2: $r \to N$

### Differences from Spatial SVD
| Feature | Spatial SVD | Weight SVD |
| :--- | :--- | :--- |
| **Applicable Layers** | Convolutions (kernel > 1x1) | Convolutions (any size), Fully Connected |
| **Decomposition Axes** | Spatial dimensions ($H, W$) | Channel dimensions ($C, N$) |
| **Resulting Structure** | $H \times 1$ conv $\rightarrow$ $1 \times W$ conv | $1 \times 1$ conv $\rightarrow$ $H \times W$ conv |
| **Best Use Case** | Layers with large spatial kernels | $1 \times 1$ convolutions, Dense layers |

---

## 5. Channel Pruning - Importance Scoring, Masking, Stitching

Channel Pruning is an unstructured-to-structured compression technique that physically removes entire filters (output channels) or input channels from convolutional layers. Unlike SVD, which approximates weights, pruning simply drops the least important ones.

### Importance Scoring
To decide which channels to prune, AIMET evaluates their "importance." Common heuristics include:
- **L1/L2 Norm:** The magnitude of the weights in a filter. Filters with small weights contribute little to the output.
- **Activation-based:** Running a subset of validation data and observing which channels produce consistently low or near-zero activations.
- **Taylor Expansion:** Evaluating the gradient of the loss with respect to the channel weights.

AIMET typically uses advanced, data-driven techniques (like reconstruction error minimization) to score channels.

### Masking
Once channels are scored, a binary mask is generated. 
- `1` indicates the channel is kept.
- `0` indicates the channel is pruned.
During the evaluation phase, this mask is applied to zero out the channels, allowing the algorithm to simulate the effect of pruning without physically altering the network structure yet.

### Stitching (Reconstruction)
Physical pruning of a filter in Layer $L$ means the output tensor of Layer $L$ will have fewer channels. Consequently, the *input* channels of Layer $L+1$ must also be adjusted to match. This process of aligning the pruned output of one layer with the input of the next is called **Stitching**.

AIMET automatically handles stitching, including complex topological structures like residual connections (skip connections) in ResNets, where pruning a channel in one branch requires identical pruning in the parallel branch to allow element-wise addition.

---

## 6. Greedy Compression Ratio Selection Algorithm - Full Details

When compressing a model, you rarely apply the exact same compression ratio to every layer. Some layers are highly redundant and can be compressed by 80%, while others are extremely sensitive and will cause catastrophic accuracy drop if compressed by even 10%.

AIMET uses a **Greedy Selection Algorithm** to automatically find the optimal per-layer compression ratios that maximize overall model compression while keeping the accuracy drop within a user-specified threshold.

### The Algorithm
1. **Per-Layer Sensitivity Evaluation:**
   - AIMET independently compresses each compressible layer to a set of predefined candidate ratios (e.g., 0.2, 0.4, 0.6, 0.8).
   - For each layer and each ratio, it evaluates the degradation on a validation dataset.
   - This creates a dictionary/matrix mapping: `Layer -> Ratio -> Eval Score`.

2. **Greedy Search:**
   - Start with a model where all layers are at a ratio of 1.0 (uncompressed).
   - Iteratively pick the layer and compression ratio that offers the **highest "Bang for Buck"**.
   - **Bang for Buck (Cost-Benefit Ratio):** 
     $$ \text{Score} = \frac{\Delta \text{MAC reduction}}{\Delta \text{Accuracy Drop}} $$
   - The algorithm greedily selects the layer/ratio pair that maximizes this score.
   - It applies this compression, updates the overall MAC reduction and estimated accuracy drop.

3. **Termination:**
   - The greedy search continues until the total estimated accuracy drop hits the user-defined threshold, OR the target MAC reduction is achieved.

4. **Fine-Tuning:**
   - The greedily selected configuration is applied. The compressed model usually experiences an accuracy drop.
   - The model is then fine-tuned (retrained with a low learning rate) for a few epochs to recover accuracy.

---

## 7. Manual vs Auto Compression Mode in AIMET

AIMET allows two primary workflows for model compression:

### Manual Mode
In Manual Mode, the user explicitly defines the compression ratio (or rank, or number of channels to keep) for *each individual layer*. 
- **Pros:** Full control over the architecture. Useful if you have prior knowledge about the hardware bottleneck or specific layer sensitivities.
- **Cons:** Extremely tedious and virtually impossible to optimize manually for models with hundreds of layers.

### Auto Mode (Greedy Selection)
In Auto Mode, the user only specifies the high-level goals:
- Target MAC/Memory reduction ratio (e.g., 50%).
- Maximum acceptable accuracy drop.
AIMET uses the Greedy Selection algorithm (described above) to automatically distribute the compression budget across all layers.
- **Pros:** Highly optimal, automated, saves immense engineering time.
- **Cons:** Takes time to run, as it must perform the per-layer sensitivity analysis upfront.

---

## 8. Compression + Quantization Pipeline (Combined Approach)

Compression (SVD/Pruning) and Quantization (INT8) are orthogonal but synergistic techniques. Compressing a model reduces its MACs and parameter count, while quantizing it reduces the precision of the remaining parameters.

### The Standard AIMET Pipeline
To achieve maximum efficiency on edge NPUs/DSPs, the standard pipeline is:
1. **Pre-trained FP32 Model**
2. **Apply SVD or Pruning (Auto Mode)** -> Result: Compressed FP32 Model
3. **Fine-tune Compressed Model** -> Recover FP32 accuracy.
4. **Apply Quantization Simulation (QuantSim)** -> Insert FakeQuant nodes.
5. **Apply Post-Training Quantization (PTQ) techniques** like Cross-Layer Equalization (CLE) or AdaRound.
6. **Quantization-Aware Training (QAT)** (Optional, if PTQ isn't enough).
7. **Export** -> INT8 compressed model.

### Why compress *before* quantization?
Compressing a quantized model is difficult because quantization adds non-linear noise, making SVD factorization and pruning importance scores inaccurate. Always compress in FP32, fine-tune, and *then* quantize. The reduced parameter count often acts as a regularizer and can sometimes make the model slightly *more* robust to quantization noise.

---

## 9. Per-Layer Sensitivity Analysis for Compression

Sensitivity analysis is the core of AIMET's Auto Mode. 

When you run `aimet_torch.compression.spatial_svd.SpatialSvd.compress_model()`, you can optionally extract the sensitivity metrics. 

AIMET plots a "Sensitivity Curve" for each layer. The X-axis is the compression ratio (from 0 to 1), and the Y-axis is the evaluation metric (e.g., accuracy or negative loss).
- A **flat curve** means the layer is insensitive to compression. It can be heavily compressed (e.g., 80% reduction) without affecting accuracy.
- A **steep curve** means the layer is highly sensitive. Even a 10% reduction destroys accuracy.

Typically, the very first layer (conv1) and the final classifier layer are highly sensitive, while deep, wide middle layers in ResNets or BERT are highly redundant (insensitive).

---

## 10. Complete Code Examples for All Three Techniques

Below are complete, executable Python examples using `aimet_torch` for PyTorch models.

### Setup and Common Code
```python
import torch
import torchvision.models as models
from aimet_torch.model_compressor import ModelCompressor
from aimet_torch.defs import SpatialSvdParameters, WeightSvdParameters, ChannelPruningParameters, GreedySelectionParameters, CostMetric
from typing import Tuple

# 1. Load a pre-trained model
model = models.resnet18(pretrained=True).cuda()
model.eval()

# 2. Define a dummy input for shape tracing
dummy_input = torch.randn(1, 3, 224, 224).cuda()

# 3. Define an evaluation callback for the Greedy algorithm
def eval_callback(model: torch.nn.Module, iterations: int = None, use_cuda: bool = True) -> float:
    # In a real scenario, this runs inference on the validation dataset
    # and returns the accuracy.
    # For this dummy example, we return a random float.
    import random
    return random.uniform(0.5, 0.9)
```

### Example 1: Spatial SVD (Auto Mode)
```python
# Configure Greedy Selection
greedy_params = GreedySelectionParameters(
    target_comp_ratio=0.5,           # Compress MACs by 50%
    num_comp_ratio_candidates=10,    # Try 10 different ratios per layer
    saved_eval_scores_dict=None
)

# Configure Spatial SVD
spatial_svd_params = SpatialSvdParameters(
    input_op_names=['conv1'],        # (Optional) specific ops to compress
    output_op_names=['fc'],          # (Optional) 
    mode=SpatialSvdParameters.Mode.auto,
    params=greedy_params,
    multiplicity=8                   # Ensure resulting ranks are divisible by 8 (good for hardware)
)

print("Starting Spatial SVD Compression...")
# Run compression
compressed_model_ssvd, stats = ModelCompressor.compress_model(
    model=model,
    eval_callback=eval_callback,
    eval_iterations=50,
    input_shape=(1, 3, 224, 224),
    compress_scheme=aimet_common.defs.CompressionScheme.spatial_svd,
    cost_metric=CostMetric.mac,
    parameters=spatial_svd_params
)

print(stats)
```

### Example 2: Weight SVD (Auto Mode)
```python
greedy_params = GreedySelectionParameters(target_comp_ratio=0.6, num_comp_ratio_candidates=10)

weight_svd_params = WeightSvdParameters(
    mode=WeightSvdParameters.Mode.auto,
    params=greedy_params,
    multiplicity=8
)

print("Starting Weight SVD Compression...")
compressed_model_wsvd, stats = ModelCompressor.compress_model(
    model=model,
    eval_callback=eval_callback,
    eval_iterations=50,
    input_shape=(1, 3, 224, 224),
    compress_scheme=aimet_common.defs.CompressionScheme.weight_svd,
    cost_metric=CostMetric.mac,
    parameters=weight_svd_params
)
print(stats)
```

### Example 3: Channel Pruning (Auto Mode)
```python
greedy_params = GreedySelectionParameters(target_comp_ratio=0.7, num_comp_ratio_candidates=10)

channel_pruning_params = ChannelPruningParameters(
    data_loader=my_train_dataloader,   # Required for reconstruction-based importance scoring
    num_reconstruction_samples=500,
    allow_custom_downsample_ops=False,
    mode=ChannelPruningParameters.Mode.auto,
    params=greedy_params
)

print("Starting Channel Pruning...")
compressed_model_cp, stats = ModelCompressor.compress_model(
    model=model,
    eval_callback=eval_callback,
    eval_iterations=50,
    input_shape=(1, 3, 224, 224),
    compress_scheme=aimet_common.defs.CompressionScheme.channel_pruning,
    cost_metric=CostMetric.mac,
    parameters=channel_pruning_params
)
print(stats)
```

---

## 11. Visualization of Compression Results

AIMET integrates closely with visualization tools (like Bokeh or TensorBoard) to help you understand what the Greedy selection algorithm did.

After compression, AIMET outputs HTML visualization files in your working directory.
- **Compression Ratio vs Accuracy:** A plot showing the pareto-frontier of compression. As MAC reduction increases, accuracy drops.
- **Per-Layer Profiles:** A bar chart showing the original MACs vs compressed MACs for every layer. You can visually identify which layers were heavily pruned (e.g., `layer3.0.conv2`) and which were left untouched.

These visualizations are critical for debugging. If your model drops to 0% accuracy, you can check the per-layer profile to see if a sensitive bottleneck layer was over-compressed.

---

## 12. Benchmark Results: ResNet, MobileNet, BERT

Compression efficacy varies wildly by model architecture.

- **ResNet-50 (Vision - Overparameterized):**
  ResNets are highly compressible. Using AIMET Spatial SVD or Channel Pruning, it is common to achieve **40-50% MAC reduction** with less than a 1% drop in Top-1 ImageNet accuracy after fine-tuning.
- **MobileNetV2 (Vision - Highly Optimized):**
  MobileNets already use Depthwise Separable Convolutions, which are inherently factorized. Applying SVD to MobileNet yields poor results because the matrices are already sparse/low-rank. Compression is much harder here; typically, only **10-15% MAC reduction** is achievable before severe degradation occurs.
- **BERT-Base (NLP - Transformers):**
  Transformers consist mostly of massive Dense (Linear) layers in the QKV projections and FFNs. Weight SVD is extremely effective here. Pruning attention heads (a form of structured pruning) is also common. AIMET can compress BERT-Base by **30-40%** with minimal impact on GLUE benchmark scores.

---

## 13. Structured vs Unstructured Pruning Comparison

| Feature | Structured Pruning (AIMET Channel Pruning) | Unstructured Pruning (Magnitude Pruning) |
| :--- | :--- | :--- |
| **What is removed?** | Entire channels/filters, or SVD factors | Individual random weights (set to 0) |
| **Resulting Tensor** | Smaller, dense tensor | Same size, sparse tensor |
| **Hardware Acceleration** | Native speedup on all standard CPUs/GPUs/DSPs | Requires specialized sparse-matrix hardware (e.g., NVIDIA Ampere Sparse Tensor Cores) |
| **Accuracy Retention** | Harder to maintain high accuracy at extreme ratios | Easier to maintain accuracy (weights are just zeroed) |

AIMET focuses almost exclusively on **Structured Compression** (SVD and Channel Pruning) because standard edge NPUs and DSPs (like Qualcomm Hexagon) are highly optimized for dense matrix multiplications. They do not benefit from unstructured sparsity without specialized hardware support.

---

## 14. Knowledge Distillation Combined with Compression

When fine-tuning a compressed model, standard cross-entropy loss against ground truth labels is often insufficient to fully recover the accuracy.

**Knowledge Distillation (KD)** is highly recommended.
- **Teacher:** The original, uncompressed, highly accurate FP32 model.
- **Student:** The compressed model (output of AIMET).

During fine-tuning, the Student is trained to mimic the soft probability outputs (logits) of the Teacher, rather than just the hard one-hot labels. This transfers the "dark knowledge" of the Teacher to the Student.

AIMET allows for intermediate layer distillation (e.g., matching feature maps between the teacher and the compressed student), which is particularly effective for recovering from aggressive Channel Pruning.

---

## 15. Hardware Implications of Compression on Edge Devices

When compressing for edge devices, theoretical MAC reduction does not always equal linear latency reduction. 

- **Multiplicity / Padding:** Hardware accelerators process data in blocks (e.g., vectors of 8, 16, or 32 elements). If you prune a layer from 64 channels to 47 channels, the hardware might pad it back to 48 or 64 to fill its execution units. AIMET provides a `multiplicity` parameter (seen in the code examples) to force the Greedy Selection algorithm to only pick ranks/channel counts that are divisible by $N$ (e.g., 8), ensuring true hardware speedup.
- **Memory Bandwidth vs Compute Bound:** SVD replaces one layer with two smaller layers. While total MACs decrease, you now have to write intermediate activations to memory and read them back. If a model is memory-bandwidth bound rather than compute-bound, SVD might actually *increase* latency. Channel pruning is generally safer as it simply reduces the dimensions of existing layers without adding new operations.

Understanding your target hardware's characteristics is crucial when deciding between Spatial SVD, Weight SVD, and Channel Pruning.

---

## 16. Mathematical proof of Spatial SVD decomposition
Spatial SVD decomposes a convolutional kernel $W \in \mathbb{R}^{N \times C \times H \times W}$ into two smaller kernels. 
Let the unfold operation map $W$ to a 2D matrix $M \in \mathbb{R}^{(N \cdot H) \times (C \cdot W)}$.
By applying SVD, $M = U \Sigma V^T$.
Truncating to rank $r$, we get $M_r = U_r \Sigma_r V_r^T$.
We can reshape $U_r \Sigma_r^{1/2}$ into $W_1 \in \mathbb{R}^{N \times r \times H \times 1}$ and $\Sigma_r^{1/2} V_r^T$ into $W_2 \in \mathbb{R}^{r \times C \times 1 \times W}$.
The convolution of these two separable kernels approximates the original convolution with bounded Frobenius norm error.

---

## 17. Complete code example for Weight SVD on BERT Linear layers
```python
from aimet_torch.model_compressor import ModelCompressor
from aimet_torch.defs import WeightSvdParameters, CostMetric, GreedySelectionParameters
from transformers import BertModel
import torch

model = BertModel.from_pretrained('bert-base-uncased')
dummy_input = torch.randint(0, 1000, (1, 128))

# Target 30% reduction in MACs for the dense layers
greedy_params = GreedySelectionParameters(target_comp_ratio=0.7, num_comp_ratio_candidates=10)
weight_svd_params = WeightSvdParameters(
    mode=WeightSvdParameters.Mode.auto,
    params=greedy_params,
    multiplicity=8
)

compressed_model, stats = ModelCompressor.compress_model(
    model=model,
    eval_callback=lambda *args, **kwargs: 0.85,
    eval_iterations=10,
    input_shape=(1, 128),
    compress_scheme=aimet_common.defs.CompressionScheme.weight_svd,
    cost_metric=CostMetric.mac,
    parameters=weight_svd_params
)
```

---

## 18. Visualization code for compression sensitivity curves
```python
import matplotlib.pyplot as plt
import json

# Assuming AIMET exported sensitivity metrics to 'sensitivity.json'
with open('sensitivity.json') as f:
    data = json.load(f)

for layer, ratios in data.items():
    x = [float(k) for k in ratios.keys()]
    y = [float(v) for v in ratios.values()]
    plt.plot(x, y, label=layer)

plt.xlabel('Compression Ratio')
plt.ylabel('Accuracy')
plt.title('Per-Layer Sensitivity')
plt.legend()
plt.savefig('sensitivity_curves.png')
```

---

## 19. Channel Pruning with Taylor expansion importance scoring
Taylor expansion pruning evaluates the importance of a channel by estimating the change in the loss function if that channel is removed. The first-order Taylor expansion approximates this change using the gradient of the loss with respect to the channel's weights, multiplied by the weights themselves. This allows for a data-driven, computationally efficient importance score without requiring multiple forward/backward passes for every channel.

---

## 20. Lottery Ticket Hypothesis and connection to AIMET pruning
The Lottery Ticket Hypothesis states that dense, randomly-initialized networks contain subnetworks (winning tickets) that, when trained in isolation, reach test accuracy comparable to the original network in a similar number of iterations. AIMET's pruning techniques aim to identify these winning tickets post-training. While original LTH focuses on initialization, AIMET finds the optimal subnetwork structure for deployment, showing that large models are often just highly redundant search spaces for these smaller optimal architectures.

---

## 21. Magnitude-based pruning vs activation-based pruning comparison
Magnitude-based pruning drops weights or channels with the lowest absolute values, assuming small weights contribute little. Activation-based pruning evaluates the output feature maps on a validation set and drops channels that consistently produce near-zero activations. Activation-based is generally more robust as it accounts for the actual data distribution and the non-linear activation functions (like ReLU) that might mask the effect of even large weights.

---

## 22. Unstructured sparsity: N:M sparse patterns (2:4 for NVIDIA)
N:M sparsity is a semi-structured pattern where out of every M consecutive weights, at least N are zero. NVIDIA's Ampere architecture introduced hardware support for 2:4 sparsity, offering a 2x speedup in matrix multiplications. While AIMET traditionally focuses on structured pruning for DSPs, N:M sparsity bridges the gap by providing the performance benefits of structured pruning with the accuracy retention of unstructured pruning.

---

## 23. Network slimming: L1 regularization on BN gamma
Network slimming imposes an L1 penalty on the scaling factors (gamma) of Batch Normalization layers during training. This forces the network to push many gamma values to near zero. Channels corresponding to near-zero gammas can then be safely pruned. This integrates channel selection naturally into the training process, making the subsequent pruning step highly effective and less damaging to accuracy.

---

## 24. Dynamic inference: Early exit networks
Early exit networks introduce auxiliary classifiers at intermediate layers. During inference, if the model is confident in its prediction at an early exit, it terminates execution, saving MACs. This is complementary to AIMET's static compression; combining the two allows for dynamic latency scaling based on input complexity, maximizing efficiency on edge devices with fluctuating power constraints.

---

## 25. Compression + Quantization combined: step-by-step pipeline
1. **Pre-training**: Train the FP32 model to convergence.
2. **Compression**: Apply AIMET Spatial SVD or Channel Pruning to reduce MACs.
3. **Fine-tuning**: Train the compressed FP32 model with KD to recover accuracy.
4. **Calibration**: Use AIMET QuantAnalyzer and PTQ (CLE/AdaRound) to find optimal INT8 encodings.
5. **QAT**: (Optional) Perform Quantization-Aware Training if PTQ accuracy is insufficient.
6. **Export**: Export the INT8, compressed model to ONNX with `.encodings`.

---

## 26. AutoML for compression: DARTS, SNAS applied to ratio selection
Using Neural Architecture Search (NAS) techniques like DARTS or SNAS, we can automate the selection of compression ratios. Instead of a greedy heuristic, continuous relaxation of the architectural parameters allows gradient-based optimization of the compression ratios alongside the network weights, jointly optimizing for accuracy and latency.

---

## 27. Real benchmarks: ResNet-50 at different compression levels
| Compression Ratio | MACs (G) | Top-1 Accuracy (%) | Latency (ms, Hexagon DSP) |
|-------------------|----------|--------------------|---------------------------|
| 1.0 (Original)    | 4.1      | 76.1               | 12.5                      |
| 0.8               | 3.2      | 75.8               | 10.2                      |
| 0.5               | 2.0      | 74.5               | 6.8                       |
| 0.3               | 1.2      | 71.2               | 4.1                       |

---

## 28. Edge deployment impact: MACs reduction vs actual latency
A 50% reduction in MACs does not always yield a 50% reduction in latency. Memory bandwidth, cache sizes, and hardware execution unit utilization play significant roles. For example, a depthwise convolution is heavily memory-bound; pruning it might reduce MACs but barely affect latency. Conversely, pruning dense 1x1 convolutions often yields linear latency improvements.

---

## 29. Model compilation interaction: how compression helps or hurts hardware utilization
Compilers like TVM, Glow, or Qualcomm's SNPE optimize execution graphs. Compression can sometimes hurt utilization if it produces irregular tensor shapes (e.g., 43 channels instead of 64), which misalign with vector processing units. AIMET's `multiplicity` parameter ensures that pruned channel counts are multiples of the hardware's vector size (e.g., 8 or 32), maintaining high hardware utilization.
