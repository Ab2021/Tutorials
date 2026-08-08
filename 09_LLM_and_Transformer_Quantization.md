# LLM and Transformer Quantization: From BERT to LLaMA

The advent of Large Language Models (LLMs) and advanced Vision Transformers has revolutionized artificial intelligence. However, their sheer size — spanning billions to trillions of parameters — creates an unprecedented bottleneck in memory bandwidth, computational power, and energy consumption. Quantization has emerged as the premier technique for compressing these colossal models, reducing their precision from 16-bit or 32-bit floating point down to 8-bit, 4-bit, and sometimes even lower, without disproportionate losses in performance. 

This comprehensive guide deeply explores the intricacies, challenges, algorithms, and practical implementations of Transformer quantization. We journey from encoder-only architectures like BERT up to modern, massive decoder-only models like LLaMA, alongside dives into Vision Transformers (ViTs) and Diffusion Models. We also examine state-of-the-art frameworks like AIMET, GPTQ, AWQ, and SmoothQuant.

---

## 1. Challenges of Quantizing Large Language Models

Quantizing LLMs is significantly more challenging than quantizing Convolutional Neural Networks (CNNs). While CNNs have well-behaved, normally distributed activations, Transformers exhibit extreme pathological behaviors as they scale in size.

### 1.1 Activation Outlier Problem
The most notorious challenge in LLM quantization is the **activation outlier phenomenon**. As model size exceeds approximately 6 billion parameters (seen in models like OPT-13B, LLaMA-7B, etc.), specific feature dimensions in the activation tensors begin to exhibit extreme outlier values. These outliers can be over 100× larger in magnitude than the majority of the activation values. 
- **Systematic Nature:** These outliers are not random noise; they consistently appear in the same channel dimensions across different tokens and inputs. 
- **Impact on Quantization:** If we apply standard per-tensor asymmetric or symmetric quantization to activations, the massive scale factor required to accommodate the outlier pushes the rest of the values into a tiny dynamic range, effectively destroying the resolution of the normal features. If we clip the outliers, we destroy the model's predictive capability because these outlier dimensions encode critical semantic information.

### 1.2 Attention Mechanism Quantization Complexity
The self-attention mechanism consists of several matrix multiplications: \( Q \times K^T \) and \( A \times V \). Quantizing these intermediate tensors is non-trivial:
- **Softmax Precision:** The softmax operation is highly sensitive. Quantizing the inputs to softmax (attention scores) often leads to significant degradation because exponential functions amplify quantization noise. 
- **Dynamic Range Fluctuations:** The dynamic range of the \( Q \times K^T \) output varies dramatically depending on the sequence length and context, making static quantization calibration very difficult.

#### 1.2.1 Attention Score Outlier Problem with Distributions and Visualizations
A deeper look into attention scores reveals profound quantization challenges. After the \( Q \times K^T \) operation, the resulting attention score matrix often contains massive outliers before the softmax application. 
- **The "Massive Activation" Phenomenon:** Certain tokens, particularly punctuation like commas or periods, and the initial `[BOS]` (Beginning of Sequence) token, often accumulate massive attention scores. The model uses these tokens as a "sink" to dump unnecessary attention weight when it doesn't need to attend to anything else.
- **Visualizing the Distribution:** If we plot the distribution of attention scores (pre-softmax), it often looks like a narrow bell curve around 0, with a long, sparse tail stretching out to values like 50 or 100. When quantized to INT8, the resolution between values like 0.1 and 0.5 (which are critical for subtle attention weighting) is entirely lost because the scale factor is dominated by the outlier values of 100.
- **Impact on Softmax:** Because softmax is an exponential function, small errors in the pre-softmax scores are exponentially amplified. An INT8 quantization error of just 0.5 can shift the post-softmax probability mass drastically, causing the model to attend to completely wrong tokens.

### 1.3 Key-Value (KV) Cache Quantization
During autoregressive generation, LLMs cache the Key (K) and Value (V) tensors for past tokens to prevent redundant computation. For long context windows, the KV cache grows linearly and often eclipses the model weights in memory footprint.
- Quantizing the KV cache to 8-bit or 4-bit is essential for serving multiple users concurrently (increasing batch size). However, V cache contains outliers similar to general activations, and K cache quantization can disrupt the precise attention weighting.

#### 1.3.1 KV Cache Quantization Technical Deep-Dive
KV Cache quantization is rapidly becoming the most critical optimization for high-throughput LLM serving.
- **Asymmetric vs. Symmetric for KV:** Key tensors often exhibit zero-mean distributions, making them amenable to symmetric quantization. However, Value tensors, often being the output of a non-linear activation or projection, frequently exhibit non-zero means and asymmetric distributions. Therefore, applying asymmetric quantization to the V cache yields significantly better perplexity preservation.
- **Group-wise Quantization:** Instead of per-tensor or per-channel, modern approaches use group-wise quantization across the token sequence dimension. For instance, computing a dynamic scale and zero-point for every block of 64 or 128 tokens. This captures the local dynamic range changes as the context evolves.
- **Token-wise Dynamic Quantization:** The absolute best accuracy is achieved by computing scale and offset per-token on the fly. As a new token is generated, its K and V vectors are quantized individually before being appended to the cache. When attention is computed, they are dequantized on-the-fly in SRAM.
- **Impact of RoPE:** Rotary Position Embeddings (RoPE) complicate K cache quantization. RoPE rotates the keys based on their position. It is generally advised to quantize the Keys *after* RoPE has been applied, storing the rotated keys in the cache, to avoid compounding quantization noise with rotational math.

### 1.4 Embedding Quantization
The token embeddings form a massive lookup table at the entry of the model. While weights can often be quantized to 4-bit easily, embedding layers can be highly sensitive because errors introduced here propagate and amplify through every subsequent layer.

---

## 2. BERT Quantization with AIMET

BERT (Bidirectional Encoder Representations from Transformers) is an encoder-only model primarily used for natural language understanding (NLU) tasks like classification and question answering. Qualcomm's AI Model Efficiency Toolkit (AIMET) provides robust tools for BERT quantization.

### 2.1 Complete Workflow (PTQ and QAT)

AIMET offers both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT).

**Post-Training Quantization (PTQ):**
1. **Cross-Layer Equalization (CLE):** Not highly applicable for Transformers due to the lack of continuous convolutions, but useful if linear layers are adjacent without non-linearities.
2. **AdaRound (Adaptive Rounding):** Highly effective for BERT. Instead of rounding weights to the nearest quantization bin (Round-to-Nearest), AdaRound optimizes the rounding decision by minimizing the layer-wise output Mean Squared Error (MSE) using a small calibration dataset.

**Quantization-Aware Training (QAT):**
If PTQ fails to meet accuracy targets (especially for INT4/INT8 mixed precision), QAT is employed. Fake quantization nodes are inserted into the graph. During fine-tuning on the downstream task (e.g., SQuAD), the model learns to adapt its weights to the quantization noise.

### 2.2 GLUE Benchmark Results
On the General Language Understanding Evaluation (GLUE) benchmark (tasks like MNLI, QQP, SST-2), quantizing BERT-Base to W8A8 (8-bit weights and activations) using AIMET's PTQ results in less than a 0.5% drop in accuracy. Even moving to W4A8 for weight-intensive layers can maintain >98% of the FP32 accuracy if AdaRound is utilized.

### 2.3 SQuAD Benchmark Results
Question Answering (SQuAD v1.1/v2.0) is highly sensitive to token representations. W8A8 typically preserves the F1 score within a 1% margin. However, W4A4 results in complete model collapse without extensive QAT and knowledge distillation from a full-precision teacher.

### 2.4 Per-layer Sensitivity for BERT
Sensitivity analysis reveals that in BERT:
- **First and Last Layers** are extremely sensitive. The embedding layer and the final classification head usually remain in FP16 or 8-bit.
- **Feed-Forward Networks (FFN)** are generally more robust to quantization than the multi-head attention (MHA) projections.

---

## 3. Quantizing Decoder-Only Models (GPT-2, LLaMA)

Generative decoder-only models dominate the current landscape. Because their primary operation is autoregressive generation (token-by-token), they are intensely **memory-bandwidth bound**.

### 3.1 Weight-Only Quantization (W4A16 / W8A16)
In memory-bound scenarios (batch size = 1), the arithmetic logic units (ALUs) sit idle waiting for weights to be loaded from HBM. Therefore, quantizing **only the weights** to 4-bit (INT4, NF4) or 8-bit while keeping activations in FP16 provides massive speedups (up to 3-4x) and memory savings. The math is performed in FP16 (weights are dequantized in registers before the MAC operation). This mitigates the activation outlier problem entirely.

### 3.2 Weight + Activation Quantization (W8A8)
For high-throughput serving (large batch sizes), the computation becomes compute-bound. Here, W8A8 is necessary to utilize INT8 Tensor Cores (e.g., on NVIDIA GPUs) or specialized integer NPUs. W8A8 requires addressing the activation outliers, leading to methods like SmoothQuant.

### 3.3 KV Cache Quantization
Quantizing the KV cache to INT8 or INT4 allows serving larger context lengths. Methods involve dynamic per-token quantization (calculating scale factors on the fly) or using asymmetric grouped quantization to capture the dynamic range of KV values across sequence dimensions.

---

## 4. GPTQ Algorithm

GPTQ (Accurate Post-Training Quantization for Generative Pre-trained Transformers) is a state-of-the-art method for W4A16 quantization, allowing models like LLaMA-65B to run on a single GPU.

### 4.1 Mathematical Derivation (Hessian-based)
GPTQ is inspired by Optimal Brain Quantization (OBQ). The goal is to find quantized weights \( \hat{W} \) that minimize the squared error of the layer output:
\[ \text{argmin}_{\hat{W}} || WX - \hat{W}X ||_2^2 \]
Using Taylor expansion, the optimal update for the remaining unquantized weights after quantizing a specific weight depends on the inverse Hessian matrix of the activations \( H = 2 XX^T \). 

### 4.2 Layer-wise Sequential Quantization
GPTQ quantizes the weights one column at a time. When a weight in column \( i \) is quantized, the error is calculated and compensated for by updating the remaining unquantized weights in columns \( j > i \).

### 4.3 Lazy Batch Updates
To make this computationally feasible for massive LLMs, GPTQ introduces "Lazy Batch-Updates". Instead of updating the entire remaining matrix after every single column quantization, GPTQ processes columns in blocks (e.g., 128 columns). The updates are accumulated and applied using optimized dense matrix multiplications, reducing the algorithm time from days to hours.

### 4.4 AIMET's Sequential MSE Comparison
AIMET's AdaRound acts similarly by optimizing rounding based on MSE, but GPTQ specifically leverages the Cholesky decomposition of the inverse Hessian for rapid, deterministic weight compensation without gradient descent, making it perfectly suited for models with billions of parameters.

---

## 5. AWQ (Activation-aware Weight Quantization)

AWQ observes that not all weights are equally important. A small fraction of weights (typically 0.1% to 1%) have a disproportionate impact on model performance.

### 5.1 Salient Weight Identification
AWQ identifies "salient" weights by looking at the activation magnitudes. Weights that multiply with massive activation outliers are critical. If we quantize these salient weights poorly, the resulting error is multiplied by the massive activation outlier, causing catastrophic output deviation.

### 5.2 Scale Factor Computation
Instead of keeping salient weights in FP16 (which requires sparse matrix operations), AWQ scales up the salient weights and scales down the corresponding activation channels. 
\[ Y = (W \cdot S) \times (X \cdot S^{-1}) \]
By scaling up the weights, they utilize more quantization bins, effectively reducing their quantization error. The scaling factors \( S \) are found through an automatic search process to minimize output MSE.

### 5.3 Per-group Quantization
AWQ inherently relies on group-wise quantization (e.g., group size 128). This reduces the parameter overhead of the scale factors and zero-points while providing fine-grained control over the quantization grid.

---

## 6. SmoothQuant

SmoothQuant is an elegant, math-equivalent transformation that enables W8A8 quantization without custom hardware kernels.

### 6.1 The Outlier Migration Idea
As established, weights are easy to quantize (few outliers), but activations are hard (massive systematic outliers). SmoothQuant migrates the quantization difficulty from the activations to the weights. 
It applies a per-channel scaling factor \( s \) to divide the activations (smoothing out the outliers) and multiplies the corresponding weight channels by \( s \) to maintain mathematical equivalence.

### 6.2 Per-channel Smoothing Factor
The smoothing factor is defined as:
\[ s_j = \max(|X_j|)^\alpha / \max(|W_j|)^{1-\alpha} \]
Where \( \alpha \) (usually 0.5) balances the difficulty between weights and activations. Once smoothed, the activations lose their extreme outliers and can be seamlessly quantized using standard INT8 symmetric per-tensor quantization, enabling standard INT8 matrix multiplication on commodity hardware.

---

## 7. BitsAndBytes (LLM.int8(), QLoRA)

BitsAndBytes is an ecosystem that democratized LLM finetuning and inference.

- **LLM.int8():** Solves the outlier problem dynamically. It uses thresholding to separate the matrix multiplication into two parts: a 16-bit matrix multiplication for the outlier columns (usually <1% of data) and an 8-bit multiplication for the rest. They are summed at the end.
- **QLoRA:** Introduces the 4-bit NormalFloat (NF4) data type, which is theoretically optimal for normally distributed weights. QLoRA loads the base model in NF4, keeps it frozen, and adds small FP16 Low-Rank Adapters (LoRA) for fine-tuning. It also introduces Double Quantization to compress the quantization statistics themselves, allowing a 65B model to be fine-tuned on a single 48GB GPU.

---

## 8. Quantization for Diffusion Models (Stable Diffusion)

Stable Diffusion relies heavily on UNet architectures containing cross-attention and self-attention.
- **Time-step Sensitivity:** The activation distribution in Diffusion models changes drastically depending on the time-step of the diffusion process. Early steps (heavily noised) have different dynamics than later steps.
- **PTQ Challenges:** Standard PTQ often introduces structural artifacts in the generated images. Multi-step calibration is required.
- **Weight/Activation Bits:** W8A8 is generally the baseline. W4A8 requires advanced QAT with distillation to maintain image structural integrity (FID and CLIP scores).

### 8.1 Stable Diffusion INT8 Quantization (UNet, VAE, CLIP separately)
Quantizing Stable Diffusion is a multi-modal challenge because it consists of three distinct models, each with different sensitivities.

#### The Text Encoder (CLIP)
- **Sensitivity:** High. CLIP encodes the text prompt into a latent representation. Small quantization errors here change the fundamental meaning of the prompt, leading to image generation that completely misses the user's intent.
- **Strategy:** Often kept in FP16 or conservatively quantized to W8A16. If W8A8 is required, SmoothQuant is highly recommended to preserve the outlier embeddings.

#### The UNet (Denoising Engine)
- **Sensitivity:** Moderate to High, but varies by layer. The UNet is the computationally heaviest part. It contains ResNet blocks and Spatial/Cross Attention blocks.
- **Strategy:** 
  - **Downsampling/Upsampling blocks:** Generally robust to INT8.
  - **Cross-Attention blocks:** Highly sensitive. The key and value projections for the text embeddings must be handled carefully. 
  - **Time-embedding layers:** Extremely sensitive. The time embeddings dictate the noise schedule. Quantization errors here cause the model to generate the wrong noise level, leading to deep structural artifacts (e.g., extra limbs, broken geometry). These are often pinned to FP16.
  - **PTQ vs QAT:** Simple INT8 PTQ often results in severe color banding and structural loss. Advanced PTQ techniques like AdaRound or full QAT using a subset of the LAION dataset for calibration are usually necessary for acceptable visual fidelity.

#### The VAE (Variational Autoencoder)
- **Sensitivity:** Extremely High. The VAE decodes the latent space back into pixel space. 
- **Strategy:** Quantizing the VAE is notoriously difficult. INT8 quantization almost always results in splotchy artifacts, checkerboarding, or loss of fine textures (like skin pores or fabric details). In most edge deployment scenarios, the VAE is kept entirely in FP16. Because it only runs once at the very end of the generation process, the compute cost of leaving it in FP16 is minimal compared to the UNet.

---

## 9. Vision Transformers (ViT) quantization

Vision Transformers exhibit a specific "attention spike" problem. After the softmax operation, attention maps often have a value extremely close to 1.0 for the CLS token, while other values are near 0.
- **Log2 Quantization:** Using logarithmic quantization for the attention maps is sometimes more effective than linear quantization.
- **Patch Representation:** The variance across patches can be high. LayerNorm inputs in ViTs are highly sensitive. Using mixed precision where LayerNorm and Softmax operate in FP16 while Linear layers are INT8 is standard practice.

---

## 10. On-device LLM deployment (Snapdragon + LLaMA)

Running LLaMA-7B on mobile edge devices (like Qualcomm Snapdragon) imposes strict power and thermal constraints (typically < 5 Watts).
- **NPU Acceleration:** The Hexagon NPU relies heavily on INT8/INT4 math. SmoothQuant and AdaRound are critical here to bake the scaling factors into the weights offline.
- **Memory Bandwidth:** Mobile LPDDR5 has a fraction of the bandwidth of GPU HBM. 4-bit weight quantization is mandatory to achieve acceptable token generation rates (e.g., 10-15 tokens/sec).
- **Grouped Query Attention (GQA):** Architectures like LLaMA-2 70B and LLaMA-3 use GQA, which inherently reduces the KV cache size, making it much friendlier for on-device deployment.

### 10.1 LLaMA-2 7B Deployment Pipeline on Snapdragon (Detailed)
Deploying a model like LLaMA-2 7B on a Snapdragon 8 Gen 3 requires a precise pipeline to fit the model within memory and thermal constraints.

1. **Model Preparation (HuggingFace):** Start with the PyTorch FP16 weights.
2. **Weight Quantization (W4A16 or W4A8):** Use AIMET or GPTQ to compress the 7B parameters from 14GB down to ~3.5GB. W4A8 is preferred to leverage the Hexagon HTX (Tensor Core) units fully.
3. **KV Cache Configuration:** Pre-allocate the KV cache buffer for the maximum supported context length (e.g., 2048 tokens). Implement dynamic INT8 quantization for the KV cache to keep it under 500MB.
4. **ONNX Export:** Export the model in two parts to handle dynamic sequence lengths efficiently:
   - **Prefill Model:** Handles the initial prompt processing in parallel. Takes sequence length `N` as input.
   - **Decode Model:** Handles autoregressive generation. Takes sequence length `1` as input and relies on the KV cache.
5. **QAIRT Conversion:** Use the `qairt-converter` to compile both ONNX graphs into `.dlc` containers. Provide the AIMET JSON encodings to ensure precise hardware translation.
6. **Context Binary Generation:** Serialize the `.dlc` into `.bin` to eliminate graph compilation time on the device, ensuring the app boots instantly.
7. **Execution via QNN SDK:** Write a native C++ Android app using the QNN SDK. Load the `.bin` into the Hexagon NPU. Manage the KV cache buffer purely in device memory (VTCM or mapped DDR) to avoid costly CPU-NPU data transfers between token generations.

---

## 11. AIMET for NLP: Complete Code Examples

### 11.1 Complete AIMET Code for BERT PTQ (QuantSim setup + calibration + export)

Below is a complete, detailed example of using AIMET to perform AdaRound and QuantSim Post-Training Quantization on a HuggingFace BERT model.

```python
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.qc_quantize_op import QuantScheme

# 1. Load the pre-trained FP32 model and tokenizer
model_name = 'textattack/bert-base-uncased-SST-2'
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()
tokenizer = BertTokenizer.from_pretrained(model_name)

# 2. Prepare Calibration DataLoader (crucial for PTQ/AdaRound)
class DummyCalibrationDataset(Dataset):
    def __init__(self, size=128, seq_len=128):
        self.size = size
        self.seq_len = seq_len
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        # Generate dummy input ids and attention masks
        return (torch.randint(0, 30000, (self.seq_len,)), 
                torch.ones(self.seq_len, dtype=torch.long))

calib_dataset = DummyCalibrationDataset(size=512)
calib_dataloader = DataLoader(calib_dataset, batch_size=8)

# 3. Configure AdaRound Parameters for W4A8
# AdaRound optimizes weight rounding (better than nearest-rounding)
params = AdaroundParameters(
    data_loader=calib_dataloader,
    num_batches=64, # 64 batches of size 8 = 512 samples
    default_num_iterations=10000, # optimization steps per layer
    default_reg_param=0.01,
    default_beta_range=(20, 2)
)

# 4. Apply AdaRound to optimize 4-bit weights
dummy_input = (torch.randint(0, 30000, (1, 128)), torch.ones(1, 128, dtype=torch.long))
print("Starting AdaRound optimization...")
adarounded_model = Adaround.apply_adaround(
    model, 
    dummy_input=dummy_input, 
    params=params, 
    path="./adaround_output", 
    filename_prefix="bert_sst2", 
    default_param_bw=4, # 4-bit weights
    default_quant_scheme=QuantScheme.post_training_tf_enhanced
)

# 5. Create QuantSim for Full Activation Calibration
# Pass the adarounded model into QuantSim to calibrate 8-bit activations
def forward_pass_callback(model, dataloader):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            model(input_ids=input_ids, attention_mask=attention_mask)

sim = QuantizationSimModel(
    adarounded_model, 
    dummy_input=dummy_input,
    default_output_bw=8, # 8-bit activations
    default_param_bw=4,  # 4-bit weights
    quant_scheme=QuantScheme.post_training_tf_enhanced
)

# 6. Compute Encodings (Find min/max for activations)
print("Computing activation encodings...")
sim.compute_encodings(forward_pass_callback, forward_pass_callback_args=(calib_dataloader,))

# 7. Export the Quantized Model (ONNX + JSON encodings)
print("Exporting model for Qualcomm QAIRT...")
sim.export(
    path="./export_output",
    filename_prefix="bert_w4a8",
    dummy_input=dummy_input
)
print("Export complete. Ready for qairt-converter.")
```

### 11.2 Complete AIMET Code for BERT QAT Training Loop

When PTQ (AdaRound) is not enough to recover accuracy, Quantization-Aware Training (QAT) is required. QAT simulates quantization noise during training, allowing the model's weights to adjust and compensate.

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.qc_quantize_op import QuantScheme

# Assume `model` is already loaded and `train_dataloader` is prepared
# dummy_input = ...

# 1. Initialize QuantSim for QAT
sim = QuantizationSimModel(
    model, 
    dummy_input=dummy_input,
    default_output_bw=8,
    default_param_bw=8,
    quant_scheme=QuantScheme.training_range_learning_with_tf_init
)

# 2. Compute initial encodings using a forward pass
def compute_initial_encodings(model, dataloader):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            model(input_ids=batch[0], attention_mask=batch[1])
            if i >= 10: break # Small subset is enough

sim.compute_encodings(compute_initial_encodings, forward_pass_callback_args=(train_dataloader,))

# 3. Enable QAT Training Mode
qat_model = sim.model
qat_model.train()

# Use a very small learning rate for QAT (e.g., 1e-5 or 1e-6)
optimizer = AdamW(qat_model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# 4. QAT Training Loop
print("Starting QAT fine-tuning...")
epochs = 2
for epoch in range(epochs):
    for step, batch in enumerate(train_dataloader):
        input_ids, attention_mask, labels = batch
        
        optimizer.zero_grad()
        
        # Forward pass (includes simulated quantization noise)
        outputs = qat_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        loss = loss_fn(logits, labels)
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

# 5. Export the QAT-optimized model
sim.export(path="./qat_output", filename_prefix="bert_qat_w8a8", dummy_input=dummy_input)
```

---

## 12. Mixed Precision for Attention

Not all components of the attention mechanism require the same precision:
- **Query, Key, Value Projections:** Often robust to INT8 or even INT4.
- **Attention Matrix (Q * K^T):** Highly sensitive. Often kept in FP16 or quantized to a higher bit-width (INT8 symmetric).
- **Softmax Output:** The non-linear exponential distribution makes INT8 linear quantization very lossy. Techniques often use Log-quantization or retain FP16.
- **Output Projection:** The final linear layer inside the attention block is relatively robust and can be INT8.

### 12.1 FlashAttention and its Quantization Implications
FlashAttention revolutionized transformer performance by computing exact attention mathematically while minimizing costly memory reads/writes to HBM through clever tiling and recomputation.
- **Fusion Complications:** FlashAttention fuses the \(Q \times K^T\), Masking, Softmax, and \(\times V\) operations into a single massive CUDA kernel. Because these operations never write their intermediate results to global memory, inserting standard quantization/dequantization nodes between them (as typical PTQ does) breaks the FlashAttention fusion.
- **Solution:** Hardware-aware implementations of Quantized FlashAttention now exist, where the inputs (Q, K, V) are loaded in INT8, the math is performed on Tensor Cores, and scaling factors are applied entirely within SRAM registers during the tiled computation. This represents the cutting edge of LLM inference optimization.

---

## 13. Speculative Decoding and Quantization

Speculative decoding uses a small, fast "draft" model to generate multiple tokens, which are then verified in parallel by the massive "target" model.
- **Quantization Synergy:** The draft model can be heavily quantized (e.g., W4A4 or INT3) to maximize generation speed. Even if it makes errors due to extreme quantization, the FP16/INT8 target model corrects them.
- This creates an optimal hardware pipeline: the draft model runs entirely in L2/L3 cache, while the target model maximizes HBM bandwidth for parallel verification.

### 13.1 Deep Dive: Speculative Decoding with Quantized Draft Models
When combining quantization with speculative decoding, the acceptance rate of the draft tokens becomes the critical metric.
- **The Target Model:** Let's assume the target is a high-quality W8A16 LLaMA-2 70B model.
- **The Draft Model:** A highly aggressive W4A4 or even W2A16 LLaMA-68M model.
- **Trade-offs:** By quantizing the draft model so aggressively, we drastically increase its token generation speed (e.g., 200 tokens/sec). However, the extreme quantization damages its predictive power, meaning the target model will reject its guesses more frequently.
- **The Balancing Act:** If the draft model is too heavily quantized, the acceptance rate drops below 50%, and the overhead of running the draft model outweighs the parallel verification benefits. Careful PTQ of the draft model is required to maintain an acceptance rate of ~70-80% while keeping it small enough to reside in cache.

---

## 14. Benchmarks: Perplexity, Latency, Memory

When evaluating quantized LLMs, three metrics are paramount:

1. **Perplexity (PPL):** The ultimate measure of language modeling capability. 
2. **Latency (Tokens/Second):** The speed of generation.
3. **Memory Footprint:** The RAM required to hold the weights and KV cache.

### 14.1 Perplexity Tables: LLaMA-2 7B/13B/70B at INT8/INT4

The following table demonstrates the impact of different quantization schemes on the perplexity (WikiText-2) of the LLaMA-2 family. Lower is better.

| Model Size | FP16 Baseline | W8A16 (RTN) | W8A8 (SmoothQuant) | W4A16 (GPTQ) | W4A16 (AWQ) | W4A8 (AdaRound) |
|---|---|---|---|---|---|---|
| **LLaMA-2 7B** | 5.47 | 5.50 | 5.52 | 5.61 | 5.58 | 5.85 |
| **LLaMA-2 13B** | 4.88 | 4.90 | 4.93 | 5.02 | 4.98 | 5.15 |
| **LLaMA-2 70B** | 3.32 | 3.34 | 3.35 | 3.41 | 3.39 | 3.50 |

*Observations:* 
- W8A16 is nearly lossless across all scales.
- 70B models are significantly more robust to 4-bit quantization than 7B models. The larger the model, the easier it is to compress aggressively without catastrophic PPL degradation.

### 14.2 Full Latency Benchmarks: Tokens/sec on Snapdragon 8 Gen 3

Deploying on a mobile SoC like the Snapdragon 8 Gen 3 (using QNN SDK, Hexagon HTX) yields the following estimated performance for token generation (batch size 1, context length 512):

| Model | Quantization Format | Memory Footprint | Gen Speed (Tokens/Sec) | Time To First Token |
|---|---|---|---|---|
| **LLaMA-2 7B** | W4A16 (Weights INT4, Act FP16) | ~3.8 GB | 14 - 16 t/s | ~150 ms |
| **LLaMA-2 7B** | W4A8 (Weights INT4, Act INT8) | ~3.6 GB | 18 - 22 t/s | ~120 ms |
| **LLaMA-2 13B** | W4A8 | ~7.0 GB | 8 - 10 t/s | ~250 ms |

*Note: W8A8 is largely memory-bound on mobile LPDDR memory, offering minimal speedup over W4A8 for autoregressive generation where the batch size is 1.*

---

## 15. Future Directions: FP4, INT2, Binary Transformers

- **Microformats (FP4 / E2M1):** NVIDIA's Blackwell architecture introduces hardware support for FP4. Floating point representation handles outlier ranges better than integer types.
- **Sub-4-bit (INT2, Ternary):** Pushing models to 2-bit or ternary (-1, 0, 1) quantization requires massive QAT. Techniques like QuIP# use incoherent processing and lattice codebooks to maintain accuracy at 2-bits.
- **1-bit Architectures (BitNet / BiLLM):** The ultimate frontier. BitNet b1.58 proposes a native 1.58-bit model where weights are {-1, 0, 1}. This transforms matrix multiplication into pure addition/subtraction, fundamentally changing silicon design and potentially eliminating the von Neumann bottleneck.

## Additional Deep Dives

### Complete AIMET BERT Quantization Code (PTQ + QAT)
```python
# Full BERT PTQ Pipeline
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from aimet_torch.quantsim import QuantizationSimModel, QuantScheme
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters

# Step 1: Load BERT-Base for Question Answering
model_name = 'bert-base-uncased'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.eval()

# Step 2: Prepare dummy inputs for tracing
dummy_input_ids = torch.randint(0, 1000, (1, 128))
dummy_attention = torch.ones(1, 128, dtype=torch.long)
dummy_token_type = torch.zeros(1, 128, dtype=torch.long)
dummy_input = (dummy_input_ids, dummy_attention, dummy_token_type)

# Step 3: Create QuantSim
sim = QuantizationSimModel(
    model=model,
    dummy_input=dummy_input,
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    default_output_bw=8,
    default_param_bw=8,
)

# Step 4: Calibration
squad_dataset = load_dataset('squad', split='validation[:500]')

def calibrate_bert(model, args):
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(squad_dataset):
            if i >= 50: break
            inputs = tokenizer(
                sample['question'], sample['context'],
                return_tensors='pt', max_length=384, truncation=True, padding='max_length'
            )
            model(**inputs)

sim.compute_encodings(calibrate_bert, None)

# Step 5: Evaluate
def evaluate_squad(model):
    # F1 evaluation on SQuAD validation
    total_f1 = 0
    # ... evaluation loop
    return total_f1

# Step 6: Export
sim.export('./bert_int8', 'bert_qa_int8',
           dummy_input=dummy_input,
           onnx_export_args={'opset_version': 13})
print('Exported: bert_qa_int8.onnx + bert_qa_int8.encodings.json')
```

### LLaMA-2 7B Deployment on Snapdragon 8 Gen 3

**Hardware Target**: Snapdragon 8 Gen 3
- Hexagon NPU: 45 TOPS INT8
- 12GB LPDDR5X at 77 GB/s
- Target: Conversational LLM at ≥10 tokens/sec

**Model Size Analysis**:
| Format | Model Size | Fits in NPU SRAM? | Tokens/sec |
|--------|-----------|-------------------|------------|
| FP32   | 28 GB     | No (>>12GB)       | N/A        |
| FP16   | 14 GB     | No                | 2-3        |
| INT8   | 7 GB      | Partial           | 6-8        |
| INT4   | 3.5 GB    | Yes               | 12-18      |
| INT4-grouped (Q4_K_M) | 4.1 GB | Yes | 10-15 |

**Deployment Pipeline with AIMET**:
```python
from aimet_torch.quantsim import QuantizationSimModel, QuantScheme
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load LLaMA-2 7B
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    torch_dtype=torch.float16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

# Weight-only INT4 quantization (W4A16)
# AIMET Sequential MSE for weight quantization
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters

dummy_input_ids = torch.randint(0, 32000, (1, 256))
dummy_input = {'input_ids': dummy_input_ids, 'attention_mask': torch.ones(1, 256, dtype=torch.long)}

# INT4 weight quantization only (activations stay FP16)
params = AdaroundParameters(
    data_loader=calibration_loader,
    num_batches=8,
    default_num_iterations=5000,
)

quantized_model = Adaround.apply_adaround(
    model=model,
    dummy_input=dummy_input,
    params=params,
    path='./llama2_int4',
    filename_prefix='llama2_7b',
    default_param_bw=4,  # 4-bit weights
    default_quant_scheme=QuantScheme.post_training_tf_enhanced
)
```

### KV Cache Quantization Technical Deep-Dive

KV (Key-Value) Cache is critical for autoregressive LLM inference:

```
KV Cache Memory Growth:
  Sequence length 512:  2 × layers × heads × seq × head_dim × 2 (K+V)
  LLaMA-2 7B:           2 × 32 × 32 × 512 × 128 × 2 bytes (FP16)
                        = 2 × 32 × 32 × 512 × 128 × 2 = 268 MB per request!

  At 10 concurrent users: 2.68 GB just for KV cache
  At INT8: 1.34 GB (50% savings)
  At INT4: 0.67 GB (75% savings)
```

**INT8 KV Cache with AIMET**:
```python
# Custom quantizer for KV cache
class KVCacheQuantizer:
    def __init__(self, num_bits=8, symmetric=False):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.scale = None
        self.zero_point = None
    
    def calibrate(self, kv_cache_samples):
        # Collect statistics over calibration samples
        all_keys = torch.cat([s['keys'] for s in kv_cache_samples], dim=2)
        self.scale = all_keys.abs().max() / (2**(self.num_bits-1) - 1)
        self.zero_point = 0 if self.symmetric else -128
    
    def quantize(self, tensor):
        q = (tensor / self.scale).round().clamp(-128, 127)
        return q.to(torch.int8)
    
    def dequantize(self, q_tensor):
        return q_tensor.to(torch.float16) * self.scale
```

### Stable Diffusion INT8 Quantization

**Architecture Components**:
| Component | Type | Parameters | Quantization Difficulty |
|-----------|------|-----------|------------------------|
| Text Encoder (CLIP) | Transformer | 123M | Easy (INT8) |
| VAE Encoder | CNN | 34M | Easy (INT8) |
| UNet | CNN + Attention | 860M | Medium (sensitive attention) |
| VAE Decoder | CNN | 49M | Hard (quality-critical) |

```python
# Stable Diffusion Component Quantization
from diffusers import StableDiffusionPipeline
from aimet_torch.quantsim import QuantizationSimModel, QuantScheme

pipeline = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')

# 1. Quantize CLIP text encoder (easiest)
clip_sim = QuantizationSimModel(
    pipeline.text_encoder,
    dummy_input=torch.randint(0, 1000, (1, 77)),
    default_output_bw=8, default_param_bw=8
)

# 2. Quantize UNet (most impact on quality)
# Use mixed precision: attention at FP16, conv at INT8
unet_sim = QuantizationSimModel(
    pipeline.unet,
    dummy_input=(torch.randn(2, 4, 64, 64), torch.tensor([1.0]), torch.randn(2, 77, 768)),
    default_output_bw=8, default_param_bw=8
)
# Disable quantization for cross-attention (sensitive)
for name, module in unet_sim.model.named_modules():
    if 'attn' in name and hasattr(module, 'output_quantizers'):
        for q in module.output_quantizers:
            q.bitwidth = 16  # Keep attention at FP16

# 3. VAE decoder: more conservative (INT8 or keep FP16)
```

**Results: Stable Diffusion v1.5 on Snapdragon 8 Gen 3**:
| Config | FID Score | Time/Image | Memory |
|--------|-----------|-----------|--------|
| FP16   | 12.4      | 28s       | 2.8 GB |
| INT8 all | 15.1    | 11s       | 1.1 GB |
| Mixed (attn FP16, rest INT8) | 13.0 | 14s | 1.6 GB |

### Speculative Decoding with Quantized Models

Speculative decoding uses a small draft model to propose tokens, verified by the large model:

```python
# Speculative decoding pipeline
class SpeculativeDecoder:
    def __init__(self, draft_model, target_model, k=4):
        # draft_model: small INT4 model (fast)
        # target_model: large INT8 model (accurate)
        # k: tokens to speculate ahead
        self.draft = draft_model
        self.target = target_model
        self.k = k
    
    def generate(self, input_ids, max_new_tokens=200):
        generated = input_ids
        while len(generated[0]) < len(input_ids[0]) + max_new_tokens:
            # Step 1: Draft model generates k tokens
            draft_output = self.draft.generate(
                generated, max_new_tokens=self.k,
                do_sample=True, temperature=0.8
            )
            draft_tokens = draft_output[:, -self.k:]
            
            # Step 2: Target model verifies all k tokens in parallel
            combined = torch.cat([generated, draft_tokens], dim=1)
            target_logits = self.target(combined).logits
            
            # Step 3: Accept tokens that match target distribution
            # (rejection sampling)
            accepted = self._rejection_sample(draft_tokens, target_logits)
            generated = torch.cat([generated, accepted], dim=1)
        
        return generated
```

**Speedup from Speculative Decoding + Quantization**:
| Setup | Tokens/sec | Notes |
|-------|-----------|-------|
| LLaMA-2 7B FP16 | 8 t/s | Baseline |
| LLaMA-2 7B INT8 | 14 t/s | 1.75x |
| LLaMA-2 7B INT4 | 22 t/s | 2.75x |
| Speculative (70B+7B INT4) | 35 t/s | 4.4x |

### Perplexity Tables: LLaMA-2 7B/13B/70B

| Model | Format | Perplexity (WikiText-2) | Size (GB) |
|-------|--------|------------------------|----------|
| LLaMA-2 7B | FP32 | 5.47 | 28.0 |
| LLaMA-2 7B | FP16 | 5.47 | 14.0 |
| LLaMA-2 7B | INT8 | 5.51 | 7.0 |
| LLaMA-2 7B | INT4 (GPTQ) | 5.68 | 3.5 |
| LLaMA-2 7B | INT4 (AWQ) | 5.60 | 3.9 |
| LLaMA-2 13B | FP16 | 4.88 | 26.0 |
| LLaMA-2 13B | INT8 | 4.91 | 13.0 |
| LLaMA-2 13B | INT4 (AWQ) | 5.02 | 7.2 |
| LLaMA-2 70B | FP16 | 3.32 | 140.0 |
| LLaMA-2 70B | INT8 | 3.34 | 70.0 |
| LLaMA-2 70B | INT4 (AWQ) | 3.41 | 38.0 |
| LLaMA-2 70B | INT2 (QuIP#) | 3.89 | 19.0 |

### FlashAttention and Quantization

FlashAttention restructures attention computation for memory efficiency:

```
Standard Attention Memory Complexity: O(n²) 
  Full attention matrix materialized in DRAM
  For n=8192 tokens, head_dim=128: 8192² × 4 bytes = 268 MB

FlashAttention Memory Complexity: O(n) 
  Chunked computation, only tiles fit in SRAM
  Dramatically reduces DRAM bandwidth

Quantization Interaction:
  Standard attention: quantize the full n×n matrix → simple
  FlashAttention: attention matrix never materialized → harder!
  
Solution: Quantize Q, K, V projections; use FP16 for attention score
  computation within FlashAttention kernels
```
