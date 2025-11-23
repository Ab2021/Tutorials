# Day 23: Parameter-Efficient Fine-tuning (PEFT)
## Core Concepts & Theory

### The Fine-tuning Bottleneck

Full Fine-tuning (FFT) a 70B model requires updating 70B parameters.
- **Storage:** Need to save a full copy of the model for every task. (70B * 2 bytes = 140GB per checkpoint).
- **Memory:** Gradients + Optimizer states for 70B params = ~1TB VRAM.
- **Solution:** PEFT. Update only a tiny subset of parameters (<1%).

### 1. LoRA (Low-Rank Adaptation)

**Concept:**
Hypothesis: The change in weights $\Delta W$ during fine-tuning has a "low intrinsic rank".
Instead of updating $W$ (size $d \times d$), we inject trainable rank decomposition matrices $A$ and $B$.
$$ W' = W + \Delta W = W + BA $$
- $B$: $d \times r$ (initialized to 0)
- $A$: $r \times d$ (initialized to Gaussian)
- $r$: Rank (e.g., 8, 16, 64). Much smaller than $d$.

**Benefits:**
- **Memory:** Gradients/Optimizer states only for $A, B$. (Reduces VRAM by 3x).
- **Storage:** Only save $A, B$ (few MBs). Base model $W$ is frozen.
- **Switching:** Can swap adapters on the fly for different tasks without reloading the base model.

### 2. QLoRA (Quantized LoRA)

**Concept:** Combine LoRA with 4-bit Quantization.
- **Base Model:** Loaded in 4-bit (NF4 format). Frozen.
- **Adapters:** Trained in BF16/FP16.
- **Memory:** LLaMA-65B fits on a single 48GB GPU (A6000).
- **Double Quantization:** Quantize the quantization constants to save even more memory.

### 3. Prompt Tuning & Prefix Tuning

**Prompt Tuning:**
- Prepend trainable "soft prompt" vectors to the input embeddings.
- Only update these vectors.
- **Cons:** Performance lags behind LoRA for complex tasks.

**Prefix Tuning:**
- Prepend trainable vectors to the *keys and values* of every attention layer.
- **Cons:** Reduces effective context length.

### 4. Adapters (Houlsby et al.)

**Concept:** Insert small MLP layers (Adapter layers) *between* the Transformer layers.
- **Structure:** Down-project -> Non-linearity -> Up-project.
- **Cons:** Adds inference latency (extra layers). LoRA can be merged into $W$, so zero latency.

### Summary of PEFT Methods

| Method | Updates | Inference Latency | Memory Savings | Performance |
| :--- | :--- | :--- | :--- | :--- |
| **FFT** | 100% | Zero | None | Best |
| **LoRA** | <1% | Zero (Merged) | High | Near-FFT |
| **QLoRA** | <1% | High (De-quant) | Very High | Near-FFT |
| **Prompt** | <0.1% | Zero | High | Lower |
| **Adapter** | ~3% | High (Extra layers) | Medium | Good |

### Next Steps
In the Deep Dive, we will derive the LoRA update rule and implement a LoRA layer from scratch in PyTorch.
