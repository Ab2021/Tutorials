# Day 29: PEFT, LoRA, and QLoRA - Theory & Implementation

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: Parameter Efficient Fine-Tuning

## 1. Theoretical Foundation: Why PEFT?

Full Fine-Tuning (FFT) of LLaMA-7B updates 7B parameters.
*   **Storage**: Need to save 14GB checkpoint per task.
*   **Memory**: Gradients + Optimizer States for 7B params = 100GB+ VRAM.

**PEFT**: Freeze the pre-trained model. Add small trainable adapters.
*   Update only < 1% of parameters.
*   Performance matches FFT.

## 2. LoRA (Low-Rank Adaptation)

Hypothesis: Weight updates $\Delta W$ have a low intrinsic rank.
$$ W_{new} = W_{frozen} + \Delta W $$
$$ \Delta W = B A $$
*   $W \in \mathbb{R}^{d \times d}$.
*   $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$.
*   $r \ll d$ (e.g., $r=8$).
*   $A$ initialized to Gaussian, $B$ to Zero (so $\Delta W = 0$ at start).

## 3. QLoRA (Quantized LoRA)

Combines 4-bit Quantization with LoRA.
1.  **4-bit NormalFloat (NF4)**: Optimal data type for Gaussian weights.
2.  **Double Quantization**: Quantize the quantization constants.
3.  **Paged Optimizers**: Offload optimizer states to CPU RAM if GPU OOM.

Result: Fine-tune LLaMA-65B on a single 48GB GPU.

## 4. Implementation: Fine-Tuning LLaMA with LoRA

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. Load in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", 
    quantization_config=bnb_config,
    device_map="auto"
)

# 2. Prepare for LoRA
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

# 3. Train (Standard Trainer)
# ...
```

## 5. Merging LoRA Weights

After training, we have small adapter weights (100MB).
We can merge them back into the base model for inference (no latency overhead).
$$ W_{final} = W_{base} + B A \cdot \frac{\alpha}{r} $$

```python
model = model.merge_and_unload()
model.save_pretrained("merged_model")
```
