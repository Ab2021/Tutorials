# Day 172: Teaching Old Dogs New Tricks: LoRA & PEFT
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 25: Large Language Model Infrastructure

---

> **🎯 Focus Area:** Retraining 70B parameters costs $1,000,000. **LoRA (Low-Rank Adaptation)** freezes the 70B weights and trains a tiny "Adapter" (100MB) side-by-side. It costs $10 and achieves 99% of full fine-tuning performance.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the mathematics of Low-Rank decomposition ($W + BA$).
2.  **Fine-tune** a Llama-7B model on a Colab GPU using **QLoRA** (4-bit quantization).
3.  **Merge** LoRA adapters back into the base model for inference speed.
4.  **Manage** multiple adapters (SQL Adapter, Python Adapter) for a single base model.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine with GPU (16GB+ RAM) or Colab T4.

### Software Environment
- `pip install peft bitsandbytes transformers trl`.

---

## 📖 Theoretical Foundation

### 1. The LoRA Hypothesis
Neural Networks have low "intrinsic dimension".
Instead of updating the full weight matrix $W$ ($d \times d$), we add a low-rank update $\Delta W = B A$.
*   $A$: $d \times r$ (Random Init)
*   $B$: $r \times d$ (Zero Init)
*   $r$: Rank (e.g., 8, 16, 64). Much smaller than $d$.
*   **Parameters:** $2 \times d \times r$. If $d=4096, r=8$, params are reduced by 256x.

### 2. QLoRA
*   **Quantize** Base Model to 4-bit (NF4).
*   **Train** LoRA Adapters in FP16/BF16.
*   **Result:** Fine-tune 65B model on a single 48GB GPU.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Training Script (SFT)

Using Hugging Face `trl` (Transformer Reinforcement Learning) library.

#### 📁 `src/train_lora.py`
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

model_name = "meta-llama/Llama-2-7b-chat-hf"

# 1. 4-Bit Quantization Config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 2. Load Base Model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# 3. LoRA Config
peft_config = LoraConfig(
    r=16,       # Low Rank
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"] # Apply to Attention Query/Value
)

model = get_peft_model(model, peft_config)
print(f"Trainable Params: {model.print_trainable_parameters()}")

# 4. Dataset
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

# 5. Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args, # Batch size, LR, etc.
)

trainer.train()
trainer.save_model("adapters/my-chatbot")
```

### 👨‍💻 Core Implementation: Inference with Adapters

Load Base + Adapter.

#### 📁 `src/inference_lora.py`
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "meta-llama/Llama-2-7b-chat-hf"
adapter_path = "adapters/my-chatbot"

# Load Base
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load Adapter
model = PeftModel.from_pretrained(model, adapter_path)

# Run
tokenizer = AutoTokenizer.from_pretrained(base_model)
inputs = tokenizer("Hello bot!", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

### 👨‍💻 Infrastructure: Merging for Production

Running `PeftModel` adds overhead (extra matrix multiply).
**Merging:** Compute $W_{merged} = W_{base} + B A$. Save as standard model.

#### 📁 `src/merge_lora.py`
```python
model = model.merge_and_unload()
model.save_pretrained("models/my-chatbot-merged")
# Now you can serve this with vLLM directly!
```

---

## 🔬 Lab Exercise: "The Polyglot"

### Task
Multi-Adapter Serving.
1.  Train Adapter A on SQL generation ("text-to-sql").
2.  Train Adapter B on Python generation ("text-to-python").
3.  Load Base Model once (14GB VRAM).
4.  Load Adapter A (100MB) and Adapter B (100MB).
5.  **Runtime:** Receive request.
    *   If user asks "Select *...", activate Adapter A.
    *   If user asks "def func()...", activate Adapter B.
6.  **Tool:** `LoRAX` (LoRA Exchange) serving framework allows hot-swapping adapters on the fly.

---

## 📖 Advanced Theory: Rank Selection
How to choose `r`?
*   `r=8`: Good for simple style transfer (Shakespeare).
*   `r=64`: Good for complex reasoning injection (Math).
*   `r=256`: Approaching full finetuning. Diminishing returns.
*   **Alpha:** Scaling factor. Usually set $\alpha = 2r$.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Storage:** Store only the adapter weights (100MB). Do not duplicate the base model (14GB) for every version.
2.  **Catastrophic Forgetting:** LoRA mitigates this because the base weights are frozen. You retain general knowledge while gaining specific skills.
3.  **Target Modules:** Don't just target `q_proj` and `v_proj`. Targeting all linear layers (`gate_proj`, `up_proj`, `down_proj`) often yields better results (QLoRA paper).

### API Summary
```python
LoraConfig(r=16, target_modules=["all-linear"])
```

---

**Day 172 Complete** ✅

*Next: Day 173 - RAG Infrastructure - Vector Databases at Scale.*
