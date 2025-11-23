# Day 29: Efficient Fine-Tuning (PEFT, LoRA, QLoRA)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing QLoRA with Hugging Face

We will fine-tune Llama-3-8B on a single GPU using `bitsandbytes` and `peft`.

**Dependencies:** `transformers`, `peft`, `bitsandbytes`, `trl`

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Quantization Config (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 2. Load Base Model
model_name = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# 3. LoRA Config
peft_config = LoraConfig(
    r=16,       # Rank (Low rank = fewer params)
    lora_alpha=32, # Scaling factor
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"] # Apply to Attention layers
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Output: "trainable params: 4M || all params: 7B || trainable%: 0.06%"

# 4. Training
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args
)

trainer.train()
```

### LoRA Hyperparameters

*   **Rank (r):** The dimension of the low-rank matrices. Typical values: 8, 16, 64. Higher $r$ = more capacity, but more VRAM.
*   **Alpha ($\alpha$):** Scaling factor. Weight = $W + \frac{\alpha}{r} \Delta W$. Usually set $\alpha = 2r$ or $\alpha = r$.
*   **Target Modules:** Which layers to adapt?
    *   `q_proj`, `v_proj`: Standard (Attention).
    *   `all-linear`: Better performance, slightly more VRAM.

### Merging Adapters

After training, you have the base model + adapter weights (100MB).
To deploy, you usually **merge** them:
```python
model = model.merge_and_unload()
model.save_pretrained("merged_model")
```
Now it's just a standard Llama-3 model, compatible with vLLM/TGI.

### Summary

*   **QLoRA** is the gold standard for single-GPU fine-tuning.
*   **Merging** is essential for production inference speed.
*   **Rank** is the main knob to tune for capacity vs efficiency.
