# Lab 1: LoRA Fine-tuning (QLoRA)

## Objective
Fine-tune a 7B parameter model on a free Google Colab GPU (T4).
We will use **QLoRA** (Quantized Low-Rank Adaptation) to fit the model in 16GB VRAM.

## 1. Setup

```bash
pip install -q -U bitsandbytes transformers peft accelerate datasets trl
```

## 2. The Training Script (`finetune.py`)

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# 1. Config
MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
NEW_MODEL = "llama-2-7b-miniguanaco"

# 2. Quantization Config (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 3. Load Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 4. LoRA Config
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. Dataset (Guanaco - 1k samples)
dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")

# 6. Trainer
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    logging_steps=25,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
)

# 7. Train
trainer.train()
trainer.model.save_pretrained(NEW_MODEL)
```

## 3. Inference with Adapter

```python
from peft import PeftModel

# Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, ...)
# Load Adapter
model = PeftModel.from_pretrained(base_model, NEW_MODEL)

# Generate
inputs = tokenizer("What is AI?", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

## 4. Submission
Submit a screenshot of the training loss curve.
