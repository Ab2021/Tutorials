# Lab 1: Pre-training a TinyLlama

## Objective
Train a "Real" LLM (not just a toy script).
We will use the **Hugging Face Trainer** and **Accelerate** library.
This mimics how Llama-3 was trained, just on a smaller scale.

## 1. Setup

```bash
poetry add transformers datasets accelerate wandb
```

## 2. The Training Script (`pretrain.py`)

We will train a 50M parameter model on the `wikitext-2` dataset.

```python
import torch
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import wandb

# 1. Configuration
MODEL_NAME = "gpt2" # We will initialize from scratch, but use GPT2 config structure
dataset_name = "wikitext"
dataset_config = "wikitext-2-raw-v1"

# 2. Load Data
dataset = load_dataset(dataset_name, dataset_config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 3. Define Model (From Scratch)
config = AutoConfig.from_pretrained(
    MODEL_NAME,
    vocab_size=len(tokenizer),
    n_ctx=128,
    n_embd=256,   # Small embedding
    n_layer=4,    # Only 4 layers
    n_head=4      # Only 4 heads
)

model = AutoModelForCausalLM.from_config(config)
print(f"Model Parameters: {model.num_parameters() / 1e6:.2f}M")

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./tiny-llama-checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-4,
    weight_decay=0.01,
    report_to="wandb", # Log to Weights & Biases
    fp16=torch.cuda.is_available(), # Use Mixed Precision if GPU
)

# 5. Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# 6. Train
trainer.train()
trainer.save_model("./tiny-llama-final")
```

## 3. Running the Lab

1.  Login to WandB: `wandb login`.
2.  Run: `python pretrain.py`.
3.  Check the WandB dashboard. Look at the **Loss Curve**.

## 4. Inference

Write a script `infer.py` to load your model and generate text.

```python
from transformers import pipeline

generator = pipeline('text-generation', model='./tiny-llama-final', tokenizer='gpt2')
print(generator("The history of science is", max_length=50))
```

## 5. Submission
Submit the link to your WandB run.
