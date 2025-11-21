# Day 28: The Hugging Face Ecosystem - Theory & Implementation

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: Transformers, Datasets, and Accelerate

## 1. Theoretical Foundation: The Standardization of NLP

Before HF: Every paper had its own C++/Lua/Python code. Hard to reproduce.
**Hugging Face Transformers**:
*   **Unified API**: `from_pretrained()`.
*   **Model Hub**: 500k+ models.
*   **Interoperability**: PyTorch, TensorFlow, JAX.

## 2. Key Components

### Transformers
*   **Model**: The architecture (e.g., `BertModel`).
*   **Config**: Hyperparameters (`BertConfig`).
*   **Tokenizer**: Text processing (`BertTokenizer`).
*   **Pipeline**: High-level abstraction (`pipeline("sentiment-analysis")`).

### Datasets
*   Apache Arrow based. Memory-mapped.
*   Can load 1TB dataset on a 16GB RAM laptop.
*   Streaming support.

### Accelerate
*   Boilerplate remover for distributed training (Multi-GPU, TPU, FP16).

## 3. Implementation: Training with Trainer API

The `Trainer` class handles the training loop, logging, checkpointing, and mixed precision.

```python
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# 1. Load Data
dataset = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['sentence1'], batch['sentence2'], truncation=True)

dataset = dataset.map(tokenize, batched=True)

# 2. Load Model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 3. Config
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True # Mixed Precision
)

# 4. Train
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()
```

## 4. Custom Training Loop (Accelerate)

If you want full control but need Multi-GPU support.

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

for batch in train_loader:
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss) # Replaces loss.backward()
    optimizer.step()
```
