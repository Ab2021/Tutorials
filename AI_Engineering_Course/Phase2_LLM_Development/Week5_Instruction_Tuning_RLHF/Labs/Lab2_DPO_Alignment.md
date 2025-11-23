# Lab 2: DPO Alignment

## Objective
Align your fine-tuned model using **Direct Preference Optimization (DPO)**.
DPO is more stable than RLHF (PPO) and requires no Reward Model.

## 1. Dataset
We need a dataset with columns: `prompt`, `chosen`, `rejected`.
We will use `HuggingFaceH4/ultrafeedback_binarized`.

## 2. The DPO Script (`dpo_train.py`)

```python
from trl import DPOTrainer
from peft import LoraConfig

# ... Load Model & Tokenizer (Same as Lab 1) ...

# DPO Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    logging_steps=10,
    output_dir="dpo_results",
)

# DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None, # DPO uses the model itself as reference if None (with LoRA)
    args=training_args,
    beta=0.1, # The KL penalty strength
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

dpo_trainer.train()
```

## 3. Analysis
Compare the output of the SFT model (Lab 1) vs the DPO model (Lab 2) on a controversial prompt.
*   Prompt: "How do I steal a car?"
*   SFT: Might refuse or might help.
*   DPO: Should politely refuse (if aligned for safety).

## 4. Submission
Submit a comparison of outputs between SFT and DPO models.
