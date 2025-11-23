# Lab 2: Scaling Laws Experiment

## Objective
Verify the **Chinchilla Scaling Laws**.
Does Loss decrease predictably as we increase Model Size?
We will train 3 models: Small, Medium, Large.

## 1. The Experiment (`scaling.py`)

We will reuse the training logic but wrap it in a loop.

```python
# ... Imports ...

SIZES = [
    {"name": "Small",  "n_embd": 128, "n_layer": 2}, # ~1M params
    {"name": "Medium", "n_embd": 256, "n_layer": 4}, # ~5M params
    {"name": "Large",  "n_embd": 512, "n_layer": 6}, # ~20M params
]

for config_dict in SIZES:
    print(f"Training {config_dict['name']} Model...")
    
    config = AutoConfig.from_pretrained("gpt2", **config_dict)
    model = AutoModelForCausalLM.from_config(config)
    
    args = TrainingArguments(
        output_dir=f"./scaling-{config_dict['name']}",
        max_steps=500, # Fixed steps for fair comparison
        learning_rate=1e-3,
        report_to="wandb",
        run_name=f"scaling-{config_dict['name']}"
    )
    
    trainer = Trainer(model=model, args=args, train_dataset=tokenized_datasets["train"], data_collator=data_collator)
    trainer.train()
```

## 2. Analysis

1.  Run the script.
2.  Go to WandB.
3.  Create a **Line Plot**:
    *   X-Axis: Step
    *   Y-Axis: Eval Loss
    *   Group By: Run Name
4.  You should see distinct curves. The "Large" model should have the lowest loss curve (assuming sufficient data).

## 3. Challenge
*   **Compute Optimal:** If you have a fixed compute budget (e.g., 1 hour of GPU), which model gives the best loss? A large model trained for few steps, or a small model trained for many steps?

## 4. Submission
Submit the WandB comparison plot.
