# Day 112: Week 16 Review & Project - AutoBERT
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 16: Ray Tune - Hyperparameter Optimization

---

> **🎯 Focus Area:** We combine Ray Train and Ray Tune to perform a "Grandmaster-level" search for the perfect BERT hyperparameters, automating hours of manual trial-and-error.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Design** a Search Space for Transformer Fine-tuning.
2.  **Integrate** Hugging Face `Trainer` with Ray Tune (via `RayTrainable` or Callback).
3.  **Execute** an ASHA-scheduled search on a GPU cluster.
4.  **Export** the best model.

---

## 📚 Week 16 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 106 | HPO Basics | "Random is better than Grid." |
| 107 | Tune API | "I can pause and resume my experiments." |
| 108 | BayesOpt | "The algorithm learns from past failures." |
| 109 | Schedulers | "ASHA killed 80% of my bad trials in 5 minutes." |
| 110 | Train Integration | "Each trial is a full distributed job." |
| 111 | Analysis | "Loss decreases only when Batch Size > 16." |

---

## 🏗️ Final Project: "AutoBERT"

### Scenario
Fine-tune `bert-base-uncased` on IMDB Sentiment Analysis.
*   **Search Space:**
    *   Learning Rate: $1e-5$ to $1e-4$.
    *   Batch Size: 16, 32.
    *   Weight Decay: $0.0$ to $0.3$.
*   **Constraint:** Max 4 Concurrent Trials.

### Step 1: The Train Loop

We use standard PyTorch logic, wrapped in Ray Train.

#### 📁 `project/autobert.py`
```python
import ray
from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, Checkpoint
from ray.tune.schedulers import ASHAScheduler
import transformers
import evaluate
import torch
import numpy as np
import tempfile
import os

def train_func(config):
    # 1. Unpack Config
    lr = config["lr"]
    wd = config["wd"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    
    # 2. Data
    # In real life, use Ray Data. Here, HF Datasets.
    from datasets import load_dataset
    dataset = load_dataset("imdb")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize(e):
        return tokenizer(e["text"], padding="max_length", truncation=True)
        
    encoded_dataset = dataset.map(tokenize, batched=True)
    # Subset for speed
    train_ds = encoded_dataset["train"].shuffle().select(range(1000))
    eval_ds = encoded_dataset["test"].shuffle().select(range(500))

    # 3. Model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    # 4. HF Trainer
    # Ray provides a callback to report metrics automatically!
    from ray.train.huggingface.transformers import RayTrainReportCallback
    
    training_args = transformers.TrainingArguments(
        output_dir=".",
        learning_rate=lr,
        weight_decay=wd,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none" # Disable WandB internal, let Ray handle it
    )
    
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=lambda p: {"accuracy": (np.argmax(p.predictions, axis=1) == p.label_ids).mean()},
        callbacks=[RayTrainReportCallback()]
    )
    
    trainer.train()

# Driver
if __name__ == "__main__":
    ray.init()
    
    # ASHA Scheduler
    scheduler = ASHAScheduler(
        max_t=5, # 5 Epochs max
        grace_period=1,
        reduction_factor=2
    )
    
    # Tuner
    tuner = tune.Tuner(
        TorchTrainer(
            train_loop_per_worker=train_func,
            scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
        ),
        param_space={
            "train_loop_config": {
                "lr": tune.loguniform(1e-5, 1e-4),
                "wd": tune.uniform(0.0, 0.3),
                "batch_size": tune.choice([8, 16]),
                "epochs": 5
            }
        },
        tune_config=tune.TuneConfig(
            metric="eval_loss",
            mode="min",
            num_samples=8,
            scheduler=scheduler
        )
    )
    
    results = tuner.fit()
    print("Best Hyperparameters:", results.get_best_result().config)
```

### Step 2: The Scheduler Effect
*   **Epoch 1:** 8 trials running.
*   **ASHA Check:** Bottom 4 trials are killed (Bad LR usually).
*   **Epoch 2:** 4 trials running.
*   **ASHA Check:** Bottom 2 killed.
*   **Epoch 5:** The best trial finishes.

---

## 🔬 Lab Exercise: "Resource Contention"

### Task
Simulate heavy contention.
1.  Set `ScalingConfig(num_workers=4)`. (Each trial takes 4 GPUs).
2.  If you have 4 GPUs total, only 1 trial runs at a time.
3.  ASHA is less effective here because it needs *concurrency* to compare trials. If Trial B starts only after Trial A finishes, ASHA has no distribution to compare against initially.
4.  **Insight:** For HPO, smaller distributed jobs (1-2 GPUs) run in parallel are often better than huge sequential jobs.

---

## 📝 Success Criteria
1.  **Metric Flow:** You see `eval_loss` and `accuracy` in the final table.
2.  **Early Stopping:** You confirm that some trials report status `TERMINATED` after only 1 or 2 iterations.
3.  **Best Model:** You can print the config of the winner.

---

**Week 16 Complete** ✅
**Phase 6C Status: 66% Complete**

*Next Week: Week 17 - Ray Serve - High Performance Model Serving.*
