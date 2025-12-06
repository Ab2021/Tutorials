# Day 129: The Social Network for Models: Weights & Biases
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 19: Experiment Tracking & Model Registry

---

> **🎯 Focus Area:** MLflow is great for artifacts, but **Weights & Biases (W&B)** shines at Visualization and Collaboration. It turns your loss curves into shareable reports that your manager can actually understand.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Instrument** a PyTorch training loop with `wandb.init()` and `wandb.log()`.
2.  **Log** rich media (Images, Audio, 3D Point Clouds) to the dashboard.
3.  **Use** W&B Tables to debug model predictions interactively.
4.  **Execute** a Hyperparameter Sweep using the W&B Agent.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Internet Access (to reach `api.wandb.ai`).

### Software Environment
- `pip install wandb`.
- A free account at [wandb.ai](https://wandb.ai).

---

## 📖 Theoretical Foundation

### 1. MLflow vs W&B
| Feature | MLflow | Weights & Biases |
|:---|:---|:---|
| **Hosting** | Self-Hosted (Open Source) | SaaS (Proprietary/Enterprise) |
| **Focus** | Artifact Management & Lifecycle | Visualization & Collaboration |
| **Rich Media** | Basic (PNGs) | Advanced (Video, Audio, 3D, HTML) |
| **Sweeps** | Basic | Advanced (BayesOpt built-in) |

### 2. The Project Structure
*   **Project:** Top level folder.
*   **Run:** Single execution.
*   **Reports:** Markdown documents embedding live charts.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: First Run

#### 📁 `src/04_wandb_demo.py`
```python
import wandb
import random
import torch
import numpy as np

# 1. Login (Interactive or Key)
# wandb.login(key="...")

# 2. Init Experiment
# config capture hyperparameters
run = wandb.init(
    project="Phase6_Review",
    job_type="training",
    config={
        "learning_rate": 0.01,
        "batch_size": 32,
        "architecture": "CNN"
    }
)
config = wandb.config

# 3. Simulate Training Loop
epochs = 10
offset = random.random() / 5

for epoch in range(epochs):
    # Fake metrics
    acc = 1 - 2 ** -epoch - random.random() * offset
    loss = 2 ** -epoch + random.random() * offset
    
    # Log to Cloud
    wandb.log({"accuracy": acc, "loss": loss, "epoch": epoch})
    
# 4. Log Rich Media
# Image with Bounding Box
image = np.random.randint(0, 255, (28, 28, 3))
wandb.log({"examples": [wandb.Image(image, caption="Random Noise")]})

# 5. Finish
wandb.finish()
```

### 👨‍💻 Core Implementation: W&B Tables

Tables allow SQL-like querying of your test set predictions.

```python
# Create a Table
table = wandb.Table(columns=["image", "true_label", "pred_label", "confidence"])

# Add Data
for i in range(10):
    img = np.zeros((28,28))
    table.add_data(wandb.Image(img), "cat", "dog", 0.85)

# Log Table
wandb.log({"predictions": table})
```

**In Dashboard:** You can now group by "true_label" and filter where "pred_label != true_label" to see exactly which images confused the model.

---

## 🔬 Lab Exercise: "The Sweep"

### Task
Find best LR.
1.  Define `sweep.yaml`:
    ```yaml
    program: src/04_wandb_demo.py
    method: bayes
    metric:
      name: loss
      goal: minimize
    parameters:
      learning_rate:
        min: 0.0001
        max: 0.1
    ```
2.  Init Sweep: `wandb sweep sweep.yaml`. Returns a `SWEEP_ID`.
3.  Start Agent: `wandb agent <SWEEP_ID>`.
4.  **Observation:** The Agent pulls config from Cloud, runs your script, uploads results. The Cloud Controller (BayesOpt) decides the next config.

---

## 📝 Daily Summary

### Key Takeaways
1.  **System Metrics:** W&B automatically logs GPU usage, Memory, and Temperature. This helps identify bottlenecks (e.g., "GPU at 0% during data loading").
2.  **Alerts:** You can configure Slack/Email alerts if "Loss explodes" or "Run crashes".
3.  **Collaboration:** You can send a URL to your colleague: "Hey, check out Run #45, the validation loss is weird."

### API Summary
```python
wandb.init(project="...")
wandb.log({...})
wandb.watch(model) # Logs gradients
```

---

**Day 129 Complete** ✅

*Next: Day 130 - Artifacts & Storage - Handling Big Files.*
