# Day 54: Hyperparameter Optimization
## Core Concepts & Theory

### Hyperparameter Tuning Challenges

**Problem:** Finding optimal hyperparameters is expensive and time-consuming

**Key Hyperparameters:**
- Learning rate
- Batch size
- Number of layers/heads
- Hidden dimensions
- Dropout rate
- Weight decay

### 1. Grid Search

**Concept:** Exhaustively search predefined parameter grid

**Example:**
```python
learning_rates = [1e-5, 1e-4, 1e-3]
batch_sizes = [16, 32, 64]
# Total: 3 × 3 = 9 combinations
```

**Pros:** Simple, guaranteed to find best in grid
**Cons:** Exponentially expensive, doesn't scale

### 2. Random Search

**Concept:** Sample random combinations

**Benefits:**
- More efficient than grid search
- Better coverage of parameter space
- Can run in parallel

**When to use:** High-dimensional search space

### 3. Bayesian Optimization

**Concept:** Build probabilistic model of objective function

**Process:**
```
1. Sample initial points
2. Fit Gaussian Process to results
3. Use acquisition function to select next point
4. Evaluate and update model
5. Repeat
```

**Acquisition Functions:**
- **Expected Improvement (EI):** Balance exploration/exploitation
- **Upper Confidence Bound (UCB):** Optimistic exploration
- **Probability of Improvement (PI):** Conservative

### 4. Population-Based Training (PBT)

**Concept:** Evolve population of models

**Process:**
```
1. Train population of models in parallel
2. Periodically evaluate performance
3. Replace worst performers with mutated copies of best
4. Continue training
```

**Benefits:**
- Adapts hyperparameters during training
- Finds better solutions than fixed hyperparameters

### 5. Learning Rate Schedules

**Constant:**
```python
lr = 1e-4  # Fixed throughout training
```

**Step Decay:**
```python
lr = initial_lr * (decay_rate ** (epoch // step_size))
```

**Cosine Annealing:**
```python
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * epoch / max_epochs))
```

**Warmup + Decay:**
```python
if epoch < warmup_epochs:
    lr = initial_lr * (epoch / warmup_epochs)
else:
    lr = initial_lr * decay_schedule(epoch - warmup_epochs)
```

### 6. Batch Size Scaling

**Linear Scaling Rule:**
```
If batch_size increases by k, multiply learning_rate by k
```

**Example:**
- Batch 32, LR 1e-4
- Batch 256 (8x), LR 8e-4

**Gradient Accumulation Alternative:**
```
Effective batch = mini_batch × accumulation_steps
Keep same LR
```

### 7. Early Stopping

**Concept:** Stop training when validation loss stops improving

**Implementation:**
```python
patience = 5  # Wait 5 epochs
best_loss = float('inf')
counter = 0

for epoch in range(max_epochs):
    val_loss = validate()
    
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        save_checkpoint()
    else:
        counter += 1
        if counter >= patience:
            break  # Early stop
```

### 8. Hyperband

**Concept:** Adaptive resource allocation

**Process:**
```
1. Sample many configurations
2. Train each for small budget
3. Keep top performers
4. Double budget for survivors
5. Repeat until one configuration remains
```

**Benefits:**
- Efficient use of resources
- Finds good configurations quickly

### 9. Neural Architecture Search (NAS)

**Concept:** Automatically search for optimal architecture

**Methods:**
- **Reinforcement Learning:** Train controller to generate architectures
- **Evolutionary:** Evolve population of architectures
- **Gradient-Based:** DARTS (differentiable architecture search)

**Challenge:** Extremely expensive (thousands of GPU hours)

### 10. AutoML Tools

**Optuna:**
```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    model = train(lr, batch_size)
    return model.val_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

**Ray Tune:**
```python
from ray import tune

config = {
    'lr': tune.loguniform(1e-5, 1e-3),
    'batch_size': tune.choice([16, 32, 64])
}

tune.run(train_fn, config=config, num_samples=100)
```

### Real-World Examples

**GPT-3 Training:**
- **Learning Rate:** 6e-5 (with warmup)
- **Batch Size:** 3.2M tokens
- **Schedule:** Cosine decay

**LLaMA:**
- **Learning Rate:** 3e-4
- **Batch Size:** 4M tokens
- **Schedule:** Cosine with warmup

### Summary

**Optimization Strategies:**
- **Grid Search:** Small search space (<10 params)
- **Random Search:** Medium search space
- **Bayesian Optimization:** Expensive evaluations
- **PBT:** Long training runs
- **Hyperband:** Many configurations, limited budget

**Best Practices:**
- Start with learning rate (most important)
- Use warmup for large models
- Scale batch size with learning rate
- Use early stopping to save time

### Next Steps
In the Deep Dive, we will implement Bayesian optimization, PBT, and learning rate schedules with complete code examples.
