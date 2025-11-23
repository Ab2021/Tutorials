# Lab 3: Positional Encodings Visualizer

## Objective
Transformers process tokens in parallel, so they have no notion of order.
We inject **Positional Encodings**.
We will visualize the "Sinusoidal" encodings from the original paper.

## 1. The Math
$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$

## 2. The Visualizer (`pe_viz.py`)

```python
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns

def get_sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Generate
max_len = 100
d_model = 512
pe = get_sinusoidal_pe(max_len, d_model)

# Plot
plt.figure(figsize=(12, 8))
sns.heatmap(pe.numpy(), cmap="viridis")
plt.title("Sinusoidal Positional Encodings")
plt.xlabel("Embedding Dimension")
plt.ylabel("Sequence Position")
plt.show()
```

## 3. Analysis
Notice the wave-like patterns.
*   Low dimensions (left) have high frequency.
*   High dimensions (right) have low frequency.
This allows the model to learn relative positions easily.

## 4. Submission
Submit the heatmap.
