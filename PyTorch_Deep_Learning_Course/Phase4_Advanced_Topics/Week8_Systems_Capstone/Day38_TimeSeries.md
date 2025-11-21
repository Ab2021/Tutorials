# Day 38: Time Series & Transformers - Theory & Implementation

> **Phase**: 4 - Advanced Topics
> **Week**: 8 - Systems & Capstone
> **Topic**: Forecasting, Informer, and Time-Mixer

## 1. Theoretical Foundation: Beyond ARIMA

Classical (ARIMA) works for univariate, linear data.
Deep Learning (RNN/Transformer) works for multivariate, non-linear, massive datasets.
**Challenges**:
*   **Long Sequence**: History length $L$ can be 10k. Attention $O(L^2)$ is too slow.
*   **Non-Stationarity**: Distribution changes over time (Concept Drift).

## 2. Architectures

### Informer (ProbSparse Attention)
*   Observation: Attention matrix is sparse.
*   Selects top-k queries based on KL-divergence.
*   Reduces complexity to $O(L \log L)$.

### PatchTST (Patch Time Series Transformer)
*   **Patching**: Group time steps into patches (like ViT).
*   **Channel Independence**: Process each variable (e.g., Temperature, Pressure) independently. Share weights.
*   SOTA performance.

## 3. Implementation: Simple Transformer for Forecasting

```python
import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, pred_len):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.project_in = nn.Linear(input_dim, d_model)
        self.project_out = nn.Linear(d_model, pred_len * input_dim)
        
    def forward(self, src):
        # src: [Batch, Seq Len, Input Dim]
        src = self.project_in(src)
        output = self.transformer_encoder(src)
        # Take last token representation
        output = output[:, -1, :]
        prediction = self.project_out(output)
        return prediction # [Batch, Pred Len * Input Dim]
```

## 4. Decomposition (Trend + Seasonality)

Crucial for Time Series.
**Autoformer**:
*   Built-in Decomposition Block.
*   Separates series into $X_{trend} + X_{seasonal}$.
*   Transformer processes them separately.
