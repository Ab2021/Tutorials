# Day 10: PyTorch Lightning - Theory & Implementation

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: Structuring Code, Boilerplate Removal, and Scalability

## 1. Theoretical Foundation: The Abstraction Ladder

Raw PyTorch is flexible but verbose.
*   **Research Code**: Often messy, hard to reproduce, hard to scale (DDP requires rewrite).
*   **PyTorch Lightning (PL)**: A wrapper that organizes PyTorch code.
    *   **LightningModule**: The Model + Optimization logic.
    *   **Trainer**: The Loop + Hardware logic.

### Why use it?
1.  **Reproducibility**: Standard structure.
2.  **Scalability**: Switch from CPU to Multi-GPU to TPU by changing 1 flag.
3.  **Best Practices**: Auto-enables checkpointing, logging, mixed precision.

## 2. The `LightningModule`

It groups the 3 core components:
1.  **Model**: `__init__`, `forward`.
2.  **Optimizer**: `configure_optimizers`.
3.  **Loop**: `training_step`.

```python
import lightning as L
import torch.nn as nn
import torch.optim as optim

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(10, 1)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss) # Auto-logging
        return loss
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
```

## 3. The `Trainer`

The Trainer handles the engineering: Loops, Device placement, Saving, Logging.

```python
# Run on 4 GPUs with Mixed Precision
trainer = L.Trainer(
    max_epochs=10,
    accelerator="gpu",
    devices=4,
    strategy="ddp",
    precision="16-mixed"
)

trainer.fit(model, train_loader)
```

## 4. Callbacks

Inject logic into the loop (like Hooks, but higher level).
*   `ModelCheckpoint`: Save best model.
*   `EarlyStopping`: Stop if val loss plateaus.
*   `LearningRateMonitor`: Log LR.

```python
from lightning.pytorch.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=3)
trainer = L.Trainer(callbacks=[early_stop])
```

## 5. DataModules

Encapsulate data logic (Download, Split, Transform, DataLoader).

```python
class MNISTDataModule(L.LightningDataModule):
    def setup(self, stage):
        # Download and split
        self.mnist_train = ...
        self.mnist_val = ...
        
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)
```
