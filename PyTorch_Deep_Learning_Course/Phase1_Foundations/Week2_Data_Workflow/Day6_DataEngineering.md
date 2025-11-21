# Day 6: Data Engineering - Theory & Implementation

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: Datasets, Dataloaders, and Efficient I/O

## 1. Theoretical Foundation: The ETL Pipeline

Deep Learning is often I/O bound, not Compute bound.
The **ETL (Extract, Transform, Load)** pipeline must feed the GPU faster than the GPU can consume data.

### Map-style vs Iterable-style Datasets
1.  **Map-style (`__getitem__`)**:
    *   Random access ($O(1)$).
    *   Requires knowing the length (`__len__`).
    *   Good for: Images, Text Classification (where data fits in memory or random seek is fast).
2.  **Iterable-style (`__iter__`)**:
    *   Sequential access.
    *   Good for: Streaming data (Logs, Audio), Massive datasets (Petabytes) stored in sharded files (WebDataset).

### Multiprocessing & GIL
Python's Global Interpreter Lock (GIL) prevents true parallelism on threads.
PyTorch `DataLoader` uses `multiprocessing` to spawn separate processes (workers).
Each worker has its own copy of the dataset and Python interpreter.

## 2. Implementation: Custom Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

class MyImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # 1. Read Metadata
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        
        # 2. Load Data (Lazy Loading)
        image = Image.open(img_path).convert('RGB')
        
        # 3. Transform (Augmentation)
        if self.transform:
            image = self.transform(image)
            
        return image, label
```

## 3. The DataLoader

The `DataLoader` orchestrates the fetching.
*   **Batching**: Stacking N samples into a tensor.
*   **Shuffling**: Critical for SGD.
*   **Collate Function**: How to merge samples.

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # Parallel workers
    pin_memory=True,    # Fast GPU transfer
    drop_last=True      # Drop incomplete last batch
)
```

## 4. Custom Collate Function

Default collate handles tensors and numbers. Fails on variable-length data or custom objects.

```python
def my_collate(batch):
    # batch is a list of tuples [(img1, lbl1), (img2, lbl2), ...]
    imgs = [item[0] for item in batch]
    lbls = [item[1] for item in batch]
    
    # Custom logic: Pad images to max size in batch
    imgs = pad_sequence(imgs) 
    
    return torch.stack(imgs), torch.tensor(lbls)
```

## 5. Handling Imbalanced Data

If Class A has 1000 samples and Class B has 10 samples, the model will just predict A.
**WeightedRandomSampler**: Samples Class B more frequently.

```python
weights = [0.1, 0.9] # Probability for each sample
sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights))
loader = DataLoader(dataset, sampler=sampler) # Shuffle must be False
```
