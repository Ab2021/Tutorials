# Lab 1: Modern ML Stack Setup & PyTorch Basics

## Objective
Set up a professional, reproducible Deep Learning environment and implement a basic training loop.
This lab establishes the foundation for the entire course.

## 1. Environment Setup (The "Right" Way)

We will use **Poetry** for dependency management and **Docker** for reproducibility.

### 1.1. Prerequisites
*   Install [Python 3.10+](https://www.python.org/downloads/)
*   Install [Poetry](https://python-poetry.org/docs/#installation)
*   Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Optional but recommended)
*   CUDA Toolkit (if you have an NVIDIA GPU)

### 1.2. Initialize Project
```bash
mkdir ai_eng_lab1
cd ai_eng_lab1
poetry init -n --name "ai_eng_lab1" --description "AI Engineering Course Lab 1"
```

### 1.3. Add Dependencies
We will use PyTorch (CPU version for compatibility, or CUDA if available).
```bash
# Add PyTorch (Check https://pytorch.org/get-started/locally/ for your specific command)
poetry add torch torchvision torchaudio

# Add Utilities
poetry add numpy pandas matplotlib scikit-learn tqdm tensorboard
```

### 1.4. Virtual Environment
Poetry creates a virtualenv automatically.
```bash
poetry shell
```

---

## 2. PyTorch Training Loop Implementation

We will train a simple Multi-Layer Perceptron (MLP) on the MNIST dataset.
**Goal:** Achieve >98% accuracy.

### 2.1. The Model (`model.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(SimpleMLP, self).__init__()
        # Layer 1: Input -> Hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) # Batch Norm for stability
        
        # Layer 2: Hidden -> Hidden
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Layer 3: Hidden -> Output
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.2) # Regularization

    def forward(self, x):
        # x shape: [batch_size, 1, 28, 28] -> Flatten to [batch_size, 784]
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x
```

### 2.2. The Dataset (`data.py`)

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST Mean/Std
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader
```

### 2.3. The Trainer (`train.py`)

This is where the magic happens. We will include **TensorBoard** logging.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import SimpleMLP
from data import get_dataloaders

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
LR = 0.001
BATCH_SIZE = 64

def train():
    # Setup
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    model = SimpleMLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter("runs/mnist_experiment_1")

    print(f"Training on {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update Progress Bar
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
            
            # Log to TensorBoard every 100 batches
            if i % 100 == 0:
                writer.add_scalar('training loss', running_loss / 100, epoch * len(train_loader) + i)
                running_loss = 0.0

        # Evaluation Phase
        evaluate(model, test_loader, writer, epoch)

    # Save Model
    torch.save(model.state_dict(), "mnist_mlp.pth")
    print("Model saved to mnist_mlp.pth")
    writer.close()

def evaluate(model, test_loader, writer, epoch):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    writer.add_scalar('test accuracy', accuracy, epoch)
    writer.add_scalar('test loss', avg_loss, epoch)

if __name__ == "__main__":
    train()
```

---

## 3. Running the Lab

1.  Run the training script:
    ```bash
    python train.py
    ```
2.  Watch the progress bar.
3.  Launch TensorBoard to visualize loss curves:
    ```bash
    tensorboard --logdir=runs
    ```
    Open `http://localhost:6006` in your browser.

## 4. Challenge (Optional)
*   **Convolutional Neural Network (CNN):** Modify `model.py` to use `nn.Conv2d` layers instead of `nn.Linear`. Does accuracy improve?
*   **Learning Rate Scheduler:** Add `torch.optim.lr_scheduler.StepLR` to decay the learning rate.

## 5. Submission
Submit your `runs` folder screenshot and your final `test accuracy`.
