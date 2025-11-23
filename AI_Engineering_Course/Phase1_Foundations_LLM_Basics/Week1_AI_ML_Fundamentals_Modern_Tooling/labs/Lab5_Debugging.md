# Lab 5: Debugging Neural Networks

## Objective
Neural Networks fail silently. They don't crash; they just don't learn.
We will build tools to detect **Dead ReLUs** and **Vanishing Gradients**.

## 1. The Monitor (`monitor.py`)

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Bad Initialization (Causes Dead ReLUs)
class BadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        
        # Initialize weights to large negative numbers
        nn.init.constant_(self.fc1.weight, -10.0) 
        nn.init.constant_(self.fc1.bias, -10.0)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 2. Hook Function
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# 3. Run
model = BadNet()
model.fc1.register_forward_hook(get_activation('fc1'))

x = torch.randn(1, 100)
output = model(x)

# 4. Visualize
plt.hist(activations['fc1'].flatten().numpy())
plt.title("FC1 Activations")
plt.show()

# Check for Dead ReLUs (All zeros)
if activations['fc1'].max() <= 0:
    print("WARNING: Dead ReLUs detected!")
```

## 2. Gradient Flow
Register a `register_full_backward_hook` to inspect gradients.
If gradients are all zero or extremely small, you have vanishing gradients.

## 3. Challenge
Fix the initialization in `BadNet` using `nn.init.kaiming_normal_` and verify the activations are no longer dead.

## 4. Submission
Submit the histogram of activations *after* fixing the initialization.
