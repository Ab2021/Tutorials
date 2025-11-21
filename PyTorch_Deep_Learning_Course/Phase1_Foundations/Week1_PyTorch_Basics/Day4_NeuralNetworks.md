# Day 4: Neural Network Architecture - Theory & Implementation

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: `nn.Module`, Layers, and Object-Oriented Deep Learning

## 1. Theoretical Foundation: The Layer Abstraction

### The Neuron (Perceptron)
$y = \sigma(w^T x + b)$
*   $w$: Weights (Synaptic strength).
*   $b$: Bias (Activation threshold).
*   $\sigma$: Non-linearity (Firing rate).

### The Layer
A collection of neurons operating in parallel.
Mathematically, it's an Affine Transformation followed by a Non-linearity.
$$ Y = \sigma(X W^T + b) $$

### The Deep Network
Composition of layers.
$$ F(x) = f_N(...f_2(f_1(x))...) $$
**Universal Approximation Theorem**: A network with a single hidden layer (and sufficient width) can approximate any continuous function.
**Depth Efficiency**: Deep networks requires exponentially fewer parameters than shallow networks to represent complex hierarchical functions.

## 2. PyTorch `nn.Module`

`nn.Module` is the base class for all neural network modules. It provides:
1.  **Parameter Tracking**: Automatically finds all `nn.Parameter` attributes.
2.  **Device Management**: `.to(device)` moves all parameters recursively.
3.  **State Management**: `.state_dict()` for saving/loading.
4.  **Hooks**: For inspecting internals.

```python
import torch
import torch.nn as nn

class MyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Define Layers (State)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Define Computation Graph (Behavior)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MyMLP(784, 128, 10)
```

## 3. Common Layers

### Linear (Dense / Fully Connected)
`nn.Linear(in_features, out_features)`
*   Matrix Multiplication.
*   Parameters: Weights $(out, in)$ and Bias $(out)$.

### Convolutional
`nn.Conv2d(in_channels, out_channels, kernel_size)`
*   Sliding window operation.
*   Preserves spatial structure.
*   **Weight Sharing**: Same kernel applied everywhere. Translation Equivariance.

### Recurrent
`nn.LSTM`, `nn.GRU`
*   Processes sequences. Maintains hidden state.

### Normalization
`nn.BatchNorm2d`, `nn.LayerNorm`
*   Stabilizes training by standardizing activations (Mean 0, Var 1).
*   Smoothes the loss landscape.

### Dropout
`nn.Dropout(p=0.5)`
*   Regularization. Randomly zeroes out neurons during training.
*   Prevents co-adaptation of features.

## 4. `nn.Sequential` vs Subclassing

### `nn.Sequential`
Quick way to stack layers.
```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)
```
*   *Pros*: Concise.
*   *Cons*: Linear topology only. No skip connections or multiple inputs.

### Subclassing (Recommended)
Full control over `forward`.
```python
class ResBlock(nn.Module):
    def forward(self, x):
        return x + self.conv(x) # Skip connection
```

## 5. Parameter Initialization

How you initialize weights matters immensely.
*   **Zero**: Bad. Symmetry problem. All neurons learn same thing.
*   **Random Normal**: Better, but variance can explode/vanish.
*   **Xavier (Glorot)**: Keeps variance constant across layers. Good for Tanh/Sigmoid.
*   **Kaiming (He)**: Good for ReLU.

```python
# PyTorch default is usually Kaiming Uniform for Linear/Conv
nn.init.kaiming_normal_(model.fc1.weight)
```
