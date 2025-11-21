# Day 6: Neural Networks Fundamentals

## Overview
Neural networks form the foundation of modern computer vision. This lesson covers the mathematical principles, architectures, and training procedures for feedforward neural networks.

## 1. Biological Motivation

### The Perceptron (1958)
**Mathematical model of a neuron:**
$$ y = f\left(\sum_{i=1}^n w_i x_i + b\right) = f(\mathbf{w}^T \mathbf{x} + b) $$

Where:
- $\mathbf{x} \in \mathbb{R}^n$: Input vector
- $\mathbf{w} \in \mathbb{R}^n$: Weight vector
- $b \in \mathbb{R}$: Bias term
- $f$: Activation function

### Activation Functions

**1. Sigmoid:**
$$ \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z)) $$

**Properties:**
- Range: $(0, 1)$
- Smooth, differentiable
- **Problem:** Vanishing gradients for $|z| > 3$

**2. Hyperbolic Tangent:**
$$ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z) $$

**Properties:**
- Range: $(-1, 1)$
- Zero-centered (better than sigmoid)
- Still suffers from vanishing gradients

**3. ReLU (Rectified Linear Unit):**
$$ \text{ReLU}(z) = \max(0, z), \quad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases} $$

**Advantages:**
- No vanishing gradient for $z > 0$
- Computationally efficient
- Sparse activation

**Problem:** Dead neurons ($z \leq 0$ always)

**4. Leaky ReLU:**
$$ \text{LeakyReLU}(z) = \max(\alpha z, z), \quad \alpha = 0.01 $$

**5. ELU (Exponential Linear Unit):**
$$ \text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases} $$

**6. Swish/SiLU:**
$$ \text{Swish}(z) = z \cdot \sigma(z) $$

**Modern choice:** ReLU for hidden layers, softmax for output (classification).

## 2. Multi-Layer Perceptron (MLP)

### Architecture
**Layer-wise computation:**
$$ \mathbf{h}^{(1)} = f^{(1)}(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}) $$
$$ \mathbf{h}^{(2)} = f^{(2)}(\mathbf{W}^{(2)} \mathbf{h}^{(1)} + \mathbf{b}^{(2)}) $$
$$ \vdots $$
$$ \mathbf{y} = f^{(L)}(\mathbf{W}^{(L)} \mathbf{h}^{(L-1)} + \mathbf{b}^{(L)}) $$

**Universal Approximation Theorem:**
A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$, under mild assumptions on the activation function.

### Implementation

```python
import numpy as np
from typing import List, Tuple

class Layer:
    """Fully connected layer."""
    
    def __init__(self, input_dim: int, output_dim: int, activation='relu'):
        # He initialization for ReLU
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        self.activation = activation
        
        # Cache for backpropagation
        self.cache = {}
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.cache['X'] = X
        Z = X @ self.W + self.b
        self.cache['Z'] = Z
        
        if self.activation == 'relu':
            A = np.maximum(0, Z)
        elif self.activation == 'sigmoid':
            A = 1 / (1 + np.exp(-Z))
        elif self.activation == 'tanh':
            A = np.tanh(Z)
        elif self.activation == 'softmax':
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        else:  # linear
            A = Z
        
        self.cache['A'] = A
        return A
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        """Backward pass."""
        Z = self.cache['Z']
        X = self.cache['X']
        m = X.shape[0]
        
        # Activation gradient
        if self.activation == 'relu':
            dZ = dA * (Z > 0)
        elif self.activation == 'sigmoid':
            A = self.cache['A']
            dZ = dA * A * (1 - A)
        elif self.activation == 'tanh':
            A = self.cache['A']
            dZ = dA * (1 - A**2)
        elif self.activation == 'softmax':
            # Assuming cross-entropy loss, dZ = A - Y
            dZ = dA
        else:  # linear
            dZ = dA
        
        # Parameter gradients
        self.dW = (X.T @ dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Input gradient
        dX = dZ @ self.W.T
        
        return dX

class MLP:
    """Multi-layer perceptron."""
    
    def __init__(self, layer_dims: List[int], activations: List[str]):
        """
        Args:
            layer_dims: [input_dim, hidden1, hidden2, ..., output_dim]
            activations: Activation for each layer
        """
        self.layers = []
        for i in range(len(layer_dims) - 1):
            layer = Layer(layer_dims[i], layer_dims[i+1], activations[i])
            self.layers.append(layer)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation."""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    def backward(self, dA: np.ndarray):
        """Backward propagation."""
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
    
    def update_parameters(self, learning_rate: float):
        """Gradient descent update."""
        for layer in self.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db
```

## 3. Loss Functions

### Classification

**Binary Cross-Entropy:**
$$ L = -\frac{1}{m} \sum_{i=1}^m [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)] $$

**Categorical Cross-Entropy:**
$$ L = -\frac{1}{m} \sum_{i=1}^m \sum_{c=1}^C y_{i,c} \log(\hat{y}_{i,c}) $$

**Gradient (softmax + cross-entropy):**
$$ \frac{\partial L}{\partial z_c} = \hat{y}_c - y_c $$

### Regression

**Mean Squared Error:**
$$ L = \frac{1}{2m} \sum_{i=1}^m (\hat{y}_i - y_i)^2 $$

**Gradient:**
$$ \frac{\partial L}{\partial \hat{y}} = \frac{1}{m}(\hat{y} - y) $$

```python
def cross_entropy_loss(Y_pred: np.ndarray, Y_true: np.ndarray) -> Tuple[float, np.ndarray]:
    """Categorical cross-entropy loss."""
    m = Y_true.shape[0]
    
    # Clip predictions to avoid log(0)
    Y_pred_clipped = np.clip(Y_pred, 1e-7, 1 - 1e-7)
    
    # Loss
    loss = -np.sum(Y_true * np.log(Y_pred_clipped)) / m
    
    # Gradient (for softmax output)
    dY = (Y_pred - Y_true) / m
    
    return loss, dY

def mse_loss(Y_pred: np.ndarray, Y_true: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean squared error loss."""
    m = Y_true.shape[0]
    
    loss = np.sum((Y_pred - Y_true)**2) / (2 * m)
    dY = (Y_pred - Y_true) / m
    
    return loss, dY
```

## 4. Training Procedure

### Mini-Batch Gradient Descent

```python
def train(model: MLP, X_train, Y_train, X_val, Y_val, 
          epochs=100, batch_size=32, learning_rate=0.01):
    """Training loop."""
    m = X_train.shape[0]
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]
        
        # Mini-batch training
        epoch_loss = 0
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            
            # Forward pass
            Y_pred = model.forward(X_batch)
            
            # Compute loss
            loss, dY = cross_entropy_loss(Y_pred, Y_batch)
            epoch_loss += loss
            
            # Backward pass
            model.backward(dY)
            
            # Update parameters
            model.update_parameters(learning_rate)
        
        # Validation
        Y_val_pred = model.forward(X_val)
        val_loss, _ = cross_entropy_loss(Y_val_pred, Y_val)
        val_acc = np.mean(np.argmax(Y_val_pred, axis=1) == np.argmax(Y_val, axis=1))
        
        history['train_loss'].append(epoch_loss / (m // batch_size))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {history['train_loss'][-1]:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
    
    return history
```

## 5. Example: MNIST Classification

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data.values, mnist.target.values.astype(int)

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode labels
Y = np.zeros((len(y), 10))
Y[np.arange(len(y)), y] = 1

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create model
model = MLP(
    layer_dims=[784, 256, 128, 10],
    activations=['relu', 'relu', 'softmax']
)

# Train
history = train(model, X_train, Y_train, X_test, Y_test, 
                epochs=50, batch_size=128, learning_rate=0.1)

# Evaluate
Y_pred = model.forward(X_test)
accuracy = np.mean(np.argmax(Y_pred, axis=1) == np.argmax(Y_test, axis=1))
print(f"Test Accuracy: {accuracy:.4f}")
```

## Summary
Neural networks use layered transformations with non-linear activations to learn complex mappings. Training via backpropagation and gradient descent optimizes parameters to minimize loss.

**Next:** Convolutional Neural Networks for spatial data.
