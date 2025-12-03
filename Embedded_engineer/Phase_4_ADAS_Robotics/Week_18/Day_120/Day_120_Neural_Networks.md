# Day 120: Neural Networks Basics (Review)
## Phase 4: ADAS & Robotics Systems | Week 18: Deep Learning for Perception

---

> **📝 Day 120 Focus:**
> Classical Computer Vision (Edge Detection, HOG) is dead for high-level tasks. **Deep Learning** is the engine of modern ADAS. Today, we review the building blocks: Neurons, Layers, and the magic of **Backpropagation**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the architecture of a Multi-Layer Perceptron (MLP).
2.  **Differentiate** Activation Functions (ReLU, Sigmoid, Softmax).
3.  **Calculate** Loss (MSE, Cross-Entropy).
4.  **Implement** a simple Neural Network in **PyTorch**.
5.  **Train** the network to classify simple data.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Linear Algebra:** Matrix Multiplication ($W \cdot x + b$).
-   **Calculus:** Chain Rule (for Backprop).

### Hardware Requirements
-   **GPU (Optional):** NVIDIA CUDA helps, but CPU is fine for today.

### Software Stack
-   **Python:** `torch`, `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Artificial Neuron

$$ y = f(\sum w_i x_i + b) $$
-   **Weights ($w$):** Learnable parameters.
-   **Bias ($b$):** Shift.
-   **Activation ($f$):** Non-linearity. Without this, a deep network is just a single linear matrix.

### 🔹 Part 2: Activation Functions

1.  **Sigmoid:** $1 / (1 + e^{-x})$. Squeezes to $[0, 1]$. Good for probability. Bad for deep layers (Vanishing Gradient).
2.  **ReLU (Rectified Linear Unit):** $\max(0, x)$. The standard. Fast, solves vanishing gradient.
3.  **Softmax:** Converts a vector of logits into probabilities summing to 1. Used in output layer for classification.

### 🔹 Part 3: Training Loop

1.  **Forward Pass:** Compute prediction $\hat{y}$.
2.  **Loss Calculation:** Error $L = (y - \hat{y})^2$.
3.  **Backward Pass:** Compute gradients $\nabla L$ w.r.t weights (Chain Rule).
4.  **Optimizer Step:** Update weights $w = w - \eta \nabla L$ (Gradient Descent).

---

## 💻 Implementation: PyTorch MLP

**Scenario:**
-   Classify 2D points (Spiral Dataset).
-   Input: $(x, y)$.
-   Output: Class 0, 1, or 2.

### 🛠️ Setup
Create `week18_day120` and `pytorch_mlp.py`.

```bash
mkdir -p ~/ros2_ws/src/week18_day120
cd ~/ros2_ws/src/week18_day120
touch pytorch_mlp.py
```

### 👨‍💻 Code: The First Network

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Data Generation (Spiral) ---
def create_spiral_data(N, K):
    X = np.zeros((N*K, 2)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y

# --- 2. Model Definition ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # Input (2) -> Hidden (100) -> Hidden (100) -> Output (3)
        self.layers = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 3) # 3 Classes
        )
        
    def forward(self, x):
        return self.layers(x)

def main():
    # Setup Data
    N = 100 # points per class
    K = 3   # classes
    X_np, y_np = create_spiral_data(N, K)
    
    # Convert to Tensors
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    
    # Setup Model
    model = SimpleMLP()
    criterion = nn.CrossEntropyLoss() # Includes Softmax
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training Loop
    epochs = 1000
    loss_history = []
    
    print("Training...")
    for i in range(epochs):
        # 1. Forward
        logits = model(X)
        loss = criterion(logits, y)
        
        # 2. Backward
        optimizer.zero_grad() # Clear old gradients
        loss.backward()       # Compute new gradients
        optimizer.step()      # Update weights
        
        loss_history.append(loss.item())
        
        if i % 100 == 0:
            print(f"Epoch {i}: Loss {loss.item():.4f}")
            
    # --- Visualization ---
    # Decision Boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid_tensor)
        Z = torch.argmax(Z, dim=1).reshape(xx.shape)
        
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title("Decision Boundary")
    
    plt.subplot(1, 2, 2)
    plt.plot(loss_history)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Linear Trap

### Lab Objectives
1.  Run the script.
2.  **Observation:** The model learns the spiral shape perfectly.
3.  **Experiment:**
    -   Remove `nn.ReLU()` from the model definition.
    -   `self.layers = nn.Sequential(nn.Linear(2, 100), nn.Linear(100, 100), nn.Linear(100, 3))`
    -   **Result:** The decision boundary becomes a straight line (or 3 lines). The loss stays high.
    -   **Lesson:** A stack of linear layers is mathematically equivalent to a *single* linear layer ($W_3 W_2 W_1 = W_{total}$). You **need** non-linearity to learn curves.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Loss Not Decreasing
**Symptom:** Loss stays constant.
**Cause:** Learning Rate too high (overshooting) or too low (stuck).
**Solution:** Try `lr=0.001` or `lr=0.1`.

#### 2. Shape Errors
**Symptom:** `RuntimeError: size mismatch`.
**Cause:** Output of Layer 1 (100) doesn't match Input of Layer 2.
**Solution:** Check `nn.Linear(In, Out)`. Out of L1 must equal In of L2.

---

## ⚡ Optimization & Best Practices

### 1. GPU Acceleration
PyTorch makes this easy.
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X = X.to(device)
y = y.to(device)
```
-   For small MLPs, CPU is faster (less overhead). For CNNs, GPU is 50x faster.

### 2. Overfitting
If the model memorizes the training data but fails on new data.
-   **Solution:** Dropout (`nn.Dropout(0.5)`), Weight Decay (L2 Regularization), or Early Stopping.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the purpose of the Optimizer (Adam/SGD)?
    *   **A:** To update the weights based on the gradients to minimize the loss.
2.  **Q:** Why do we use Cross-Entropy for classification?
    *   **A:** It penalizes confident wrong answers heavily. MSE is better for regression (predicting a number).
3.  **Q:** What is a Tensor?
    *   **A:** A multi-dimensional array (like numpy array) that lives on the GPU and tracks gradients.

### Challenge Task
**Task:** Regression.
1.  Change data to $y = x^2$.
2.  Change Output Layer to `nn.Linear(100, 1)` (Predict 1 number).
3.  Change Loss to `nn.MSELoss()`.
4.  Train and plot the parabola.

---

## 📚 Further Reading & References
-   [PyTorch Blitz Tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
-   [Neural Networks and Deep Learning (Nielsen)](http://neuralnetworksanddeeplearning.com/)

---

**Day 120 Complete** | Phase 4: ADAS & Robotics Systems | Week 18: Deep Learning for Perception
