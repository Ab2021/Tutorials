# Day 27.2: Quantum Neural Networks - A Practical Implementation

## Introduction: Merging Quantum Circuits and Neural Networks

Having established the fundamentals of quantum computing, we now arrive at the exciting intersection of quantum mechanics and machine learning: the **Quantum Neural Network (QNN)**. A QNN is a hybrid quantum-classical model where a classical optimization algorithm is used to train the parameters of a quantum circuit. The goal is to leverage quantum phenomena like superposition and entanglement to potentially achieve advantages over purely classical neural networks.

These models are often called **Variational Quantum Algorithms (VQAs)** or **Parameterized Quantum Circuits (PQCs)**. The core idea is to use a quantum computer as a co-processor. A classical computer handles the data, passes parameters to the quantum circuit, and updates those parameters based on a loss function calculated from the quantum measurement outcomes.

This guide will provide a hands-on walkthrough of building and training a QNN for a simple classification task using PennyLane and PyTorch.

**Today's Learning Objectives:**

1.  **Understand the QNN Architecture:** Grasp the hybrid quantum-classical workflow of a typical QNN.
2.  **Build a Data-Driven Quantum Circuit:** Learn how to combine data encoding and a parameterized model circuit into a single QNode.
3.  **Integrate with PyTorch:** Seamlessly convert a PennyLane QNode into a `torch.nn.Module` that can be used within a standard PyTorch training loop.
4.  **Train a QNN:** Implement a full training pipeline, including defining a loss function, an optimizer, and a loop to train the quantum circuit's parameters.
5.  **Evaluate and Visualize Results:** Assess the performance of the trained QNN and visualize the decision boundary it has learned.

---

## Part 1: The Architecture of a Quantum Neural Network

A QNN is a hybrid system. The workflow looks like this:

1.  **Classical Data Input:** Start with a classical dataset (e.g., images, feature vectors).
2.  **Quantum Circuit (The Model):**
    *   **Data Encoding:** A fixed (non-trainable) part of the circuit encodes the classical data point into a quantum state. We'll use Angle Encoding for this.
    *   **Parameterized Model:** A trainable part of the circuit, composed of parameterized gates (e.g., rotations), processes the quantum state. These parameters are the "weights" of our QNN.
    *   **Measurement:** The circuit is measured, producing a classical output (e.g., an expectation value).
3.  **Classical Post-Processing & Loss Calculation:**
    *   The classical output from the quantum circuit might be post-processed (e.g., scaled or shifted).
    *   A classical loss function (e.g., Mean Squared Error, Cross-Entropy) compares the model's output to the true label.
4.  **Classical Optimization:**
    *   An optimizer (e.g., Adam, SGD) running on the classical computer calculates the gradients of the loss function with respect to the circuit parameters.
    *   The optimizer updates the parameters.
5.  **Iteration:** Repeat the process until the model converges.

![QNN Diagram](https://pennylane.ai/qml/_images/qnn_torch.png)

---

## Part 2: Building the Quantum Circuit for Classification

We will design a quantum circuit that takes a 2D feature vector as input and outputs a single expectation value, which we will interpret as a classification score.

Our circuit will have two main components:
*   **State Preparation (Encoding):** We'll use `qml.AngleEmbedding` to encode the 2D input data.
*   **Model Layers:** We'll use a simple, repeating layer of parameterized rotations and entangling CNOT gates. This is often called a "variational ansatz."

### 2.1. Code: Defining the Quantum Components

```python
import pennylane as qml
from pennylane import numpy as np

print("--- Part 2: Defining the Quantum Circuit ---")

# Define the quantum device
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# Define a single layer of our variational circuit
def layer(weights):
    # weights is a (n_qubits, 3) array
    for i in range(n_qubits):
        qml.RX(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i)
        qml.RZ(weights[i, 2], wires=i)
    qml.CNOT(wires=[0, 1])

# Define the full quantum circuit (QNode)
@qml.qnode(dev)
def quantum_circuit(weights, x):
    # weights is a (n_layers, n_qubits, 3) array
    # x is the 2D input feature vector
    
    # 1. State Preparation / Data Encoding
    qml.AngleEmbedding(x, wires=range(n_qubits))
    
    # 2. Parameterized Model Layers
    for w in weights:
        layer(w)
        
    # 3. Measurement
    return qml.expval(qml.PauliZ(0))

# --- Test the circuit with dummy data ---
n_layers = 2
weights_init = 0.01 * np.random.randn(n_layers, n_qubits, 3)
x_init = np.array([0.5, 0.3])

# Draw the circuit
print("Circuit Diagram:")
print(qml.draw(quantum_circuit)(weights_init, x_init))

# Run the circuit
output = quantum_circuit(weights_init, x_init)
print(f"\nInitial output for dummy data: {output:.4f}")
```

---

## Part 3: Integrating with PyTorch

PennyLane's `qml.qnn.TorchLayer` makes it incredibly easy to convert our QNode into a PyTorch layer. This allows us to use PyTorch's automatic differentiation and optimizers to train our quantum circuit.

### 3.1. Code: Creating a PyTorch QNN Layer

```python
import torch
import torch.nn as nn

print("\n--- Part 3: Integrating with PyTorch ---")

# Define the weight shapes for our circuit
weight_shapes = {"weights": (n_layers, n_qubits, 3)}

# Create the PyTorch layer
# The QNode is passed, along with the shapes of the trainable weights
qnn_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# Now, let's wrap it in a standard nn.Sequential model
# We can add classical pre- or post-processing layers here if needed
model = nn.Sequential(qnn_layer)

print("PyTorch Model:")
print(model)

# --- Test the forward pass with PyTorch tensors ---
# Note that inputs must now be torch.Tensor
x_torch = torch.tensor([0.5, 0.3])
output_torch = model(x_torch)

print(f"\nOutput from PyTorch model: {output_torch.item():.4f}")
```

---

## Part 4: Training the QNN

Now for the main event: training our hybrid model on a real task. We will generate a synthetic dataset of two concentric circles, a classic non-linearly separable problem.

### 4.1. Code: Data Preparation, Training Loop, and Loss

```python
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

print("\n--- Part 4: Training the QNN ---")

# --- 1. Generate Data ---
X, y = make_circles(n_samples=200, factor=0.5, noise=0.1, random_state=42)
# Rescale labels from {0, 1} to {-1, 1} to match the Pauli-Z expectation value range
y_rescaled = torch.tensor(y * 2 - 1, dtype=torch.float32)
X_torch = torch.tensor(X, dtype=torch.float32)

# --- 2. Define Loss and Optimizer ---
# We'll use a simple square loss
def loss_fn(predictions, targets):
    return torch.mean((predictions - targets) ** 2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# --- 3. Training Loop ---
epochs = 20
batch_size = 20
losses = []

for epoch in range(epochs):
    for i in range(0, len(X), batch_size):
        # Get batch
        X_batch = X_torch[i : i + batch_size]
        y_batch = y_rescaled[i : i + batch_size]
        
        # Forward pass
        predictions = model(X_batch).squeeze() # Squeeze to remove extra dim
        
        # Calculate loss
        loss = loss_fn(predictions, y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    losses.append(loss.item())
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# --- 4. Plot Loss ---
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
```

---

## Part 5: Evaluation and Visualization

After training, we need to see how well our QNN learned to classify the data. We can do this by calculating its accuracy and, more intuitively, by plotting its decision boundary.

### 5.1. Code: Calculating Accuracy and Plotting the Boundary

```python
print("\n--- Part 5: Evaluation and Visualization ---")

# --- 1. Calculate Accuracy ---
with torch.no_grad():
    raw_predictions = model(X_torch).squeeze()
    # Convert expectation values (-1 to 1) to class labels (0 or 1)
    final_predictions = (torch.sign(raw_predictions) + 1) / 2
    accuracy = torch.mean((final_predictions == torch.tensor(y, dtype=torch.float32)).float())
    print(f"Final Accuracy: {accuracy.item() * 100:.2f}%")

# --- 2. Plot Decision Boundary ---
def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(8, 8))
    # Plot data points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="red", label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="Class 1")

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    # Get predictions for each point in the mesh
    grid_data = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid_data).reshape(xx.shape)
    
    # Plot the contour
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap=plt.cm.RdBu, alpha=0.5)
    plt.title("QNN Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

plot_decision_boundary(model, X, y)
```

## Conclusion

In this guide, we successfully built and trained a Quantum Neural Network from scratch. We saw how to define a quantum circuit as our model, seamlessly integrate it into a PyTorch workflow, and train it using standard deep learning tools. The final decision boundary plot shows that our simple, two-qubit model was able to learn the non-linear relationship in the data.

**Key Takeaways:**

1.  **QNNs are Hybrid:** They combine the strengths of classical computers (data handling, optimization) and quantum computers (processing in high-dimensional spaces).
2.  **PennyLane is the Glue:** Libraries like PennyLane provide the essential bridge, allowing quantum circuits to be differentiable and pluggable into ML frameworks.
3.  **The Workflow is Familiar:** The training loop for a QNN is remarkably similar to that of a classical neural network, making it accessible to ML practitioners.
4.  **Data Encoding Matters:** The `AngleEmbedding` layer was our bridge from classical vectors to quantum states.
5.  **Potential for Advantage:** While this was a toy problem, it demonstrates the core mechanism by which quantum circuits can act as powerful, non-linear function approximators, just like their classical counterparts.

This hands-on experience forms the foundation for tackling more complex problems and exploring more advanced QNN architectures.

## Self-Assessment Questions

1.  **Hybrid Model:** What are the respective roles of the classical computer and the quantum computer in the QNN training loop?
2.  **Parameters:** In our `quantum_circuit`, what are the trainable parameters, and what is the non-trainable input?
3.  **TorchLayer:** What is the primary purpose of the `qml.qnn.TorchLayer` wrapper?
4.  **Loss Function:** Why did we rescale the labels `y` from `{0, 1}` to `{-1, 1}` before calculating the loss?
5.  **Expressivity:** How could you increase the complexity or "expressivity" of our QNN model to potentially solve a more difficult classification problem?
