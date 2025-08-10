# Day 1.2: Mathematical Foundations - A Practical Implementation Guide

## Introduction: Math in Code

In the previous section, we built a neural network and saw the mechanics of training. Underpinning all of that is a set of core mathematical concepts. This guide will bridge the gap between abstract mathematical theory and its practical application in deep learning code. We won't be solving complex equations on paper; instead, we'll be using PyTorch to see how these mathematical ideas come to life.

Understanding this connection is crucial. When you know what the code is doing mathematically, you can debug more effectively, design better models, and read research papers with greater confidence.

**Today's Learning Objectives:**

1.  **Linear Algebra in Action:**
    *   Represent data using PyTorch Tensors.
    *   Understand how a neural network layer is just a series of linear algebra operations (dot products, matrix multiplications).
    *   Visualize vector and matrix operations.

2.  **Calculus in Action (The Magic of Autograd):**
    *   Understand what a gradient is intuitively.
    *   Use PyTorch's `autograd` to automatically compute gradients for any function.
    *   Connect the concept of a gradient to the process of model training (gradient descent).

3.  **Probability & Statistics in Action:**
    *   Understand how loss functions are rooted in probability (e.g., MSE and the Gaussian assumption).
    *   See how activation functions like Softmax produce probability distributions.
    *   Initialize model weights using different statistical distributions.

---

## Part 1: Linear Algebra - The Language of Data

At its core, deep learning is about transforming tensors of numbers. Linear algebra is the language we use to describe these transformations.

*   **Scalar:** A single number (e.g., `5`).
*   **Vector:** A 1D array of numbers (e.g., a single data point with multiple features, or the weights of a single neuron).
*   **Matrix:** A 2D array of numbers (e.g., a batch of data, or the weights of a neural network layer).
*   **Tensor:** A generalization to N dimensions.

### 1.1. PyTorch Tensors: Our Primary Tool

PyTorch's `Tensor` is the fundamental data structure, similar to a NumPy array but with two key superpowers: GPU acceleration and automatic differentiation (`autograd`).

```python
import torch

# --- Creating Tensors ---
# A scalar
scalar = torch.tensor(42)
print(f"Scalar: {scalar}")
print(f"Scalar dimensions: {scalar.dim()}\n")

# A vector
vector = torch.tensor([1.0, 2.0, 3.0])
print(f"Vector: {vector}")
print(f"Vector dimensions: {vector.dim()}\n")

# A matrix
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Matrix:\n{matrix}")
print(f"Matrix dimensions: {matrix.dim()}\n")

# A 3D tensor
tensor_3d = torch.randn(2, 3, 4) # Shape: (depth, rows, columns)
print(f"3D Tensor (shape {tensor_3d.shape}):\n{tensor_3d}")
print(f"3D Tensor dimensions: {tensor_3d.dim()}\n")
```

### 1.2. A Neural Network Layer as a Matrix Operation

Let's revisit the linear part of a neuron: `z = w1*x1 + w2*x2 + ... + b`.

If we have a *batch* of data and a *layer* of neurons, this becomes a matrix multiplication.

*   Let `X` be our input data matrix, where each row is a data sample.
*   Let `W` be our weight matrix, where each column corresponds to a neuron's weights.
*   Let `b` be our bias vector.

The forward pass for an entire layer is just: `Z = X @ W + b`

*   The `@` symbol in PyTorch is used for matrix multiplication.

Let's see this in code.

```python
import torch
import torch.nn as nn

# --- Manual Matrix Multiplication ---
# Let's imagine a batch of 3 data samples, each with 4 features.
# Shape: (batch_size, num_features)
X = torch.randn(3, 4)

# Let's create a layer with 2 neurons. Each neuron needs 4 weights (one for each input feature).
# The weight matrix shape will be (num_features, num_neurons).
W = torch.randn(4, 2)

# We need one bias for each neuron.
# Shape: (num_neurons,)
b = torch.randn(2)

# Perform the linear transformation
Z = X @ W + b

print("--- Linear Layer as Matrix Multiplication ---")
print(f"Input shape: {X.shape}")
print(f"Weight matrix shape: {W.shape}")
print(f"Bias vector shape: {b.shape}")
print(f"Output shape: {Z.shape}\n")

# --- Using PyTorch's `nn.Linear` Layer ---
# This does the exact same thing, but it handles the weight and bias creation for us.
# It's the standard, convenient way to do it.

# Create a layer that takes 4 input features and has 2 output features (neurons)
linear_layer = nn.Linear(in_features=4, out_features=2)

# Perform the forward pass
Z_pytorch = linear_layer(X)

print("--- Using nn.Linear ---")
print(f"Input shape: {X.shape}")
print(f"Output shape from nn.Linear: {Z_pytorch.shape}")

# You can inspect the automatically created weights and biases
print(f"  - Layer weights shape: {linear_layer.weight.shape}")
print(f"  - Layer bias shape: {linear_layer.bias.shape}")
```

Notice that `nn.Linear` stores its weight matrix transposed compared to our manual `W`. This is a common convention. The underlying math (`X @ W.T + b`) is the same.

---

## Part 2: Calculus - The Engine of Learning

Calculus, specifically differential calculus, is how we optimize our neural network. The core idea is to find the **gradient** of the loss function. A gradient is a vector that points in the direction of the steepest ascent of a function. To minimize our loss, we just need to take small steps in the *opposite* direction of the gradient.

### 2.1. PyTorch Autograd: Your Personal Gradient Calculator

This is arguably the most magical part of PyTorch. You don't need to manually derive the gradients like we did in the NumPy example. You just need to tell PyTorch which tensors you want it to track.

You do this by setting `requires_grad=True` on a tensor.

```python
import torch

# --- A Simple Example ---
# Let's create a simple function: y = x^2
# The derivative is dy/dx = 2x

# Create a tensor and tell PyTorch to track its gradients
x = torch.tensor(3.0, requires_grad=True)

# Define the function
y = x**2

# Now, the magic part: perform the backward pass
y.backward()

# The calculated gradient is stored in the .grad attribute of the original tensor
print("--- Autograd Example 1: y = x^2 ---")
print(f"Value of x: {x.item()}")
print(f"Calculated gradient dy/dx at x=3: {x.grad.item()}")
print(f"(Expected: 2 * 3 = 6)\n")

# --- A More Complex Example ---
# Let's see how it works with multiple tensors and a more complex function.
# Let a = 2, b = 3, and our function be c = a^2 + 3*b

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a**2 + 3*b

# We want to find dc/da and dc/db
# dc/da = 2a = 4
# dc/db = 3

c.backward()

print("--- Autograd Example 2: c = a^2 + 3*b ---")
print(f"Value of a: {a.item()}, Value of b: {b.item()}")
print(f"Calculated gradient dc/da: {a.grad.item()} (Expected: 4)")
print(f"Calculated gradient dc/db: {b.grad.item()} (Expected: 3)\n")

# --- Gradients in a Neural Network Context ---
# When we call `loss.backward()` in a PyTorch training loop, this is exactly
# what's happening. PyTorch calculates the gradient of the loss with respect
# to every single model parameter (weights and biases) that has `requires_grad=True`.

# Let's demonstrate with a tiny model.
model = nn.Linear(in_features=3, out_features=1)

# Create some dummy data
x_sample = torch.randn(1, 3)
y_true = torch.tensor([[5.0]])

# Get a prediction
y_pred = model(x_sample)

# Calculate loss
loss = (y_pred - y_true)**2

# Calculate gradients
loss.backward()

print("--- Autograd in a Model ---")
print("A single data sample passes through a linear layer.")
print("We calculate the squared error loss and call .backward().")
print("PyTorch then computes the gradients for the layer's weights and bias:")

# The gradients are now stored in the .grad attribute of the parameters
print(f"  - Gradient for weights:\n{model.weight.grad}")
print(f"  - Gradient for bias: {model.bias.grad}")
```

### 2.2. The Chain Rule in Action

Autograd works by building a **computation graph**. Every operation you perform creates a node in this graph. When you call `.backward()`, PyTorch traverses this graph backward from the final node (the loss) to the leaf nodes (the model parameters), applying the chain rule at each step to compute the gradients.

This is why you don't have to worry about the math. You define the model's architecture (the forward pass), and PyTorch takes care of the backward pass automatically.

---

## Part 3: Probability & Statistics - The Foundation of Uncertainty

Deep learning is fundamentally about dealing with uncertainty. We want to build models that can make good predictions on new, unseen data. Probability and statistics provide the tools to frame this problem.

### 3.1. Loss Functions as Probability

Have you ever wondered *why* we use Mean Squared Error (MSE) for our loss function? It's not arbitrary. Using MSE is equivalent to assuming that our data has a **Gaussian (or Normal) distribution** around the true value. We are essentially performing Maximum Likelihood Estimation (MLE), finding the model parameters that make our observed data most probable under this Gaussian assumption.

Similarly, for classification problems, we often use **Cross-Entropy Loss**. This is equivalent to assuming our data follows a Bernoulli or Multinoulli distribution. It measures the "distance" between the predicted probability distribution from our model and the true distribution.

### 3.2. Activation Functions as Probability Distributions

Some activation functions have a direct probabilistic interpretation.

*   **Sigmoid:** As we saw, it squashes output to (0, 1). This is perfect for the output of a binary classifier, as it can be interpreted as the predicted probability of the positive class.

*   **Softmax:** This is a generalization of the sigmoid to multiple classes. It takes a vector of raw scores (logits) and converts it into a probability distribution where all the elements are between 0 and 1 and sum to 1.

```python
import torch
import torch.nn as nn

# Let's say our model predicts raw scores for 3 classes: Cat, Dog, Fish
# These raw scores are called "logits"
logits = torch.tensor([2.0, 1.0, 0.1])

# The softmax function will convert these scores into probabilities
softmax = nn.Softmax(dim=0)
probabilities = softmax(logits)

print("--- Softmax for Probabilities ---")
print(f"Raw logits: {logits}")
print(f"Probabilities: {probabilities}")
print(f"Sum of probabilities: {probabilities.sum()}")
print(f"The model predicts class 'Cat' with the highest probability ({probabilities[0]:.2f}).")
```

### 3.3. Weight Initialization as Sampling

The way we initialize our weights is also a statistical process. We are essentially drawing samples from a probability distribution. The choice of distribution can have a significant impact on training performance.

*   **Random Normal/Uniform:** Simple and common, but can lead to problems in deep networks (vanishing/exploding gradients).
*   **Xavier/Glorot Initialization:** A more intelligent scheme that takes into account the number of input and output neurons to keep the signal variance roughly constant as it passes through the layer. This is the default for `nn.Linear` in PyTorch.
*   **Kaiming/He Initialization:** An adaptation of Xavier initialization specifically for ReLU activation functions.

Let's see how to apply a different initialization.

```python
import torch.nn.init as init

# Create a linear layer
layer = nn.Linear(10, 5)

print("\n--- Weight Initialization ---")
print("Default (Xavier) initialization:")
print(layer.weight)

# Apply Kaiming initialization
init.kaiming_uniform_(layer.weight, nonlinearity='relu')
print("\nAfter Kaiming initialization:")
print(layer.weight)
```

## Conclusion: The Holy Trinity of Deep Learning Math

We have seen how the three core areas of mathematics are not just abstract theories but are actively implemented in the tools we use every day.

*   **Linear Algebra** is the skeleton of deep learning. It provides the structure for representing data and transformations through matrix operations (`nn.Linear`).
*   **Calculus** is the engine. It provides the mechanism for learning and optimization through gradients (`autograd` and `loss.backward()`).
*   **Probability and Statistics** is the soul. It provides the framework for dealing with uncertainty, from designing loss functions (`nn.MSELoss`, `nn.CrossEntropyLoss`) and activation functions (`nn.Softmax`) to initializing our models (`init.kaiming_uniform_`).

A solid, intuitive grasp of these concepts in code will empower you to move beyond simply using models to designing, debugging, and understanding them on a much deeper level.

## Self-Assessment Questions

1.  **Matrix Multiplication:** If you have an input batch of shape `[32, 128]` (32 samples, 128 features) and you pass it to an `nn.Linear` layer with `out_features=10`, what will be the shape of the output? What is the shape of the layer's weight matrix?
2.  **`requires_grad`:** What happens if you create a tensor with `requires_grad=False` and then include it in a calculation for a loss function before calling `loss.backward()`?
3.  **Autograd:** What is the one-line command in PyTorch that triggers the entire backpropagation and gradient calculation process?
4.  **Softmax:** If a model outputs logits `[ -0.5, 2.5, 0.1 ]` for three classes, which class is it most confident about? Why?
5.  **Initialization:** Why is initializing all weights to zero a bad idea?
6.  **Loss Functions:** If you are building a model to predict house prices (a regression task), which loss function from this guide would be a good choice? If you were building a model to classify images into 100 different categories, what would be a better choice?

