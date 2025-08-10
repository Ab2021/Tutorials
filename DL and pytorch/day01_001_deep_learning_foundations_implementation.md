# Day 1.1: Deep Learning Foundations - A Practical Implementation Guide

## Introduction: From Theory to Code

Welcome to the first practical implementation guide of this course! The theoretical markdown files provide the "what" and the "why" of deep learning. These implementation files will provide the "how." We will take the core concepts and translate them into working code, starting from the most fundamental building blocks.

This guide is designed for beginners. We will assume very little prior knowledge of deep learning frameworks. Our goal is to demystify the magic behind neural networks by building them from scratch using Python and NumPy first, and then showing how to do the same thing more efficiently using PyTorch.

**Today's Learning Objectives:**

1.  **Build a Single Neuron:** Understand and implement the fundamental computational unit of a neural network from scratch.
2.  **Understand the Forward Pass:** Manually calculate the output of a simple network.
3.  **Grasp the Concept of a Loss Function:** Learn how we measure a network's performance.
4.  **Understand the Backward Pass (Backpropagation):** Manually calculate gradients to understand how a network learns.
5.  **Implement Gradient Descent:** Use the calculated gradients to update the network's weights and improve its performance.
6.  **Build a Complete Neural Network from Scratch:** Combine all the above concepts to solve the classic XOR problem using only NumPy.
7.  **Transition to PyTorch:** Re-implement the same XOR network in PyTorch to appreciate the power and convenience of a modern deep learning framework.

---

## Part 1: The Anatomy of a Neuron (The LEGO Brick of Deep Learning)

A neural network, no matter how complex, is built from a simple component: the neuron (or node). A neuron takes one or more inputs, performs a calculation, and produces a single output.

The calculation has two steps:

1.  **Linear Step:** A weighted sum of the inputs, plus a bias. This is a simple linear transformation.
    `z = (w1*x1 + w2*x2 + ... + wn*xn) + b`
    *   `x1, x2, ...`: The inputs to the neuron.
    *   `w1, w2, ...`: The weights. Each weight represents the *importance* of its corresponding input. A larger weight means the input has a stronger influence on the output.
    *   `b`: The bias. This is a constant that allows the neuron to shift its output up or down, independent of the inputs.

2.  **Activation Step:** The result of the linear step (`z`) is passed through a non-linear function called an **activation function**. This is the key that allows neural networks to learn complex, non-linear patterns. Without it, a neural network would just be a simple linear model.

    `output = activation(z)`

One of the most common activation functions is the **Sigmoid function**. It squashes any input value into a range between 0 and 1. This is very useful for binary classification tasks, where the output can be interpreted as a probability.

`Sigmoid(z) = 1 / (1 + e^(-z))`

### 1.1. Implementing a Neuron in NumPy

Let's build a single neuron with 2 inputs.

```python
import numpy as np

# Let's define the sigmoid activation function
def sigmoid(z):
    """Calculates the sigmoid of a number or a numpy array."""
    return 1 / (1 + np.exp(-z))

# Let's define the neuron itself as a class
class Neuron:
    def __init__(self, num_inputs):
        """
        Initializes a neuron with random weights and a zero bias.
        - num_inputs: The number of inputs the neuron will receive.
        """
        # We initialize the weights randomly. This is a crucial step to break symmetry
        # and ensure that different neurons in a network learn different things.
        # The weights are often initialized from a standard normal distribution.
        self.weights = np.random.randn(num_inputs)
        
        # The bias is often initialized to zero.
        self.bias = 0
        
        print(f"Neuron created with {num_inputs} inputs.")
        print(f"  - Initial weights: {self.weights}")
        print(f"  - Initial bias: {self.bias}")

    def forward(self, inputs):
        """
        Performs the forward pass of the neuron.
        - inputs: A NumPy array of input values.
        """
        # Step 1: Linear step (weighted sum + bias)
        # We use np.dot for the dot product between weights and inputs.
        linear_combination = np.dot(self.weights, inputs) + self.bias
        
        # Step 2: Activation step
        output = sigmoid(linear_combination)
        
        return output

# --- Usage Example ---
# Create a neuron that accepts 2 inputs
neuron = Neuron(num_inputs=2)

# Provide some sample inputs
# Let's say our inputs are x1 = 2.0 and x2 = 3.0
inputs = np.array([2.0, 3.0])

# Perform the forward pass to get the neuron's output
output = neuron.forward(inputs)

print(f"\nInput to the neuron: {inputs}")
print(f"Output of the neuron: {output}")
```

---

## Part 2: Building a Simple Network (Connecting the LEGOs)

A neural network is just a collection of these neurons organized into layers.

*   **Input Layer:** This isn't really a layer of neurons. It just represents the raw input data.
*   **Hidden Layers:** These are the layers of neurons between the input and output. This is where the "deep" in "deep learning" comes from. The more hidden layers, the deeper the network.
*   **Output Layer:** This is the final layer of neurons that produces the network's prediction.

Let's build a very simple network architecture:
*   2 input features.
*   1 hidden layer with 2 neurons.
*   1 output layer with 1 neuron.

This network will take 2 numbers as input and produce 1 number as output.

### 2.1. Implementing a Network from Scratch

```python
import numpy as np

# We'll reuse our sigmoid function from before
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class SimpleNeuralNetwork:
    def __init__(self):
        """
        Initializes the network with a fixed architecture:
        - 2 input neurons
        - 1 hidden layer with 2 neurons
        - 1 output layer with 1 neuron
        """
        # --- Weights and Biases for the Hidden Layer ---
        # W_h1: weights for the first neuron in the hidden layer (2 weights for 2 inputs)
        self.W_h1 = np.random.randn(2)
        # W_h2: weights for the second neuron in the hidden layer (2 weights for 2 inputs)
        self.W_h2 = np.random.randn(2)
        # b_h1, b_h2: biases for the hidden layer neurons
        self.b_h1 = 0
        self.b_h2 = 0

        # --- Weights and Biases for the Output Layer ---
        # W_o1: weights for the output neuron (2 weights, because it takes input from the 2 hidden neurons)
        self.W_o1 = np.random.randn(2)
        # b_o1: bias for the output neuron
        self.b_o1 = 0

    def forward(self, inputs):
        """
        Performs the forward pass of the entire network.
        - inputs: A NumPy array with 2 features.
        """
        # --- Hidden Layer Calculations ---
        # Calculate the output of the first hidden neuron
        h1_linear = np.dot(self.W_h1, inputs) + self.b_h1
        h1_output = sigmoid(h1_linear)

        # Calculate the output of the second hidden neuron
        h2_linear = np.dot(self.W_h2, inputs) + self.b_h2
        h2_output = sigmoid(h2_linear)

        # The outputs of the hidden layer become the inputs for the output layer
        hidden_layer_outputs = np.array([h1_output, h2_output])

        # --- Output Layer Calculation ---
        o1_linear = np.dot(self.W_o1, hidden_layer_outputs) + self.b_o1
        network_output = sigmoid(o1_linear)

        return network_output

# --- Usage Example ---
# Create an instance of our network
network = SimpleNeuralNetwork()

# Provide some sample inputs
inputs = np.array([2.0, 3.0])

# Get the network's prediction
prediction = network.forward(inputs)

print("---")
print("Simple Neural Network (Forward Pass)")
print(f"Input to the network: {inputs}")
print(f"Prediction from the network: {prediction}")
```

So far, our network produces a prediction, but it's just a random guess because the weights were initialized randomly. To make it learn, we need to introduce the concepts of loss, backpropagation, and optimization.

---

## Part 3: How a Network Learns (The Core Logic)

Learning in a neural network is an iterative process:

1.  **Forward Pass:** Make a prediction.
2.  **Calculate Loss:** Compare the prediction to the true answer to see how wrong it is. The function that does this is the **loss function**.
3.  **Backward Pass (Backpropagation):** Figure out *which weights and biases were most responsible* for the error. This is done by calculating the **gradient** of the loss with respect to each weight and bias. The gradient is just a derivative that tells us the direction and magnitude of the steepest ascent. To decrease the loss, we need to move in the opposite direction of the gradient.
4.  **Update Weights (Gradient Descent):** Slightly adjust all the weights and biases in the direction that minimizes the loss. The size of this adjustment is controlled by the **learning rate**.

We repeat these four steps many times, and the network's predictions will gradually get better.

### 3.1. The Loss Function: Mean Squared Error (MSE)

A common loss function for regression tasks (where the output is a continuous value) is the Mean Squared Error.

`MSE = (1/n) * sum((y_true - y_pred)^2)`

*   `y_true`: The actual, correct answer.
*   `y_pred`: The network's prediction.
*   `n`: The number of samples.

For a single sample, it's just `(y_true - y_pred)^2`.

### 3.2. Backpropagation: The Chain Rule

Backpropagation is essentially a big application of the chain rule from calculus. We want to find how the loss `L` changes when we change a weight `w`, i.e., `dL/dw`. We do this by working backward from the loss through the network, layer by layer, calculating the gradients at each step.

Let's define the derivative of our sigmoid function, as we'll need it for backpropagation.

`d(sigmoid(z))/dz = sigmoid(z) * (1 - sigmoid(z))`

### 3.3. Putting It All Together: A Full Training Loop from Scratch

We will now train our network to solve the XOR problem. XOR (exclusive OR) is a classic problem because it's not linearly separable. A simple linear model can't solve it, but our simple neural network can!

XOR Truth Table:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

```python
import numpy as np

# --- Activation Function and its Derivative ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# --- Loss Function and its Derivative ---
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)

# --- The Neural Network Class with Training Logic ---
class XorNeuralNetwork:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

        # --- Initialize weights and biases ---
        # Hidden Layer (2 neurons)
        self.W_h = np.random.randn(2, 2) # Shape: (num_inputs, num_neurons)
        self.b_h = np.zeros(2)

        # Output Layer (1 neuron)
        self.W_o = np.random.randn(2, 1) # Shape: (num_inputs_from_hidden, num_neurons)
        self.b_o = np.zeros(1)

    def forward(self, inputs):
        # This method will now also store intermediate values needed for backpropagation
        self.inputs = inputs

        # Hidden layer forward pass
        self.h_linear = np.dot(self.inputs, self.W_h) + self.b_h
        self.h_output = sigmoid(self.h_linear)

        # Output layer forward pass
        self.o_linear = np.dot(self.h_output, self.W_o) + self.b_o
        self.o_output = sigmoid(self.o_linear)

        return self.o_output

    def backward(self, y_true):
        # --- Calculate Gradients (The core of backpropagation) ---
        
        # Start from the end and work backwards
        # 1. Gradient of the loss with respect to the network's final output
        d_loss_d_pred = mse_loss_derivative(y_true, self.o_output)
        
        # 2. Gradient of the final output with respect to the output layer's linear part
        d_pred_d_o_linear = sigmoid_derivative(self.o_linear)
        d_loss_d_o_linear = d_loss_d_pred * d_pred_d_o_linear

        # 3. Gradient of the output layer's linear part with respect to its weights and biases
        d_o_linear_d_W_o = self.h_output.T
        d_loss_d_W_o = np.dot(d_o_linear_d_W_o, d_loss_d_o_linear)
        d_loss_d_b_o = d_loss_d_o_linear.sum(axis=0)

        # 4. Gradient of the output layer's linear part with respect to the hidden layer's output
        d_o_linear_d_h_output = self.W_o.T
        d_loss_d_h_output = np.dot(d_loss_d_o_linear, d_o_linear_d_h_output)

        # 5. Gradient of the hidden layer's output with respect to its linear part
        d_h_output_d_h_linear = sigmoid_derivative(self.h_linear)
        d_loss_d_h_linear = d_loss_d_h_output * d_h_output_d_h_linear

        # 6. Gradient of the hidden layer's linear part with respect to its weights and biases
        d_h_linear_d_W_h = self.inputs.T
        d_loss_d_W_h = np.dot(d_h_linear_d_W_h, d_loss_d_h_linear)
        d_loss_d_b_h = d_loss_d_h_linear.sum(axis=0)

        # --- Update Weights and Biases (Gradient Descent) ---
        self.W_o -= self.learning_rate * d_loss_d_W_o
        self.b_o -= self.learning_rate * d_loss_d_b_o
        self.W_h -= self.learning_rate * d_loss_d_W_h
        self.b_h -= self.learning_rate * d_loss_d_b_h

    def train(self, X, y, epochs):
        print("\n---")
        print("Starting Training (NumPy from Scratch)")
        for epoch in range(epochs):
            # In a real scenario, we would shuffle the data and process in batches.
            # For this simple example, we process the whole dataset at once.
            y_pred = self.forward(X)
            self.backward(y)

            if epoch % 1000 == 0:
                loss = mse_loss(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# --- Prepare the XOR data ---
# Input data (4 samples, 2 features each)
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Target data (4 samples, 1 output each)
y_train = np.array([[0], [1], [1], [0]])

# --- Create and Train the Network ---
xor_network = XorNeuralNetwork(learning_rate=0.1)
xor_network.train(X_train, y_train, epochs=10000)

# --- Test the Trained Network ---
print("\n---")
print("Testing the Trained Network")
for x_input, y_true in zip(X_train, y_train):
    prediction = xor_network.forward(x_input)
    print(f"Input: {x_input}, Prediction: {prediction[0]:.4f}, True: {y_true[0]}")

```

Congratulations! You've just built and trained a neural network from scratch. You now understand the fundamental mechanics of how deep learning works.

---

## Part 4: The Easy Way - Rebuilding with PyTorch

Building networks from scratch is insightful, but it's not practical for real-world problems. It's slow, error-prone, and doesn't run on GPUs. This is where frameworks like PyTorch come in.

PyTorch provides:

*   **Tensors:** A data structure similar to NumPy arrays but with the ability to run on GPUs for massive speedups.
*   **`nn.Module`:** A base class for building neural network models, which helps organize layers and parameters.
*   **Pre-built Layers:** Optimized implementations of layers like `nn.Linear` (for our weighted sum + bias) and activation functions.
*   **Autograd:** An automatic differentiation engine. This is the magic part. You just define the forward pass, and PyTorch automatically calculates all the gradients for you during the backward pass. No more manual `backward()` method!
*   **Optimizers:** Implementations of optimization algorithms like `SGD` (Stochastic Gradient Descent) and `Adam`.

Let's solve the XOR problem again, this time with PyTorch.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# --- Prepare the XOR data as PyTorch Tensors ---
# Note the `dtype=torch.float32` which is standard for neural network inputs.
X_train_torch = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_train_torch = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# --- Define the Network using PyTorch's Building Blocks ---
class XorNetPyTorch(nn.Module):
    def __init__(self):
        super(XorNetPyTorch, self).__init__()
        # We define the layers in the constructor.
        # PyTorch will automatically track the weights and biases of these layers.
        
        # A linear layer that takes 2 inputs and produces 2 outputs (our hidden layer)
        self.hidden_layer = nn.Linear(in_features=2, out_features=2)
        
        # The sigmoid activation function
        self.activation = nn.Sigmoid()
        
        # A linear layer that takes 2 inputs (from the hidden layer) and produces 1 output
        self.output_layer = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        # The forward pass defines how the data flows through the layers.
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        x = self.activation(x) # We use sigmoid on the output as well for this example
        return x

# --- Create the Model, Loss Function, and Optimizer ---
# Instantiate the model
model_torch = XorNetPyTorch()

# PyTorch has pre-built loss functions. MSELoss is the same as our manual one.
loss_function = nn.MSELoss()

# PyTorch has pre-built optimizers. SGD stands for Stochastic Gradient Descent.
# We tell it which parameters to optimize (model_torch.parameters()).
optimizer = optim.SGD(model_torch.parameters(), lr=0.1)

# --- The PyTorch Training Loop ---
print("\n---")
print("Starting Training (PyTorch)")
epochs = 10000
for epoch in range(epochs):
    # 1. Clear the gradients from the previous iteration
    optimizer.zero_grad()

    # 2. Forward pass: get the model's predictions
    y_pred_torch = model_torch(X_train_torch)

    # 3. Calculate the loss
    loss = loss_function(y_pred_torch, y_train_torch)

    # 4. Backward pass: PyTorch's autograd calculates the gradients automatically!
    loss.backward()

    # 5. Update the weights using the optimizer
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --- Test the Trained PyTorch Network ---
print("\n---")
print("Testing the Trained PyTorch Network")
# We use torch.no_grad() to tell PyTorch we don't need to calculate gradients for this part.
with torch.no_grad():
    for x_input, y_true in zip(X_train_torch, y_train_torch):
        prediction = model_torch(x_input)
        print(f"Input: {x_input.numpy()}, Prediction: {prediction.item():.4f}, True: {y_true.item()}")

```

## Conclusion: NumPy vs. PyTorch

You have now seen how to build and train a neural network in two different ways. The NumPy implementation was verbose and required manual gradient calculation, but it revealed the core mechanics of deep learning. The PyTorch implementation was concise, powerful, and much closer to how real-world deep learning is done.

**Key Takeaways:**

*   A neural network is a collection of simple neurons organized in layers.
*   Learning is an iterative process of making a prediction, calculating the error (loss), figuring out who is to blame (backpropagation), and making a small correction (optimization).
*   Building from scratch is invaluable for understanding, but frameworks like PyTorch are essential for practical application due to their automatic differentiation, GPU support, and pre-built, optimized components.

In the next implementation guides, we will build exclusively with PyTorch, now that you have a solid understanding of the fundamentals happening under the hood.

## Self-Assessment Questions

1.  **Weights and Biases:** What are the roles of weights and biases in a neuron? Why do we initialize weights randomly?
2.  **Activation Functions:** What is the purpose of an activation function? What would happen if we didn't use one?
3.  **Loss Function:** What is a loss function? Can you name the one we used in this guide?
4.  **Backpropagation:** In one sentence, what is the goal of backpropagation?
5.  **Gradient Descent:** What is the "learning rate," and how does it affect training?
6.  **PyTorch `autograd`:** What is the biggest advantage of using PyTorch's `autograd` engine compared to our NumPy implementation?
7.  **`nn.Module`:** What is the purpose of subclassing `nn.Module` when building a model in PyTorch?
8.  **Optimizers:** What two things do you need to provide to a PyTorch optimizer (like `optim.SGD`) when you create it?
