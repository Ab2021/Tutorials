# Day 17: Neural Networks Foundations

> **Phase**: 2 - Core Algorithms
> **Week**: 4 - Unsupervised & Deep Learning
> **Focus**: The Building Blocks of Deep Learning
> **Reading Time**: 50 mins

---

## 1. From Perceptron to MLP

### 1.1 The Perceptron
The atom of deep learning. A linear classifier.
$$y = \text{step}(w \cdot x + b)$$
*   **Limitation**: Can only solve linearly separable problems. Cannot solve XOR.

### 1.2 Multi-Layer Perceptron (MLP)
Stacking perceptrons creates a network.
*   **Hidden Layers**: Intermediate representations.
*   **Universal Approximation Theorem**: An MLP with just one hidden layer (and enough neurons) can approximate *any* continuous function. (In theory).

---

## 2. Activation Functions: The Spark of Life

Without activation functions, a Neural Network is just a giant Linear Regression.
$$W_2(W_1 x) = (W_2 W_1)x = W_{new} x$$
Stacking linear layers collapses into a single linear layer. Non-linearity is essential.

### 2.1 Sigmoid / Tanh
*   **Old School**. S-shaped.
*   **Problem**: Saturation. At tails, gradient is near 0. Causes Vanishing Gradient.

### 2.2 ReLU (Rectified Linear Unit)
$$f(x) = \max(0, x)$$
*   **The Standard**.
*   **Pros**: Computationally free. No vanishing gradient for $x > 0$.
*   **Cons**: Dead ReLU. If $x < 0$ always, the neuron dies (gradient is 0) and never recovers. Leaky ReLU fixes this.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Initialization
**Scenario**: You initialize all weights to 0.
**Result**: Every neuron computes the same output. They get the same gradient. They update identically. The network acts like a single neuron. Symmetry is not broken.
**Solution**:
*   **He Initialization (Kaiming)**: For ReLU. Random values scaled by $\sqrt{2/n_{in}}$.
*   **Xavier Initialization (Glorot)**: For Sigmoid/Tanh.

### Challenge 2: Overfitting
**Scenario**: MLP has 1 million parameters. Dataset has 10k rows. It memorizes the data.
**Solution**:
*   **Dropout**: Randomly kill 50% of neurons during training. Forces redundancy.
*   **Early Stopping**: Stop when validation loss starts rising.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: Why is non-linearity necessary in a Neural Network?**
> **Answer**: Without non-linear activations, a deep network is mathematically equivalent to a single linear transformation (matrix multiplication). Non-linearity allows the network to learn complex boundaries and warp the input space to make classes separable.

**Q2: What is the "Dead ReLU" problem?**
> **Answer**: If a ReLU neuron's weights are updated such that the weighted sum is always negative for all inputs, it outputs 0. The gradient of 0 is 0. The weights never update again. The neuron is "dead." High learning rates often cause this. Solution: Leaky ReLU or lower learning rate.

**Q3: Why do we prefer Deep networks over Wide networks?**
> **Answer**: While a wide shallow network is a universal approximator, it is inefficient. Deep networks learn **hierarchical features** (edges -> shapes -> objects). This compositionality allows them to represent complex functions with exponentially fewer parameters than a shallow network.

---

## 5. Further Reading
- [Neural Networks and Deep Learning (Nielsen)](http://neuralnetworksanddeeplearning.com/)
- [Playground.tensorflow.org](https://playground.tensorflow.org/)
