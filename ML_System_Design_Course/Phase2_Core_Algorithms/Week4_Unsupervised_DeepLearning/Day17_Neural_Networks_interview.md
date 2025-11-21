# Day 17: Neural Networks - Interview Questions

> **Topic**: Deep Learning Foundations
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Explain the Perceptron. What are its limitations?
**Answer:**
*   Single neuron model. $y = step(w^T x + b)$.
*   **Limitation**: Can only solve **Linearly Separable** problems. Cannot solve XOR.

### 2. What is a Multi-Layer Perceptron (MLP)?
**Answer:**
*   Stack of perceptrons with non-linear activation functions.
*   Input Layer -> Hidden Layers -> Output Layer.
*   Universal Function Approximator.

### 3. Why do we need Non-linear Activation Functions?
**Answer:**
*   Without them, a stack of linear layers is just one big linear layer ($W_2(W_1 x) = W_{new} x$).
*   Non-linearity allows learning complex boundaries.

### 4. Explain the Universal Approximation Theorem.
**Answer:**
*   A feedforward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of $R^n$.

### 5. What is Backpropagation?
**Answer:**
*   Algorithm to compute gradients of the Loss w.r.t weights.
*   Uses Chain Rule.
*   Forward pass computes Loss. Backward pass propagates error.

### 6. Explain the ReLU activation function. Why is it better than Sigmoid?
**Answer:**
*   $ReLU(x) = \max(0, x)$.
*   **Pros**: No vanishing gradient for $x > 0$. Computationally cheap. Sparsity.
*   **Sigmoid**: Saturates at tails (gradient $\approx$ 0). Expensive ($e^x$).

### 7. What is the "Dead ReLU" problem? How do you fix it?
**Answer:**
*   If a neuron outputs 0 for all inputs (weights become negative), gradient is 0. It never updates. It dies.
*   **Fix**: Leaky ReLU ($0.01x$ for $x<0$) or ELU. Initialize bias to small positive number (0.01).

### 8. Explain Weight Initialization. Why not initialize all weights to zero?
**Answer:**
*   **Zero**: All neurons in a layer compute the same output and get same gradient. They never diverge. Symmetry problem.
*   **Random**: Breaks symmetry.

### 9. What is Xavier (Glorot) Initialization?
**Answer:**
*   Scales weights based on number of inputs ($n_{in}$) and outputs ($n_{out}$).
*   Keeps variance of activations constant across layers.
*   Good for Sigmoid/Tanh.

### 10. What is He Initialization?
**Answer:**
*   Similar to Xavier but optimized for **ReLU**.
*   Variance $= 2 / n_{in}$.

### 11. What is Batch Normalization? How does it work?
**Answer:**
*   Normalizes layer inputs to Mean 0, Var 1. Then scales and shifts ($\gamma, \beta$).
*   Reduces **Internal Covariate Shift**.
*   Allows higher learning rates. Acts as regularization.

### 12. What is Dropout? Why does it work?
**Answer:**
*   Randomly zero out neurons during training with probability $p$.
*   **Why**: Prevents co-adaptation. Forces network to learn robust features. Ensemble effect.

### 13. What is the difference between Epoch, Batch, and Iteration?
**Answer:**
*   **Epoch**: One pass through entire dataset.
*   **Batch**: Subset of data processed at once.
*   **Iteration**: One update step (one batch).

### 14. How do you choose the number of hidden layers and neurons?
**Answer:**
*   **Heuristic**: Start small. Overfit a small batch. Then increase size and add regularization.
*   **Grid Search** / **Hyperopt**.

### 15. What is Internal Covariate Shift?
**Answer:**
*   Distribution of layer inputs changes as previous layer weights change.
*   Next layer has to constantly adapt to new distribution. Slows training.

### 16. Explain the Softmax output layer.
**Answer:**
*   Converts raw scores (logits) into probabilities summing to 1.
*   Used for Multi-class classification.

### 17. What is the difference between Cross-Entropy and MSE for classification?
**Answer:**
*   **CE**: Penalizes wrong confident predictions heavily. Convex for Softmax.
*   **MSE**: Slower convergence. Non-convex with Softmax.

### 18. What are Hyperparameters in a Neural Network?
**Answer:**
*   Parameters set *before* training (LR, Batch Size, Layers, Dropout rate).
*   Not learned by Gradient Descent.

### 19. How do you handle Overfitting in Deep Learning?
**Answer:**
*   More Data.
*   Data Augmentation.
*   Regularization (L1/L2, Dropout).
*   Early Stopping.
*   Smaller Model.

### 20. What is Transfer Learning?
**Answer:**
*   Take a model trained on a large task (ImageNet).
*   Freeze early layers (feature extractors).
*   Fine-tune last layers on your small dataset.
