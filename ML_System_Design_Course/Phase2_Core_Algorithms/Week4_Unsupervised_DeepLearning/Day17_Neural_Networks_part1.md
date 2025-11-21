# Day 17 (Part 1): Advanced Neural Networks

> **Phase**: 6 - Deep Dive
> **Topic**: Anatomy of a Neuron
> **Focus**: Initialization, Activations, and Theory
> **Reading Time**: 60 mins

---

## 1. Initialization Theory

Why not init with Zero? Why not Random Normal(0, 1)?

### 1.1 Symmetry Breaking
*   **Zero**: All neurons compute same gradient. Update same way. Act like 1 neuron.
*   **Large Weights**: Saturation (Tanh/Sigmoid). Vanishing gradient.

### 1.2 Xavier (Glorot) vs. He (Kaiming)
*   **Goal**: Keep variance of activations constant across layers.
*   **Xavier**: $\text{Var}(W) = 2 / (n_{in} + n_{out})$. Good for Tanh/Sigmoid.
*   **He**: $\text{Var}(W) = 2 / n_{in}$. Good for ReLU.
    *   *Reason*: ReLU kills half the neurons (output 0). We need to double the variance to compensate.

---

## 2. Modern Activations

### 2.1 GELU (Gaussian Error Linear Unit)
*   $x \Phi(x)$. Smooth approximation of ReLU.
*   **Used in**: BERT, GPT.
*   **Why?**: Non-monotonic, smooth derivatives.

### 2.2 Swish (SiLU)
*   $x \cdot \sigma(x)$.
*   Discovered by NAS (Neural Architecture Search).

---

## 3. Tricky Interview Questions

### Q1: Explain the Universal Approximation Theorem.
> **Answer**: A 2-layer MLP (1 hidden layer) with sufficient width and non-linear activation can approximate *any* continuous function on a compact set.
> *   **Caveat**: It doesn't say *how* to learn it, or how many neurons (could be exponential). Deep networks are more parameter efficient than wide ones.

### Q2: How to fix Dead ReLU?
> **Answer**:
> 1.  **Leaky ReLU**: Small slope for $x < 0$. Gradient flows.
> 2.  **Lower Learning Rate**: High LR pushes weights to a region where ReLU is always off.
> 3.  **He Initialization**.

### Q3: Why do we need Bias terms?
> **Answer**: Allows shifting the activation function. Without bias, the hyperplane must pass through the origin ($w^T x = 0$). Bias allows $w^T x + b = 0$.

---

## 4. Practical Edge Case: Internal Covariate Shift
*   **Theory**: Distribution of layer inputs changes during training.
*   **Fix**: Batch Normalization. (Though recent papers argue it smoothens the loss landscape rather than fixing shift).

