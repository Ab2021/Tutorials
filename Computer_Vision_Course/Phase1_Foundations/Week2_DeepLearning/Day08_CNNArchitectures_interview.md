# Day 8 Interview Questions: Training CNNs

## Q1: Why do we use activation functions like ReLU?
**Answer:**
To introduce **non-linearity**.
*   Without non-linear activations, a stack of linear layers is mathematically equivalent to a single linear layer ($W_2(W_1 x) = W_{combined} x$).
*   Non-linearity allows the network to approximate complex functions (Universal Approximation Theorem).

## Q2: Explain the Vanishing Gradient problem. How does ReLU help?
**Answer:**
*   **Problem:** In deep networks with Sigmoid/Tanh, gradients are multiplied by the derivative of the activation during backprop. Since $\sigma'(x) \in [0, 0.25]$, multiplying many small numbers causes gradients to vanish to zero in early layers.
*   **ReLU:** Derivative is either 0 or 1. For positive inputs, gradient flows unchanged (1). This prevents vanishing gradients in the positive regime.

## Q3: What is the difference between L1 and L2 regularization?
**Answer:**
*   **L2 (Weight Decay):** Penalty $\lambda \sum w^2$. Forces weights to be small/diffuse. Gradient is linear ($2w$).
*   **L1 (Lasso):** Penalty $\lambda \sum |w|$. Forces weights to be sparse (many zeros). Gradient is constant ($\pm 1$).

## Q4: Why is Batch Normalization helpful?
**Answer:**
1.  **Reduces Internal Covariate Shift:** Stabilizes the distribution of inputs to each layer.
2.  **Faster Convergence:** Allows higher learning rates.
3.  **Regularization:** Adds slight noise (mean/std computed on mini-batch), acting as a weak regularizer.
4.  **Less Sensitive to Initialization.**

## Q5: What is the difference between SGD and Adam?
**Answer:**
*   **SGD:** Updates weights using a fixed learning rate. Can get stuck in saddle points or oscillate.
*   **Adam:** Adaptive learning rate per parameter.
    *   Uses **Momentum** (history of gradients) to accelerate.
    *   Uses **RMSProp** (history of squared gradients) to scale updates (smaller steps for steep gradients, larger for flat).
    *   Generally converges faster but might generalize slightly worse than well-tuned SGD.

## Q6: What is Dropout and how does it work during inference?
**Answer:**
*   **Training:** Randomly sets neurons to zero with probability $p$. Prevents co-adaptation.
*   **Inference:** All neurons are active. To preserve expected magnitude, weights are scaled by $(1-p)$ (or inputs scaled by $1/(1-p)$ during training, which is Inverted Dropout).

## Q7: Explain the "Linear Scaling Rule" for batch size.
**Answer:**
If you increase the batch size by a factor of $k$, you should increase the learning rate by a factor of $k$ (or $\sqrt{k}$).
*   Larger batches give a less noisy estimate of the gradient, so we can take larger steps with confidence.

## Q8: What is Label Smoothing?
**Answer:**
A regularization technique where one-hot targets (e.g., $[0, 1]$) are replaced with soft targets (e.g., $[0.05, 0.95]$).
*   Prevents the model from becoming over-confident (predicting probability 1.0).
*   Improves generalization and calibration.

## Q9: Why do we need a validation set?
**Answer:**
To evaluate the model on unseen data during training for:
1.  **Hyperparameter Tuning:** Choosing LR, batch size, architecture.
2.  **Early Stopping:** Stopping training when validation loss starts increasing (overfitting), even if training loss is decreasing.

## Q10: Implement Cross-Entropy Loss from scratch.
**Answer:**
```python
def cross_entropy(logits, targets):
    # logits: (N, C), targets: (N,) class indices
    
    # 1. Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) # Stability
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # 2. NLL
    N = logits.shape[0]
    correct_logprobs = -np.log(probs[range(N), targets])
    loss = np.sum(correct_logprobs) / N
    
    return loss
```
