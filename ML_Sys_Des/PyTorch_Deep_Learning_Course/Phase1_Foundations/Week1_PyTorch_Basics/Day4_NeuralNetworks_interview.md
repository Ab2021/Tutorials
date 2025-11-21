# Day 4: Neural Networks - Interview Questions

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: Architecture, Layers, and Initialization

### 1. What is the difference between `nn.ModuleList` and a Python `list` of modules?
**Answer:**
*   `nn.ModuleList` registers the modules it contains with the parent `nn.Module`. This ensures they appear in `.parameters()` and move to GPU when `.to(device)` is called.
*   A Python `list` does not register them. The optimizer won't see the parameters, and they won't train.

### 2. Why do we need Non-Linearities (Activation Functions)?
**Answer:**
*   Without non-linearities, a stack of Linear layers is mathematically equivalent to a single Linear layer ($W_2(W_1 x) = (W_2 W_1)x = W_{new} x$).
*   Non-linearities allow the network to approximate complex, non-convex functions (Universal Approximation).

### 3. Explain the "Dying ReLU" problem.
**Answer:**
*   If a ReLU neuron outputs 0 (input < 0), its gradient is 0.
*   If the weights update such that the neuron *always* sees negative inputs, it will never output non-zero again. It "dies".
*   **Fix**: LeakyReLU (small slope for negative), ELU, or better initialization (He Init).

### 4. How does Xavier (Glorot) Initialization work?
**Answer:**
*   Designed to keep the variance of activations and gradients constant across layers.
*   Draws weights from $N(0, \frac{2}{n_{in} + n_{out}})$.
*   Ideal for symmetric activations like Tanh or Sigmoid (linear regime near 0).

### 5. How does Kaiming (He) Initialization work?
**Answer:**
*   Designed for ReLU (which halves the variance because it zeroes out half the inputs).
*   Draws weights from $N(0, \frac{2}{n_{in}})$.
*   Prevents signal from vanishing in deep ReLU networks.

### 6. What is the difference between `model.train()` and `model.eval()`?
**Answer:**
*   They set the `training` boolean flag in the module.
*   **Dropout**: Active in train, disabled (identity) in eval.
*   **BatchNorm**: Updates running stats and uses batch stats in train. Uses frozen running stats in eval.
*   Does NOT affect gradients (use `torch.no_grad()` for that).

### 7. Why is Batch Normalization effective?
**Answer:**
*   **Internal Covariate Shift**: Reduces the shifting of distribution of layer inputs.
*   **Smoothing**: Makes the loss landscape smoother (Lipschitz constant).
*   Allows higher learning rates.
*   Acts as a weak regularizer.

### 8. What is a "Skip Connection" (Residual Connection)?
**Answer:**
*   $y = F(x) + x$.
*   Allows gradients to flow directly through the identity path ($+x$) during backprop.
*   Solves Vanishing Gradient problem in very deep networks (ResNets).

### 9. How do you calculate the number of parameters in a Convolutional Layer?
**Answer:**
*   Shape: $(C_{out}, C_{in}, K, K)$.
*   Weights: $C_{out} \times C_{in} \times K \times K$.
*   Bias: $C_{out}$.
*   Total: $(C_{in} \times K^2 + 1) \times C_{out}$.

### 10. What is "Global Average Pooling"?
**Answer:**
*   Takes the average of each feature map ($H \times W$) to produce a single number per channel.
*   Converts $(N, C, H, W) \to (N, C)$.
*   Used in modern architectures (ResNet) to replace the massive Fully Connected layers at the end, reducing parameters and overfitting.

### 11. What is `nn.Parameter`?
**Answer:**
*   A Tensor subclass.
*   When assigned as an attribute to a Module, it is automatically registered as a parameter.
*   `requires_grad=True` by default.

### 12. Explain the difference between `F.cross_entropy` and `nn.CrossEntropyLoss`.
**Answer:**
*   Functionally identical.
*   `nn.CrossEntropyLoss` is a Class (Module), useful for passing to high-level trainers.
*   `F.cross_entropy` is a Function.
*   **Note**: Both combine `LogSoftmax` and `NLLLoss` (Negative Log Likelihood). Do not apply Softmax before them!

### 13. How do you handle different input sizes in a Fully Connected network?
**Answer:**
*   You can't. FC layers have fixed weight matrices.
*   You must resize/crop inputs to fixed size.
*   Or use Global Pooling (SPP) to convert variable size feature maps to fixed vector.

### 14. What is "Weight Sharing"?
**Answer:**
*   Using the same parameter object in multiple places in the graph.
*   Example: CNNs (kernel shared across space). RNNs (weights shared across time). Autoencoders (tied weights between encoder/decoder).
*   Gradients from all usages sum up.

### 15. How do you implement a custom layer?
**Answer:**
*   Subclass `nn.Module`.
*   Initialize parameters in `__init__` using `nn.Parameter`.
*   Implement `forward(self, x)`.

### 16. What is "Dropout"? Why do we scale by $1/(1-p)$ during training?
**Answer:**
*   Randomly zero out neurons with probability $p$.
*   **Inverted Dropout**: We scale the remaining neurons by $1/(1-p)$ during training so that the *expected sum* remains the same.
*   This avoids needing to scale weights during test time.

### 17. What is the Receptive Field?
**Answer:**
*   The region of the input image that affects a particular neuron in the output.
*   Increases linearly with depth for Conv layers.
*   Dilated convolutions increase receptive field exponentially without adding parameters.

### 18. What is `register_buffer` used for?
**Answer:**
*   Storing state that is part of the model but not trained via SGD.
*   Example: `running_mean` in BatchNorm, or positional encodings in Transformers.
*   Saved in `state_dict`.

### 19. Can a Linear Layer learn a Non-Linear function?
**Answer:**
*   No. It is a linear map.
*   It can only learn linear decision boundaries (hyperplanes).

### 20. What is the difference between 1D, 2D, and 3D Convolutions?
**Answer:**
*   **1D**: Slides over time (Text, Audio). Input $(N, C, L)$.
*   **2D**: Slides over H and W (Images). Input $(N, C, H, W)$.
*   **3D**: Slides over D, H, W (Video, MRI). Input $(N, C, D, H, W)$.
