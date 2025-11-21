# Day 20: Mobile Optimization - Interview Questions

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: Efficient Architectures, Pruning, and Edge AI

### 1. What is "Depthwise Separable Convolution"?
**Answer:**
*   Factorizes a standard convolution into:
    1.  **Depthwise**: Spatial filtering ($K \times K \times 1$) per channel.
    2.  **Pointwise**: Channel mixing ($1 \times 1 \times C$).
*   Reduces computation by factor of $N + K^2$ roughly (usually 8-9x).

### 2. Why is Unstructured Pruning often slower on GPUs?
**Answer:**
*   GPUs are SIMD (Single Instruction Multiple Data) devices optimized for dense matrix multiplication.
*   Random zeros break memory coalescing and cache locality.
*   Unless you use specialized sparse kernels (cuSPARSE) and high sparsity, the overhead of indexing sparse matrices outweighs the FLOP reduction.

### 3. What is "Knowledge Distillation"?
**Answer:**
*   Transferring "Dark Knowledge" from a Teacher to a Student.
*   The Student learns to match the Teacher's logits (soft targets), which contain information about inter-class relationships.

### 4. Explain "Channel Shuffle" in ShuffleNet.
**Answer:**
*   In Grouped Convolutions, information doesn't flow between groups.
*   Channel Shuffle permutes the output channels so that the next layer's groups take inputs from different groups of the previous layer.
*   Enables information flow without the cost of dense $1 \times 1$ convs.

### 5. What is "Hard Swish"?
**Answer:**
*   Approximation of Swish ($x \cdot \sigma(x)$).
*   $x \cdot \frac{ReLU6(x+3)}{6}$.
*   Swish uses Sigmoid, which is expensive on mobile (requires exp). Hard Swish uses ReLU (piecewise linear), which is cheap.

### 6. What is "Lottery Ticket Hypothesis"?
**Answer:**
*   A randomly initialized dense network contains a sub-network (winning ticket) that is initialized such that—when trained in isolation—it can match the test accuracy of the original network.
*   Justifies pruning *after* training.

### 7. How does "Quantization Aware Training" differ from "Post-Training Quantization"?
**Answer:**
*   **PTQ**: Train FP32 $\to$ Convert INT8. Fast, but accuracy loss.
*   **QAT**: Train with simulated quantization noise. Slow, but best accuracy.

### 8. What is "GhostNet"?
**Answer:**
*   Observation: Feature maps in CNNs often have redundant duplicates (ghosts).
*   Idea: Generate half the features using expensive Conv, and generate the other half using cheap Linear Transformations (shifts/rotations) of the first half.

### 9. What is "CoreML"?
**Answer:**
*   Apple's framework for running ML on iOS/macOS.
*   Leverages the Apple Neural Engine (ANE) for hardware acceleration.

### 10. Why use "Inverted Residuals" in MobileNetV2?
**Answer:**
*   Standard Residuals: Wide $\to$ Narrow $\to$ Wide.
*   Inverted: Narrow $\to$ Wide $\to$ Narrow.
*   The skip connection connects the thin layers (Bottlenecks).
*   Memory efficient because the large tensors (Wide) are not materialized in memory for the skip connection.

### 11. What is "Temperature" in Distillation?
**Answer:**
*   A hyperparameter $T$ used to soften the softmax distribution.
*   High $T$: Output becomes uniform (reveals information about tiny probabilities of incorrect classes).
*   Low $T$: Output becomes one-hot (standard softmax).

### 12. What is "Hardware-Aware NAS"?
**Answer:**
*   Searching for an architecture where the objective function includes the *measured latency* on the target device.
*   Ensures the model is fast on the specific hardware (e.g., DSP vs GPU).

### 13. What is "TFLite"?
**Answer:**
*   Google's format for mobile inference.
*   Uses a FlatBuffer format (memory mappable) for fast loading.
*   Optimized kernels for ARM NEON.

### 14. What is the "Linear Bottleneck" in MobileNetV2?
**Answer:**
*   Removing the non-linearity (ReLU) at the end of the bottleneck layer.
*   ReLU destroys information in low-dimensional spaces. Keeping it linear preserves the manifold.

### 15. Can you prune a model during training?
**Answer:**
*   Yes. Techniques like L1 regularization on weights push them to zero.
*   "Iterative Magnitude Pruning": Train $\to$ Prune $\to$ Retrain $\to$ Prune.

### 16. What is "SqueezeNet"?
**Answer:**
*   An early efficient network (2016).
*   Used "Fire Modules" (Squeeze $1 \times 1$ $\to$ Expand $1 \times 1 + 3 \times 3$).
*   Achieved AlexNet accuracy with 50x fewer parameters.

### 17. What is "ExecuTorch"?
**Answer:**
*   PyTorch's next-gen edge runtime.
*   Focuses on portability and small binary size (<50KB runtime).
*   Uses `torch.export` to capture the graph reliably.

### 18. Why is "Sigmoid" bad for fixed-point arithmetic?
**Answer:**
*   Sigmoid involves `exp()`, which is transcendental.
*   Hard to implement efficiently with INT8 lookup tables.
*   ReLU / Hard-Sigmoid are preferred.

### 19. What is "Federated Learning"?
**Answer:**
*   Training models on user devices (Edge) without sending data to the server.
*   Devices send *gradients* (updates) to the server, which aggregates them.
*   Privacy-preserving.

### 20. What is "On-Device Learning"?
**Answer:**
*   Updating the model on the phone (e.g., FaceID adapting to your beard).
*   Requires efficient backprop or simple head-tuning on the device.
