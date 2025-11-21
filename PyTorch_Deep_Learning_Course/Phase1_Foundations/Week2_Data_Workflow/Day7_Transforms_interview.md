# Day 7: Transforms - Interview Questions

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: Augmentation, Normalization, and Robustness

### 1. Why do we normalize images with Mean and Std?
**Answer:**
*   **Optimization**: It centers the data around 0 with unit variance. This makes the loss surface more spherical (well-conditioned Hessian), allowing SGD to converge faster with higher learning rates.
*   **Architecture**: Activation functions like Tanh/Sigmoid are linear near 0 and saturate at extremes. Centering keeps activations in the linear regime.

### 2. What is the difference between Invariance and Equivariance?
**Answer:**
*   **Invariance**: Output doesn't change with transformation. $f(Tx) = f(x)$. (e.g., Classifier: "Cat" is still "Cat" if rotated).
*   **Equivariance**: Output transforms with input. $f(Tx) = T f(x)$. (e.g., Segmentation: Mask rotates with image).

### 3. Explain "MixUp" augmentation. Why does it work?
**Answer:**
*   Linearly interpolates two images and their labels.
*   Encourages the model to behave linearly in-between training examples.
*   Regularizes the decision boundary, preventing it from being too sharp or confident on OOD data.

### 4. When should you NOT use Random Horizontal Flip?
**Answer:**
*   When horizontal orientation matters semantically.
*   Examples: Digit recognition (MNIST) - 5 vs 2 (sometimes), Text recognition, Traffic signs (Turn Left vs Turn Right).

### 5. What is "Test Time Augmentation" (TTA)?
**Answer:**
*   Running inference on multiple augmented versions of the same input (e.g., original + flipped + crops).
*   Averaging the predictions.
*   Reduces variance and improves robustness.

### 6. Why is resizing masks with "Bilinear" interpolation bad?
**Answer:**
*   Masks are categorical (integers 0, 1, 2...).
*   Bilinear interpolation produces floats (e.g., average of 0 and 1 is 0.5).
*   This destroys class labels.
*   **Fix**: Use "Nearest Neighbor" for masks.

### 7. What is "Color Jitter"?
**Answer:**
*   Randomly changing Brightness, Contrast, Saturation, and Hue.
*   Makes model invariant to lighting conditions (Sunny vs Cloudy).

### 8. How does "Cutout" or "Random Erasing" help?
**Answer:**
*   Randomly masks out a square patch of the image.
*   Forces the model not to rely on a single feature (e.g., "ears" of a cat) but to use context and other features.
*   Similar effect to Dropout, but in input space.

### 9. What is the advantage of GPU augmentation?
**Answer:**
*   Parallelism.
*   Avoids CPU bottleneck.
*   Avoids PCIe transfer overhead (if data is already on GPU or decoded there).

### 10. What is "AutoAugment"?
**Answer:**
*   Using Reinforcement Learning to search for the optimal sequence of augmentations for a specific dataset.
*   Removes manual tuning of augmentation policies.

### 11. Why do we convert images to Tensor `[C, H, W]`?
**Answer:**
*   PyTorch uses Channel-First format.
*   PIL/OpenCV use Channel-Last `[H, W, C]`.
*   `ToTensor()` handles this permutation and scales `[0, 255]` to `[0, 1]`.

### 12. What is "Center Crop" used for?
**Answer:**
*   Usually during Validation/Testing.
*   We resize image to 256, then Center Crop 224.
*   Ensures we evaluate on the central (most salient) part of the image without distortion.

### 13. Can augmentation hurt performance?
**Answer:**
*   Yes, if the augmentation destroys class semantics.
*   Example: Rotating a "6" into a "9".
*   Example: Too much color jitter on medical images (X-rays) where intensity carries meaning.

### 14. What is "Differentiable Augmentation"?
**Answer:**
*   Augmentations implemented as differentiable tensor operations.
*   Allows backpropagating gradients through the augmentation.
*   Used in GAN training (DiffAugment) to prevent discriminator overfitting on limited data.

### 15. Explain "Five Crop" testing.
**Answer:**
*   Taking 4 corner crops + 1 center crop (and their flips = 10 crops).
*   Averaging predictions.
*   Standard protocol for ImageNet evaluation.

### 16. What is the "Manifold Intrusion" problem in MixUp?
**Answer:**
*   Mixing a "Cat" and a "Dog" might result in an image that looks like a "Monkey" (on the manifold), but label is 50% Cat / 50% Dog.
*   Rare, but theoretically possible. MixUp assumes linear interpolation in pixel space = linear interpolation in label space.

### 17. How do you handle augmentation for Object Detection?
**Answer:**
*   Must transform Bounding Boxes (Bbox) along with image.
*   If crop removes an object, remove the bbox.
*   If flip, invert x-coordinates of bbox.
*   TorchVision v2 handles this.

### 18. What is "RandAugment"?
**Answer:**
*   A simplified version of AutoAugment.
*   Instead of learning a policy, it uniformly samples $N$ augmentations with magnitude $M$.
*   Much cheaper to tune (grid search N and M).

### 19. Why use `antialias=True` in Resize?
**Answer:**
*   Downsampling without filtering causes aliasing (Jagged edges, Moire patterns).
*   Antialiasing applies a low-pass filter before subsampling.
*   Improves shift invariance and quality.

### 20. What is "Sample Pairing"?
**Answer:**
*   Averaging two random images.
*   Label is the label of the first image.
*   Surprisingly works as a regularizer.
