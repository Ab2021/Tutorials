# Day 20 Interview Questions: Project Review

## Q1: Why is fine-grained classification harder than generic classification?
**Answer:**
*   **Generic (ImageNet):** Classes are visually distinct (Dog vs Car vs Apple). Inter-class variance is high.
*   **Fine-Grained (CUB-200):** Classes are visually similar (Sparrow vs Finch). Inter-class variance is low, while Intra-class variance (pose, background) can be high.
*   Requires the model to focus on subtle, local details rather than global shape.

## Q2: Why did you choose ResNet-50 over VGG-16?
**Answer:**
*   **Performance:** ResNet-50 generally achieves higher accuracy due to residual connections preventing vanishing gradients.
*   **Efficiency:** ResNet-50 has fewer parameters (~25M) than VGG-16 (~138M) because it uses Global Average Pooling instead of massive Dense layers.
*   **Training Speed:** ResNet converges faster.

## Q3: How does increasing image resolution help?
**Answer:**
*   Fine-grained features (e.g., the color of a bird's eye ring) might occupy only a few pixels in a $224 \times 224$ image.
*   Downsampling (Pooling) further destroys this information.
*   At $448 \times 448$, these features are preserved through deeper layers, allowing the network to use them for discrimination.

## Q4: What is the purpose of Label Smoothing in this project?
**Answer:**
*   It prevents the model from becoming over-confident (predicting probability 1.0 for the ground truth).
*   In fine-grained tasks, classes are ambiguous. A bird might look 90% like Species A and 10% like Species B.
*   Label smoothing encourages the model to learn a softer distribution, improving generalization and calibration.

## Q5: Explain the "Bilinear Pooling" concept.
**Answer:**
*   Standard pooling (Avg/Max) summarizes features independently.
*   Bilinear pooling computes the outer product of feature vectors at each location, then sums them.
*   $F = \sum x_i x_i^T$.
*   This captures **pairwise correlations** between feature channels (e.g., "Feature A present" AND "Feature B present"), which is powerful for describing complex textures and parts.

## Q6: How would you handle class imbalance in this dataset?
**Answer:**
*   **Weighted Loss:** Assign higher weights to minority classes in the Cross-Entropy loss.
*   **Oversampling:** Sample minority class images more frequently during training.
*   **Focal Loss:** Automatically down-weights easy examples and focuses on hard (minority) ones.

## Q7: What did you learn from visualizing Grad-CAM?
**Answer:**
*   It showed that the model focuses on the **head and wings** of the bird to make decisions.
*   If the model was focusing on the background (e.g., sky or branch), it would indicate overfitting or bias, suggesting the need for better segmentation or background removal.

## Q8: Why use Cosine Annealing scheduler?
**Answer:**
*   It smoothly decreases the learning rate from a max value to a min value following a cosine curve.
*   Unlike Step Decay (which drops abruptly), Cosine Annealing keeps the LR high for longer (exploration) and then decays rapidly (exploitation), often finding better minima.

## Q9: Did you use Transfer Learning? Why?
**Answer:**
**Yes.**
*   Training from scratch on 11k images would lead to massive overfitting.
*   Pretraining on ImageNet (1.2M images) allows the model to learn robust low-level filters (edges, textures) and high-level semantic features.
*   We only need to "fine-tune" these features to distinguish between bird species.

## Q10: How would you deploy this model to a mobile phone?
**Answer:**
1.  **Quantization:** Convert weights from Float32 to Int8 (4x size reduction).
2.  **Pruning:** Remove unimportant weights.
3.  **Distillation:** Train a smaller MobileNetV3 student to mimic this ResNet-50 teacher.
4.  **Export:** Convert to ONNX or TFLite format.
