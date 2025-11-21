# Day 9 Interview Questions: Transfer Learning

## Q1: What is the difference between Feature Extraction and Fine-Tuning?
**Answer:**
*   **Feature Extraction:** Freezes the backbone (convolutional base) and only trains the classifier head. The backbone acts as a fixed feature calculator.
*   **Fine-Tuning:** Unfreezes some or all of the backbone layers and trains them with a low learning rate. Allows the features to adapt to the new dataset.

## Q2: Why do we use a lower learning rate for fine-tuning?
**Answer:**
*   The pretrained weights are already good (converged on ImageNet).
*   A high learning rate would cause large updates, destroying the learned filters ("catastrophic forgetting").
*   We want to make small adjustments ("polishing") to adapt to the new domain.

## Q3: If I have a small dataset similar to ImageNet, what strategy should I use?
**Answer:**
**Feature Extraction (Linear Probe).**
*   Since data is small, fine-tuning might overfit.
*   Since data is similar, ImageNet features are likely already optimal.
*   Just train a linear classifier on top.

## Q4: If I have a large dataset different from ImageNet (e.g., Medical X-Rays), what strategy?
**Answer:**
**Fine-Tune the whole network (or train from scratch).**
*   ImageNet features (dogs/cats) might not be relevant for X-Rays.
*   Large data allows training many parameters without overfitting.
*   Usually, initializing with ImageNet weights is still better than random initialization (faster convergence).

## Q5: What is "Catastrophic Forgetting"?
**Answer:**
When a neural network forgets previously learned information (Source domain) upon learning new information (Target domain).
*   In transfer learning, if we fine-tune too aggressively, we lose the general feature extraction capabilities learned from ImageNet.

## Q6: How does Domain Adaptation differ from Transfer Learning?
**Answer:**
*   **Transfer Learning:** We have labels for the Target domain. We want to maximize accuracy on Target.
*   **Domain Adaptation:** Often Unsupervised. We have labeled Source data and **unlabeled** Target data. We want to adapt the model to work on Target (e.g., Synthetic to Real).

## Q7: Can I use a model pretrained on $224 \times 224$ images for $512 \times 512$ input?
**Answer:**
**Yes.**
*   **CNNs:** Convolutional layers are translation invariant and handle any size. The Global Average Pooling layer before the classifier handles variable spatial dimensions, outputting a fixed vector size $(1 \times 1 \times C)$.
*   **ViTs:** Need to interpolate positional embeddings.

## Q8: What is "Discriminative Learning Rate"?
**Answer:**
Using different learning rates for different layers.
*   **Early layers:** Capture edges/textures (universal). Use very low LR (or freeze).
*   **Late layers:** Capture high-level semantics (task-specific). Use higher LR.

## Q9: Why is ImageNet the standard for pretraining?
**Answer:**
*   **Scale:** 1.2 million images.
*   **Diversity:** 1000 classes covering a wide range of objects.
*   **Features:** Models trained on ImageNet learn robust, general-purpose visual features that transfer well to almost any other visual task.

## Q10: Implement layer freezing in PyTorch.
**Answer:**
```python
# Freeze first 6 layers
child_counter = 0
for child in model.children():
    if child_counter < 6:
        for param in child.parameters():
            param.requires_grad = False
    child_counter += 1
```
