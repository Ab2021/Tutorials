# Day 19: CNNs & Computer Vision Foundations

> **Phase**: 2 - Core Algorithms
> **Week**: 4 - Unsupervised & Deep Learning
> **Focus**: Vision & Spatial Hierarchies
> **Reading Time**: 50 mins

---

## 1. The Convolutional Operation

MLPs flatten images into vectors, destroying spatial structure. CNNs preserve it.

### 1.1 Convolution
*   **Filter (Kernel)**: A small matrix (e.g., 3x3) that slides over the image.
*   **Feature Map**: The output. Represents "where" a feature (edge, texture) exists.
*   **Parameter Sharing**: The same filter is used everywhere. Drastically reduces parameters compared to Dense layers.

### 1.2 Pooling
*   **Max Pooling**: Takes the max value in a window (2x2).
*   **Purpose**:
    1.  Reduces spatial dimensions (Compression).
    2.  **Translation Invariance**: Small shifts in input don't change the output.

---

## 2. Architectures that Changed the World

### 2.1 ResNet (Residual Networks)
*   **Problem**: Deep networks (20+ layers) were hard to train due to vanishing gradients.
*   **Solution**: Skip Connections ($y = F(x) + x$). The gradient can flow through the identity path $x$ unimpeded. Allowed training of 100+ layer networks.

### 2.2 Transfer Learning
*   **Idea**: Don't train from scratch. Use a ResNet pretrained on ImageNet (14M images).
*   **Strategy**: Freeze the "Backbone" (feature extractor). Only train the final "Head" (classifier) on your small dataset.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Input Size
**Scenario**: Images come in different sizes. CNNs need fixed input.
**Solution**:
*   **Resize/Crop**: Standard (e.g., 224x224).
*   **Global Average Pooling**: At the end of the network, average the entire feature map. Allows accepting any image size.

### Challenge 2: Data Scarcity
**Scenario**: You have only 500 images of "Defective Parts".
**Solution**:
*   **Heavy Augmentation**: Rotate, flip, color jitter, cutout.
*   **Transfer Learning**: Fine-tune a pretrained model.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: What is Translation Invariance vs. Equivariance?**
> **Answer**:
> *   **Invariance** (Pooling): If I shift the cat in the image, the output (label "Cat") stays the same.
> *   **Equivariance** (Convolution): If I shift the input, the feature map shifts by the same amount.

**Q2: Calculate the Receptive Field.**
> **Answer**: The receptive field is the region of the input image that a particular neuron "sees". Deeper neurons have larger receptive fields. A 3x3 conv on top of a 3x3 conv gives a 5x5 receptive field.

**Q3: Why use 3x3 filters instead of 7x7?**
> **Answer**: A stack of three 3x3 layers has the same receptive field (7x7) as one 7x7 layer, but with fewer parameters and more non-linearities (ReLU between layers), making it more discriminative and efficient.

---

## 5. Further Reading
- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/)
- [Visualizing CNN Features](https://distill.pub/2017/feature-visualization/)
