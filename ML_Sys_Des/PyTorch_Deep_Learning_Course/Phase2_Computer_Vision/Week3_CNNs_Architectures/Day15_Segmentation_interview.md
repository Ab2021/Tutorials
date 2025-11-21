# Day 15: Segmentation - Interview Questions

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: U-Net, Mask R-CNN, and Metrics

### 1. What is the difference between Semantic and Instance Segmentation?
**Answer:**
*   **Semantic**: Classifies every pixel. All "cars" are the same class. Cannot distinguish individual cars.
*   **Instance**: Detects individual objects and segments them. "Car 1" is different from "Car 2".

### 2. Why does U-Net have Skip Connections?
**Answer:**
*   The Encoder downsamples the image, losing spatial information (fine details) to gain semantic information.
*   The Decoder upsamples, but cannot recover the lost details from the low-res feature map alone.
*   Skip connections copy high-res features from the Encoder directly to the Decoder, allowing precise localization of boundaries.

### 3. What is "Dice Loss"? Why use it over Cross Entropy?
**Answer:**
*   $Dice = \frac{2 |A \cap B|}{|A| + |B|}$.
*   Cross Entropy evaluates every pixel equally. In segmentation, 90% of pixels might be background. A model predicting "All Background" gets 90% accuracy (but 0.1 loss).
*   Dice Loss focuses on the overlap of the *foreground* class, handling class imbalance naturally.

### 4. Explain "Transposed Convolution".
**Answer:**
*   A learnable upsampling layer.
*   It broadcasts each input pixel to a region defined by the kernel.
*   Mathematically equivalent to the backward pass of a standard convolution.

### 5. What are "Checkerboard Artifacts"?
**Answer:**
*   Grid-like patterns in the output of Transposed Convolutions.
*   Caused when the kernel size is not divisible by the stride, leading to uneven overlap of the kernel outputs.
*   Fix: Use Bilinear Upsampling + Conv.

### 6. What is "RoI Align" in Mask R-CNN?
**Answer:**
*   An improvement over RoI Pooling.
*   RoI Pooling quantizes coordinates (rounds to integer), causing misalignment between the mask and the original image.
*   RoI Align uses bilinear interpolation to compute exact values at floating-point coordinates.

### 7. What is "Atrous Spatial Pyramid Pooling" (ASPP)?
**Answer:**
*   Used in DeepLab.
*   Applies dilated convolutions with different rates in parallel.
*   Captures multi-scale context (local and global) without resizing the image.

### 8. How do you handle "Class Imbalance" in segmentation?
**Answer:**
*   **Loss**: Dice Loss, Focal Loss, Tversky Loss.
*   **Sampling**: Patch-based training (sample patches centered on foreground objects).

### 9. What is "Panoptic Segmentation"?
**Answer:**
*   Unifies Semantic and Instance segmentation.
*   Assigns a unique label (Category + Instance ID) to every pixel.
*   Handles both "Stuff" (Sky, Road) and "Things" (Cars, People).

### 10. Why is U-Net symmetric?
**Answer:**
*   To ensure that the feature map sizes at corresponding Encoder and Decoder levels match, allowing for concatenation (Skip Connections).

### 11. What is the output shape of a Segmentation network?
**Answer:**
*   $(N, C, H, W)$, where $C$ is the number of classes.
*   Each pixel has a probability distribution over classes.

### 12. What is "Mean IoU" (mIoU)?
**Answer:**
*   Calculate IoU for each class: $\frac{TP}{TP + FP + FN}$.
*   Average over all classes.
*   Standard metric for Semantic Segmentation.

### 13. How does Mask R-CNN generate masks?
**Answer:**
*   It has a separate "Mask Head" (FCN) that runs on the RoI features.
*   It predicts a $28 \times 28$ binary mask for each class.
*   The mask is then resized to the bounding box size.

### 14. What is "Dilated Convolution"?
**Answer:**
*   Convolution with holes.
*   Increases Receptive Field without downsampling.
*   Allows dense feature extraction at high resolution.

### 15. Can you use a Transformer for Segmentation?
**Answer:**
*   Yes (SegFormer, Mask2Former).
*   Transformers are excellent at capturing global context (Self-Attention), which helps in resolving ambiguous regions.

### 16. What is "Tversky Loss"?
**Answer:**
*   Generalization of Dice Loss.
*   Allows weighting False Positives and False Negatives differently.
*   $T = \frac{TP}{TP + \alpha FP + \beta FN}$.

### 17. How do you evaluate Instance Segmentation?
**Answer:**
*   **Mask mAP**: Similar to Box mAP, but IoU is calculated on the Mask overlap, not the Box overlap.

### 18. What is the "Bottleneck" in U-Net?
**Answer:**
*   The lowest resolution layer between Encoder and Decoder.
*   Contains the most abstract, high-level semantic representation of the image.

### 19. Why do we use $1 \times 1$ convolution at the end of U-Net?
**Answer:**
*   To map the feature vector (e.g., 64 channels) at each pixel to the desired number of classes (e.g., 1 channel for binary mask).

### 20. What is "Conditional Random Field" (CRF) post-processing?
**Answer:**
*   A probabilistic graphical model used to refine segmentation boundaries.
*   It ensures that pixels with similar color/intensity likely have the same label.
*   Used in early DeepLab versions to sharpen edges.
