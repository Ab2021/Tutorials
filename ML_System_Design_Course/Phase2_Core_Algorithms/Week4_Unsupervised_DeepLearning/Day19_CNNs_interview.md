# Day 19: CNNs - Interview Questions

> **Topic**: Computer Vision
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Why use CNNs for images instead of MLPs?
**Answer:**
*   **Parameter Sharing**: Same filter applied everywhere. Drastically reduces parameters.
*   **Translation Invariance**: Recognizes object regardless of position.
*   **Local Connectivity**: Exploits spatial structure.

### 2. Explain the Convolution operation.
**Answer:**
*   Sliding a kernel (filter) over the input.
*   Element-wise multiplication and sum.
*   Detects features (edges, textures).

### 3. What is Padding? Why is it used?
**Answer:**
*   Adding zeros around the border.
*   **Valid**: No padding. Output shrinks.
*   **Same**: Output size = Input size. Preserves spatial dimensions.

### 4. What is Stride?
**Answer:**
*   Step size of the filter.
*   Stride 2 halves the spatial dimension (Downsampling).

### 5. What is Pooling (Max vs Average)?
**Answer:**
*   Downsampling operation.
*   **Max**: Takes max value in window. Captures most prominent feature. Invariant to small shifts.
*   **Average**: Takes mean. Smooths.

### 6. How do you calculate the output size of a Conv layer?
**Answer:**
*   $W_{out} = \lfloor \frac{W_{in} - K + 2P}{S} \rfloor + 1$.

### 7. What is a Receptive Field?
**Answer:**
*   The region of the input image that affects a particular neuron in the output.
*   Deeper layers have larger receptive fields (see the whole image).

### 8. Explain the architecture of ResNet. Why does it work?
**Answer:**
*   **Skip Connections**: Add input to output ($y = F(x) + x$).
*   Allows gradients to flow directly to earlier layers (Identity mapping).
*   Solves Vanishing Gradient for very deep networks (100+ layers).

### 9. What is a 1x1 Convolution? Why is it useful?
**Answer:**
*   Convolution with 1x1 kernel.
*   Acts as a linear projection across channels.
*   Used to **reduce dimensionality** (channels) or add non-linearity without changing spatial size.

### 10. What is Inception Module (GoogleNet)?
**Answer:**
*   Parallel branches with different filter sizes (1x1, 3x3, 5x5).
*   Concatenates outputs.
*   Lets network decide optimal scale.

### 11. What is Depthwise Separable Convolution (MobileNet)?
**Answer:**
*   Factorizes standard conv into:
    1.  **Depthwise**: Spatial conv per channel.
    2.  **Pointwise**: 1x1 conv to mix channels.
*   Drastically reduces computation (FLOPS). Good for mobile.

### 12. What is Transfer Learning in CNNs?
**Answer:**
*   Use VGG/ResNet pretrained on ImageNet.
*   Replace last FC layer. Retrain.

### 13. What is Data Augmentation for images?
**Answer:**
*   Random Rotation, Flip, Crop, Color Jitter.
*   Increases effective dataset size. Improves invariance.

### 14. Explain R-CNN, Fast R-CNN, and Faster R-CNN.
**Answer:**
*   **R-CNN**: Extract regions (Selective Search) -> CNN -> SVM. Slow.
*   **Fast R-CNN**: CNN on whole image -> ROI Pooling -> FC.
*   **Faster R-CNN**: Replaces Selective Search with **Region Proposal Network (RPN)**. End-to-end.

### 15. What is YOLO (You Only Look Once)?
**Answer:**
*   Single-stage detector.
*   Divides image into grid. Each cell predicts Bounding Box + Class.
*   Fast real-time detection.

### 16. What is Semantic Segmentation vs Instance Segmentation?
**Answer:**
*   **Semantic**: Classify every pixel (Sky, Road, Car). All cars are same color.
*   **Instance**: Detect individual objects. Car 1 is different from Car 2.

### 17. What is U-Net?
**Answer:**
*   Encoder-Decoder architecture with Skip Connections.
*   Used for Segmentation (Biomedical).
*   Preserves high-resolution details.

### 18. What is the Vanishing Gradient problem in CNNs?
**Answer:**
*   Same as MLPs. Solved by ResNet (Skip connections) and Batch Norm.

### 19. How does a CNN handle different image sizes?
**Answer:**
*   Standard CNN with FC layers requires fixed size.
*   **FCN (Fully Convolutional Network)** or **Global Average Pooling** allows arbitrary input size.

### 20. What is Global Average Pooling?
**Answer:**
*   Replaces Flatten + FC layers.
*   Takes average of each feature map.
*   Reduces parameters. Prevents overfitting.
