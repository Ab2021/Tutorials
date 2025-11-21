# Day 20 Deep Dive: Improving Fine-Grained Classification

## 1. Higher Resolution
*   Fine-grained details (beak, eyes) disappear at $224 \times 224$.
*   Increasing input size to $448 \times 448$ or $512 \times 512$ significantly boosts accuracy.
*   **Trade-off:** Slower training, smaller batch size.

## 2. Bilinear CNNs (B-CNN)
**Specialized Architecture for Fine-Grained.**
*   Instead of Global Average Pooling (1st order statistic), use **Outer Product** of features (2nd order statistic).
*   $x = \sum_{i} f_i f_i^T$.
*   Captures complex feature interactions (e.g., "red spot" AND "wing").
*   **Result:** SOTA on CUB-200 for a long time.

## 3. Attention-Based Methods
*   **Weakly Supervised Data Augmentation Network (WS-DAN):**
    *   Uses attention maps to crop discriminative parts (e.g., head) and zoom in.
    *   Feeds both original and cropped images to the network.
*   **Vision Transformers:**
    *   ViT inherently captures global context and local details via attention.
    *   Usually outperforms CNNs on fine-grained tasks if pretrained properly (ImageNet-21k).

## 4. MixUp & CutMix
**Regularization is Key.**
*   CUB-200 is small (11k images). Overfitting is the main enemy.
*   **MixUp:** $x' = \lambda x_i + (1-\lambda) x_j$.
*   **CutMix:** Paste patch of $x_j$ onto $x_i$.
*   These force the model to look at multiple parts of the object.

## 5. Error Analysis
**Common Mistakes:**
*   **Similar Species:** "Common Tern" vs "Forster's Tern".
*   **Background Clutter:** Bird hidden in leaves.
*   **Pose Variation:** Flying vs Sitting.
*   **Fix:** Hard Negative Mining (train more on confused pairs).

## Summary
To push performance beyond the baseline ResNet, we need techniques that explicitly model part interactions (Bilinear Pooling) or focus on discriminative regions (Attention/Cropping).
