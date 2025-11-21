# Day 13: Transfer Learning - Interview Questions

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: Fine-tuning, Adaptation, and Regularization

### 1. When should you freeze the backbone vs fine-tune it?
**Answer:**
*   **Freeze (Feature Extraction)**: Small dataset (prevents overfitting), Similar domain.
*   **Fine-tune**: Large dataset, Distant domain (need to adapt features).

### 2. What is "Differential Learning Rate"?
**Answer:**
*   Using different learning rates for different parts of the network.
*   Lower LR for early layers (generic features, fragile).
*   Higher LR for late layers/head (task-specific, need to learn fast).

### 3. Why do we replace the final Fully Connected layer?
**Answer:**
*   The pre-trained FC layer outputs ImageNet classes (1000).
*   Our target task has $N$ classes.
*   The weights in the old FC layer are specific to ImageNet categories and useless for our task.

### 4. What is "Catastrophic Forgetting"?
**Answer:**
*   The tendency of a neural network to completely forget previously learned information upon learning new information.
*   Happens because the weights are overwritten to minimize loss on the new task.

### 5. How does LoRA work?
**Answer:**
*   Low-Rank Adaptation.
*   Instead of updating the full weight matrix $W$, we learn a delta $\Delta W = BA$.
*   $B$ and $A$ are low-rank matrices.
*   We freeze $W$ and only train $B$ and $A$.
*   Efficient storage and training.

### 6. Explain "Linear Probing".
**Answer:**
*   Training a linear classifier on top of the frozen features of a pre-trained model.
*   Used to evaluate the quality of representations (Self-Supervised Learning evaluation).

### 7. What is the risk of fine-tuning with a small batch size?
**Answer:**
*   **BatchNorm Instability**: BN statistics (mean/var) are noisy with small batches.
*   Updating BN stats can destroy the pre-trained feature distribution.
*   Fix: Freeze BN layers (`eval()` mode) or use GroupNorm.

### 8. What is "Domain Adaptation"?
**Answer:**
*   Techniques to adapt a model trained on Source Domain to Target Domain where labels might be scarce.
*   Example: Train on GTA V (Sim), Test on Real World (Real).
*   Methods: Adversarial training, Feature alignment.

### 9. Why is ImageNet pre-training so effective?
**Answer:**
*   ImageNet forces the model to learn a rich hierarchy of features (Edges $\to$ Shapes $\to$ Objects).
*   These features are largely transferable to other visual tasks.

### 10. What is "Warmup" in the context of fine-tuning?
**Answer:**
*   Starting with a very low LR and increasing it.
*   Crucial when fine-tuning to avoid "shocking" the pre-trained weights with large gradients from the randomly initialized head.

### 11. Can you transfer from a ResNet to a Vision Transformer?
**Answer:**
*   No. The architectures are different. Weights don't map.
*   You can only transfer knowledge via **Distillation** (Teacher-Student).

### 12. What is "Knowledge Distillation"?
**Answer:**
*   Training a small Student model to mimic the outputs (logits) of a large Teacher model.
*   Loss = Soft targets (KL Div) + Hard targets (Cross Entropy).

### 13. How do you handle input size mismatch?
**Answer:**
*   CNNs (Convolutional parts) are agnostic to input size.
*   The issue is the FC layer or Positional Embeddings.
*   **Global Average Pooling** solves the FC issue.
*   **Interpolation** solves Positional Embedding issue (in ViTs).

### 14. What is "Test-Time Adaptation"?
**Answer:**
*   Updating the model (usually just BN stats) on the test sample itself before prediction.
*   Helps with distribution shift.

### 15. Why might a pre-trained model perform worse than random initialization?
**Answer:**
*   **Negative Transfer**: Source and Target domains are conflicting (e.g., transferring from MNIST (Text) to Chest X-Ray (Texture)).
*   The features learned are misleading for the new task.

### 16. What is "Co-tuning"?
**Answer:**
*   Training on both Source and Target datasets simultaneously (if Source data is available).
*   Prevents forgetting and improves transfer.

### 17. Explain "Layer-wise Freezing".
**Answer:**
*   Unfreezing layers gradually from top to bottom during training.
*   Allows the head to stabilize before modifying the backbone.

### 18. What is the "Head" of the network?
**Answer:**
*   The final layers specific to the task (Classification layer, Bounding Box regressor).
*   The "Body" or "Backbone" is the feature extractor.

### 19. How does "Weight Decay" affect fine-tuning?
**Answer:**
*   It pulls weights towards zero.
*   In L2-SP, we modify it to pull weights towards *pre-trained values*.

### 20. What is "Self-Supervised Pre-training" (SSL)?
**Answer:**
*   Training on data without labels (SimCLR, MAE, DINO).
*   Produces features that often transfer *better* than Supervised ImageNet pre-training for downstream tasks like segmentation.
