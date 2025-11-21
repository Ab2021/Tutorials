# Day 17 Interview Questions: Self-Supervised Learning

## Q1: Why does SimCLR require a large batch size?
**Answer:**
*   SimCLR uses the other images in the current batch as **negative samples**.
*   A larger batch size provides more negatives, making the task harder and forcing the model to learn better representations.
*   Small batches lead to poor performance because the negatives are not diverse enough.

## Q2: How does MoCo solve the batch size problem?
**Answer:**
*   It maintains a **Queue** (Memory Bank) of past representations to serve as negatives.
*   This decouples the number of negatives from the mini-batch size.
*   You can have a queue of 65k negatives while using a batch size of 256.

## Q3: What is the difference between SimCLR and BYOL?
**Answer:**
*   **SimCLR:** Contrastive. Requires negative pairs. Tries to push different images apart.
*   **BYOL:** Non-contrastive. Only uses positive pairs. Tries to predict the representation of the augmented view. Relies on asymmetry (Predictor + Stop Gradient) to prevent collapse.

## Q4: Explain the "Masking" strategy in MAE.
**Answer:**
*   MAE divides the image into patches and randomly masks a high percentage (e.g., 75%).
*   The Encoder only sees the **visible** patches (saving computation).
*   The Decoder takes the encoded visible patches + learnable **mask tokens** and tries to reconstruct the original pixels of the masked patches.

## Q5: Why is Strong Data Augmentation critical for Contrastive Learning?
**Answer:**
*   If augmentations are weak, the model can cheat.
*   Example: If we only use color jitter, the model might just match color histograms.
*   **Random Crop** is the most important augmentation. It forces the model to recognize that a "dog head" and "dog leg" belong to the same object class.

## Q6: What is "Mode Collapse" in SSL?
**Answer:**
When the encoder maps **all** inputs to the same constant vector output.
*   In this case, the distance between positive pairs is 0 (perfect), but the representation is useless.
*   Contrastive learning prevents this by penalizing similarity to negatives.

## Q7: What is the "Projection Head" and why do we throw it away?
**Answer:**
*   It is a small MLP (Linear-ReLU-Linear) placed after the encoder.
*   The contrastive loss is applied to the output of this head.
*   **Why throw away?** The head learns to be invariant to augmentations (e.g., rotation, color), which might be useful information for downstream tasks. The representation *before* the head retains more information.

## Q8: How do we evaluate Self-Supervised models?
**Answer:**
1.  **Linear Probing:** Freeze the encoder, train a linear classifier on labeled data (ImageNet).
2.  **Fine-Tuning:** Unfreeze encoder, fine-tune on labeled data (1% or 10% labels).
3.  **Transfer Learning:** Test on different datasets (detection, segmentation).

## Q9: What is the difference between Autoencoders and MAE?
**Answer:**
*   **Standard Autoencoder:** Compresses the *entire* input into a bottleneck.
*   **MAE:** Masks most of the input. It is a *denoising* autoencoder that reconstructs missing parts. The high masking ratio (75%) makes it a difficult reasoning task, not just copying.

## Q10: Explain the Momentum Update in MoCo.
**Answer:**
The Key Encoder parameters $\theta_k$ are updated as a moving average of the Query Encoder parameters $\theta_q$:
$$ \theta_k \leftarrow m \theta_k + (1-m) \theta_q $$
*   $m \approx 0.999$.
*   This ensures the key encoder changes slowly and consistently, providing stable negative samples in the queue.
