# Day 17: Self-Supervised Learning

## 1. The Motivation
**Problem:** Labeling data is expensive and slow.
**Solution:** Learn representations from unlabeled data by solving "pretext tasks".
*   **Old School:** Jigsaw puzzles, Colorization, Rotation prediction.
*   **Modern:** Contrastive Learning & Masked Image Modeling.

## 2. Contrastive Learning (SimCLR, 2020)
**Idea:** Maximize agreement between differently augmented views of the same image (Positives) and minimize agreement between different images (Negatives).

**Framework:**
1.  **Augmentation:** Take image $x$, create two views $\tilde{x}_i, \tilde{x}_j$ (crop, color jitter).
2.  **Encoder:** $h = f(\tilde{x})$ (ResNet).
3.  **Projection Head:** $z = g(h)$ (MLP). Loss is calculated on $z$.
4.  **Loss (InfoNCE):**
    $$ L = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)} $$

**Key Findings:**
*   Strong data augmentation is crucial.
*   Large batch sizes (e.g., 4096) are needed for enough negatives.
*   Projection head improves representation quality.

## 3. Momentum Contrast (MoCo, 2020)
**Problem:** SimCLR needs huge batches for negatives. GPU memory limit.
**Solution:** Decouple batch size from number of negatives using a **Queue**.

**Mechanism:**
*   **Query Encoder ($q$):** Updated by backprop.
*   **Key Encoder ($k$):** Updated by momentum (moving average of $q$).
*   **Queue:** Stores dictionary of past keys (negatives).
*   Allows massive number of negatives (e.g., 65k) with small mini-batches.

## 4. BYOL (Bootstrap Your Own Latent, 2020)
**Shocking Discovery:** You don't need negatives!
**Mechanism:**
*   **Online Network:** Predicts the target representation.
*   **Target Network:** Moving average of Online.
*   **Loss:** Minimize distance between Online prediction and Target projection.
*   **Why no collapse?** Asymmetry (Predictor head in Online only) and Momentum update prevent trivial constant solutions.

## 5. Masked Autoencoders (MAE, 2021)
**Idea:** BERT for Vision.
1.  **Masking:** Randomly mask 75% of image patches.
2.  **Encoder:** Process *only* visible patches (efficient!).
3.  **Decoder:** Reconstruct the missing pixels from latent representation + mask tokens.
4.  **Loss:** MSE on masked pixels.

**Result:** Learns rich, holistic representations. Scales incredibly well.

## Summary
Self-Supervised Learning (SSL) allows pre-training on billions of unlabelled images. MAE and Contrastive Learning are the current SOTA paradigms.
