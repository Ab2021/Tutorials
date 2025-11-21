# Day 9 Deep Dive: Transfer Learning Nuances

## 1. Batch Normalization in Fine-Tuning
**Trap:** When freezing the backbone, should you freeze BN statistics (mean/var)?
*   **Yes (model.eval()):** Use ImageNet stats. Good if target data is similar.
*   **No (model.train()):** Update stats on target data. Good if target domain is different (e.g., Medical images).
*   **PyTorch Default:** `model.eval()` freezes dropout and BN stats.

## 2. Resolution Discrepancy
Pretrained models are usually $224 \times 224$.
*   **Fine-tuning at higher resolution:** (e.g., $384 \times 384$).
*   Performance usually improves.
*   **Issue:** Positional embeddings (ViT) or Global Pooling (CNN) need adaptation.
*   **Fix:** Interpolate positional embeddings.

## 3. Linear Probing vs Fine-Tuning (LP-FT)
**Paper:** "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution" (Kumar et al., 2022).
*   **Problem:** Fine-tuning immediately can distort good features.
*   **Solution (LP-FT):**
    1.  **Linear Probe:** Train head only until convergence.
    2.  **Fine-Tune:** Unfreeze all and train with low LR.
*   Achieves better accuracy and robustness.

## 4. Test-Time Augmentation (TTA)
Not training, but a transfer technique.
*   At inference, run image with multiple augmentations (flip, crop).
*   Average predictions.
*   Boosts accuracy by 1-2%.

## 5. Knowledge Distillation
Transfer knowledge from a large Teacher (ResNet101) to a small Student (ResNet18).
$$ L = \alpha L_{CE}(y, \hat{y}) + (1-\alpha) L_{KL}(\sigma(z_t/T), \sigma(z_s/T)) $$
*   **Soft Targets:** Student learns "dark knowledge" (e.g., a dog looks a bit like a cat).

## Summary
Fine-tuning is an art. Managing BN, resolution, and learning rate schedules (LP-FT) separates good models from great ones.
