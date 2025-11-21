# Day 17 Deep Dive: Contrastive Loss & Collapse

## 1. InfoNCE Loss Derivation
**InfoNCE** (Noise Contrastive Estimation) aims to identify the positive sample from a set of noise samples.
$$ L = -\log \frac{\exp(z_i \cdot z_j / \tau)}{\exp(z_i \cdot z_j / \tau) + \sum_{neg} \exp(z_i \cdot z_{neg} / \tau)} $$
*   **Numerator:** Pull positive pair together (Alignment).
*   **Denominator:** Push positive away from negatives (Uniformity).
*   **Temperature ($\tau$):** Controls how "peaked" the distribution is. Low $\tau$ focuses on hardest negatives.

## 2. The Collapse Problem
**Trivial Solution:** The network outputs a constant vector $c$ for all inputs.
*   Distance is always 0. Loss is minimized (in non-contrastive methods).
*   **Contrastive methods** avoid this by pushing negatives apart.
*   **Non-contrastive (BYOL/SimSiam)** avoid this via:
    *   **Stop Gradient:** Target network is fixed.
    *   **Predictor Head:** Adds asymmetry.

## 3. SwAV (Swapping Assignments between Views)
**Clustering-based SSL.**
*   Instead of comparing features directly, compare cluster assignments.
*   Compute "codes" (cluster assignments) for view 1 and view 2.
*   Predict code of view 1 using feature of view 2.
*   **Multi-Crop Strategy:** Use 2 large crops + 6 small crops. Greatly boosts performance.

## 4. MAE vs Contrastive
*   **Contrastive (SimCLR/MoCo):** Invariant to augmentation. Good for semantic classification.
*   **Reconstructive (MAE):** Predicts pixel details. Good for dense tasks (segmentation/detection) and fine-grained recognition.

## 5. DINO (Self-Distillation with no Labels)
**ViT-specific SSL.**
*   Student and Teacher networks.
*   Output probability distributions (Softmax).
*   Minimize Cross-Entropy between Student and Teacher.
*   **Centering & Sharpening:** Prevents collapse.
*   **Result:** Attention maps automatically segment objects!

```python
# SimCLR Loss Implementation
def nt_xent_loss(z1, z2, temperature=0.5):
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    
    # Mask out self-similarity
    mask = torch.eye(2*N, dtype=torch.bool).to(z.device)
    sim.masked_fill_(mask, -float('inf'))
    
    # Positive pairs
    target = torch.arange(2*N).to(z.device)
    target[0:N] = target[0:N] + N
    target[N:2*N] = target[N:2*N] - N
    
    loss = F.cross_entropy(sim, target)
    return loss
```
