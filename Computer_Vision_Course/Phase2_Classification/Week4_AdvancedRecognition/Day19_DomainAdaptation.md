# Day 19: Domain Adaptation

## 1. The Domain Shift Problem
**Scenario:**
*   **Source Domain ($D_S$):** Labeled data (e.g., Synthetic images from GTA5).
*   **Target Domain ($D_T$):** Unlabeled data (e.g., Real world Cityscapes).
*   **Problem:** A model trained on $D_S$ fails on $D_T$ because the distributions $P(X_S) \neq P(X_T)$ differ (lighting, texture, camera).

**Goal:** Learn a model that performs well on $D_T$ using labeled $D_S$ and unlabeled $D_T$.

## 2. Adversarial Adaptation (DANN)
**Domain-Adversarial Neural Network (Ganin et al., 2016).**
**Idea:** Learn features that are **discriminative** for the class (dog vs cat) but **indistinguishable** for the domain (source vs target).

**Architecture:**
1.  **Feature Extractor ($G_f$):** Maps input to feature vector.
2.  **Label Predictor ($G_y$):** Predicts class label (Standard Cross-Entropy).
3.  **Domain Classifier ($G_d$):** Predicts domain (Source=0, Target=1).

**Gradient Reversal Layer (GRL):**
*   Forward pass: Identity (does nothing).
*   Backward pass: Multiplies gradient by $-\lambda$.
*   **Effect:**
    *   $G_d$ tries to minimize domain classification error (distinguish domains).
    *   $G_f$ tries to **maximize** domain classification error (fool $G_d$).
    *   Result: $G_f$ learns domain-invariant features.

```python
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)
```

## 3. Image-to-Image Translation (CycleGAN)
**Idea:** Translate Source images to look like Target images (Pixel-level adaptation).
*   Train a GAN to convert GTA5 $\to$ Cityscapes.
*   Train a classifier on the "Fake Cityscapes" images (which now have labels).

**Cycle Consistency Loss:**
*   $G: X \to Y$, $F: Y \to X$.
*   $F(G(x)) \approx x$.
*   Ensures content is preserved during translation (e.g., a car stays a car).

## 4. Maximum Mean Discrepancy (MMD)
**Statistic-based approach.**
*   Minimize the distance between the mean embeddings of Source and Target distributions in a Kernel Hilbert Space.
*   $L = L_{class} + \lambda || \mu_S - \mu_T ||^2$.

## Summary
Domain Adaptation is crucial for deploying models in the real world where data differs from training sets. DANN (Feature-level) and CycleGAN (Pixel-level) are the two main strategies.
