# Day 32 Deep Dive: Training Stability & Mode Collapse

## 1. Mode Collapse
**Problem:** The Generator finds *one* output that fools the Discriminator and produces it endlessly.
*   Example: Generating only "Shoe" images when trained on MNIST Fashion.
*   **Why?** $G$ is lazy. If "Shoe" works, why learn "Shirt"?
*   **Fix:** Minibatch Discrimination, Wasserstein Loss.

## 2. Vanishing Gradients
**Problem:** If $D$ is too perfect ($D(x)=1, D(G(z))=0$), the gradient $\nabla \log(1-D(G(z)))$ vanishes. $G$ stops learning.
**Solution:**
*   **Non-Saturating Loss:** Maximize $\log D(G(z))$ instead of minimizing $\log(1-D(G(z)))$.
*   **Label Smoothing:** Train $D$ with target 0.9 instead of 1.0.
*   **Noise:** Add noise to inputs of $D$.

## 3. Wasserstein GAN (WGAN)
**Idea:** Replace the "Discriminator" (Classifier) with a "Critic" (Regression).
*   Measures the **Earth Mover's Distance** (Wasserstein Distance) between real and fake distributions.
*   **Loss:** $L = \mathbb{E}[D(x)] - \mathbb{E}[D(G(z))]$.
*   **Constraint:** $D$ must be 1-Lipschitz continuous.
    *   **WGAN-CP:** Weight Clipping (Bad).
    *   **WGAN-GP:** Gradient Penalty (Good). Enforce $||\nabla D(\hat{x})||_2 = 1$.

## 4. Evaluation Metrics
*   **Inception Score (IS):** Measures quality and diversity. High is good.
*   **Fréchet Inception Distance (FID):** Measures distance between feature distributions of Real and Fake data. Low is good. **Standard metric.**

## Summary
Training GANs is notoriously unstable ("Black Magic"). WGAN-GP and FID are essential tools for modern GAN development.
