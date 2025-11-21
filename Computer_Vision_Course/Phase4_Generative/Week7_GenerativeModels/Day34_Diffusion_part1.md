# Day 34 Deep Dive: Sampling & DDIM

## 1. Sampling (Inference)
Training is fast (parallel over $t$), but sampling is slow (sequential).
Algorithm:
1.  Start with $x_T \sim \mathcal{N}(0, I)$.
2.  For $t = T, \dots, 1$:
    *   Predict noise $\epsilon_\theta(x_t, t)$.
    *   Estimate mean $\mu_\theta$.
    *   Sample $x_{t-1} = \mu_\theta + \sigma_t z$.
3.  Return $x_0$.
*   Requires 1000 forward passes of the U-Net!

## 2. DDIM (Denoising Diffusion Implicit Models)
**Goal:** Speed up sampling (e.g., 1000 steps $\to$ 50 steps).
*   **Idea:** The forward process in DDPM is Markovian ($x_t$ depends only on $x_{t-1}$).
*   DDIM generalizes this to a **Non-Markovian** process that has the same marginals $q(x_t|x_0)$.
*   This allows us to skip steps during inference (deterministic sampling).
*   Result: High quality samples in 10-50 steps.

## 3. Classifier Guidance
**Goal:** Control the generation (e.g., "Generate a Dog").
*   Train a separate classifier $f(x_t, t)$ on noisy images.
*   During sampling, shift the mean by the gradient of the classifier:
    $$ \hat{\epsilon} = \epsilon_\theta(x_t) - w \cdot \nabla_{x_t} \log p(y|x_t) $$
*   Pushes the image towards the class $y$.

## 4. Classifier-Free Guidance
**Goal:** Guidance without training a separate classifier.
*   Train the diffusion model conditionally $p(x|y)$ AND unconditionally $p(x|\varnothing)$ (by dropping labels 10% of time).
*   **Extrapolation:**
    $$ \hat{\epsilon} = \epsilon_\theta(x_t | \varnothing) + w \cdot (\epsilon_\theta(x_t | y) - \epsilon_\theta(x_t | \varnothing)) $$
*   If $w > 1$, we exaggerate the features that make the image look like class $y$.
*   Used in DALL-E 2, Stable Diffusion, Imagen.

## Summary
DDIM made diffusion practical. Classifier-Free Guidance made it controllable and high-fidelity, beating GANs on ImageNet generation.
