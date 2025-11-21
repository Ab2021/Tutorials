# Day 17: GANs - Deep Dive

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: WGAN, Gradient Penalty, and StyleGAN

## 1. Wasserstein GAN (WGAN)

Standard GAN minimizes Jensen-Shannon (JS) Divergence.
JS Divergence is bad when distributions don't overlap (gradients vanish).
**Wasserstein Distance (Earth Mover's Distance)**:
*   Cost to move pixels from distribution G to D.
*   Provides smooth gradients everywhere.

**Changes**:
1.  Remove Sigmoid from D (Output is a scalar score, not probability).
2.  Loss: $D(x) - D(G(z))$.
3.  **Constraint**: D must be **1-Lipschitz** continuous.

## 2. Enforcing Lipschitz: Gradient Penalty (WGAN-GP)

Original WGAN used Weight Clipping (bad).
**Gradient Penalty**:
Add a penalty if the gradient norm of D deviates from 1.
$$ L = L_{orig} + \lambda E[(||\nabla D(\hat{x})||_2 - 1)^2] $$
*   $\hat{x}$: Random points interpolated between Real and Fake.
*   Standard for stable GAN training.

## 3. StyleGAN (NVIDIA)

The state-of-the-art for high-res faces.
Key Innovations:
1.  **Mapping Network**: Maps $z \to w$ (Intermediate Latent Space). Disentangles features.
2.  **AdaIN (Adaptive Instance Norm)**: Injects $w$ into every layer to control style (normalization statistics).
3.  **Progressive Growing**: Train on $4 \times 4$, then $8 \times 8$, ..., up to $1024 \times 1024$.
4.  **Noise Injection**: Adds random noise at each layer to control stochastic details (hair placement, freckles).

## 4. Conditional GAN (cGAN)

Generating specific classes (e.g., "Generate a Cat").
Feed Class Label $y$ to both G and D.
*   G: Concatenate $z$ and $y$ (embedding).
*   D: Concatenate $x$ and $y$.
*   **Pix2Pix**: Image-to-Image translation (Edges $\to$ Photo).

## 5. CycleGAN

Unpaired Image-to-Image translation (Horse $\to$ Zebra without paired examples).
**Cycle Consistency Loss**:
$$ F(G(x)) \approx x $$
*   Horse $\to$ Zebra $\to$ Horse should return original image.
*   Prevents mode collapse and enforces structure preservation.
