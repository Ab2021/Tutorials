# Day 33: Advanced GANs (CycleGAN & StyleGAN)

## 1. Image-to-Image Translation
**Goal:** Translate an image from Domain $X$ (e.g., Horse) to Domain $Y$ (e.g., Zebra).

### Pix2Pix (2017)
*   **Paired Data:** Requires aligned pairs (e.g., Sketch $\leftrightarrow$ Photo).
*   **Architecture:** Conditional GAN (cGAN).
    *   $G(x, z) \to y$.
    *   $D(x, y)$ checks if pair is real.
*   **Loss:** Adversarial Loss + L1 Loss ($||y - G(x)||_1$).

### CycleGAN (2017)
*   **Unpaired Data:** No aligned pairs needed (just a folder of Horses and a folder of Zebras).
*   **Cycle Consistency Loss:**
    *   $G: X \to Y$ and $F: Y \to X$.
    *   $x \to G(x) \to F(G(x)) \approx x$.
    *   "If I translate English to French and back, I should get the same sentence."
*   **Loss:** $L_{GAN} + \lambda L_{cyc}$.

## 2. StyleGAN (2019)
**Goal:** Generate high-resolution, photorealistic faces with controllable attributes.
**Key Innovations:**
1.  **Mapping Network:** Maps $z \in \mathcal{Z}$ (Normal) to $w \in \mathcal{W}$ (Disentangled).
    *   8 FC layers.
2.  **AdaIN (Adaptive Instance Normalization):**
    *   Injects style $w$ into the generator at every scale.
    *   $\text{AdaIN}(x, y) = \sigma(y) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(y)$.
3.  **Progressive Growing:** Starts generating $4 \times 4$, then $8 \times 8$, up to $1024 \times 1024$. (Later replaced by StyleGAN2 architecture).
4.  **Noise Injection:** Adds random noise after each layer to generate stochastic details (hair, freckles).

## 3. Style Mixing
*   Use latent code $w_1$ for coarse layers (Pose, Face Shape).
*   Use latent code $w_2$ for fine layers (Color, Texture).
*   **Result:** Face of Person A with hair/color of Person B.

## Summary
CycleGAN solved the data problem (no pairs needed). StyleGAN solved the quality and control problem, becoming the gold standard for image synthesis.
