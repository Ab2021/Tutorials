# Day 33 Interview Questions: Advanced GANs

## Q1: What is the main difference between Pix2Pix and CycleGAN?
**Answer:**
*   **Pix2Pix:** Supervised. Requires **paired** training data (e.g., same photo in Day and Night). Uses L1 loss + Adversarial loss.
*   **CycleGAN:** Unsupervised. Requires **unpaired** data (e.g., set of Day photos, set of Night photos). Uses Cycle Consistency loss + Adversarial loss.

## Q2: Explain Cycle Consistency Loss.
**Answer:**
*   It enforces that if we translate an image to the other domain and back, we should recover the original image.
*   $x \to G(x) \to F(G(x)) \approx x$.
*   Without this, the generator $G$ could map $x$ to *any* random image in domain $Y$ that looks real, ignoring the content of $x$. This loss preserves content.

## Q3: What is the purpose of the Mapping Network in StyleGAN?
**Answer:**
*   To disentangle the latent space.
*   The input noise $z$ must follow a fixed Gaussian distribution.
*   The intermediate latent code $w$ is free to learn a distribution that matches the density of the real data features.
*   This makes semantic editing (e.g., adding glasses) linear and independent in $W$ space.

## Q4: How does StyleGAN generate coarse and fine details separately?
**Answer:**
*   It injects the style vector $w$ at different layers of the synthesis network.
*   **Early layers ($4 \times 4$ - $8 \times 8$):** Control pose, face shape, general style.
*   **Middle layers ($16 \times 16$ - $32 \times 32$):** Control facial features, eyes, hair style.
*   **Fine layers ($64 \times 64$ - $1024 \times 1024$):** Control color scheme, micro-texture.

## Q5: What is AdaIN?
**Answer:**
Adaptive Instance Normalization.
*   It aligns the mean and variance of the content features with the mean and variance of the style features.
*   It is the core mechanism for style transfer.

## Q6: Why does CycleGAN use 2 Generators and 2 Discriminators?
**Answer:**
*   We need to translate both ways: $X \to Y$ and $Y \to X$.
*   $G: X \to Y$, $D_Y$ (Checks if $Y$ is real).
*   $F: Y \to X$, $D_X$ (Checks if $X$ is real).
*   Total 4 networks.

## Q7: What is the "Truncation Trick" in StyleGAN?
**Answer:**
*   Sampling from the tails of the distribution (extreme values of $w$) produces weird/unrealistic images.
*   We move $w$ closer to the mean $\bar{w}$: $w' = \bar{w} + \psi (w - \bar{w})$.
*   If $\psi < 1$, diversity decreases but quality improves.
*   If $\psi = 0$, it always generates the "average face".

## Q8: How does Pix2Pix ensure the output matches the input structure?
**Answer:**
*   It uses **L1 Loss** between the generated image and the ground truth target.
*   $L1 = ||y_{true} - G(x)||_1$.
*   This forces the generator to respect the low-frequency structure (edges, shapes) of the target.

## Q9: What is "Perceptual Path Length" (PPL)?
**Answer:**
*   A metric used in StyleGAN to measure disentanglement.
*   It measures how much the image changes perceptually when we interpolate between two points in latent space.
*   A smooth, disentangled space has low PPL (changes are gradual and consistent).

## Q10: Implement AdaIN logic.
**Answer:**
```python
def adain(x, style_mean, style_std):
    # x: (N, C, H, W)
    # style_mean, style_std: (N, C, 1, 1)
    
    x_mean = x.mean(dim=[2, 3], keepdim=True)
    x_std = x.std(dim=[2, 3], keepdim=True) + 1e-8
    
    return style_std * (x - x_mean) / x_std + style_mean
```
