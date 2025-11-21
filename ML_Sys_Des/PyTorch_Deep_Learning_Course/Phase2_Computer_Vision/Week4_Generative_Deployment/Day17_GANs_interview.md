# Day 17: GANs - Interview Questions

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: Adversarial Learning, Stability, and Architectures

### 1. Why is GAN training unstable?
**Answer:**
*   It's a Minimax game (saddle point optimization), not a minimization problem.
*   G and D must balance. If D gets too strong, G gets no gradients. If G gets too strong, D gives up.
*   Oscillations and Mode Collapse are common.

### 2. What is "Mode Collapse"?
**Answer:**
*   The Generator produces limited variety (e.g., only generates one specific face).
*   Happens because G finds a single output that successfully fools D, and keeps producing it. D adapts, G moves to another single point. They play "Whac-A-Mole".

### 3. Explain the difference between BCE Loss and Wasserstein Loss.
**Answer:**
*   **BCE**: $-\log(D(x)) - \log(1-D(G(z)))$. Saturates if D is perfect.
*   **Wasserstein**: $D(x) - D(G(z))$. Linear. Provides useful gradients even when distributions are far apart. Requires Lipschitz constraint.

### 4. Why do we use LeakyReLU in the Discriminator?
**Answer:**
*   To allow gradients to flow even when the input is negative.
*   In D, we want gradients from both real and fake samples to flow back to G. A dead ReLU would block information.

### 5. What is "Label Smoothing" in GANs?
**Answer:**
*   Replacing label 1.0 (Real) with 0.9.
*   Prevents D from being over-confident.
*   If D is too confident, gradients vanish.

### 6. How does CycleGAN work without paired data?
**Answer:**
*   It learns two mappings: $G: X \to Y$ and $F: Y \to X$.
*   **Cycle Consistency**: $F(G(x)) \approx x$.
*   This constraint forces the mapping to preserve the content of $x$ while changing the style to $Y$.

### 7. What is "Spectral Normalization"?
**Answer:**
*   A technique to enforce Lipschitz continuity in the Discriminator.
*   Normalizes the weight matrix $W$ by its largest singular value (Spectral Norm) at each layer.
*   Stabilizes training (standard in BigGAN/SNGAN).

### 8. Why do we detach `fake_imgs` when training the Discriminator?
**Answer:**
*   We only want to update D's weights to classify fakes better.
*   We do *not* want to update G's weights during D's training step.
*   `detach()` cuts the computational graph.

### 9. What is the "Receptive Field" importance in GANs?
**Answer:**
*   The Discriminator needs a large receptive field to judge global coherence (e.g., does the face have two eyes?).
*   PatchGAN discriminators classify $N \times N$ patches as real/fake, enforcing local texture realism.

### 10. Explain "Progressive Growing" in StyleGAN.
**Answer:**
*   Start training with $4 \times 4$ images.
*   Stabilize.
*   Add layers to G and D to handle $8 \times 8$. Fade them in smoothly.
*   Repeat until $1024 \times 1024$.
*   Makes training high-res GANs feasible.

### 11. What is "Fréchet Inception Distance" (FID)?
**Answer:**
*   The standard metric for GAN evaluation.
*   Computes distance between feature distributions of Real and Fake images (extracted from InceptionV3).
*   Lower FID = Better quality and diversity.

### 12. Why is Tanh used in the Generator output?
**Answer:**
*   To bound the output pixel values to $[-1, 1]$.
*   Matches the input data normalization (usually mean 0.5, std 0.5).

### 13. What is "Pix2Pix"?
**Answer:**
*   A conditional GAN for paired image-to-image translation.
*   Uses L1 Loss (pixel-wise) + GAN Loss (realism).
*   L1 ensures correctness, GAN ensures sharpness.

### 14. What is the "Latent Space" $z$ in GANs?
**Answer:**
*   The input noise vector (usually Gaussian).
*   The Generator learns to map this noise distribution to the image manifold.
*   Interpolating in $z$ produces smooth morphs between images.

### 15. How does "Gradient Penalty" enforce 1-Lipschitz?
**Answer:**
*   A function is 1-Lipschitz if its gradient norm is $\le 1$ everywhere.
*   WGAN-GP penalizes the model if the gradient norm moves away from 1.

### 16. What is "Inception Score" (IS)?
**Answer:**
*   Older metric.
*   Measures two things:
    1.  **Quality**: Images look like specific objects (Low entropy conditional class prob).
    2.  **Diversity**: Images cover all classes (High entropy marginal class prob).
*   Flawed because it doesn't compare to real data statistics.

### 17. What is "AdaIN" (Adaptive Instance Normalization)?
**Answer:**
*   Used in StyleGAN.
*   Aligns the mean and variance of content features to match the mean and variance of style features.
*   $AdaIN(x, y) = \sigma(y) (\frac{x - \mu(x)}{\sigma(x)}) + \mu(y)$.

### 18. Why use Strided Convolutions instead of Pooling in GANs?
**Answer:**
*   Pooling loses information (location).
*   Strided Convolutions allow the network to learn its own spatial downsampling/upsampling.

### 19. What is "Truncation Trick" in sampling?
**Answer:**
*   Sampling $z$ from a truncated normal distribution (cutting off tails).
*   Trades diversity for quality. (Avoids weird artifacts from extreme latent points).

### 20. Can GANs be used for Data Augmentation?
**Answer:**
*   Yes. Generating synthetic data for rare classes (e.g., medical lesions) to train classifiers.
