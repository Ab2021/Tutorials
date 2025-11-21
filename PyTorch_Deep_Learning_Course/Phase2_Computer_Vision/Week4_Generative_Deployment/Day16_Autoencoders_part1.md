# Day 16: Autoencoders - Deep Dive

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: VQ-VAE, ELBO, and Disentanglement

## 1. The Math of VAE (ELBO)

We want to maximize $P(x)$ (Likelihood of data).
Intractable. So we maximize the **Evidence Lower Bound (ELBO)**.
$$ \log P(x) \ge E_{z \sim Q}[\log P(x|z)] - D_{KL}(Q(z|x) || P(z)) $$
*   Term 1: Reconstruction Likelihood (How well does $z$ explain $x$?).
*   Term 2: KL Divergence (How close is our approximate posterior $Q$ to the prior $P$?).

## 2. Vector Quantized VAE (VQ-VAE)

Standard VAEs produce blurry images because of the Gaussian noise and MSE loss.
**VQ-VAE** uses a **Discrete Latent Space**.
1.  **Codebook**: A list of $K$ embedding vectors $e_1, ..., e_K$.
2.  **Encoder**: Outputs continuous vector $z_e(x)$.
3.  **Quantization**: Snap $z_e(x)$ to the nearest codebook vector $e_k$.
    $$ z_q(x) = e_k \text{ where } k = \text{argmin}_j ||z_e(x) - e_j|| $$
4.  **Decoder**: Reconstructs from $z_q(x)$.
5.  **Straight-Through Estimator**: Since `argmin` is non-differentiable, we copy gradients from decoder input to encoder output directly.

Result: Extremely sharp images. Used in DALL-E 1 and Stable Diffusion (Latent Diffusion).

## 3. Disentangled Representations ($\beta$-VAE)

We want each dimension of $z$ to control a specific semantic factor (e.g., $z_1$=Rotation, $z_2$=Color).
**$\beta$-VAE**: Increase the weight of the KL term by $\beta > 1$.
$$ L = Recon + \beta \cdot D_{KL} $$
*   Forces the latent distribution to be very close to isotropic Gaussian (uncorrelated dimensions).
*   Trade-off: Higher $\beta$ = Better disentanglement but worse reconstruction (blurrier).

## 4. Masked Autoencoders (MAE)

The modern BERT-style autoencoder for Vision.
1.  Patchify image.
2.  **Mask** 75% of patches randomly.
3.  **Encoder**: Process *only* visible patches (efficient!).
4.  **Decoder**: Reconstruct missing patches from latent representation.
5.  Learns powerful representations for Transfer Learning.
