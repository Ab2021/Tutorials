# Day 31 Deep Dive: VQ-VAE & Disentanglement

## 1. The Blur Problem in VAEs
VAEs tend to generate blurry images.
*   **Reason:** The MSE loss assumes pixel independence and Gaussian noise. It averages out high-frequency details.
*   **Also:** The Gaussian prior is too simple for complex image distributions.

## 2. Vector Quantized VAE (VQ-VAE)
**Idea:** Learn a **discrete** latent space instead of a continuous one.
*   **Codebook:** A list of $K$ embedding vectors $e_1, \dots, e_K$.
*   **Encoder:** Outputs continuous vector $z_e(x)$.
*   **Quantization:** Find the nearest neighbor in the codebook: $z_q(x) = e_k$ where $k = \arg \min ||z_e(x) - e_j||$.
*   **Decoder:** Reconstructs from $z_q(x)$.
*   **Result:** Sharp images! Used in DALL-E 1.

## 3. Disentangled Representations ($\beta$-VAE)
**Goal:** Each dimension of $z$ should control a single semantic factor (e.g., $z_1$=Rotation, $z_2$=Color).
*   **Method:** Increase the weight $\beta$ of the KL divergence term ($L = L_{recon} + \beta L_{KL}$ with $\beta > 1$).
*   **Effect:** Forces the latent dimensions to be independent (uncorrelated), leading to better interpretability.
*   **Trade-off:** Higher $\beta$ leads to worse reconstruction quality (blurrier).

## 4. Denoising Autoencoder (DAE)
**Idea:** Train the network to remove noise.
*   **Input:** Corrupted image $\tilde{x} = x + \text{noise}$.
*   **Target:** Clean image $x$.
*   **Benefit:** Forces the model to learn the manifold of "real" data and robust features.

## Summary
VQ-VAE bridged the gap between Autoencoders and high-fidelity generation, paving the way for modern discrete generative models.
