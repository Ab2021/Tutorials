# Day 35: Latent Diffusion Models (Stable Diffusion)

## 1. The Computational Bottleneck
Standard Diffusion (DDPM) operates in **Pixel Space**.
*   Image: $512 \times 512 \times 3$.
*   Dimension: ~786,000 values.
*   Problem: Every step of diffusion requires processing this massive tensor. Slow and expensive.

## 2. Latent Diffusion Models (LDM)
**Idea:** Move the diffusion process to a compressed **Latent Space**.
1.  **Perceptual Compression (Autoencoder):**
    *   Train a VQ-GAN or KL-VAE to compress image $x$ to latent $z$.
    *   Compression factor $f=8$.
    *   $512 \times 512 \times 3 \to 64 \times 64 \times 4$.
    *   Dimension: ~16,000 values (50x reduction!).
2.  **Latent Diffusion:**
    *   Train the diffusion model (U-Net) to denoise $z_t$ in latent space.
    *   Much faster training and inference.
3.  **Decoding:**
    *   Map denoised $z_0$ back to pixel space using the Decoder.

## 3. Conditioning (Text-to-Image)
How do we tell the model *what* to generate?
*   **Cross-Attention:**
    *   Text Prompt $\to$ CLIP Encoder $\to$ Embeddings Sequence.
    *   U-Net Intermediate Layers $\to$ Query.
    *   Text Embeddings $\to$ Key/Value.
    *   $Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V$.
*   This injects semantic information into the spatial features of the U-Net.

## 4. Stable Diffusion Architecture
*   **Autoencoder:** KL-regularized VAE ($f=8$).
*   **U-Net:** 860M parameters. ResNet blocks + Self-Attention + Cross-Attention.
*   **Text Encoder:** CLIP ViT-L/14.

```python
# Pseudo-code for LDM Inference
z_T = torch.randn(1, 4, 64, 64) # Random latent noise
prompt = "A photo of an astronaut riding a horse"
c = clip_encoder(prompt) # Text embeddings

for t in reversed(range(T)):
    # Predict noise in latent space
    eps = unet(z_t, t, context=c)
    z_t_minus_1 = step(z_t, eps, t)

# Decode to pixels
image = vae_decoder(z_0)
```

## Summary
Latent Diffusion democratized AI art by making high-resolution generation possible on consumer GPUs (8GB VRAM). It separates the "perceptual compression" (VAE) from the "semantic generation" (Diffusion).
