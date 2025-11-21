# Day 31: Diffusion Models - Theory & Implementation

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: DDPM, Latent Diffusion, and Stable Diffusion

## 1. Theoretical Foundation: Denoising Diffusion Probabilistic Models (DDPM)

Generative Models:
*   **GANs**: Adversarial (Generator vs Discriminator). Unstable.
*   **VAEs**: Probabilistic (Encoder-Decoder). Blurry.
*   **Diffusion**: Iterative Denoising. High quality, stable, slow.

**The Process**:
1.  **Forward Process ($q$)**: Gradually add Gaussian noise to an image $x_0$ until it becomes pure noise $x_T$.
    $$ x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon $$
2.  **Reverse Process ($p$)**: Train a Neural Network to predict the noise $\epsilon$ added at step $t$, to recover $x_{t-1}$.
    $$ x_{t-1} \approx \frac{1}{\sqrt{\alpha_t}} (x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)) $$

## 2. Latent Diffusion Models (Stable Diffusion)

Pixel-space diffusion is expensive ($256 \times 256 \times 3$).
**LDM**:
1.  Train a VAE (Autoencoder) to compress image to Latent Space ($64 \times 64 \times 4$).
2.  Perform Diffusion in Latent Space (Cheaper).
3.  Decode back to pixels.

## 3. Implementation: Minimal DDPM Training Loop

```python
import torch
import torch.nn as nn

class SimpleDiffusion(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.timesteps = 1000
        # Define beta schedule...
        
    def forward(self, x_0):
        # 1. Sample random timestep t
        t = torch.randint(0, self.timesteps, (x_0.shape[0],)).to(x_0.device)
        
        # 2. Sample noise
        epsilon = torch.randn_like(x_0)
        
        # 3. Add noise (Forward Process)
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon
        x_t = self.q_sample(x_0, t, epsilon)
        
        # 4. Predict noise
        epsilon_pred = self.unet(x_t, t)
        
        # 5. Loss (MSE)
        return nn.MSELoss()(epsilon_pred, epsilon)

# Training
# optimizer.zero_grad()
# loss = diffusion(images)
# loss.backward()
# optimizer.step()
```

## 4. Conditioning (Text-to-Image)

How do we control the generation?
**Cross-Attention** in the U-Net.
*   $Q = \text{Image Features}$.
*   $K, V = \text{Text Embeddings (CLIP)}$.
*   The U-Net looks at the text to decide how to denoise.
