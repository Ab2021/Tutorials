# Day 34: Diffusion Models (DDPM)

## 1. The Idea
**GANs:** Map noise to image in one shot. Unstable.
**Diffusion:** Slowly destroy the image by adding noise, then learn to reverse the process step-by-step.
*   **Forward Process ($q$):** $x_0 \to x_1 \to \dots \to x_T$. Add Gaussian noise at each step until $x_T$ is pure noise.
*   **Reverse Process ($p$):** $x_T \to x_{T-1} \to \dots \to x_0$. Learn a neural network to remove noise at each step.

## 2. Forward Process (No Training)
We define a noise schedule $\beta_t$.
$$ q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I) $$
Crucially, we can sample $x_t$ directly from $x_0$:
$$ x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon $$
where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod \alpha_s$.

## 3. Reverse Process (Training)
We want to estimate $p_\theta(x_{t-1} | x_t)$.
Since $q$ is Gaussian, $p$ is also Gaussian for small steps.
$$ p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) $$
**Simplified Objective:**
Instead of predicting the mean $\mu$, we predict the **noise** $\epsilon$ that was added.
$$ L_{simple} = \mathbb{E}_{t, x_0, \epsilon} [ || \epsilon - \epsilon_\theta(x_t, t) ||^2 ] $$
*   Input: Noisy image $x_t$ and time step $t$.
*   Output: Predicted noise $\epsilon_\theta$.
*   Target: Actual noise $\epsilon$.

## 4. Architecture
**U-Net** is the standard choice.
*   **Time Embedding:** We must tell the network *which* step $t$ we are at (is it pure noise or almost clean?).
*   Sinusoidal embeddings (like Transformers) are added to the features.

```python
import torch
import torch.nn as nn

class SimpleDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet() # Standard U-Net

    def forward(self, x_0):
        t = torch.randint(0, 1000, (x_0.shape[0],)).to(x_0.device)
        noise = torch.randn_like(x_0)
        
        # Forward diffusion
        alpha_bar = self.get_alpha_bar(t)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        loss = nn.MSELoss()(predicted_noise, noise)
        return loss
```

## Summary
Diffusion models trade speed for stability and quality. By breaking generation into 1000 small, easy steps, they avoid the mode collapse of GANs.
