# Day 31: Autoencoders (AE & VAE)

## 1. Autoencoder Basics
**Goal:** Learn a compressed representation (latent code) of the input data.
*   **Unsupervised Learning:** Input $x$, Target $x$.
*   **Architecture:**
    *   **Encoder:** $z = E(x)$. Maps input to latent space (bottleneck).
    *   **Decoder:** $\hat{x} = D(z)$. Reconstructs input from latent code.
*   **Loss:** Reconstruction Loss (MSE). $L = ||x - \hat{x}||^2$.
*   **Use Cases:** Denoising, Dimensionality Reduction, Anomaly Detection.

## 2. Variational Autoencoder (VAE)
**Problem with AE:** The latent space is not continuous. Interpolating between two points in $z$ produces garbage.
**Solution:** VAE forces the latent space to be a Gaussian distribution.
*   **Encoder:** Predicts mean $\mu$ and variance $\sigma^2$ of the distribution $q(z|x)$.
*   **Sampling:** Sample $z \sim \mathcal{N}(\mu, \sigma^2)$ using the **Reparameterization Trick**: $z = \mu + \sigma \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$.
*   **Decoder:** Reconstructs $\hat{x}$ from sampled $z$.

## 3. VAE Loss Function
$$ L = L_{recon} + \beta L_{KL} $$
1.  **Reconstruction Loss:** MSE or BCE. Ensures output looks like input.
2.  **KL Divergence:** Measures difference between learned distribution $q(z|x)$ and standard normal $\mathcal{N}(0, 1)$.
    *   Forces the latent space to be smooth and continuous.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim) # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim) # Log Variance
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

## Summary
AEs learn to copy. VAEs learn to generate. By imposing a probabilistic prior on the latent space, VAEs allow us to sample new, valid data points.
