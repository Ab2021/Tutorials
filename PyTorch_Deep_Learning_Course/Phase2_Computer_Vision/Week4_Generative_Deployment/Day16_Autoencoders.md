# Day 16: Autoencoders & VAEs - Theory & Implementation

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: Latent Space, Reconstruction, and Variational Inference

## 1. Theoretical Foundation: Unsupervised Learning

We don't have labels $y$. We only have data $x$.
Goal: Learn a compressed representation (Latent Code $z$) of $x$.

### The Autoencoder (AE)
1.  **Encoder**: $z = E(x)$. Compresses input to low-dim bottleneck.
2.  **Decoder**: $\hat{x} = D(z)$. Reconstructs input from bottleneck.
3.  **Loss**: MSE $||x - \hat{x}||^2$.

### The Manifold Hypothesis
High-dimensional data (Images) lies on a low-dimensional manifold.
The Autoencoder learns the coordinate system of this manifold.

## 2. Variational Autoencoder (VAE)

Standard AEs have a "hole" problem. The latent space is not continuous.
Interpolating between two points in $z$ might yield garbage.
**VAE** forces the latent space to be a **Gaussian Distribution**.

1.  **Encoder**: Predicts Mean $\mu$ and Variance $\sigma^2$.
2.  **Reparameterization Trick**: Sample $z = \mu + \sigma \cdot \epsilon$, where $\epsilon \sim N(0, 1)$.
3.  **Decoder**: Reconstruct $\hat{x}$ from sampled $z$.
4.  **Loss**: Reconstruction Loss + KL Divergence (Regularizer).
    $$ L = ||x - \hat{x}||^2 + D_{KL}(N(\mu, \sigma) || N(0, 1)) $$

## 3. Implementation: VAE in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3)) # Pixels [0, 1]
        
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss Function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

## 4. Applications
*   **Denoising**: Train to map Noisy Input $\to$ Clean Output.
*   **Anomaly Detection**: High reconstruction error = Anomaly.
*   **Generation**: Sample $z \sim N(0, 1)$, decode to generate new images.
