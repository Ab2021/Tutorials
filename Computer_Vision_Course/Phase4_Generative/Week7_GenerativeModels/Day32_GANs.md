# Day 32: Generative Adversarial Networks (GANs)

## 1. The Adversarial Game
**Goal:** Generate data indistinguishable from real data.
**Players:**
1.  **Generator ($G$):** Creates fake data from noise $z$. Tries to fool $D$.
2.  **Discriminator ($D$):** Classifies data as Real or Fake. Tries to catch $G$.

**Analogy:** Counterfeiter ($G$) vs Police ($D$).

## 2. The Minimax Loss
$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}} [\log D(x)] + \mathbb{E}_{z \sim p_{z}} [\log (1 - D(G(z)))] $$
*   **Discriminator Goal:** Maximize probability of assigning 1 to real data and 0 to fake data.
*   **Generator Goal:** Minimize $\log(1 - D(G(z)))$, which is equivalent to maximizing $\log D(G(z))$ (Fooling $D$).

## 3. DCGAN (Deep Convolutional GAN)
The first stable architecture for image generation (2015).
**Guidelines:**
*   Replace Pooling with **Strided Convolutions** (Discriminator) and **Transposed Convolutions** (Generator).
*   Use **Batch Normalization** in both G and D.
*   Use **ReLU** in Generator (except Tanh output).
*   Use **LeakyReLU** in Discriminator.

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # ... (More layers) ...
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh() # Output range [-1, 1]
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # ...
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```

## Summary
GANs produce sharper images than VAEs because they don't minimize pixel-wise MSE. Instead, they optimize a "perceptual" loss learned by the Discriminator.
