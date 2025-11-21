# Day 17: GANs (Generative Adversarial Networks) - Theory & Implementation

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: Adversarial Training, DCGAN, and Stability

## 1. Theoretical Foundation: The Minimax Game

Two networks fighting each other:
1.  **Generator ($G$)**: Creates fake images from noise $z$. Tries to fool $D$.
2.  **Discriminator ($D$)**: Classifies images as Real or Fake. Tries to catch $G$.

### The Objective Function
$$ \min_G \max_D V(D, G) = E_{x \sim p_{data}}[\log D(x)] + E_{z \sim p_z}[\log(1 - D(G(z)))] $$
*   $D$ wants to maximize likelihood of correct classification.
*   $G$ wants to minimize $D$'s success (maximize $\log D(G(z))$).
*   **Nash Equilibrium**: $G$ produces perfect distribution $p_{data}$, and $D$ outputs 0.5 everywhere (cannot distinguish).

## 2. DCGAN (Deep Convolutional GAN)

The architecture that made GANs work for images.
*   **No Pooling**: Use Strided Conv (Discriminator) and Transposed Conv (Generator).
*   **Batch Norm**: In both G and D.
*   **Activations**: ReLU in G, LeakyReLU (0.2) in D. Tanh in G output.

## 3. Implementation: DCGAN Training Loop

```python
import torch
import torch.nn as nn

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x): return self.main(x)

# Training Loop
criterion = nn.BCELoss()
opt_d = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_g = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

for real_imgs, _ in loader:
    # 1. Train Discriminator
    netD.zero_grad()
    # Real
    label_real = torch.ones(batch_size, 1).to(device)
    output = netD(real_imgs)
    errD_real = criterion(output, label_real)
    errD_real.backward()
    # Fake
    noise = torch.randn(batch_size, 100, 1, 1).to(device)
    fake_imgs = netG(noise)
    label_fake = torch.zeros(batch_size, 1).to(device)
    output = netD(fake_imgs.detach()) # Detach to stop grad to G
    errD_fake = criterion(output, label_fake)
    errD_fake.backward()
    opt_d.step()
    
    # 2. Train Generator
    netG.zero_grad()
    label_real = torch.ones(batch_size, 1).to(device) # G wants D to say "Real"
    output = netD(fake_imgs) # No detach here!
    errG = criterion(output, label_real)
    errG.backward()
    opt_g.step()
```

## 4. Common Problems

### Mode Collapse
The Generator finds one image that fools D and produces *only* that image.
Diversity is lost.

### Vanishing Gradients
If D is too good, $D(G(z)) \approx 0$. The gradient $\nabla \log(1 - D(G(z)))$ vanishes.
**Fix**: Train G to maximize $\log D(G(z))$ (Non-saturating loss).
