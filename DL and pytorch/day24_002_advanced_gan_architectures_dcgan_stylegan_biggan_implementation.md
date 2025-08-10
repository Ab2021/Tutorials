# Day 24.2: Advanced GAN Architectures (DCGAN, StyleGAN, BigGAN) - A Practical Guide

## Introduction: From Blurry Blobs to Photorealism

The original GAN, built with simple MLPs, demonstrated the power of adversarial training but could only produce small, blurry images. The journey to generating high-resolution, photorealistic images has been driven by a series of architectural breakthroughs. These advanced architectures introduced key innovations to stabilize training and give the generator more control over the image synthesis process.

This guide provides a practical overview of three of the most influential GAN architectures that paved the way for modern image generation: **DCGAN**, **StyleGAN**, and **BigGAN**.

**Today's Learning Objectives:**

1.  **Revisit DCGAN:** Solidify the understanding of the architectural guidelines (transposed convolutions, batch norm, LeakyReLU) that made deep convolutional GANs stable.
2.  **Explore StyleGAN:** Learn about the style-based generator, the mapping network, and adaptive instance normalization (AdaIN) that enable unprecedented control and realism.
3.  **Understand BigGAN:** Grasp the key techniques (like the truncation trick and self-attention) that allowed GANs to be scaled up to massive sizes for high-fidelity generation.
4.  **Implement a DCGAN:** Build a complete DCGAN from scratch in PyTorch to generate MNIST digits, reinforcing the architectural principles.

---

## Part 1: DCGAN - Stable Convolutional GANs

As we saw in Day 19, the **Deep Convolutional GAN (DCGAN)** was the first major breakthrough that made it possible to train stable GANs with CNNs.

**Recap of DCGAN Architectural Guidelines:**
*   **Generator:** Uses **transposed convolutions** for up-sampling, `BatchNorm2d`, and `ReLU` activation, with a final `Tanh` output.
*   **Discriminator:** Replaces pooling with **strided convolutions** for down-sampling, uses `BatchNorm2d`, and `LeakyReLU` activation.

These guidelines created a robust baseline that solved many of the initial stability problems and became the foundation for future research.

---

## Part 2: StyleGAN - Unprecedented Control and Realism

**StyleGAN** (and its successor, StyleGAN2), developed by NVIDIA, represents a paradigm shift in generator design. It produces stunningly photorealistic human faces and offers an incredible level of control over the generated image.

**Key Innovations:**

1.  **Style-Based Generator:** This is the core idea. Instead of feeding the latent code `z` directly into the generator, StyleGAN first maps `z` to an intermediate latent space `W` using a non-linear **mapping network** (an MLP). This intermediate latent code `w` is then used to control the "style" of the image at each level of the generator.

2.  **Adaptive Instance Normalization (AdaIN):** The `w` vector is transformed and then injected into the generator at each convolutional block via an AdaIN layer. The AdaIN layer normalizes the feature maps and then applies a scale and bias learned from `w`. This allows `w` to control the style (colors, textures, lighting) of the features at that specific resolution.

3.  **Progressive Growing:** The generator starts by producing very low-resolution images (e.g., 4x4) and is trained until stable. Then, new blocks are progressively added to both the generator and discriminator to increase the resolution (8x8, 16x16, etc.), with each new block being faded in smoothly. This makes training on high-resolution images much more stable.

**The Result:**
*   **Disentanglement:** The intermediate latent space `W` is much more disentangled than the input space `Z`. This means that different dimensions in `w` tend to correspond to distinct, high-level attributes of the face (e.g., pose, hair style, age, glasses).
*   **Style Mixing:** You can use one `w` vector to control the coarse styles (low-resolution layers) and another `w` vector to control the fine styles (high-resolution layers), allowing you to mix the pose of one face with the hair and skin texture of another.

![StyleGAN Generator](https://i.imgur.com/J4g9Y7L.png)

---

## Part 3: BigGAN - The Power of Scale

**The Question:** Can we get better results simply by making GANs bigger? The **BigGAN** project from Google DeepMind showed that the answer is yes, but it requires specific techniques to manage the training of such a massive model.

**Key Innovations:**

1.  **Massive Scale:** BigGAN used models that were 2x to 4x wider (more channels) and deeper than previous GANs, and were trained on the huge ImageNet dataset with very large batch sizes (e.g., 2048).

2.  **Self-Attention Module:** It incorporated the **self-attention** mechanism (from Transformers) into both the generator and discriminator. This allows the model to capture long-range dependencies across the image. For example, when generating a dog, the attention mechanism helps ensure that the texture of the fur is consistent across the entire body, not just in local patches.

3.  **The Truncation Trick:** This is a simple but powerful technique used during *inference* (not training) to improve the quality of individual samples at the cost of reduced variety.
    *   Instead of sampling the initial noise vector `z` from a standard normal distribution, you sample it from a **truncated** normal distribution (i.e., you clamp the values, preventing them from being too far from the mean).
    *   A small truncation value leads to very high-quality, but less diverse ("typical") images. A value of 1.0 is standard sampling.

**The Result:** BigGAN was able to generate high-fidelity, diverse images for all 1000 ImageNet classes, demonstrating that GANs, like Transformers, benefit greatly from scaling up.

---

## Part 4: DCGAN Implementation for MNIST (Refresher)

Let's reinforce our understanding by implementing the foundational DCGAN architecture again. This code is a complete, runnable example for generating 64x64 MNIST digits.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

print("--- Part 4: DCGAN Implementation for MNIST ---")

# --- 1. Setup and Parameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
image_size = 64
channels = 1
batch_size = 128
lr = 0.0002
beta1 = 0.5
num_epochs = 5 # For a real run, use 20-25 epochs

# --- 2. Data Loading ---
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = dsets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 3. Model Definitions (Generator and Discriminator) ---
# (Using the exact same Generator and Discriminator classes from Day 19.2)
class Generator(nn.Module):
    def __init__(self): super().__init__(); self.main = nn.Sequential(nn.ConvTranspose2d(latent_dim, 64 * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(64 * 8), nn.ReLU(True), nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(64 * 4), nn.ReLU(True), nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(64 * 2), nn.ReLU(True), nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False), nn.Tanh())
    def forward(self, x): return self.main(x)

class Discriminator(nn.Module):
    def __init__(self): super().__init__(); self.main = nn.Sequential(nn.Conv2d(channels, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(64 * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(64 * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(64 * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False))
    def forward(self, x): return self.main(x)

# --- 4. Training Setup ---
netG = Generator().to(device)
netD = Discriminator().to(device)
criterion = nn.BCEWithLogitsLoss()
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# --- 5. Training Loop ---
print("Starting DCGAN Training...")
for epoch in range(num_epochs):
    for i, data in enumerate(data_loader, 0):
        # (Training steps are identical to Day 19.2)
        netD.zero_grad()
        real_cpu = data[0].to(device); b_size = real_cpu.size(0)
        label = torch.full((b_size,), 1., device=device)
        output = netD(real_cpu).view(-1); errD_real = criterion(output, label); errD_real.backward()
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device); fake = netG(noise)
        label.fill_(0.); output = netD(fake.detach()).view(-1); errD_fake = criterion(output, label); errD_fake.backward()
        optimizerD.step()
        netG.zero_grad(); label.fill_(1.)
        output = netD(fake).view(-1); errG = criterion(output, label); errG.backward()
        optimizerG.step()
    print(f'Epoch [{epoch+1}/{num_epochs}] finished.')

print("Finished Training.")

# --- 6. Visualize Results ---
with torch.no_grad():
    fake_images = netG(fixed_noise).detach().cpu()

plt.figure(figsize=(8,8)); plt.axis("off"); plt.title("Generated MNIST Digits")
plt.imshow(np.transpose(vutils.make_grid(fake_images, padding=2, normalize=True),(1,2,0)))
plt.show()
```

## Conclusion

The evolution from DCGAN to StyleGAN and BigGAN showcases the incredible progress in generative modeling. Researchers have moved from simply achieving stable training to creating models that offer fine-grained artistic control and can generate images with a level of fidelity that is often indistinguishable from reality.

**Key Architectural Trends:**

*   **Increasing Control:** Moving from a simple noise vector input to a disentangled, style-based latent space (StyleGAN).
*   **Increasing Scale:** Dramatically increasing the model size, batch size, and dataset size to improve quality and diversity (BigGAN).
*   **Attention for Global Coherence:** Incorporating self-attention mechanisms to capture long-range dependencies within an image (BigGAN).
*   **Progressive Training:** Starting small and progressively adding layers to generate higher-resolution images, which stabilizes training (ProGAN, StyleGAN).

These advanced architectures are the engines behind many of the generative AI tools we see today and continue to be an active and exciting area of research.

## Self-Assessment Questions

1.  **DCGAN:** What type of layer does a DCGAN use for up-sampling in the generator?
2.  **StyleGAN:** What is the purpose of the mapping network in StyleGAN?
3.  **AdaIN:** What does the Adaptive Instance Normalization (AdaIN) layer do?
4.  **BigGAN:** What two key techniques did BigGAN use to successfully train at a massive scale?
5.  **Truncation Trick:** What is the "truncation trick," and what is the trade-off involved when you use it?

