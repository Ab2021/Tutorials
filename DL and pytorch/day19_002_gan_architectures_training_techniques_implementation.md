# Day 19.2: GAN Architectures & Training Techniques - A Practical Guide

## Introduction: From Simple GANs to Deep Convolutional GANs

The simple MLP-based GAN we built in the previous guide is great for understanding the core concepts, but it's not powerful enough to generate high-quality images. To do that, we need to incorporate the power of Convolutional Neural Networks.

The **Deep Convolutional GAN (DCGAN)** was a landmark paper that proposed a set of architectural guidelines for building stable and effective GANs using CNNs. These guidelines provided a stable baseline that much of the subsequent research was built upon.

This guide provides a practical overview of the DCGAN architecture and the key training techniques that are essential for stabilizing the notoriously difficult GAN training process.

**Today's Learning Objectives:**

1.  **Understand the DCGAN Architecture:** Learn the specific architectural guidelines for the Generator (using transposed convolutions) and the Discriminator (using strided convolutions).
2.  **Implement a DCGAN:** Build a complete DCGAN in PyTorch capable of generating simple images (e.g., from the MNIST dataset).
3.  **Learn GAN Training Best Practices:** Understand key techniques for stabilizing training, such as using specific activation functions, applying Batch Normalization, and using the Adam optimizer.
4.  **Visualize the Generator's Progress:** See how to save and visualize the images produced by the generator at different stages of training to monitor its learning process.

---

## Part 1: The DCGAN Architecture

The DCGAN paper proposed several key architectural changes from a standard CNN:

**For the Generator (The Artist):**
1.  **Use Transposed Convolutions (`nn.ConvTranspose2d`) for up-sampling.** This is the learnable way to go from a low-dimensional noise vector to a high-resolution image.
2.  **Use Batch Normalization (`nn.BatchNorm2d`) in all layers except the output.** This helps to stabilize the gradient flow and prevent mode collapse.
3.  **Use ReLU (`nn.ReLU`) activation in all layers except the output.**
4.  **Use Tanh (`nn.Tanh`) activation in the output layer.** This scales the output pixels to be in the range `[-1, 1]`, which is a common convention for image data in GANs.

**For the Discriminator (The Critic):**
1.  **Replace all pooling layers with Strided Convolutions (`nn.Conv2d` with `stride=2`).** This makes the entire network differentiable and allows it to learn its own spatial down-sampling.
2.  **Use Batch Normalization (`nn.BatchNorm2d`) in all layers except the input and output.**
3.  **Use Leaky ReLU (`nn.LeakyReLU`) activation in all layers.** This helps to prevent sparse gradients and allows gradients to flow more easily, which is crucial for the discriminator's health.
4.  **Use a Sigmoid activation in the output layer** (or no activation if using `BCEWithLogitsLoss`).

---

## Part 2: Implementing a DCGAN for MNIST

Let's build a DCGAN to generate handwritten digits. We will use the MNIST dataset as our source of real images.

### 2.1. Data Loading

First, we load the MNIST dataset and apply the necessary transforms. Since the Generator's output will be `Tanh` activated `[-1, 1]`, we must normalize our real images to the same range.

```python
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

print("--- Part 2.1: Data Loading for DCGAN ---")

# --- Parameters ---
image_size = 64 # We will resize the 28x28 MNIST images to 64x64
batch_size = 128

# --- Transforms ---
# We need to resize and normalize the images to the [-1, 1] range
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # (value - 0.5) / 0.5 -> [-1, 1]
])

# --- Load MNIST Dataset ---
mnist_dataset = dsets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)

print("MNIST data loaded and normalized.")
```

### 2.2. The Generator and Discriminator Models

Now we implement the DCGAN architecture guidelines.

```python
print("\n--- Part 2.2: DCGAN Model Implementation ---")

# --- Parameters ---
latent_dim = 100  # Size of the input noise vector
gf_dim = 64       # Generator feature map size
df_dim = 64       # Discriminator feature map size
channels = 1      # MNIST is grayscale

# --- The Generator ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: (N, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, gf_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gf_dim * 8),
            nn.ReLU(True),
            # State: (N, gf_dim*8, 4, 4)
            nn.ConvTranspose2d(gf_dim * 8, gf_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim * 4),
            nn.ReLU(True),
            # State: (N, gf_dim*4, 8, 8)
            nn.ConvTranspose2d(gf_dim * 4, gf_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim * 2),
            nn.ReLU(True),
            # State: (N, gf_dim*2, 16, 16)
            nn.ConvTranspose2d(gf_dim * 2, gf_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim),
            nn.ReLU(True),
            # State: (N, gf_dim, 32, 32)
            nn.ConvTranspose2d(gf_dim, channels, 4, 2, 1, bias=False),
            nn.Tanh() # Output: (N, channels, 64, 64)
        )
    def forward(self, x):
        return self.main(x)

# --- The Discriminator ---
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (N, channels, 64, 64)
            nn.Conv2d(channels, df_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, df_dim, 32, 32)
            nn.Conv2d(df_dim, df_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, df_dim*2, 16, 16)
            nn.Conv2d(df_dim * 2, df_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, df_dim*4, 8, 8)
            nn.Conv2d(df_dim * 4, df_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (N, df_dim*8, 4, 4)
            nn.Conv2d(df_dim * 8, 1, 4, 1, 0, bias=False),
            # Output: (N, 1, 1, 1) -> a single logit
        )
    def forward(self, x):
        return self.main(x)

print("Generator and Discriminator models defined.")
```

### 2.3. The Training Loop

The training loop is conceptually the same as for the simple GAN, but now we are working with image batches.

```python
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

print("\n--- Part 2.3: DCGAN Training ---")

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG = Generator().to(device)
netD = Discriminator().to(device)

criterion = nn.BCEWithLogitsLoss()

# Create fixed noise for visualization
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

# Use the Adam optimizer, as recommended
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# --- Training Loop (Simplified) ---
num_epochs = 5 # A full training run takes longer
img_list = []

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(data_loader, 0):
        # --- 1. Train Discriminator ---
        netD.zero_grad()
        # a) On real data
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), 1., dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        
        # b) On fake data
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(0.)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        
        errD = errD_real + errD_fake
        optimizerD.step()

        # --- 2. Train Generator ---
        netG.zero_grad()
        label.fill_(1.) # Generator wants discriminator to think fake images are real
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()
        
        if i % 100 == 0:
            print(f'[{epoch+1}/{num_epochs}][{i}/{len(data_loader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

    # After each epoch, save the generator's output on the fixed_noise
    with torch.no_grad():
        fake_images = netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(fake_images, padding=2, normalize=True))

print("Finished Training.")

# --- Visualize Results ---
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Generator's Progress")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
```

## Conclusion

The DCGAN architecture provided a stable and scalable blueprint for building GANs that could work with image data. By combining the adversarial training framework with established best practices from convolutional networks, it paved the way for the stunning generative results we see today.

**Key Takeaways and Training Tips:**

1.  **Architecture Matters:** Use transposed convolutions for up-sampling in the generator and strided convolutions for down-sampling in the discriminator.
2.  **Normalization is Key:** Batch Normalization is crucial for stabilizing the training of deep GANs.
3.  **Use Leaky ReLU in the Discriminator:** This prevents sparse gradients and helps the discriminator stay strong.
4.  **Use Adam:** The Adam optimizer is generally the recommended choice for training GANs.
5.  **Monitor the Losses:** The Generator loss (`errG`) and Discriminator loss (`errD`) can be noisy. If `errG` goes to zero, the generator is overpowering the discriminator. If `errD` goes to zero, the discriminator is too strong, and the generator isn't getting a useful gradient signal. A healthy training process involves a balance between the two.

These techniques are the foundation for training almost all modern GAN architectures.

## Self-Assessment Questions

1.  **Generator Upsampling:** What type of layer does a DCGAN generator use to increase the spatial size of its feature maps?
2.  **Discriminator Downsampling:** What does a DCGAN discriminator use instead of traditional pooling layers for down-sampling?
3.  **Activation Functions:** What activation function is recommended for the hidden layers of the discriminator? What about the output layer of the generator?
4.  **Normalization:** Why is Batch Normalization important in a DCGAN?
5.  **Loss Monitoring:** During training, you notice your generator's loss is consistently very high, and your discriminator's loss is consistently near zero. What does this indicate about the training dynamic?

