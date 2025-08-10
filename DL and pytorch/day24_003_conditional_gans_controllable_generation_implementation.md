# Day 24.3: Conditional GANs & Controllable Generation - A Practical Guide

## Introduction: Steering the Generation

A standard Generative Adversarial Network (GAN) is like a talented but wild artist. It can generate beautiful, realistic images, but you have no control over *what* it generates. You provide random noise, and you get a random sample from the data distribution it learned. To make GANs truly useful as a creative tool, we need a way to direct the generation process.

**Conditional GANs (cGANs)** provide the solution. They extend the GAN framework by providing both the Generator and the Discriminator with an extra piece of **conditional information**, `y`. This information can be a class label, a piece of text, or even another image, and it acts as a steering wheel, allowing us to control the output of the Generator.

This guide provides a practical deep dive into the cGAN architecture and its applications.

**Today's Learning Objectives:**

1.  **Understand the Conditional GAN Architecture:** See how the conditional input `y` is fed into both the Generator and the Discriminator.
2.  **Implement a cGAN from Scratch:** Build a complete cGAN that can generate specific handwritten digits from the MNIST dataset based on a class label.
3.  **Explore Text-to-Image Generation:** Understand how the cGAN concept is extended to generate images from text descriptions.
4.  **Grasp Image-to-Image Translation (Pix2Pix):** Revisit this powerful framework as a form of conditional GAN where the input image is the condition.

---

## Part 1: The Conditional GAN (cGAN) Framework

The modification to the original GAN framework is simple but powerful.

*   **Generator (G):** Instead of taking only a noise vector `z` as input, it now also takes the conditional information `y`. Its goal is to generate an image `x` that is both realistic and consistent with the condition `y`. The new input is `[z, y]`. `x_fake = G(z, y)`.

*   **Discriminator (D):** Instead of only taking an image `x` as input, it now also takes the conditional information `y`. Its goal is to determine if the input image `x` is a real image belonging to the class `y`. The new input is `[x, y]`. `prediction = D(x, y)`.

**The Loss Function:**
The min-max game remains the same, but the inputs to `D` and `G` are now conditioned on `y`.

`min_G max_D V(D, G) = E_{x,y ~ p_data(x,y)}[log(D(x, y))] + E_{z ~ p_z(z), y ~ p_y(y)}[log(1 - D(G(z, y), y))]`

![cGAN Diagram](https://i.imgur.com/3h4Y5fG.png)

---

## Part 2: Implementing a cGAN for MNIST

Let's build a cGAN that can generate a specific digit (0-9) on command. The condition `y` will be the integer class label for the digit we want to generate.

### 2.1. Data Loading

We load the MNIST dataset as usual. We will need both the images and their labels.

```python
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

print("--- Part 2: Implementing a cGAN for MNIST ---")

# --- Parameters ---
image_size = 28 * 28 # Flattened images
latent_dim = 100
num_classes = 10

# --- Data Loading ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
mnist_dataset = dsets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True)
```

### 2.2. The Conditional Generator and Discriminator

We need to modify our simple MLP-based models to accept the class label as an additional input. A common way to do this is to use an `nn.Embedding` layer to convert the integer label into a dense vector, and then concatenate this vector with the main input.

```python
# --- The Conditional Generator ---
class cGenerator(nn.Module):
    def __init__(self):
        super(cGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, image_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Create label embeddings
        c = self.label_embedding(labels)
        # Concatenate noise vector and label embedding
        x = torch.cat([z, c], 1)
        return self.model(x)

# --- The Conditional Discriminator ---
class cDiscriminator(nn.Module):
    def __init__(self):
        super(cDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(image_size + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img, labels):
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        # Create label embeddings
        c = self.label_embedding(labels)
        # Concatenate image and label embedding
        x = torch.cat([img_flat, c], 1)
        return self.model(x)

print("Conditional Generator and Discriminator models defined.")
```

### 2.3. The cGAN Training Loop

The training loop is very similar to a standard GAN, but we must now feed the labels to both networks.

```python
import torch.optim as optim

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = cGenerator().to(device)
disc = cDiscriminator().to(device)
optimizer_g = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()

# --- Training Loop (Simplified) ---
num_epochs = 10 # For a real run, use more epochs

print("\nStarting cGAN training...")
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(data_loader):
        batch_size = imgs.size(0)
        imgs, labels = imgs.to(device), labels.to(device)
        
        real_label_val = torch.ones(batch_size, 1, device=device)
        fake_label_val = torch.zeros(batch_size, 1, device=device)

        # --- Train Discriminator ---
        optimizer_d.zero_grad()
        # On real data
        d_out_real = disc(imgs, labels)
        loss_d_real = criterion(d_out_real, real_label_val)
        # On fake data
        noise = torch.randn(batch_size, latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        fake_imgs = gen(noise, gen_labels)
        d_out_fake = disc(fake_imgs.detach(), gen_labels)
        loss_d_fake = criterion(d_out_fake, fake_label_val)
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizer_d.step()

        # --- Train Generator ---
        optimizer_g.zero_grad()
        d_out_on_fake = disc(fake_imgs, gen_labels)
        loss_g = criterion(d_out_on_fake, real_label_val)
        loss_g.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {loss_d.item():.4f}, G Loss: {loss_g.item():.4f}")

print("Training finished.")
```

### 2.4. Generating Conditional Images

After training, we can ask the generator to produce an image of a specific digit.

```python
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# --- Generate one of each digit ---
gen.eval()
with torch.no_grad():
    noise = torch.randn(num_classes, latent_dim, device=device)
    # Create labels from 0 to 9
    target_labels = torch.arange(0, num_classes, device=device)
    
    generated_digits = gen(noise, target_labels).view(-1, 1, 28, 28)

# --- Visualize ---
plt.figure(figsize=(10, 2))
plt.title("Conditionally Generated MNIST Digits")
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_digits, nrow=num_classes, padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()
```

---

## Part 3: Other Conditional Applications

The cGAN framework is extremely flexible and forms the basis for many powerful applications.

*   **Text-to-Image Synthesis:**
    *   **Condition `y`:** A text description of an image.
    *   **How it works:** The text is first passed through a text encoder (like a pre-trained BERT or a specific text encoder like CLIP) to get a sentence embedding. This embedding vector is then used as the conditional input `y` for the GAN.
    *   **Models:** StackGAN, AttnGAN, and more recently, diffusion models like DALL-E 2 and Stable Diffusion have become state-of-the-art.

*   **Image-to-Image Translation (Pix2Pix):**
    *   **Condition `y`:** An entire source image (e.g., a satellite map).
    *   **How it works:** The generator is a U-Net that takes the source image as input. The discriminator (a PatchGAN) sees both the source image and the target image (either real or fake) and must decide if the pair is realistic.

## Conclusion

Conditional GANs are a crucial evolution of the original GAN framework. By introducing a conditional input, they transform GANs from a random generative process into a powerful and controllable tool for creating specific, desired outputs.

**Key Takeaways:**

1.  **Control is the Goal:** cGANs allow us to direct the output of the generator by providing a conditional signal `y`.
2.  **Condition Both Networks:** It is essential to feed the conditional information `y` to *both* the generator (so it knows what to create) and the discriminator (so it knows what to expect).
3.  **Embeddings for Conditions:** For discrete conditions like class labels, `nn.Embedding` is the standard way to create a vector representation to be used as input.
4.  **A Foundation for Advanced Tasks:** The cGAN framework is the basis for many state-of-the-art applications, including text-to-image synthesis and paired image-to-image translation.

This ability to control the generation process makes GANs not just a tool for mimicking a data distribution, but a powerful instrument for creative and goal-oriented content creation.

## Self-Assessment Questions

1.  **cGAN vs. GAN:** What are the two main architectural differences between a standard GAN and a conditional GAN?
2.  **Generator Input:** In our MNIST cGAN, what two tensors are concatenated to form the final input to the generator's MLP?
3.  **Discriminator Input:** In our MNIST cGAN, what two tensors are concatenated to form the final input to the discriminator's MLP?
4.  **Text-to-Image:** In a text-to-image model, what is used as the conditional input `y`?
5.  **Pix2Pix:** In the Pix2Pix framework, what is the role of the input image (e.g., the satellite map)?
