# Day 32 Interview Questions: GANs

## Q1: Why is the GAN loss function called a "Minimax" game?
**Answer:**
*   It involves two players with opposing goals.
*   $D$ wants to **Maximize** the objective $V(D, G)$ (Accuracy of classification).
*   $G$ wants to **Minimize** the objective $V(D, G)$ (Accuracy of $D$).
*   Equilibrium is reached when $D$ outputs 0.5 everywhere (cannot distinguish).

## Q2: What is Mode Collapse?
**Answer:**
*   A failure mode where the Generator produces limited varieties of samples (e.g., only generating the digit "1" in MNIST).
*   It happens because $G$ maps the entire latent space $Z$ to a single high-probability point in the data distribution to cheat $D$.

## Q3: Why use LeakyReLU in the Discriminator but ReLU in the Generator?
**Answer:**
*   **Discriminator:** We want gradients to flow even when the input is negative (fake data might look "negative" in feature space). Dead ReLUs block gradients from $D$ to $G$.
*   **Generator:** ReLU is sufficient for internal layers. Tanh is used at the output to map to $[-1, 1]$.

## Q4: Explain the difference between WGAN and standard GAN.
**Answer:**
*   **Standard GAN:** Uses Jensen-Shannon Divergence. Prone to vanishing gradients if distributions don't overlap.
*   **WGAN:** Uses Wasserstein Distance (Earth Mover's Distance). Provides meaningful gradients everywhere, even when distributions are disjoint.
*   **Implementation:** Remove Sigmoid from $D$. Use linear output. Clip weights or use Gradient Penalty.

## Q5: What is FID (Fréchet Inception Distance)?
**Answer:**
*   The standard metric for GANs.
*   Pass Real and Fake images through Inception-v3. Extract features from the pooling layer.
*   Assume features follow a multidimensional Gaussian distribution $(\mu, \Sigma)$.
*   Calculate distance: $||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$.
*   Lower FID = Better quality and diversity.

## Q6: Why do we use Transposed Convolutions in the Generator?
**Answer:**
*   The Generator needs to upsample a small latent vector ($1 \times 1 \times 100$) to a full image ($64 \times 64 \times 3$).
*   Transposed Convolution (Fractionally Strided Conv) learns to upsample by broadcasting weights, effectively reversing a convolution.

## Q7: What is "Label Smoothing" in GANs?
**Answer:**
*   Instead of training $D$ with hard labels (1 for Real, 0 for Fake), use soft labels (e.g., 0.9 for Real).
*   This prevents $D$ from becoming too confident, which would cause gradients to vanish (saturation of Sigmoid).

## Q8: Can GANs be used for Super-Resolution?
**Answer:**
**Yes (SRGAN).**
*   Generator takes Low-Res image, outputs High-Res.
*   Discriminator checks if the High-Res image looks real.
*   Adds a "Perceptual Loss" (VGG features) + Adversarial Loss.
*   Result: Adds realistic high-frequency textures that MSE loss would smooth out.

## Q9: Why is training GANs hard?
**Answer:**
*   **Non-convergence:** The players might oscillate forever instead of reaching equilibrium.
*   **Hyperparameter sensitivity:** Learning rates of $G$ and $D$ must be balanced carefully (Two-Time Scale Update Rule - TTUR).
*   **Mode Collapse.**

## Q10: Implement the Non-Saturating Generator Loss.
**Answer:**
```python
# Instead of min log(1 - D(G(z)))
# We use max log(D(G(z))) -> min -log(D(G(z)))

criterion = nn.BCELoss()
# Create labels as 1 (Real) because we want D to think they are real
labels = torch.ones(batch_size, 1).to(device)
output = discriminator(fake_images)
loss_G = criterion(output, labels)
```
