# Day 31 Interview Questions: Autoencoders

## Q1: What is the "Reparameterization Trick" and why is it needed?
**Answer:**
*   In VAEs, we need to sample $z$ from a distribution $\mathcal{N}(\mu, \sigma^2)$.
*   Sampling is a stochastic operation and is **not differentiable**. We cannot backpropagate through a random node.
*   **Trick:** Express $z$ as a deterministic function of inputs and noise: $z = \mu + \sigma \cdot \epsilon$.
*   Now, gradients can flow through $\mu$ and $\sigma$. The randomness is offloaded to $\epsilon$ (which doesn't need gradients).

## Q2: Why do Autoencoders have a "bottleneck"?
**Answer:**
*   If the latent dimension is equal to or larger than the input dimension, the AE can simply learn the Identity function (copy input to output) without learning any useful features.
*   The bottleneck forces the network to compress the information, keeping only the most important variations (like PCA).

## Q3: What is the difference between AE and VAE?
**Answer:**
*   **AE:** Deterministic. Maps input to a single point in latent space. Latent space is irregular. Good for compression/denoising.
*   **VAE:** Probabilistic. Maps input to a distribution. Latent space is smooth/continuous. Good for generation (sampling).

## Q4: Explain the KL Divergence term in VAE loss.
**Answer:**
*   It acts as a regularizer.
*   It penalizes the encoder if the learned distribution $q(z|x)$ deviates too much from the standard normal $\mathcal{N}(0, 1)$.
*   Without it, the encoder would cheat by placing distributions far apart to minimize overlap (reconstruction error), destroying the ability to sample new data.

## Q5: How does VQ-VAE handle the non-differentiable quantization step?
**Answer:**
*   The `argmin` operation is not differentiable.
*   **Straight-Through Estimator:**
    *   Forward pass: Use quantized values $z_q$.
    *   Backward pass: Copy gradients from decoder input $z_q$ directly to encoder output $z_e$, bypassing the quantization step.
    *   $\frac{\partial L}{\partial z_e} \approx \frac{\partial L}{\partial z_q}$.

## Q6: Can Autoencoders be used for Anomaly Detection?
**Answer:**
**Yes.**
*   Train AE on "Normal" data only.
*   At test time, pass a new sample through the AE.
*   If the **Reconstruction Error** is high, it means the AE has never seen this pattern before $\to$ Anomaly.

## Q7: What is "Posterior Collapse" in VAEs?
**Answer:**
*   When the decoder is too powerful (e.g., an RNN or PixelCNN), it ignores the latent code $z$ and generates data purely from its own autoregressive history.
*   The KL loss drops to 0, and $z$ becomes useless noise.
*   **Fix:** KL Annealing (start with $\beta=0$, slowly increase) or weaken the decoder.

## Q8: Implement the VAE Loss function.
**Answer:**
```python
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD
```

## Q9: Why use Log Variance instead of Variance?
**Answer:**
*   Variance $\sigma^2$ must be positive.
*   Predicting $\sigma^2$ directly requires an activation like ReLU/Softplus, which can be unstable.
*   Predicting $\log(\sigma^2)$ allows the network to output any real number $(-\infty, \infty)$, which is numerically more stable. We then exponentiate it to get positive variance.

## Q10: What is a Sparse Autoencoder?
**Answer:**
*   An AE with a sparsity penalty on the latent activations (e.g., L1 regularization).
*   Forces most neurons in the bottleneck to be zero for any given input.
*   Learns a basis set of features (like edges in V1 visual cortex).
