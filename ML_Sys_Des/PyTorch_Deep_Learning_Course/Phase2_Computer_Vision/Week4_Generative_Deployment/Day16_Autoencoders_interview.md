# Day 16: Autoencoders - Interview Questions

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: VAEs, Latent Space, and Unsupervised Learning

### 1. What is the difference between an Autoencoder and a PCA?
**Answer:**
*   **PCA**: Linear transformation. Finds orthogonal axes of maximum variance. Equivalent to a Linear Autoencoder with MSE loss.
*   **Autoencoder**: Non-linear (ReLU/Sigmoid). Can learn complex, curved manifolds.

### 2. Why do VAEs produce blurry images?
**Answer:**
*   **MSE Loss**: Averages out pixel values. If the model is uncertain between black and white, it predicts gray.
*   **Gaussian Prior**: Forces the latent space to be smooth, preventing sharp transitions in the manifold that correspond to edges.

### 3. Explain the "Reparameterization Trick".
**Answer:**
*   We cannot backpropagate through a random sampling operation $z \sim N(\mu, \sigma)$.
*   Trick: Move the randomness to an external input $\epsilon \sim N(0, 1)$.
*   $z = \mu + \sigma \cdot \epsilon$.
*   Now $z$ is a deterministic function of $\mu$ and $\sigma$ (which have gradients) and a constant $\epsilon$.

### 4. What is "Posterior Collapse" in VAEs?
**Answer:**
*   When the decoder ignores the latent code $z$ and models the data solely based on its own autoregressive power (if using an RNN/PixelCNN decoder).
*   The KL term drives $Q(z|x)$ to equal the prior $P(z)$, so $z$ carries no information about $x$.

### 5. What is the purpose of the KL Divergence term in VAE loss?
**Answer:**
*   Regularization.
*   It forces the learned latent distribution to be close to a Standard Normal $N(0, 1)$.
*   This ensures the latent space is dense and continuous, allowing for valid sampling/generation.

### 6. How does VQ-VAE differ from standard VAE?
**Answer:**
*   **Discrete Latent Space**: Uses a codebook of vectors instead of a continuous distribution.
*   **Deterministic**: No sampling noise during inference.
*   **Sharpness**: Produces much sharper images because it avoids the averaging effect of Gaussian priors.

### 7. What is "Denoising Autoencoder"?
**Answer:**
*   Input: Corrupted image (Noise, Dropout, Masking).
*   Target: Clean image.
*   Forces the model to learn the structure of the data manifold to "project" the noisy point back onto the manifold.

### 8. What is the "Bottleneck"?
**Answer:**
*   The layer with the smallest dimension in the AE.
*   Forces the model to compress information, preventing it from simply learning the Identity function (copy-paste).

### 9. Explain "ELBO" (Evidence Lower Bound).
**Answer:**
*   The objective function maximized in Variational Inference.
*   Since we can't compute the true log-likelihood $\log P(x)$, we compute a lower bound.
*   Maximizing ELBO is equivalent to minimizing the KL divergence between the approximate posterior and the true posterior.

### 10. What is "Disentanglement"?
**Answer:**
*   The property where single latent units are sensitive to changes in single generative factors (e.g., one neuron controls smile, another controls hair color).
*   $\beta$-VAE promotes this.

### 11. How do you use an Autoencoder for Anomaly Detection?
**Answer:**
*   Train AE on normal data.
*   At test time, compute Reconstruction Error $||x - \hat{x}||^2$.
*   High error $\implies$ The model has never seen this pattern $\implies$ Anomaly.

### 12. What is a "Sparse Autoencoder"?
**Answer:**
*   Adds a sparsity penalty (L1 regularization) to the activations of the hidden layer.
*   Forces the model to represent data using a small number of active neurons.
*   Alternative to tight bottlenecks.

### 13. Why is the VAE latent space "Continuous"?
**Answer:**
*   Because we train with noise ($z = \mu + \sigma \epsilon$).
*   The decoder must be able to reconstruct $x$ not just from $\mu$, but from the neighborhood $\mu \pm \sigma$.
*   This forces similar inputs to map to nearby regions in latent space.

### 14. What is "Straight-Through Estimator"?
**Answer:**
*   Used in VQ-VAE to backpropagate through the non-differentiable `argmin` quantization.
*   Forward: $z_q$. Backward: Gradient of $z_e$ is set to Gradient of $z_q$ (Identity pass-through).

### 15. Can Autoencoders be used for dimensionality reduction?
**Answer:**
*   Yes. The bottleneck layer serves as the reduced representation.
*   Often better than PCA for non-linear data.

### 16. What is "Contractive Autoencoder"?
**Answer:**
*   Adds a penalty on the Jacobian of the encoder.
*   Forces the latent representation to be invariant to small perturbations in input.

### 17. How does Stable Diffusion use Autoencoders?
**Answer:**
*   It uses a VQ-VAE (or KL-VAE) to compress images from Pixel Space ($512 \times 512$) to Latent Space ($64 \times 64$).
*   The Diffusion process happens in this compressed Latent Space (Latent Diffusion), which is much faster.

### 18. What is the "Reconstruction Loss"?
**Answer:**
*   Measures difference between Input and Output.
*   Images: MSE or Binary Cross Entropy (if pixels normalized to [0,1]).

### 19. Why is VAE a "Generative" model?
**Answer:**
*   Because we can sample new data.
*   Sample $z \sim N(0, 1)$, pass through Decoder $\to$ New Image.
*   Standard AE cannot do this reliably because we don't know the distribution of the latent space.

### 20. What is "Masked Autoencoder" (MAE)?
**Answer:**
*   State-of-the-art Self-Supervised learner.
*   Masks 75% of image patches. Encoder sees only 25%. Decoder reconstructs pixels.
*   Learns very strong representations.
