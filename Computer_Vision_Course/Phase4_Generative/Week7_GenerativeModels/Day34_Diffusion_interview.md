# Day 34 Interview Questions: Diffusion Models

## Q1: Why are Diffusion Models slower than GANs?
**Answer:**
*   **GANs:** Single-step generation. $z \to x$. Very fast.
*   **Diffusion:** Multi-step generation. $x_T \to x_{T-1} \to \dots \to x_0$.
*   Requires running the heavy U-Net 50-1000 times to generate one image.

## Q2: What is the "Noise Schedule" $\beta_t$?
**Answer:**
*   It defines how much noise is added at each step of the forward process.
*   Linear schedule: $\beta_t$ increases linearly from $10^{-4}$ to $0.02$.
*   Cosine schedule: Smoother transition, prevents destroying information too quickly.

## Q3: Explain the objective function of DDPM.
**Answer:**
*   It is a re-weighted Variational Lower Bound (ELBO).
*   Simplifies to **Mean Squared Error** between the actual noise $\epsilon$ added to the image and the noise predicted by the network $\epsilon_\theta(x_t, t)$.
*   "Learn to denoise."

## Q4: Why do we need Time Embeddings?
**Answer:**
*   The U-Net parameters are shared across all time steps $t=1 \dots 1000$.
*   The task at $t=1000$ (pure noise) is very different from $t=1$ (clean image).
*   Time embeddings (sinusoidal or learned) tell the network the noise level, allowing it to adapt its function (e.g., focus on coarse structure at high noise, fine texture at low noise).

## Q5: What is the difference between Classifier Guidance and Classifier-Free Guidance?
**Answer:**
*   **Classifier Guidance:** Requires training an extra noisy classifier. Complicates pipeline.
*   **Classifier-Free:** Requires training the diffusion model with and without labels (jointly). Simpler and usually yields better results.

## Q6: How does DDIM speed up sampling?
**Answer:**
*   It redefines the diffusion process to be deterministic (ODE-based) rather than stochastic (SDE-based).
*   This allows taking larger steps (skipping intermediate $t$) without incurring significant discretization error.

## Q7: Why does Diffusion beat GANs on Mode Coverage?
**Answer:**
*   GANs suffer from Mode Collapse (optimizing for "fooling" D, not covering distribution).
*   Diffusion models optimize the Likelihood (ELBO) of the data.
*   They are forced to assign probability mass to *all* training data points, ensuring better diversity.

## Q8: What is the "Reparameterization" in the Forward Process?
**Answer:**
*   Instead of adding noise step-by-step iteratively ($x_0 \to x_1 \to \dots$), we can jump directly to $x_t$.
*   $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$.
*   This allows efficient training (we can sample any random $t$ and train on it instantly).

## Q9: What is "Latent Diffusion"?
**Answer:**
*   Running diffusion on pixels ($512 \times 512$) is expensive.
*   Latent Diffusion (Stable Diffusion) first compresses the image into a small latent space ($64 \times 64$) using a VQ-VAE/KL-VAE.
*   Diffusion is performed in this latent space.
*   Drastically reduces computation.

## Q10: Implement the Forward Diffusion step.
**Answer:**
```python
def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
        
    sqrt_alpha_bar_t = extract(sqrt_alpha_bar, t, x_0.shape)
    sqrt_one_minus_alpha_bar_t = extract(sqrt_one_minus_alpha_bar, t, x_0.shape)
    
    return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
```
