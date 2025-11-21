# Day 31: Diffusion Models - Deep Dive

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: Schedulers, CFG, and ControlNet

## 1. Classifier-Free Guidance (CFG)

How to make the image follow the prompt strongly?
Train the model with dropout on the text condition (10% of time, prompt is empty string).
During inference:
$$ \epsilon_{final} = \epsilon_\theta(x_t, \text{empty}) + w \cdot (\epsilon_\theta(x_t, \text{text}) - \epsilon_\theta(x_t, \text{empty})) $$
*   $w$: Guidance Scale (usually 7.5).
*   Extrapolates the direction from "unconditional" to "conditional".

## 2. Schedulers (Samplers)

DDPM requires 1000 steps. Too slow.
**DDIM (Denoising Diffusion Implicit Models)**:
*   Deterministic sampling.
*   Skips steps. Can generate in 50 steps.
**Euler Ancestral / DPM++**:
*   Treats the reverse process as an ODE (Ordinary Differential Equation).
*   Uses advanced solvers to reach the solution in 20 steps.

## 3. ControlNet

Adding spatial conditioning (Edges, Pose, Depth).
*   Freeze the original Stable Diffusion U-Net.
*   Create a trainable copy of the Encoder blocks.
*   Add "Zero Convolutions" (initialized to 0) to connect them.
*   Allows controlling the structure of the generated image.

## 4. VAE (Variational Autoencoder) Role

The VAE in Stable Diffusion is crucial.
*   **KL-Divergence Regularization**: Ensures the latent space is smooth (Gaussian).
*   **Magical Number 8**: The compression factor is usually 8 ($512 \to 64$).
*   If VAE is bad, faces look distorted and text is unreadable.

## 5. Noise Prediction vs Velocity Prediction

*   **Epsilon-Prediction**: Predict the noise $\epsilon$. Standard.
*   **V-Prediction**: Predict velocity $v$. Better for distillation and video.
