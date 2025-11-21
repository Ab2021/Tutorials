# Day 31: Diffusion Models - Interview Questions

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: Generative AI, Math, and Architecture

### 1. How does Diffusion differ from GANs?
**Answer:**
*   **GAN**: Minimax game between Generator and Discriminator. One-shot generation. Unstable (Mode Collapse).
*   **Diffusion**: Iterative denoising process. Maximizes Likelihood (ELBO). Stable training. Slower inference.

### 2. What is the "Forward Process" in Diffusion?
**Answer:**
*   A Markov chain that gradually adds Gaussian noise to the data according to a variance schedule $\beta_t$.
*   $q(x_t | x_{t-1}) = N(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$.
*   At $T \to \infty$, $x_T$ is pure Isotropic Gaussian noise.

### 3. What does the Neural Network predict in DDPM?
**Answer:**
*   It predicts the **noise** $\epsilon$ that was added to the image at step $t$.
*   Alternatively, it can predict $x_0$ directly, but predicting noise is numerically more stable.

### 4. What is "Latent Diffusion"?
**Answer:**
*   Performing the diffusion process in the compressed latent space of a VAE instead of pixel space.
*   Reduces computational complexity (e.g., $64 \times 64$ instead of $512 \times 512$).
*   Used in Stable Diffusion.

### 5. What is "Classifier-Free Guidance"?
**Answer:**
*   A technique to improve adherence to the text prompt.
*   We run the model twice: once with the prompt, once with an empty prompt.
*   We push the prediction away from the unconditional prediction towards the conditional one.
*   Scale factor $> 1$ (e.g., 7.5).

### 6. Why is Diffusion slow?
**Answer:**
*   It requires iterative refinement (e.g., 50 steps).
*   The U-Net must be run 50 times to generate one image.
*   GANs run only once.

### 7. What is "DDIM"?
**Answer:**
*   Denoising Diffusion Implicit Models.
*   A sampling method that makes the reverse process deterministic (non-Markovian).
*   Allows skipping steps (e.g., 1000 $\to$ 50) with minimal quality loss.

### 8. What is "ControlNet"?
**Answer:**
*   An adapter architecture to add spatial conditioning (Canny edges, Pose, Depth) to diffusion models.
*   Uses "Zero Convolutions" to gradually inject control without breaking the pre-trained model.

### 9. What is the "Reparameterization Trick" in the Forward Process?
**Answer:**
*   Allows sampling $x_t$ directly from $x_0$ without iterating $t$ times.
*   $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$.
*   Crucial for efficient training (we can sample any random $t$).

### 10. What is "CLIP" role in Stable Diffusion?
**Answer:**
*   CLIP Text Encoder converts the text prompt into embeddings.
*   These embeddings are fed into the U-Net via Cross-Attention layers.

### 11. What is "DreamBooth"?
**Answer:**
*   A fine-tuning technique to teach the model a specific subject (e.g., your dog).
*   Uses a rare token identifier (e.g., "a [V] dog") and "Prior Preservation Loss" to prevent forgetting generic dogs.

### 12. What is "LoRA" in the context of Diffusion?
**Answer:**
*   Same as in LLMs. Low-Rank Adaptation applied to the Cross-Attention layers of the U-Net.
*   Allows storing styles/characters in small files (100MB).

### 13. What is "In-painting"?
**Answer:**
*   Restoring or changing a specific part of an image.
*   In Diffusion: We mask the image. During reverse steps, we keep the known pixels fixed (add noise to them) and only let the model generate the masked pixels.

### 14. Why do Diffusion models struggle with hands/text?
**Answer:**
*   **Hands**: Rare in training data to see all 5 fingers clearly separated; high variance in pose.
*   **Text**: The model works in pixel space/latent space, not symbolic space. It tries to "draw" letters based on visual patterns, not spell them.

### 15. What is "Negative Prompt"?
**Answer:**
*   Text provided to the "unconditional" branch of Classifier-Free Guidance.
*   Instead of empty string, we say "ugly, blurry".
*   The model moves *away* from this concept.

### 16. What is "VAE" in Stable Diffusion?
**Answer:**
*   Variational Autoencoder.
*   Encoder compresses image to latent. Decoder reconstructs image from latent.
*   The Diffusion happens in the middle.

### 17. What is "Noise Schedule"?
**Answer:**
*   The function defining $\beta_t$ (Linear, Cosine, Sigmoid).
*   Determines how fast signal is destroyed.
*   Cosine schedule is better for pixel values.

### 18. What is "Distillation" in Diffusion?
**Answer:**
*   Teaching a student model to generate in 1 or 4 steps by mimicking the teacher's multi-step trajectory.
*   Examples: Consistency Models, LCM (Latent Consistency Models).

### 19. What is the U-Net architecture?
**Answer:**
*   Encoder-Decoder with Skip Connections.
*   Preserves high-frequency details (spatial information) via skips.
*   Ideal for image-to-image tasks (noise-to-image is effectively image-to-image).

### 20. What is "Stable Diffusion XL" (SDXL)?
**Answer:**
*   Larger U-Net (2.6B params).
*   Two text encoders (CLIP ViT-L + OpenCLIP ViT-G).
*   Refiner model for high-frequency details.
*   Trained at multiple aspect ratios.
