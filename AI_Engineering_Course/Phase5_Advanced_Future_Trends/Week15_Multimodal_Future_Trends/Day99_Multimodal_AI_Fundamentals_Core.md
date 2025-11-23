# Day 99: Multimodal AI Fundamentals
## Core Concepts & Theory

### Beyond Text

The world is not just text. It is Images, Audio, Video, and Sensors.
**Multimodal AI** processes multiple modalities simultaneously.
*   **VLM (Vision-Language Model):** Text + Image (e.g., GPT-4V, LLaVA).
*   **Audio-Language:** Text + Audio (e.g., Whisper, AudioLM).

### 1. The Alignment Problem

How do you make an LLM understand an image?
*   **Projection:** You cannot feed pixels to a Transformer. You feed **Embeddings**.
*   **Vision Encoder:** Use a CNN (ResNet) or ViT to turn an image into a vector $V_{img}$.
*   **Projection Layer:** A linear layer $W$ that maps $V_{img}$ to the same dimension as the Text Embeddings $V_{text}$.
*   **Result:** The LLM sees the image as just another "token".

### 2. CLIP (Contrastive Language-Image Pre-training)

The foundation of modern Multimodal AI (OpenAI, 2021).
*   **Idea:** Train a Text Encoder and an Image Encoder jointly.
*   **Objective:** Maximize the cosine similarity between the image of a "Dog" and the text "A photo of a dog".
*   **Result:** A shared latent space where text and images live together.

### 3. Vision Transformers (ViT)

Replacing CNNs with Transformers.
*   **Patching:** Cut the image into $16 \times 16$ patches.
*   **Flattening:** Treat each patch as a token.
*   **Attention:** Apply self-attention to patches. (Global context from layer 1).

### 4. Flamingo & LLaVA

*   **Flamingo (DeepMind):** Interleaved text and images. Used "Perceiver Resampler" to handle variable number of images.
*   **LLaVA (Large Language and Vision Assistant):** Fine-tuned Llama-2 on GPT-4 generated image-text pairs.

### Summary

Multimodal AI is the path to AGI. It grounds the model in the physical world. "A picture is worth 1000 tokens."
