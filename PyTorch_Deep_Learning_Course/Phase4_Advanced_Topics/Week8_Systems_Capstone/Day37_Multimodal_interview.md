# Day 37: Multimodal Models - Interview Questions

> **Phase**: 4 - Advanced Topics
> **Week**: 8 - Systems & Capstone
> **Topic**: Vision-Language Models, Alignment, and Architecture

### 1. How does CLIP achieve Zero-Shot classification?
**Answer:**
*   It doesn't predict class ID (0-999).
*   It predicts the similarity between the image and the text "A photo of a {label}".
*   We check similarity for all possible labels and pick the highest.

### 2. What is the "Modality Gap"?
**Answer:**
*   Even after alignment, image embeddings and text embeddings often cluster in different regions of the hypersphere.
*   Makes arithmetic (Image - Text) difficult.
*   Solved by better temperature scaling or projection layers.

### 3. Explain "Visual Instruction Tuning" (LLaVA).
**Answer:**
*   Creating a dataset of (Image, Instruction, Response).
*   Fine-tuning an LLM to output the Response given the Image features and Instruction.
*   Turns the LLM into a Chatbot that can "see".

### 4. What is "Perceiver Resampler"?
**Answer:**
*   Component in Flamingo.
*   Takes a variable number of image features (e.g., from a video) and converts them into a fixed number of visual tokens (e.g., 64).
*   Reduces computational cost for the LLM.

### 5. Why do we freeze the LLM in LLaVA/BLIP-2?
**Answer:**
*   To prevent **Catastrophic Forgetting** of language capabilities.
*   To save training cost (only train the connector/adapter).

### 6. What is "Whisper"?
**Answer:**
*   OpenAI's ASR (Automatic Speech Recognition) model.
*   Transformer Encoder-Decoder.
*   Trained on weakly supervised data (internet audio).
*   Robust to accents and noise.

### 7. What is "Stable Diffusion" text encoder?
**Answer:**
*   CLIP ViT-L/14.
*   It provides the text embeddings for the Cross-Attention layers in the U-Net.

### 8. What is "Gated Cross-Attention"?
**Answer:**
*   Used in Flamingo.
*   Adds cross-attention to visual tokens *inside* the LLM layers.
*   Uses a learnable gate initialized to 0, so the model starts by ignoring the image and behaving like a pure LLM.

### 9. What is "ImageBind"?
**Answer:**
*   Meta's model aligning 6 modalities (Image, Text, Audio, Depth, Thermal, IMU) into a single embedding space.
*   Uses Image as the binding modality.

### 10. How do you handle Video in LLMs?
**Answer:**
*   Treat video as a sequence of images.
*   Sample frames (e.g., 8 frames).
*   Encode each frame with ViT.
*   Average the features or use a temporal encoder (Perceiver).

### 11. What is "Tokenization" for Images?
**Answer:**
*   **Patchify**: Splitting image into $16 \times 16$ patches (ViT).
*   **VQ-VAE**: Discrete tokens from a codebook (DALL-E 1).

### 12. What is "Contrastive Loss" (InfoNCE)?
**Answer:**
*   $L = -\log \frac{\exp(sim(i, i)/\tau)}{\sum \exp(sim(i, j)/\tau)}$.
*   Pulls positive pairs together, pushes negative pairs apart.

### 13. What is "Grounding"?
**Answer:**
*   Linking text concepts to specific pixels/bounding boxes in the image.
*   "The man in the red shirt" $\to$ [Box coords].

### 14. What is "OCR-free" document understanding?
**Answer:**
*   Models like Donut or LLaVA that read text directly from the image pixels without an external OCR engine.

### 15. What is "SigLIP"?
**Answer:**
*   Sigmoid Loss for Language Image Pre-training.
*   Replaces Softmax with independent Sigmoids.
*   More memory efficient, scales to larger batch sizes.

### 16. What is "Projection Layer"?
**Answer:**
*   A Linear layer (or MLP) that maps Image Embedding dimension (e.g., 768) to LLM Embedding dimension (e.g., 4096).

### 17. What is "Zero-Shot Object Detection"?
**Answer:**
*   Using CLIP to detect objects.
*   "Find the [red ball]".
*   Models like OWL-ViT or Grounding DINO.

### 18. What is "Multimodal Chain of Thought"?
**Answer:**
*   "Let's think step by step" applied to images.
*   "First, identify the object. Then, read the text on it. Finally, answer the question."

### 19. What is "Fuyu"?
**Answer:**
*   A multimodal model that feeds raw image patches directly into the Linear Projection (no ViT encoder).
*   Simplifies architecture. Supports arbitrary resolutions.

### 20. What is "CLIP-T"?
**Answer:**
*   Text Encoder of CLIP.
*   Often used as a frozen feature extractor for downstream tasks.
