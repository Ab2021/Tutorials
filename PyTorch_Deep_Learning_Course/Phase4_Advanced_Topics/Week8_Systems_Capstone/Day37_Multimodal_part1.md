# Day 37: Multimodal - Deep Dive

> **Phase**: 4 - Advanced Topics
> **Week**: 8 - Systems & Capstone
> **Topic**: Flamingo, SigLIP, and Audio

## 1. Flamingo (DeepMind)

How to fuse vision into a frozen LLM?
**Perceiver Resampler**: Compresses variable number of visual features into fixed number of tokens.
**Gated Cross-Attention**: Injects visual tokens into LLM layers.
*   Initialized with $\tanh(\alpha)$ where $\alpha=0$.
*   Starts as identity function (preserves LLM knowledge). Gradually learns to attend to images.

## 2. SigLIP (Sigmoid Loss for Language Image Pre-training)

CLIP uses Softmax (Categorical Cross Entropy). Requires large batch size (32k) for good negatives.
**SigLIP**: Uses Sigmoid (Binary Cross Entropy) for every pair.
*   Scales better.
*   Doesn't require global sync of negatives.
*   Used in Google's PaliGemma.

## 3. BLIP-2 (Bootstrapping Language-Image Pre-training)

Uses a **Q-Former** (Querying Transformer) to bridge the gap between frozen Image Encoder and frozen LLM.
*   Q-Former extracts text-relevant visual features.
*   Lightweight and efficient.

## 4. Audio Spectrogram Transformer (AST)

Audio is just an image (Spectrogram).
*   Convert Audio $\to$ Mel Spectrogram.
*   Patchify (like ViT).
*   Feed to Transformer.
*   **Whisper**: Encoder-Decoder Transformer trained on 680k hours of audio for ASR.

## 5. Multimodal RAG

Retrieving images based on text, or text based on images.
*   Store images in Vector DB using CLIP embeddings.
*   Query: "Find me the chart showing Q3 revenue".
*   Retrieve image $\to$ Feed to LLaVA $\to$ Answer.
