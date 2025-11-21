# Day 39: Multimodal Vision (CLIP & VLMs)

## 1. The Shift to Multimodal
**Old Way:** Train on ImageNet (Fixed 1000 classes).
**New Way:** Train on Image-Text pairs (Web scale). Learn to align Visual and Textual representations.
**Benefit:** Zero-Shot Transfer. Can classify *any* concept described in text.

## 2. CLIP (Contrastive Language-Image Pre-training)
**Architecture:**
1.  **Image Encoder:** ViT or ResNet. $I \to E_I$.
2.  **Text Encoder:** Transformer. $T \to E_T$.
3.  **Projection:** Map both to shared embedding space (dim=512).
**Training:**
*   Batch of $N$ pairs $(I_i, T_i)$.
*   Maximize cosine similarity of diagonal $(I_i, T_i)$ (Positives).
*   Minimize similarity of off-diagonal $(I_i, T_j)$ (Negatives).
*   **Contrastive Loss:** InfoNCE.

## 3. Zero-Shot Classification with CLIP
How to classify an image without training?
1.  **Prompt Engineering:** Create text prompts for each class: "A photo of a {dog}", "A photo of a {cat}".
2.  **Encode:** Pass prompts through Text Encoder $\to$ $T_{dog}, T_{cat}$.
3.  **Compare:** Pass image through Image Encoder $\to$ $I$.
4.  **Predict:** Class = $\arg \max (I \cdot T_{class})$.

```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("dog.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
```

## 4. Flamingo (DeepMind)
**Goal:** Visual Question Answering (VQA) and Captioning.
*   **Architecture:** Frozen Vision Encoder + Frozen LLM (Chinchilla).
*   **Perceiver Resampler:** Compresses variable number of visual features into fixed number of tokens.
*   **Gated Cross-Attention:** Injects visual tokens into the LLM layers.
*   **Few-Shot:** Can learn a new task (e.g., "Count the coins") from just a few examples in the prompt.

## Summary
CLIP bridged the gap between Computer Vision and NLP, enabling models that "understand" images in the context of natural language.
