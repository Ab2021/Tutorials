# Day 99: Multimodal AI Fundamentals
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a CLIP Search Engine

We will build a semantic search engine that finds images using text.

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Load Model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Prepare Data
images = [Image.open("dog.jpg"), Image.open("cat.jpg")]
text = ["A photo of a dog", "A photo of a cat", "A photo of a car"]

# 3. Process
inputs = processor(text=text, images=images, return_tensors="pt", padding=True)

# 4. Forward Pass
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # [2, 3]
probs = logits_per_image.softmax(dim=1)

print("Probabilities:", probs)
# Output:
# Image 0 (Dog): [0.99, 0.01, 0.00] -> Matches "A photo of a dog"
```

### LLaVA Architecture (Visual Instruction Tuning)

How LLaVA works:
1.  **Vision Encoder:** CLIP ViT-L/14 (Frozen).
2.  **Projection:** A simple Linear Layer (Trainable).
3.  **LLM:** Vicuna (Llama-2 fine-tune) (Trainable).
4.  **Training:**
    *   **Stage 1 (Pre-training):** Train Projection only. Align features.
    *   **Stage 2 (Fine-tuning):** Train Projection + LLM on "Visual Chat" data.

### Multimodal RAG

Retrieving images.
1.  **Index:** Embed all images in your PDF using CLIP. Store in Vector DB.
2.  **Query:** User asks "Show me the chart about revenue."
3.  **Retrieve:** Embed query -> Find closest Image Embedding.
4.  **Generate:** Pass image to GPT-4V to explain it.

### Summary

*   **Embeddings:** The bridge between modalities.
*   **Freezing:** We often freeze the massive Vision Encoder and LLM, and only train the "Glue" (Projection Layer) to save compute.
