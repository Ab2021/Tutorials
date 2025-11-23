# Lab 1: Multimodal RAG

## Objective
Search for images using text.
We will use **CLIP**.

## 1. Setup

```bash
pip install transformers torch pillow
```

## 2. The Search Engine (`clip_search.py`)

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# 1. Load Model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Index Images
images = [Image.open("cat.jpg"), Image.open("dog.jpg")] # Ensure these exist
inputs = processor(images=images, return_tensors="pt", padding=True)
image_embeddings = model.get_image_features(**inputs)

# 3. Query
query = "A cute puppy"
text_inputs = processor(text=[query], return_tensors="pt", padding=True)
text_embeddings = model.get_text_features(**text_inputs)

# 4. Similarity
sim = torch.cosine_similarity(text_embeddings, image_embeddings)
best_match_idx = sim.argmax().item()

print(f"Best match for '{query}': Image {best_match_idx}")
```

## 3. Submission
Submit the index of the matching image.
