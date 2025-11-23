# Day 73: Multi-Modal RAG & Agents
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Multi-Modal RAG Implementation (Image + Text)

Indexing images and text, then retrieving both.

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

# 1. Load CLIP
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
tokenizer = CLIPTokenizer.from_pretrained(model_id)

class MultiModalIndex:
    def __init__(self):
        self.image_vectors = []
        self.images = [] # Store paths or blobs
        
    def add_image(self, image_path):
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # Normalize
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        self.image_vectors.append(image_features)
        self.images.append(image_path)
        
    def search(self, text_query, k=1):
        inputs = tokenizer([text_query], padding=True, return_tensors="pt")
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Cosine Similarity
        scores = []
        for img_vec in self.image_vectors:
            score = (text_features @ img_vec.T).item()
            scores.append(score)
            
        # Top K
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.images[i] for i in top_indices]

# Usage
# index = MultiModalIndex()
# index.add_image("chart.png")
# result = index.search("Show me the sales chart")
```

### 2. VLM Agent (Screen Parser)

Using GPT-4o to analyze a screenshot and decide action.

```python
import base64
import requests

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def screen_agent_step(screenshot_path, goal):
    base64_image = encode_image(screenshot_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Goal: {goal}. What should I click? Return JSON {{x, y}}."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

# Usage
# action = screen_agent_step("desktop.png", "Open Chrome")
# print(action) # Output: {"x": 100, "y": 200}
```

### 3. PDF Image Extractor (Unstructured)

Extracting images from PDF for indexing.

```python
from unstructured.partition.pdf import partition_pdf

def extract_images_from_pdf(pdf_path):
    # Extracts elements including images
    elements = partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path="./extracted_images"
    )
    
    return elements

# Usage
# elements = extract_images_from_pdf("report.pdf")
# for el in elements:
#     if el.category == "Image":
#         print(f"Found image: {el.metadata.image_path}")
```

### 4. ColPali Concept (Late Interaction)

Conceptual implementation of multi-vector retrieval.

```python
# ColPali represents an image as a BAG of vectors (patches), not a single vector.
# Query is also a BAG of vectors (tokens).
# Score = Sum_over_query_tokens(Max_over_image_patches(similarity))

def max_sim_operator(query_vectors, image_patch_vectors):
    # query_vectors: [Q, D]
    # image_patch_vectors: [P, D]
    
    # Similarity Matrix: [Q, P]
    sim_matrix = query_vectors @ image_patch_vectors.T
    
    # Max over patches (dim 1)
    max_scores = torch.max(sim_matrix, dim=1).values
    
    # Sum over query tokens
    total_score = torch.sum(max_scores)
    
    return total_score
```
